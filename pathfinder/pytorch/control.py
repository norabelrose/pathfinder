from astropy.constants import g0
from astropy.units import Quantity, quantity_input
from collections import namedtuple
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from torch import nn, Tensor
from torchdiffeq import odeint, odeint_adjoint
from typing import Tuple, Union

from .ephemeris import Ephemeris

import astropy.units as u
import torch


Trajectory = namedtuple('Trajectory', ['r', 'v', 'm', 'lr', 'lv', 'lm', 'u'])

class Control(nn.Module):
    # t0 is the initial launch date in Modified Julian Days
    @quantity_input(isp='s', max_thrust='N', wet_mass='kg', dry_mass='kg')
    def __init__(
        self,
        source: Union[Body, Orbit],
        target: Union[Body, Orbit],
        isp: Quantity,
        max_thrust: Quantity,
        wet_mass: Quantity,
        dry_mass: Quantity,

        t0: float,
        tof: float,
        costate0: Tensor = None,
        alpha: float = 0.0,
        bounded: bool = True
    ):
        super().__init__()

        self.isp = isp
        self.max_thrust = max_thrust
        self.wet_mass = torch.from_numpy(wet_mass)
        self.dry_mass = torch.from_numpy(dry_mass)

        self._max_thrust = torch.from_numpy(max_thrust.to('au*kg/d**2')) / self.wet_mass
        self._v_e = torch.from_numpy((isp * g0).to(u.au / u.d)).view(1)

        attractors = {x.parent if isinstance(x, Body) else x.attractor for x in (source, target)}
        assert len(attractors) == 1, "Source and target must orbit around the same attractor"
        self.bodies = list(attractors | {x for x in (source, target) if isinstance(x, Body)})
        self.mus = [torch.from_numpy(body.k.to('au**3/d**2')) for body in self.bodies]
        self.radii = [torch.from_numpy(body.R.to('au')) for body in self.bodies]

        self.attractor = Ephemeris(attractors.pop())
        self.source = Ephemeris(source)
        self.target = Ephemeris(target)

        # Figuring out a good initial guess for the costate variables is tricky.
        # λr is simply randomly initialized, and the dynamics don't seem to be
        # particularly sensitive to it. We initialize λv and λm so that the craft
        # will start the trajectory with prograde maximum thrust.
        if costate0 is None:
            _, v0 = self.source(t0)

            costate0 = torch.cat([
                torch.randn(3),
                v0 / v0.norm(),
                self._v_e * 2.0
            ])
        
        t0 = torch.tensor(t0) if not torch.is_tensor(t0) else t0
        tof = torch.tensor(tof) if not torch.is_tensor(tof) else tof
        self.alpha = alpha
        self.bounded = bounded
        self.costate0 = nn.Parameter(costate0)
        self.t0 = nn.Parameter(t0)
        self.log_tof = nn.Parameter(tof.log())
    
    # We store the time of flight in logspace under the hood to ensure it stays nonnegative
    @property
    def tof(self) -> Tensor:
        return self.log_tof.exp()
    
    def propagate(self, *, num_nodes: int = 100):
        state, costate = odeint(
            func=self,
            t=torch.linspace(0.0, 1.0, num_nodes),
            y0=(
                # State and co-state are both 7D state vectors of the form (position, velocity, mass)
                torch.cat([*self.source(self.t0), torch.tensor([1.0])]),
                self.costate0
            ),
            method='dopri5',
            atol=1e-6,
            rtol=1e-3,
            options=dict(
                first_step=1e-3
            )
        )
        
        λr, λv, λm = costate[:3], costate[3:6], costate[6]
        r, v, m = state[:3], state[3:6], state[6]

        u_mag, u_hat = self.get_controls(costate, m)
        u = u_mag * u_hat

        return Trajectory(r, v, m * self.wet_mass, λr, λv, λm, u)
    
    # Returns (thrust magnitude, thrust direction) tuple
    def get_controls(self, costate: Tensor, mass: Tensor) -> Tuple[Tensor, Tensor]:
        return _get_controls(costate, mass, self._v_e, self._max_thrust, self.alpha, self.bounded)
    
    def forward(self, t_hat: Tensor, y: Tuple[Tensor, Tensor]) -> Tensor:
        print(f"{t_hat=}")
        
        # Debugging
        x, costate = y

        # Re-dimensionalize
        tof = self.tof
        t = self.t0 + t_hat * tof
        body_positions = [body(t)[0] for body in (self.attractor, self.source, self.target)]

        # The odeint function will call us with torch.no_grad, but we need to compute the
        # gradient of the Hamiltonian wrt the state as part of the basic dynamics
        grad_was_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            x = x.detach().requires_grad_()

            # Decode position, velocity, and mass
            rv, m = x[:6], x[6]
            r, v = rv[:3], rv[3:]

            # Compute gravity
            separations = (body_pos - r for body_pos in body_positions)
            gravity = sum(
                sep * mu / sep.norm(dim=0, keepdim=True).maximum(radius) ** 3
                for sep, mu, radius in zip(separations, self.mus, self.radii)
            )
            #assert gravity.isfinite().all(), "Got infinite gravity value"

            thrust_mag, thrust_hat = self.get_controls(costate, m)
            thrust = thrust_mag * thrust_hat
            thrust_acc = torch.where(m > 0.0, thrust / m, 0.0)      # Thrust must be zero when we have zero mass

            # Compute time derivative of the state
            x_dot = torch.cat([
                v,                                                  # Change of position = velocity
                gravity + thrust_acc,                               # Change of velocity = acc. due to gravity + acc. due to thrust
                torch.where(m > 0.0, -thrust_mag / self._v_e, 0.0)   # Change of mass = -thrust / exhaust velocity
            ]) * tof
            hamiltonian = torch.sum(costate * x_dot) - thrust_mag.sum()

            # Identity: dλ/dt = -dH/dx.
            lambda_dot = -torch.autograd.grad(hamiltonian, x, create_graph=grad_was_enabled, retain_graph=grad_was_enabled)[0]

        return torch.cat([x_dot, lambda_dot])


@torch.jit.script
def _get_controls(costate: Tensor, mass: Tensor, v_e: Tensor, max_thrust: Tensor, alpha: float, bounded: bool):
    # The instantaneous thrust that maximizes the Hamiltonian can be decomposed into two components: its magnitude
    # and its direction. By the Cauchy-Schwartz inequality, the unit vector û that maximizes the inner product
    # <λv, û> is simply λv / ||λv||.
    λv_vec = costate[3:6]
    λv, λm = λv_vec.norm(dim=0, keepdim=True, p=2), costate[None, 6]
    thrust_hat = torch.where(λv != 0.0, λv_vec / λv, 0.0)

    if alpha < 1.0:
        thrust_mag = (-λv + mass * λm / v_e - alpha) / (2 * (1 - alpha))
        if bounded:
            thrust_mag.clamp_(-1.0, 1.0)
        
        # Normalize so that the magnitude is always positive
        thrust_hat *= thrust_mag.sign()
        thrust_mag.abs_()
    
    # Bang-bang (bounded) control: maximizes the Hamiltonian by computing its derivative wrt the thrust
    # magnitude, then use 0 thrust if dH/dT is negative and use the maximum thrust if dH/dT is positive
    else:
        dH_dT = 1 + λv - mass * λm / v_e
        thrust_mag = torch.where(dH_dT < 0.0, 0.0, 1.0)
    
    thrust_mag *= max_thrust
    return thrust_mag, thrust_hat
