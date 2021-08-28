from astropy.constants import g0
from astropy.time import Time
from astropy.units import def_unit, Quantity, quantity_input, Unit
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from astropy.units.core import Unit
from numpy.typing import ArrayLike
from poliastro.bodies import Body
from poliastro.ephem import Ephem
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import farnocchia
from torch import Tensor
from typing import Optional, Tuple, Union
import numpy as np
import scipy
import torch


Trajectory = namedtuple('Trajectory', ['r', 'v', 'm', 'lr', 'lv', 'lm', 'u', 't'])


@quantity_input(isp='s', max_thrust='N', wet_mass='kg', max_tof='d')
@dataclass
class LowThrustTransfer:
    source: Union[Body, Orbit]
    target: Union[Body, Orbit]

    isp: Quantity
    max_thrust: Quantity

    wet_mass: Quantity
    dry_mass: Optional[Quantity] = None     # Functions as a hard lower bound on the final mass

    start: Time = field(default_factory=lambda: Time(Time.now(), scale='tdb'))
    end: Optional[Time] = None

    def __post_init__(self):
        self.attractor = self.target.parent if isinstance(self.target, Body) else self.target.attractor
        self.r0, self.v0 = _get_rv_generic(self.source, self.start)

        # Define "natural" units to perform calculations in
        self.units = {}
        mass = def_unit('M', self.wet_mass, namespace=self.units)
        pos = Unit('au')
        vel = pos / Unit('d')
        acc = vel ** 2 / pos

        self.units['G'] = acc * pos ** 2                 # Gravitational parameter
        self.units['T'] = acc * mass                     # Thrust

        # Gravitational parameters in our custom units
        self.bodies = [x for x in (self.attractor, self.source, self.target) if isinstance(x, Body)]
        self._mus = [torch.from_numpy(body.k.to(self.units['G'])) for body in self.bodies]
        self._radii = [torch.from_numpy(body.R.to('au')) for body in self.bodies]   # Avoid gravitational singularities

        # Exhaust velocity & max thrust in natural units
        self._min_mass = self.dry_mass.to(self.units['M']).value if self.dry_mass is not None else None
        self._max_thrust = torch.tensor(self.max_thrust.to(self.units['T']).value)
        self._t0 = self.start.mjd
        self._v_e = torch.tensor((self.isp * g0).to('au/d').value)

    # Use Pontryagin's principle to turn this functional optimization problem into a BVP
    def compute(self, *, alpha: float = 0.0, bounded: bool = False):
        # Since we are using the magnitudes of the initial position and velocity vectors to define our
        # natural units, these should be unit vectors
        r0, v0 = self.r0.to('au'), self.v0.to('au/d')
        
        # Boundary conditions
        def boundary(state_a, state_b, p):
            r_a, r_b = state_a[:3], state_b[:3]
            v_a = state_a[3:6]

            # Re-dimensionalize
            t0, tof = p
            tf = Time(t0, format='mjd', scale='tdb') + tof
            rf = _get_rv_generic(self.target, tf)[0].to('au').value

            # _, _, hamiltonian = self.dynamics(tof, state_b, tof, alpha=alpha, bounded=bounded)
            return np.concatenate([
                r_a - r0.value,     # Actually start at the source location
                r_b - rf,           # Actually land where we're trying to reach
                v_a - v0.value,     # Start out with the correct velocity
                state_b[7:],        # Co-states should equal zero at termination
                # state_b[6, None] - 0.1,
                # hamiltonian         # Transversality condition: H(t_f) = 0
            ])
        
        # Initialize the BVP mesh with the solution of an IVP
        sol_init = self.forward(200.0, alpha=alpha, bounded=bounded)
        sol = scipy.integrate.solve_bvp(
            fun=partial(self.f, alpha=alpha, bounded=bounded),
            bc=boundary,
            x=sol_init.t,
            y=np.concatenate([
                sol_init.r.value,
                sol_init.v.to('au/d').value,
                sol_init.m[None].to(self.units['M']).value,
                sol_init.lr,
                sol_init.lv,
                sol_init.lm[None]
            ]),
            p=np.array([self._t0, 200.0])
        )
        return sol
    
    def backward(self, tof: float, *, alpha: float = 0.0, bounded: bool = False, num_nodes: int = 100) -> Trajectory:
        r0, v0 = self.r0.to('au'), self.v0.to('au/d')

        # Initialize with an impulsive Lambert trajectory
        end = self.end or self.start + tof
        target_orbit = _to_orbit(self.target, end)
        maneuver = Maneuver.lambert(
            _to_orbit(self.source, self.start),
            target_orbit
        )
        impulse = maneuver.impulses[0][1]
        transfer = Orbit.from_vectors(self.attractor, r0.squeeze(), v0.squeeze() + impulse)
        transfer = transfer.propagate(end)
        rf, vf = transfer.rv()

        # Don't actually initialize at the center of the target body, but right on the
        # edge of its gravitational sphere of influence
        if isinstance(self.target, Body):
            soi = target_orbit.a * (self.target.mass / self.attractor.mass) ** (2 / 5)
            rf = rf - soi * vf / np.linalg.norm(vf)
            print(f"Approximating sphere of influence of {self.target.name} as {soi.to('au'):.5f} ({soi.to('km'):.2f})")

        # Use Tsiolkovsky rocket equation to guess final mass
        mass_ratio = np.exp(np.linalg.norm(impulse) / (self.isp * g0))
        mf = (1 / mass_ratio)
        print(f"Guess for final mass as fraction of initial mass: {mf}")

        sol = scipy.integrate.solve_ivp(
            fun=partial(self.f, tof=tof, alpha=alpha, bounded=bounded),
            t_span=(1.0, 0.0),
            t_eval=np.linspace(1.0, 0.0, num_nodes),
            y0=np.concatenate([
                rf.to('au').value,
                vf.to('au/d').value,
                np.array([mf]),
                np.random.randn(7) * 1e-3
            ])
        )
        y0 = sol.y
        costate = y0[7:]
        lr, lv, lm = costate[:3], costate[3:6], costate[6]
        u_mag, u_hat = self.get_controls(torch.from_numpy(costate), torch.from_numpy(y0[6]), alpha, bounded)
        u = (u_mag * u_hat) << self.units['T']

        r, v, m = y0[:3] << Unit('au'), y0[3:6] << Unit('au/d'), y0[6] << self.units['M']
        print(f"Source position is off by {np.linalg.norm(r[:, -1] - r0).to('au')}")
        return Trajectory(
            r, v.to('m/s'), m.to('kg'), lr, lv, lm, u.to('N'), sol.t
        )
    
    def forward(self, tof: float, *, alpha: float = 0.0, bounded: bool = True, num_nodes: int = 100) -> Trajectory:
        r0, v0 = self.r0.to('au'), self.v0.to('au/d')
        thrust_hat = v0 / np.linalg.norm(v0)

        # Don't actually initialize at the center of the source body, but right on the
        # edge of its gravitational sphere of influence
        if isinstance(self.source, Body):
            end = self.end or self.start + tof
            target_orbit = _to_orbit(self.target, end)
            
            soi = target_orbit.a * (self.source.mass / self.attractor.mass) ** (2 / 5)
            r0 = r0 + soi * v0 / np.linalg.norm(v0)
            print(f"Approximating sphere of influence of {self.source.name} as {soi.to('au'):.5f} ({soi.to('km'):.2f})")
        
        # Terminate if we run out of fuel
        kwargs = dict(method='RK45')
        if self._min_mass is not None:
            event = lambda t, y: y[6] - self._min_mass
            event.terminal = True
            kwargs['events'] = event
        
        sol = scipy.integrate.solve_ivp(
            fun=partial(self.f, p=np.array([self._t0, tof]), alpha=alpha, bounded=bounded),
            t_span=(0.0, 1.0),
            t_eval=np.linspace(0.0, 1.0, num_nodes),
            y0=np.concatenate([
                r0.value,
                v0.value,
                np.array([1.0]),
                np.random.randn(3),
                thrust_hat.value,
                self._v_e.reshape(1) * 2.0
            ]),
            **kwargs
            # method='DOP853'
        )
        if events := sol.t_events:
            fuel_exhaustion = events[0]
            if len(fuel_exhaustion):
                assert len(fuel_exhaustion) == 1
                print(f"Note: Ran out of fuel after {fuel_exhaustion.squeeze() * tof << Unit('d'):.2f}")
        
        y0 = sol.y
        costate = y0[7:]
        lr, lv, lm = costate[:3], costate[3:6], costate[6]
        u_mag, u_hat = self.get_controls(torch.from_numpy(costate), torch.from_numpy(y0[6]), alpha, bounded)
        u = (u_mag * u_hat) << self.units['T']

        r, v, m = y0[:3] << Unit('au'), y0[3:6] << Unit('au/d'), y0[6] << self.units['M']
        print(f"Target position is off by {np.linalg.norm(r[:, -1] - target_orbit.r).to('au'):.5f}")
        return Trajectory(
            r.to('km'), v.to('km/s'), m.to('kg'), lr, lv, lm, u.to('N'), sol.t
        )
    
    # Dynamics for both the state and co-states
    def dynamics(
        self, t_prime: np.ndarray, state: np.ndarray, p: np.ndarray, alpha: float = 0.0, bounded: bool = True, require_grad = False
    ):
        # Wrap the state as a PyTorch tensor to use autograd
        state = torch.from_numpy(state)
        x, costate = state[:7], state[7:]
        x.requires_grad = require_grad

        # Decode position, velocity, and mass
        rv, m = x[:6], x[6]
        r, v = rv[:3], rv[3:]

        # Re-dimensionalize
        t0, tof = p
        t = Time(t0, format='mjd', scale='tdb') + t_prime * tof

        # Compute gravity
        separations = (torch.from_numpy(_get_rv_generic(body, t)[0].to('au').value).T - r for body in self.bodies)
        gravity = sum(
            sep * mu / sep.norm(dim=0, keepdim=True).maximum(radius) ** 3
            for sep, mu, radius in zip(separations, self._mus, self._radii)
        )
        assert gravity.isfinite().all(), "Got infinite gravity value"

        thrust_mag, thrust_hat = self.get_controls(costate, m, alpha, bounded)
        thrust = thrust_mag * thrust_hat
        thrust_acc = torch.where(m > 0.0, thrust / m, 0.0)      # Thrust must be zero when we have zero mass

        # Compute time derivative of the state
        x_dot = torch.cat([
            v,                                                  # Change of position = velocity
            gravity + thrust_acc,                               # Change of velocity = acc. due to gravity + acc. due to thrust
            torch.where(m > 0.0, -thrust_mag / self._v_e, 0.0)  # Change of mass = -thrust / exhaust velocity
        ]) * torch.tensor(tof)
        hamiltonian = torch.sum(costate * x_dot) - thrust_mag.sum()
        return x, x_dot, hamiltonian
    
    # Function passed to the integrator; returns time derivatives of state and co-states
    def f(self, t_prime, state, p, alpha: float = 0.0, bounded: bool = True):
        print(f"{t_prime=}")
        x, x_dot, hamiltonian = self.dynamics(t_prime, state, p, alpha=alpha, bounded=bounded, require_grad=True)

        # Identity: lambda dot = -dH/dx.
        hamiltonian.backward()
        lambda_dot = -x.grad

        return torch.cat([x_dot, lambda_dot]).detach().numpy()

    # Returns (thrust magnitude, thrust direction) tuple
    def get_controls(self, costate: ArrayLike, mass: ArrayLike, alpha: float = 0.0, bounded: bool = True) -> Tuple[Tensor, Tensor]:
        # The instantaneous thrust that maximizes the Hamiltonian can be decomposed into two components: its magnitude
        # and its direction. By the Cauchy-Schwartz inequality, the unit vector u-hat that maximizes the inner product
        # <lambda_v, u_hat> is simply lambda_v / ||lambda_v||.
        lv_vec = costate[3:6]
        lv, lm = lv_vec.norm(dim=0, keepdim=True), costate[None, 6]
        thrust_hat = torch.where(lv != 0.0, lv_vec / lv, 0.0)

        if alpha < 1.0:
            thrust_mag = (-lv + mass * lm / self._v_e - alpha) / (2 * (1 - alpha))
            if bounded:
                thrust_mag = thrust_mag.clamp(-1.0, 1.0)
            
            # Normalize so that the magnitude is always positive
            thrust_hat *= thrust_mag.sign()
            thrust_mag.abs_()
        
        # Bang-bang (bounded) control: maximizes the Hamiltonian by computing its derivative wrt the thrust
        # magnitude, then use 0 thrust if dH/dT is negative and use the maximum thrust if dH/dT is positive
        else:
            dH_dT = 1 + lv - mass * lm / self._v_e
            # dH_dT = 1 - max_thrust * lv / m - max_throughput * lm
            thrust_mag = torch.where(dH_dT < 0.0, 0.0, 1.0)
        
        thrust_mag *= self._max_thrust
        return thrust_mag, thrust_hat

# Computes a batch of position vectors for either a Body or an Orbit object
def _get_rv_generic(obj: Union[Body, Orbit], epochs: Time) -> Tuple[Quantity, Quantity]:
    if isinstance(obj, Orbit):
        tofs = epochs - obj.epoch
        r, v = farnocchia(obj.attractor.k, *obj.rv(), tofs.reshape(-1))
    elif isinstance(obj, Body):
        r, v = Ephem.from_body(obj, epochs).rv()
    else:
        raise NotImplementedError
    
    return r.squeeze(), v.squeeze()

def _to_orbit(obj: Union[Body, Orbit], t: Time) -> Orbit:
    if isinstance(obj, Orbit):
        return obj.propagate(t)
    elif isinstance(obj, Body):
        return Orbit.from_ephem(obj.parent, Ephem.from_body(obj, t), t)
