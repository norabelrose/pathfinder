from astropy.constants import g0
from astropy.units import Quantity, quantity_input
from collections import namedtuple
from dataclasses import dataclass
from poliastro.bodies import Body
from poliastro.twobody import Orbit
from typing import Tuple, Union

import astropy.units as u
import erfa
import jax.numpy as jnp
import jax


Trajectory = namedtuple('Trajectory', ['r', 'v', 'm', 'lr', 'lv', 'lm', 'u'])


@quantity_input(isp='s', max_thrust='N', wet_mass='kg', dry_mass='kg')
@dataclass
class Control:
    source: Union[Body, Orbit]
    target: Union[Body, Orbit]
    isp: Quantity
    max_thrust: Quantity
    wet_mass: Quantity
    dry_mass: Quantity

    t0: float
    tof: float
    costate0: jnp.ndarray = None
    alpha: float = 0.0
    bounded: bool = True

    def __post_init__(self):
        self._max_thrust = self.max_thrust.to('au*kg/d**2') / self.wet_mass
        self._v_e = (self.isp * g0).to(u.au / u.d).view(1)

        attractors = {x.parent if isinstance(x, Body) else x.attractor for x in (self.source, self.target)}
        assert len(attractors) == 1, "Source and target must orbit around the same attractor"
        self.bodies = list(attractors | {x for x in (self.source, self.target) if isinstance(x, Body)})
        self.mus = [body.k.to('au**3/d**2') for body in self.bodies]
        self.radii = [body.R.to('au') for body in self.bodies]
    
    def f(self, y: jnp.ndarray, t_hat: jnp.ndarray) -> jnp.ndarray:
        # Debugging
        x, costate = y

        # Re-dimensionalize
        tof = self.tof
        t = self.t0 + t_hat * tof
        body_positions = [body(t)[0] for body in (self.attractor, self.source, self.target)]

        (_, x_dot), neg_lambda_dot = _dynamics(
            x, costate, body_positions, self.mus, self.radii, tof, self._v_e, self._max_thrust, self.alpha, self.bounded
        )
        return jnp.concatenate([x_dot, -neg_lambda_dot])
    
    def propagate(self, *, num_nodes: int = 10):
        y = jax.experimental.odeint(
            self.f,
            jnp.concatenate([
                *self.source(self.t0),
                jnp.array([1.0]),

            ])
        )


JD_TO_MJD = 2400000.5
MU_SUN = -0.00029591220819207774     # Gravitational parameter of the Sun in units AU**3/days**2
PLAN94_BODY_NAME_TO_PLANET_INDEX = {
    'mercury': 1,
    'venus': 2,
    'earth-moon-barycenter': 3,
    'mars': 4,
    'jupiter': 5,
    'saturn': 6,
    'uranus': 7,
    'neptune': 8,
}


def _get_barycentric_posvelacc(body: str, t) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
     plan94_idx = PLAN94_BODY_NAME_TO_PLANET_INDEX[body.lower()]

     # Compute the barycentric state vector of the Sun by subtracting the
     # heliocentric & barycentric state vectors of the Earth
     earth_rv_helio, earth_rv_bary = erfa.epv00(t, JD_TO_MJD)
     sun_rv_bary = erfa.pvmpv(earth_rv_bary, earth_rv_helio)

     # Convert the heliocentric state vector for this body into barycentric
     # coordinates by simply adding the barycentric state vector of the Sun
     body_rv_helio = erfa.plan94(t, JD_TO_MJD, plan94_idx)
     body_rv_bary = erfa.pvppv(body_rv_helio, sun_rv_bary)

     # Approximate the acceleration vector for this body using Newton's law
     # of gravitation, considering only the Sun's mass and no other masses.
     body_r_helio = body_rv_helio['p']
     body_acc = MU_SUN * body_r_helio / jnp.linalg.norm(body_r_helio, axis=-1, keepdims=True) ** 3

     return body_rv_bary, body_acc


@jax.jit
def _get_controls(costate: jnp.ndarray, mass: jnp.ndarray, v_e: jnp.ndarray, max_thrust: jnp.ndarray, alpha: float, bounded: bool):
    # The instantaneous thrust that maximizes the Hamiltonian can be decomposed into two components: its magnitude
    # and its direction. By the Cauchy-Schwartz inequality, the unit vector û that maximizes the inner product
    # <λv, û> is simply λv / ||λv||.
    λv_vec = costate[3:6]
    λv, λm = jnp.linalg.norm(λv_vec, axis=0, keepdims=True), costate[None, 6]
    thrust_hat = jnp.where(λv != 0.0, λv_vec / λv, 0.0)

    if alpha < 1.0:
        thrust_mag = (-λv + mass * λm / v_e - alpha) / (2 * (1 - alpha))
        if bounded:
            thrust_mag = jnp.clip(thrust_mag, -1.0, 1.0)
        
        # Normalize so that the magnitude is always positive
        thrust_hat *= jnp.sign(thrust_mag)
        thrust_mag = jnp.abs(thrust_mag)
    
    # Bang-bang (bounded) control: maximizes the Hamiltonian by computing its derivative wrt the thrust
    # magnitude, then use 0 thrust if dH/dT is negative and use the maximum thrust if dH/dT is positive
    else:
        dH_dT = 1 + λv - mass * λm / v_e
        thrust_mag = jnp.where(dH_dT < 0.0, 0.0, 1.0)
    
    thrust_mag *= max_thrust
    return thrust_mag, thrust_hat


def _compute_hamiltonian(
        x: jnp.ndarray, costate: jnp.ndarray, body_positions, mus, radii, tof, v_e: jnp.ndarray, max_thrust: jnp.ndarray,
        alpha: float, bounded: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Decode position, velocity, and mass
    rv, m = x[:6], x[6]
    r, v = rv[:3], rv[3:]

    # Compute gravity
    separations = (body_pos - r for body_pos in body_positions)
    gravity = sum(
        sep * mu / jnp.maximum(jnp.linalg.norm(sep, axis=0, keepdims=True), radius) ** 3
        for sep, mu, radius in zip(separations, mus, radii)
    )

    thrust_mag, thrust_hat = _get_controls(costate, m, v_e, max_thrust, alpha, bounded)
    thrust = thrust_mag * thrust_hat
    thrust_acc = jnp.where(m > 0.0, thrust / m, 0.0)      # Thrust must be zero when we have zero mass

    # Compute time derivative of the state
    x_dot = jnp.concatenate([
        v,                                           # Change of position = velocity
        gravity + thrust_acc,                        # Change of velocity = acc. due to gravity + acc. due to thrust
        jnp.where(m > 0.0, -thrust_mag / v_e, 0.0)   # Change of mass = -thrust / exhaust velocity
    ]) * tof
    hamiltonian = jnp.dot(costate, x_dot) - thrust_mag
    return hamiltonian, x_dot

# Returns nested tuple: ((H, dx/dt), dH/dx)
_dynamics = jax.jit(jax.value_and_grad(_compute_hamiltonian, has_aux=True))