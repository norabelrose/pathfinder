from functools import partial
from numpy.typing import ArrayLike
from typing import Tuple

from .ephemeris import Ephemeris

import jax.numpy as jnp
import jax


# Optimal control
@partial(jax.jit, static_argnames=('smoothness', 'bounded'))
def instantaneous_control(costate: ArrayLike, mass: ArrayLike, smoothness: float, bounded: bool):
    # The instantaneous thrust that maximizes the Hamiltonian can be decomposed into two components: its magnitude
    # and its direction. By the Cauchy-Schwartz inequality, the unit vector û that maximizes the inner product
    # <λv, û> is simply λv / ||λv||.
    λv_vec = costate[3:6]
    λv, λm = jnp.linalg.norm(λv_vec, axis=-1, keepdims=True), costate[None, 6]
    thrust_hat = jnp.where(λv != 0.0, λv_vec / λv, 0.0)

    if smoothness > 0.0:
        thrust_mag = (-λv + mass * λm - (1 - smoothness)) / (2 * smoothness)
        if bounded:
            thrust_mag = jnp.clip(thrust_mag, -1.0, 1.0)
        
        # Normalize so that the magnitude is always positive
        thrust_hat *= jnp.sign(thrust_mag)
        thrust_mag = jnp.abs(thrust_mag)
    
    # Bang-bang (bounded) control: maximizes the Hamiltonian by computing its derivative wrt the thrust
    # magnitude, then use 0 thrust if dH/dT is negative and use the maximum thrust if dH/dT is positive
    else:
        dH_dT = 1 + λv - mass * λm
        thrust_mag = jnp.where(dH_dT < 0.0, 0.0, 1.0)
    
    return thrust_mag, thrust_hat


@partial(jax.jit, static_argnames=('smoothness', 'bounded'))
def compute_hamiltonian(
        x: ArrayLike, costate: ArrayLike, t, *,
        ephem: Ephemeris, v_e, max_thrust, smoothness: float, bounded: bool
    ) -> Tuple[ArrayLike, ArrayLike]:
    assert 0.0 <= smoothness <= 1.0, "Smoothness parameter must be in the closed interval [0.0, 1.0]"

    # Decode position, velocity, and mass
    rv, m = x[:6], x[6]
    r, v = rv[:3], rv[3:]

    gravity = ephem.gravitational_field(r, t)

    thrust_mag, thrust_hat = instantaneous_control(costate, m, smoothness, bounded)
    thrust_mag *= max_thrust
    delta_v = jnp.where(m > 0.0, thrust_mag / m, 0.0)   # Thrust must be zero when we have zero mass
    thrust_acc = delta_v * thrust_hat
    # print(f"{gravity=} {thrust_acc=} {v=} {separations=} {body_positions=} {r=} {mus=}")

    # Compute time derivative of the state
    x_dot = jnp.concatenate([
        v,                                           # Change of position = velocity
        gravity + thrust_acc,                        # Change of velocity = acc. due to gravity + acc. due to thrust
        jnp.where(m > 0.0, -thrust_mag / v_e, 0.0)   # Change of mass = -thrust / exhaust velocity
    ])

    # The path-dependent cost is a convex combination of the squared delta-V, and
    # delta-V itself. This allows us to interpolate between the smooth LQR solution
    # and the bang-bang solution that comes from directly minimizing delta-V.
    path_cost = smoothness * delta_v ** 2 - (1 - smoothness) * delta_v
    hamiltonian = costate.dot(x_dot) - path_cost
    return jnp.squeeze(hamiltonian), x_dot
