import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm


# Perform 3D axis-angle rotation using Rodrigues' formula.
# See https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Statement
@jax.jit
def rodrigues_rotate(v, k, theta):
    return v * jnp.cos(theta) + jnp.cross(k, v) * jnp.sin(theta) + k * k.dot(v) * (1 - jnp.cos(theta))



# Compute approximate gravitational sphere of influence for a body
def sphere_of_influence(k, k_minor, r, v):
    h = jnp.cross(r, v)
    e = ((v.dot(v) - k / (norm(r))) * r - r.dot(v) * v) / k
    ecc = norm(e)
    p = h.dot(h) / k
    a = p / (1 - (ecc ** 2))

    return a * (k_minor / k) ** (2 / 5)
