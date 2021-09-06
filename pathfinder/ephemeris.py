from astropy.time import Time
from collections import namedtuple
from poliastro.bodies import Body
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from typing import Mapping, Sequence, Union

from .interpolate import cubic_hermite_interp

import jax.numpy as jnp
import numpy as np
import poliastro


# Types of objects that Ephemeris can accept as objects
ObjectLike = Union[Body, Orbit, str]


# Differentiable, multi-body ephemeris with fast, online interpolation and
# support for both gravitating bodies and objects with negligible gravity.
# Like the other Pathfinder classes, all distances, times, and derived quantities
# like gravitational parameters are measured in terms of AU and days.
_Ephemeris = namedtuple('_Ephemeris', [
    'time_range', 'objects', 'body_positions', 'body_velocities', 'mus', 'radii'
])
class Ephemeris(_Ephemeris):
    @classmethod
    def from_objects(
        cls,
        objects: Mapping[str, ObjectLike],

        # For efficiency and simplicity reasons, we only support uniformly sampled ephemerides
        *, start: Time = None, end: Time, spacing: float = 1.0
    ):
        # objects = [objects] if not isinstance(objects, Sequence) else objects
        objects = {key: resolve_string_body(obj) if isinstance(obj, str) else obj for key, obj in objects.items()}

        start = Time(Time.now(), format='tdb') if start is None else start
        span = np.ceil(end.mjd - start.mjd)
        times = start + np.arange(span, step=spacing)

        # We maintain a separate copy of the positions and velocities of the gravitating bodies
        # and store them in two condensed JAX arrays for efficient computation of gravity
        body_rs, body_vs, mus, radii = [], [], [], []
        object_rvs = {}

        for key, obj in objects.items():
            ephem_getter = Ephem.from_body if isinstance(obj, Body) else Ephem.from_orbit
            r, v = ephem_getter(obj, times).rv()
            r, v = r.to('au').value, v.to('au/d').value

            if isinstance(obj, Body):
                mus.append(obj.k.to('au3/d2').value)
                radii.append(obj.R.to('au').value)
                
                body_rs.append(r)
                body_vs.append(v)
            
            object_rvs[key] = r, v

        return Ephemeris(
            jnp.array([0.0, span]),
            object_rvs,
            # We use the shape [time, xyz, body index] to facilitate broadcasting
            jnp.moveaxis(jnp.array(body_rs), 0, -1),
            jnp.moveaxis(jnp.array(body_vs), 0, -1),
            jnp.array(mus),
            jnp.array(radii)
        )
    
    def rv(self, times: jnp.ndarray, object_key: str):
        r, v = self.objects[object_key]
        return cubic_hermite_interp(times, self.time_range, r, v, assume_uniform=True, get_derivative=True, axis=-2)
    
    # Compute the summed gravitational field of all the gravitating bodies in this ephemeris at
    # a given point in space (r) and time (t)
    def gravitational_field(self, r: jnp.ndarray, t: jnp.ndarray):
        body_positions = cubic_hermite_interp(
            t, self.time_range, self.body_positions, self.body_velocities, assume_uniform=True, axis=-3
        )

        # It would be nice to be able to simply raise an error here whenever we are called with a
        # position that's *inside* the radius of a Body, but we don't do this for two reasons:
        #
        #   1. JAX simply disallows data-dependent assertions in JIT-ed functions, and
        #   2. Not penetrating into solid celestial bodies is effectively a 'constraint' on the
        #      optimization problem, which needs to be taken into account in the cost functional,
        #      but it probably isn't a good idea to simply crash the program whenever a candidate
        #      solution does pass through a celestial body as part of its trajectory.
        #
        # The formula used here, where we simply take the maximum of the Euclidean distance from the
        # center of mass and the radius of the object, actually gives roughly correct results for this
        # edge case, on the assumption that the body has uniform density. In particular, the gravitational
        # field strength fades linearly to zero as we approach the center of mass of any object.
        separations = body_positions - r[..., None]
        print(f"{separations=}")
        distances = jnp.linalg.norm(separations, axis=-2, keepdims=True)
        strengths = jnp.maximum(distances, self.radii) ** -3
        return jnp.sum(separations * self.mus * strengths, axis=-1)


# Helper functions
def resolve_string_body(body_str: str) -> Body:
    # Hackish way to turn strings into built-in poliastro Body objects
    obj = getattr(poliastro.bodies, body_str.title(), None)
    assert isinstance(obj, Body), f"'{obj}' is not a recognized solar system body"
    return obj
