from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from pathfinder.odeint import odeint
from poliastro.bodies import Sun, Mars

import jax.numpy as jnp
import numpy as np


def test_accuracy():
    sun_mu, mars_mu = Sun.k.to('au3/d2').value, Mars.k.to('au3/d2').value

    def dynamics(t, y):
        sun_rv, mars_rv = y[:6], y[6:]
        sun_r, sun_v = sun_rv[:3], sun_rv[3:]
        mars_r, mars_v = mars_rv[:3], mars_rv[3:]

        separation = mars_r - sun_r
        grav_vec = separation / jnp.linalg.norm(separation) ** 3

        return jnp.concatenate([
            sun_v,
            mars_mu * grav_vec,
            mars_v,
            -sun_mu * grav_vec
        ])
    
    dt = np.linspace(0.0, 365.0, 50)
    t = Time.now() + dt
    
    sun_r, sun_v = (x.get_xyz().value for x in get_body_barycentric_posvel('sun', t))
    mars_r, mars_v = (x.get_xyz().value for x in get_body_barycentric_posvel('mars', t))
    y0 = np.concatenate([sun_r[:, 0], sun_v[:, 0], mars_r[:, 0], mars_v[:, 0]], dtype=np.float64)
    
    our_output = odeint(func=lambda y, t: dynamics(t, y), y0=y0, t=dt)
    pred_mars_r = our_output.reshape(-1, 4, 3)[:, 2].T

    # This accuracy is as good as we can expect at the moment
    assert np.allclose(pred_mars_r, mars_r, atol=1e-3)
