from astropy.constants import GM_sun
from astropy.coordinates import get_body_barycentric, get_body_barycentric_posvel
from astropy.time import Time
from poliastro.bodies import Body, Sun
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import farnocchia
from torch import nn, Tensor
from typing import Tuple, Union

import astropy.units as u
import numpy as np
import torch


GRAV_PARAM_UNIT = u.au**3/u.d**2


class Ephemeris(nn.Module):
    def __init__(self, ref: Union[Body, Orbit, str]):
        super().__init__()
        self.ref = ref
    
    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return SampleEphem.apply(self.ref, t)


# Uses AU for distance and MJD for time
class SampleEphem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ref: Union[Body, Orbit, str], t: Tensor) -> Tuple[Tensor, Tensor]:
        time = Time(np.asarray(t), format='mjd', scale='tdb')

        if isinstance(ref, Orbit):
            r0, v0 = ref.rv()

            # Annoying hack we have to do to support both scalar and array time values
            scalar = time.isscalar
            if scalar:
                time = time[None]
            
            ctx.attractor = ref.attractor
            mu = ctx.attractor.k

            r_np, v_np = farnocchia(mu, r0, v0, time - ref.epoch)
            r = torch.from_numpy(r_np.to(u.au))
            v = torch.from_numpy(v_np.to(u.au/u.d))
            if scalar:
                r, v = r.squeeze(0), v.squeeze(0)
        else:
            if isinstance(ref, Body):
                ctx.attractor = ref.parent
                ref = ref.name
            else:
                ctx.attractor = Sun
                assert isinstance(ref, str)
            
            r_np, v_np = get_body_barycentric_posvel(ref, time)
            r = torch.from_numpy(r_np.get_xyz(-1))
            v = torch.from_numpy(v_np.get_xyz(-1))
        
        ctx.time = time
        ctx.save_for_backward(r, v)
        return r, v

    @staticmethod
    def backward(ctx, grad_r: Tensor, grad_v: Tensor):
        r, v = ctx.saved_tensors

        # This should probably only be true for the Sun- there's no particular celestial body with ephemeris
        # data that this object orbits, so we can't use Newtonian mechanics to get its instantaneous acceleration.
        # We fall back on a finite difference approximation.
        if not ctx.attractor:
            eps = 1.0                   # We use a single day as our epsilon value
            t_prime = ctx.time + eps

            _, v_prime = torch.from_numpy(get_body_barycentric(ctx.attractor.name, t_prime).get_xyz(-1))
            v_dot = (v_prime - v) / eps
        else:
            # Get the *barycentric* position of the Sun so that we can compute the acceleration due to gravity
            sun_pos = torch.from_numpy(get_body_barycentric(ctx.attractor.name, ctx.time).get_xyz(-1))
            sep = (r - sun_pos)
            
            mu = torch.from_numpy(ctx.attractor.k.to(GRAV_PARAM_UNIT))
            v_dot = mu * -sep / sep.norm(dim=-1, keepdim=True) ** 3    # Two-body gravitational acceleration
        
        return None, torch.sum(grad_r * v, dim=-1) + torch.sum(grad_v * v_dot, dim=-1)
