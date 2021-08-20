from astropy.constants import g0
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from astropy.units import def_unit, Quantity, quantity_input
from dataclasses import dataclass, field
from datetime import datetime
from poliastro.bodies import Body
from poliastro.ephem import Ephem
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from poliastro.util import time_range
from typing import Optional, Union
import numpy as np
import pykep as pk
import scipy
import torch


@quantity_input(isp='s', max_thrust='N', wet_mass='kg', max_tof='d')
@dataclass
class LowThrustTransfer:
    source: Union[Body, Orbit]
    target: Union[Body, Orbit]

    isp: Quantity
    max_thrust: Quantity
    wet_mass: Quantity

    start: Time = field(default_factory=lambda: Time(datetime.utcnow(), scale='tdb'))
    end: Optional[Time] = None

    # Use Pontryagin's principle to turn this functional optimization problem into a BVP
    def compute(self, *, alpha: float = 0.0, unbounded: bool = False) -> Maneuver:
        attractor = self.target.parent if isinstance(self.target, Body) else self.target.attractor

        # Get all the ephemerides data we need- this is a three-body problem
        ephem_range = time_range(self.start, periods=10000, end=self.end or self.start + 5.0 * 365.0)
        attr_ephem = Ephem.from_body(attractor, ephem_range)

        source = _get_ephem_generic(self.source, ephem_range)
        target = _get_ephem_generic(self.target, ephem_range)
        r0, v0 = source.rv(self.start)

        # Define "natural" units to perform the calculation in
        mass = def_unit('M', self.wet_mass)
        pos = def_unit('R', np.linalg.norm(r0))     # Distance of source body from attractor at start
        vel = def_unit('V', np.linalg.norm(v0))     # Velocity of source body wrt attractor at start
        acc = vel * vel / pos
        grav = acc * pos ** 2
        force = acc * mass

        # Gravitational parameters in our custom units
        objects = attractor, self.source, self.target
        ephems = [ephem for ephem, object in zip((attr_ephem, source, target), objects) if isinstance(object, Body)]
        print(ephems)
        mus = [torch.from_numpy(object.k.to(grav)) for object in objects if isinstance(object, Body)]
        max_thrust = self.max_thrust.to(force).value
        
        t0 = self.start.jd
        v_e = (self.isp * g0).to(vel).value     # Exhaust velocity
        m_dot = max_thrust / v_e                # Mass flow

        # Dynamics for both the state and co-states
        def dynamics(t_prime, state, tof, require_grad = False):
            # Wrap the state as a PyTorch tensor to use autograd
            state = torch.from_numpy(state)
            x, lambda_ = state[:7], state[7:]
            x.requires_grad = require_grad

            # Decode position, velocity, and mass
            rv, m = x[:6], x[6]
            r, v = rv[:3], rv[3:]

            # Re-dimensionalize
            t = Time(t0 + t_prime * tof, format='jd', scale='tdb')
            print(t.to_value('iso'))

            # Compute gravity
            separations = (torch.from_numpy(ephem.sample(t).get_xyz().to(pos).value) - r for ephem in ephems)
            gravity = sum(sep * mu / sep.norm() ** 3 for sep, mu in zip(separations, mus))

            # Compute quadratic control
            lv_vec = lambda_[:3]
            lv, lm = lv_vec.norm(), lambda_[6]
            if alpha < 1.0:
                # Thrust magnitude
                thrust_mag = (max_thrust * lv + m * (m_dot * lm - alpha)) / (2 * m * (1 - alpha)).unsqueeze(0)
                if not unbounded:
                    thrust_mag = thrust_mag.clamp(0.0, 1.0)
            
            # Bang-bang (bounded) control: minimize the Hamiltonian by computing its derivative wrt the thrust
            # magnitude, then use 0 thrust if dH/dT is positive and use the maximum thrust if dH/dT is negative
            else:
                dH_dT = 1 - max_thrust * lv / m - m_dot * lm
                thrust_mag = torch.tensor([0.0 if dH_dT >= 0.0 else 1.0])
            
            # Thrust direction
            thrust_hat = -lv_vec / lv
            thrust = thrust_mag * thrust_hat

            # Compute time derivative of the state
            x_dot = torch.cat([
                v,                       # Change of position = velocity
                gravity + thrust / m,    # Change of velocity = acc. due to gravity + thrust / mass
                -thrust_mag / v_e        # Change of mass = -thrust / exhaust velocity
            ])
            return x, x_dot, lambda_, thrust_mag
        
        # Function passed to the BVP solver; returns time derivatives of state and co-states
        def f(t_prime, state, tof):
            x, x_dot, lambda_, _ = dynamics(t_prime, state, tof, require_grad=True)

            # Identity: lambda dot = -dH/dx.
            hamiltonian = torch.sum(lambda_ * x_dot) # + thrust_mag
            hamiltonian.backward()
            lambda_dot = -x.grad

            return torch.cat([x_dot, lambda_dot]).detach().numpy()
        
        # Boundary conditions
        def boundary(state_a, state_b, tof):
            r_a, r_b = state_a[:3], state_b[:3]

            # Re-dimensionalize
            tf = Time(t0 + tof, format='jd', scale='tdb')
            print(tf.to_value('iso'))
            rf = target.sample(tf).get_xyz().to(pos).value

            _, x_dot, lambda_, thrust_mag = dynamics(tof, state_b, tof)
            hamiltonian = lambda_ @ x_dot + thrust_mag

            return np.concatenate([
                r_a - r0,       # Actually start at the source location
                r_b - rf,       # Actually land where we're trying to reach
                state_b[7],     # Co-states should equal zero at termination
                hamiltonian     # Transversality condition: H(t_f) = 0
            ])
        
        # Initialize with an impulsive Lambert trajectory
        maneuver = Maneuver.lambert(
            Orbit.from_ephem(attractor, source, self.start),
            Orbit.from_ephem(attractor, target, self.end or self.start + 2 * 365.0)
        )
        transfer = Orbit.from_vectors(attractor, r0, v0 + maneuver.impulses[0][1])
        points = transfer.sample()
        y_init = np.concatenate([
            points.get_xyz().to(pos).value,                         # Positions
            points.differentials["s"].get_d_xyz().to(vel).value,    # Velocities
            np.linspace(1.0, 0.0, 100)[None],                       # Masses (as proportion of initial wet mass)
            np.repeat(np.random.randn(7, 1), 100, axis=1)           # Co-states
        ])

        return scipy.integrate.solve_bvp(
            fun=f,
            bc=boundary,
            x=np.linspace(0, 1, 100),
            y=y_init,
            p=(self.end - self.start).jd if self.end else [200.0]    # Initial guess for time of flight
        )


    def _pretty(self, z):
        print("\nLow-thrust NEP transfer from " +
              self.p0.name + " to " + self.pf.name)
        print("\nLaunch epoch: {!r} MJD2000, a.k.a. {!r}".format(
            z[0], pk.epoch(z[0])))
        print("Arrival epoch: {!r} MJD2000, a.k.a. {!r}".format(
            z[0] + z[1], pk.epoch(z[0] + z[1])))
        print("Time of flight (days): {!r} ".format(z[1]))
        print("\nLaunch DV (km/s) {!r} - [{!r},{!r},{!r}]".format(np.sqrt(
            z[3]**2 + z[4]**2 + z[5]**2) / 1000, z[3] / 1000, z[4] / 1000, z[5] / 1000))
        print("Arrival DV (km/s) {!r} - [{!r},{!r},{!r}]".format(np.sqrt(
            z[6]**2 + z[7]**2 + z[8]**2) / 1000, z[6] / 1000, z[7] / 1000, z[8] / 1000))


def _get_ephem_generic(obj, epochs):
    func = Ephem.from_body if isinstance(obj, Body) else Ephem.from_orbit
    return func(obj, epochs)

def _safe_sample_ephem(ephem, t):
    points = ephem.epochs
    lower, upper = points[0], points[-1]
    if t < lower or t > upper:
        temp = Ephem.from
