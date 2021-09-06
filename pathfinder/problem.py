from astropy.constants import g0
from astropy.time import Time
from astropy.units import Quantity, Unit, quantity_input
from dataclasses import dataclass
from functools import partial
from numpy.typing import ArrayLike
from poliastro.bodies import Body
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from typing import Tuple, Union

from .dynamics import compute_hamiltonian
from .ephemeris import Ephemeris
# from .interpolate import cubic_hermite_interp
from .odeint import odeint
from .path import Path, PathDescriptor
from .utils import rodrigues_rotate, sphere_of_influence

import astropy.units as u
import jax.numpy as jnp
import jax
import numpy as np
import warnings


# Description of a spacecraft path-finding objective
@quantity_input(isp='s', max_thrust='N', wet_mass='kg', dry_mass='kg')
@dataclass
class Problem:
    source: Union[Body, Orbit]
    target: Union[Body, Orbit]

    isp: Quantity
    max_thrust: Quantity
    wet_mass: Quantity
    dry_mass: Quantity

    bounded: bool = True
    smoothness: float = 1.0

    earliest_start: Time = None     # If None, is set to the current time
    latest_arrival: Time = None     # If None, is set to 25 years from earliest_start

    def __post_init__(self):
        # Defaults
        if self.earliest_start is None:
            self.earliest_start = Time(Time.now(), scale='tdb')
        
        if self.latest_arrival is None:
            self.latest_arrival = self.earliest_start + 25.0 * 365.2425
        else:
             assert self.latest_arrival > self.earliest_start, "Latest arrival time must be later than the earliest start time"
        
        src_attractor = get_attractor_generic(self.source)
        tgt_attractor = get_attractor_generic(self.target)
        assert src_attractor == tgt_attractor, "Source and target must have the same attractor"

        self.attractor = src_attractor
        self.ephem = Ephemeris.from_objects(
            dict(
                attractor=self.attractor,
                source=self.source,
                target=self.target
            ),
            start=self.earliest_start,
            end=self.latest_arrival
        )
        # objects = (self.attractor, self.source, self.target)
        # bodies = [self.attractor] + [x for x in (self.source, self.target) if isinstance(x, Body)]
# 
        # # Load all the ephemerides data that we need up front; we sample at the day level
        # ephem_days = np.ceil(self.latest_arrival.mjd - self.earliest_start.mjd)
        # times = self.earliest_start + np.arange(0.0, ephem_days)
        # posvels = [get_posvel_generic(obj, times) for obj in objects]
# 
        # self.ephem_range = jnp.array([0.0, ephem_days])
        # self.ephem = jnp.stack([
        #     jnp.stack([pos.get_xyz(1).to('au') for pos, _ in posvels]),     # Positions
        #     jnp.stack([vel.get_xyz(1).to('au/d') for _, vel in posvels])    # Velocities
        # ])

        # Thrust and exhaust velocity measured in 'natural' units for computation
        self._max_thrust = self.max_thrust.to('au kg/d2') / self.wet_mass
        self._v_e = (self.isp * g0).to(u.au / u.d).reshape(1)

        # Construct and JIT compile the Hamiltonian and its gradient wrt the state
        hamiltonian = partial(
            compute_hamiltonian,
            ephem=self.ephem,
            v_e=self._v_e,
            max_thrust=self._max_thrust,
            smoothness=self.smoothness,
            bounded=self.bounded
        )
        self.dynamics = jax.value_and_grad(hamiltonian, has_aux=True)
    
    # Use simple and fast heuristics to compute an initial guess for a feasible path
    def initial_guess(self) -> PathDescriptor:
        # Start six months after the earliest start date, or 1/3 of the way through the
        # candidate launch window, whichever comes first
        t0 = min(180.0, (self.latest_arrival.mjd - self.earliest_start.mjd) // 3)

        r0, v0 = self.ephem.rv(t0, 'source')
        r_hat = r0 / jnp.linalg.norm(r0)
        v_hat = v0 / jnp.linalg.norm(v0)

        # Use one half of the synodic period of the two bodies as an initial guess for time-of-flight
        rt, vt = self.ephem.rv(t0, 'target')
        source_orbit = Orbit.from_vectors(self.attractor, r0 * Unit('au'), v0 * Unit('au/d'))
        target_orbit = Orbit.from_vectors(self.attractor, rt * Unit('au'), vt * Unit('au/d'))
        synodic_period = 1 / abs(1 / source_orbit.period.to('d') - 1 / target_orbit.period.to('d'))
        tof = synodic_period.value / 2

        # If the source is a Body object wth gravity, we need to start at the edge of the
        # object's gravitational sphere of influence to avoid singularities
        if isinstance(self.source, Body):
            soi = sphere_of_influence(self.source.parent.k.to('au3/d2').value, self.source.k.to('au3/d2').value, r0, v0)
            r0 = r0 + soi * v_hat
        
        costate0 = jnp.concatenate([
            -r_hat, # * 0.05,

            # For the velocity costate, we initialize it so that the spacecraft with starts out with
            # a thrust at a slight angle to its current velocity, within the plane of the ecliptic
            v_hat,
            # rodrigues_rotate(v_hat, jnp.cross(r_hat, v_hat), -jnp.pi / 4),

            # For the mass costate, we initialize it so that the spacecraft will start the path with
            # maximum thrust. In fact, we can arguably *define* the start of the path as the time of
            # the first thrust, and use this as a boundary condition to allow a free start time.
            jnp.array([3.0])
        ])
        return PathDescriptor(t0, costate0, tof)
    
    def f(self, y: ArrayLike, t_hat: ArrayLike, t0: float, tof: float) -> ArrayLike:
        x, costate = y[:7], y[7:]

        # Re-dimensionalize
        t = t0 + t_hat * tof

        (_, x_dot), neg_lambda_dot = self.dynamics(x, costate, t)
        return jnp.concatenate([x_dot, -neg_lambda_dot]) * tof
    
    def propagate(self, descriptor: PathDescriptor, *, num_nodes: int = 50):
        t0, costate0, tof = descriptor

        y = odeint(
            func=partial(self.f, t0=t0, tof=tof),
            y0=jnp.concatenate([
                *self.ephem.rv(t0, 'source'),
                jnp.array([1.0]),
                costate0
            ]),
            t=jnp.linspace(0.0, 1.0, num_nodes)
        ).T
        s, λ = y[:7], y[7:]
        
        if jnp.any(jnp.abs(y) > 2.9e15):
            warnings.warn(f"Got quantities larger than the radius of the observable universe while propagating {descriptor}")

        return Path(
            r=s[:3],
            v=s[3:6],
            m=s[6],
            lr=λ[:3],
            lv=λ[3:6],
            lm=λ[6]
        )
    
    def distance_loss(self, descriptor: PathDescriptor):
        t0, _, tof = descriptor
        # target_pos = cubic_hermite_interp(t0 + tof, self.ephem_range, *self.ephem[:, 2], assume_uniform=True)
        target_pos, _ = self.ephem.rv(t0 + tof, 'target')

        path = self.propagate(descriptor, num_nodes=5)
        return jnp.linalg.norm(target_pos - path.r[:, -1]) # , path
    
    def plot(self, descriptor: PathDescriptor):
        from astropy.coordinates import CartesianRepresentation
        from plotly.graph_objects import Figure
        from poliastro.plotting import OrbitPlotter3D

        figure = Figure()
        epoch = self.earliest_start + descriptor.t0
        plotter = OrbitPlotter3D(figure, dark=True)

        plotter.plot_body_orbit(self.source, epoch)
        plotter.plot_body_orbit(self.target, epoch + descriptor.tof)
        plotter.plot_trajectory(
            CartesianRepresentation(self.propagate(descriptor).r * Unit('au')),
            color='pink',
            label='Path'
        )
        return figure

# Helper functions
def get_attractor_generic(obj: Union[Body, Orbit]) -> Body:
    if isinstance(obj, Body):
        return obj.parent
    
    if isinstance(obj, Orbit):
        return obj.attractor
    
    raise NotImplementedError

def get_posvel_generic(obj: Union[Body, Orbit], times: Time) -> Tuple[Quantity, Quantity]:
    if isinstance(obj, Body):
        ephem = Ephem.from_body(obj, times)
    elif isinstance(obj, Orbit):
        ephem = Ephem.from_orbit(obj, times)
    else:
        raise NotImplementedError
    
    return ephem.rv()
