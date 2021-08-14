from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from astropy.units import Unit, Quantity, quantity_input
from numpy.typing import ArrayLike
from torch.functional import Tensor
from torchdiffeq import odeint
from typing import Callable, cast
import numpy as np
import torch


# Units of the standard gravitational parameter (gravitational constant * mass) of an object
grav_param = Unit('m**3/s**2')

# Gravitational constant value from CODATA 2018- the last 2 sig figs are uncertain
BIG_G = 6.67430e-11 * grav_param / 'kg'


# Numpy array subclass that allows us to store position and velocity information together in a
# single tensor of 7D vectors, so that it can be directly passed to ODE solvers, while also
# ensuring type/shape/unit safety and convenient access to the individual components.
class State(np.ndarray):
    @staticmethod
    @quantity_input(r='length', v='velocity', mass='mass', mu=grav_param)
    def from_components(r: Quantity, v: Quantity, mass: Quantity = None, *, mu: Quantity = None):
        assert np.issubdtype(r.dtype, np.floating) and np.issubdtype(v.dtype, np.floating), "r and v must have floating point dtypes"
        assert r.shape == v.shape, "r and v must have the same shape"
        assert r.shape[-1] == 3, "r and v must have size 3 along the last dimension"

        if mu is not None:
            # Since we use 64-bit floats which can store at least 15 decimal digits losslessly, and
            # gravitational parameters are not generally known beyond ~12 digits, we shouldn't lose
            # any precision by dividing and later multiplying by big G. By doing this, we are able
            # to use a uniform representation for both celestial bodies with significant gravitational
            # fields, and objects like spacecraft with negligible gravity.
            assert mass is None, "The `mass` and `mu` parameters are mutually exclusive"
            mass = mu / BIG_G
        else:
            assert mass is not None, "Either the `mass` or `mu` parameter should be specified"

        # Note that we always use float64 for state vectors to minimize numerical error in ODE solutions
        raw_array = np.concatenate([
            np.asarray(r.to('m')).astype(np.float64),
            np.asarray(v.to('m/s')).astype(np.float64),
            np.asarray(mass.to('kg')).astype(np.float64)
        ], axis=-1)
        return State(raw_array)
    
    _body_masses = {
        'sun': 1.988409871e30,
        'mercury': 3.301e23,
        'venus': 4.867e24,
        'earth': 5.972e24,
        'moon': 7.348e22,
        'mars': 6.416908921e23,
        'jupiter': 1.899e27,
        'saturn': 5.685e26,
        'uranus': 8.682e25,
        'neptune': 1.024e26,
        'pluto': 1.471e22
    }
    
    @classmethod
    def get_body(cls, body: str, time: Time = None):
        mass = cls._body_masses[body.lower()] * Unit('kg')
        time = time if time is not None else Time.now()

        r, v = cast(tuple, get_body_barycentric_posvel(body, time))
        return State.from_components(r.xyz, v.xyz, mass[None])  # type: ignore
    
    def __new__(cls, value: ArrayLike) -> 'State':
        return  np.asarray(value).view(cls)
    
    def __array_finalize__(self, obj):
        # Prevent the user from accidentally creating a StateVector that isn't 7D via reshaping
        # or reduction; they should first use np.asarray() to get rid of the shape checking
        assert self.shape[-1] == 7, "StateVectors must have size 7 along the last dimension"

        # If we're a new object or viewing an ndarray, nothing has to be done.
        if obj is None or obj.__class__ is np.ndarray:
            return
    
    def __array_wrap__(self, array, _):
        return array.view(State)

    @property
    def position(self) -> Quantity:
        return cast(Quantity, Quantity(np.asarray(self)[..., :3], unit='m', copy=False))
    
    @property
    def velocity(self) -> Quantity:
        return cast(Quantity, Quantity(np.asarray(self)[..., 3:-1], unit='m/s', copy=False))
    
    @property
    def mass(self) -> Quantity:
        return cast(Quantity, Quantity(np.asarray(self)[..., -1], unit='kg', copy=False))
    
    r = position
    v = velocity
    m = mass
    
    @property
    def x(self) -> Quantity:
        return cast(Quantity, self.position[..., 0])
    
    @property
    def y(self) -> Quantity:
        return cast(Quantity, self.position[..., 1])
    
    @property
    def z(self) -> Quantity:
        return cast(Quantity, self.position[..., 2])
    
    @property
    def x_dot(self) -> Quantity:
        return cast(Quantity, self.velocity[..., 0])
    
    @property
    def y_dot(self) -> Quantity:
        return cast(Quantity, self.velocity[..., 1])
    
    @property
    def z_dot(self) -> Quantity:
        return cast(Quantity, self.velocity[..., 2])
    
    def __repr__(self) -> str:
        return f"<r: ({self.x}, {self.y}, {self.z}), v: ({self.x_dot}, {self.y_dot}, {self.z_dot}), m: {self.m}>"
    
    ###     Physics methods     ###

    # Computes the instantaneous acceleration due to gravity for all bodies in a batch of StateVectors
    def compute_gravity(self) -> Quantity:
        pos = self.position                     # [N, 3]
        diffs = pos - pos[..., None, :]         # [N, N, 3]
        dists = np.linalg.norm(diffs, axis=-1)  # [N, N]

        # Mask out non-existent gravitational attractions of objects to themselves, which would
        # otherwise cause a divide by zero error (since the distance is 0)
        np.fill_diagonal(dists, np.inf)
        return BIG_G * np.sum(self.mass[None, :, None] * diffs / dists[..., None] ** 3, axis=-2)  # [N, 3]
    
    # The `forces` callable is a function (t, y) -> dv/dt
    def evolve(self, t: Tensor, forces: Callable[[Tensor, Quantity], Quantity] = None) -> 'State':
        # The base derivative only incorporates inertial motion and gravitational forces
        def derivative(t, y):
            grav_acc = np.asarray(self.compute_gravity())
            dv_dt = grav_acc if not forces else grav_acc + forces(t, y) / self.mass[..., None]

            # dr/dt = v(t);     dv/dt = sum(F(t) / m(t));   dm/dt = 0
            return np.concatenate([
                np.asarray(self.velocity), dv_dt, np.asarray(np.zeros_like(self.mass[..., None]))
            ], axis=-1)

        result = odeint(func=derivative, y0=torch.from_numpy(self), t=t, method='dopri8')
        return State(result.numpy())    # type: ignore
