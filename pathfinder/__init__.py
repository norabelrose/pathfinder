# This config option both enables float64 computation in JAX, and makes float64
# the default floating point data type. This is important for the kind of high
# precision ODE solving we need to do in Pathfinder. Also, the tests fail without
# this since reference libraries like SciPy use float64 for their calculations.
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

from .asteroid_info import *
from .problem import Problem
