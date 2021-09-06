from pathfinder.interpolate import cubic_hermite_interp
from scipy.interpolate import CubicHermiteSpline
from timeit import timeit
import numpy as np
import pytest


# Make sure that our differentiable JAX implementation of cubic Hermite splines
# gives the same output as the SciPy implementation up to numerical error
@pytest.mark.parametrize('uniform', [True, False])
@pytest.mark.parametrize(['axis', 'ndim'], [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])
def test_correctness(uniform: bool, axis: int, ndim: int):
    shape = np.random.randint(10, 100, size=ndim)
    num_nodes = shape[axis]
    lo, hi = 0.0, 100.0
    
    # Create a uniform mesh of time values
    if uniform:
        times = np.arange(num_nodes, dtype=np.float64)
    
    # Create a sorted list of randomly spaced time values
    else:
        times = np.random.uniform(lo, hi, num_nodes)
        times.sort()
    
    y = np.random.normal(size=shape)
    y_prime = np.random.normal(size=shape)

    new_times = np.random.uniform(lo, hi, num_nodes)
    our_y, our_y_prime = cubic_hermite_interp(new_times, times, y, y_prime, assume_uniform=uniform, get_derivative=True, axis=axis)

    scipy_interp = CubicHermiteSpline(times, y, y_prime, axis=axis)
    scipy_y = scipy_interp(new_times)
    scipy_y_prime = scipy_interp.derivative()(new_times)

    assert np.allclose(our_y, scipy_y)
    assert np.allclose(our_y_prime, scipy_y_prime)

    # When assume_uniform=True, we allow a shortcut usage where `x` is a 2 element vector
    # containing only the min and max `x` values, corresponding to the first and last elements
    # of the `y` and `y_prime` vectors. We need to make sure that this usage produces the same
    # output as when arange(min, max) is passed in for `x`.
    if uniform:
        shortcut_x = np.array([times[0], times[-1]])
        shortcut_y, shortcut_y_prime = cubic_hermite_interp(
            new_times, shortcut_x, y, y_prime, assume_uniform=uniform, get_derivative=True, axis=axis
        )
        assert np.allclose(our_y, shortcut_y)
        assert np.allclose(our_y_prime, shortcut_y_prime)


def test_speed():
    # Realistic situation: interpolating ephemerides for 3 bodies across 25 years
    num_days = 25 * 365
    shape = (3, num_days, 3)

    times = np.arange(num_days)
    y = np.random.normal(size=shape)
    y_prime = np.random.normal(size=shape)

    num_nodes = 1000
    new_times = np.random.uniform(0.0, num_days, num_nodes)
    new_times.sort()

    # Give SciPy a slight advantage by letting it set up the polynomial first
    scipy_spline = CubicHermiteSpline(times, y, y_prime, axis=1)
    scipy_time = timeit(lambda: scipy_spline(new_times), number=1000)

    our_time = timeit(
        lambda: cubic_hermite_interp(new_times, times, y, y_prime, assume_uniform=True, axis=1).block_until_ready(),

        # Run it once to make sure the compilation has already happened before we time it
        setup=lambda: cubic_hermite_interp(new_times, times, y, y_prime, assume_uniform=True, axis=1),
        number=1000
    )
    assert our_time < scipy_time