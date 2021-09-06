from functools import partial
import jax.numpy as jnp
import jax


# Differentiable, online, fast interpolation- O(1) in the number of ground truth points- using a cubic
# Hermite spline. If assume_uniform == True, then we will assume that the 'ground truth' x values are uniformly
# spaced, so we can find the appropriate interpolation interval in O(1) time instead of relying on searchsorted().
@partial(jax.jit, static_argnums=(4, 5, 6))
def cubic_hermite_interp(x_new, x, y, y_prime, assume_uniform = False, get_derivative = False, axis: int = 0):
    # Sanity checks
    assert y.shape == y_prime.shape, "Derivative values and y values must have the same shape"
    assert x.ndim == 1, "x must be a 1D vector"
    assert x_new.ndim <= 1, "x_new must be a 1D vector or scalar"

    axis %= y.ndim              # Normalize negative indices to positive ones
    num_nodes = y.shape[axis]

    # Simple, fast case
    if assume_uniform:
        assert len(x) >= 2, "x should be at least of length 2, containing the min and max x values"

        x_min = x[0]
        step = (x[-1] - x_min) / (num_nodes - 1)

        # The `jnp.clip()` call here is needed to prevent overshooting the bounds of the ground truth
        # array. This way, we silently extrapolate the interpolating polynomial for the intervals at the boundaries.
        k = jnp.clip(jnp.floor((x_new - x_min) / step), 0, num_nodes - 2).astype(jnp.int32)
        k_next = k + 1

        x_lo = x_min + step * k
        x_hi = x_lo + step
    
    # Slower, robust method
    else:
        assert len(x) == num_nodes, "When assume_uniform == False, x must have the same length as y and y_prime"
        k_next = jnp.clip(jnp.searchsorted(x, x_new), 1, len(x) - 1)
        k = k_next - 1
        x_lo, x_hi = jnp.take(x, k), jnp.take(x, k_next)
    
    # The lower and upper bounds of the intervals for each point
    y_lo, y_hi = jnp.take(y, k, axis), jnp.take(y, k_next, axis)
    y_prime_lo, y_prime_hi = jnp.take(y_prime, k, axis), jnp.take(y_prime, k_next, axis)
    
    x_span = x_hi - x_lo
    t = (x_new - x_lo) / x_span

    # Fully general way to get the basis functions to broadcast correctly
    target_ndim = y.ndim - axis if t.ndim > 0 else 0
    while t.ndim < target_ndim:
        t = jnp.expand_dims(t, -1)
        x_span = jnp.expand_dims(x_span, -1)

    # See https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Representations
    # Common subexpression elimination
    t_m1 = t - 1
    t_sq = t ** 2
    unity_minus_t_sq = (1 - t) ** 2
    two_t = 2 * t

    h00 = (1 + two_t) * unity_minus_t_sq
    h10 = t * unity_minus_t_sq * x_span
    h01 = t_sq * (3 - two_t)
    h11 = t_sq * t_m1 * x_span

    y_new = h00 * y_lo + h10 * y_prime_lo + h01 * y_hi + h11 * y_prime_hi

    if get_derivative:
        # We want to compute dy/dx, not dy/dt. dy/dt can be derived from the formula used for y(t) above.
        # Using the chain rule we have that dy/dx = dy/dt * dt/dx, and dt/dx = 1 / x_span. Doing some
        # algebraic manipulation and simplifying we get the formula used below.
        
        # These have a factor of x_span which gets canceled out when we multiply by dt/dx
        h10_prime = (3 * t - 1) * t_m1  # * x_span
        h11_prime = t * (3 * t - 2)     # * x_span
        
        y_prime_new = h10_prime * y_prime_lo + h11_prime * y_prime_hi + 6 * t * t_m1 * (y_lo - y_hi) / x_span
        return y_new, y_prime_new
    else:
        return y_new


# Use interpolation to 'predict' the odd-numbered points given the even-numbered ones and compute the residuals
def interp_residuals(x, y, y_prime):
    pred = cubic_hermite_interp(x_new=x[1::2], x=x[::2], y=y[::2], y_prime=y_prime[::2])
    return pred - y[1::2]
