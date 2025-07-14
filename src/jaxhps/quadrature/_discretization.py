from functools import partial


import jax.numpy as jnp
import jax
import numpy as np


jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnums=(0,))
def chebyshev_points(n: int) -> jax.Array:
    """
    Returns n Chebyshev points over the interval [-1, 1]

    out[i] = cos(pi * (n-1 - i) / (n-1)) for i={0,...,n-1}

    The left side of the interval is returned first.

    Args:
        n (int): number of Chebyshev points to return

    Returns:
        jax.Array: The sampled points in [-1, 1]
    """
    cos_args = jnp.arange(n, dtype=jnp.float64) / (n - 1)
    angles = jnp.flipud(jnp.pi * cos_args)
    pts = jnp.cos(angles)

    # # Normalize by 1 / sqrt(1 - x^2)
    # weights = jnp.sin(angles) ** 2 / (n - 1) * np.pi
    # nrms = jnp.sqrt(1 - pts**2)
    # weights = weights / nrms
    # weights = weights.at[0].set(0.0)
    # weights = weights.at[-1].set(0.0)
    return pts


@partial(jax.jit, static_argnums=(0,))
def first_kind_chebyshev_points(n: int) -> jax.Array:
    """
    Returns n Chebyshev points of the first kind over the interval [-1, 1].

    out[k] = cos(pi / n * ((n-1) - k + 1/2)) for k={0,...,n-1}

    Args:
        n (int): number of Chebyshev points to return

    Returns:
        jax.Array: The sampled points in [-1, 1]
    """
    cos_args = (n - 1 - jnp.arange(n, dtype=jnp.float64) + 0.5) / n
    angles = jnp.pi * cos_args
    pts = jnp.cos(angles)
    return pts


@partial(jax.jit, static_argnums=(0,))
def chebyshev_weights(n: int, bounds: jnp.array) -> jnp.array:
    """
    Generates weights for a Chebyshev quadrature rule with n points over the interval [a, b].

    Uses the Clenshaw-Curtis quadrature rule, specifically the version used in Chebfun. See [3]_.

    Args:
        n (int): Number of quadrature points
        bounds (jnp.array): Has shape (2,) and contains the interval endpoints [a, b]

    Returns:
        jnp.array: Has shape (n,) and contains the quadrature weights
    """
    a, b = bounds
    interval_len = b - a

    c = 2.0 / jnp.concatenate(
        [jnp.array([1.0]), 1.0 - jnp.arange(2, n, 2) ** 2]
    )

    if n % 2:
        # # Mirror for DCT via FFT
        start = n // 2
        c_slice = jnp.flip(c[1:start])
        c = jnp.concatenate([c, c_slice])

        w = jnp.fft.ifft(c).real
        w_out = jnp.concatenate([w, jnp.array([w[0] / 2])])
        w_out = w_out.at[0].set(w[0] / 2)
    else:
        c = 2.0 / jnp.concatenate(
            [jnp.array([1.0]), 1.0 - jnp.arange(2, n, 2) ** 2]
        )
        # Mirror for DCT via FFT
        start = n // 2 + 1
        c_slice = jnp.flip(c[1:start])
        c = jnp.concatenate([c, c_slice])

        # c = jnp.fft.ifftshift(c)
        w = jnp.fft.ifft(c).real
        w_out = jnp.concatenate([w, jnp.array([w[0] / 2])])
        w_out = w_out.at[0].set(w[0] / 2)

    # Scale by interval length
    w_out = w_out * interval_len / 2
    return w_out


@jax.jit
def affine_transform(pts: jax.Array, ab: jax.Array) -> jax.Array:
    """Affine transforms the points pts, which are assumed to be
    in the interval [-1, 1], to the interval [a, b].

    Args:
        pts (jax.Array): Has shape (n,)
        ab (jax.Array): Has shape (2,)
    Returns:
        jax.Array: Has shape (n,)
    """
    a, b = ab
    return 0.5 * (b - a) * pts + 0.5 * (a + b)


@partial(jax.jit, static_argnums=(0,))
def gauss_points(n: int) -> jax.Array:
    """
    Returns n Gauss-Legendre points over the interval [-1, 1]. This is a wrapper for ``numpy.polynomial.legendre.leggauss``.

    Args:
        n (int): Number of points

    Returns:
        jax.Array: Has shape (n,)
    """
    gauss_pts_1d = np.polynomial.legendre.leggauss(n)[0]
    return jnp.array(gauss_pts_1d)
