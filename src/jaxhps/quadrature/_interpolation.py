"""
This file contains utility functions for quadrature points and weights. It
defines Chebyshev and Gauss-Legendre quadrature points and weights.
It has utilites for interpolation maps between different quadratures, as well
as 1D differentiation matrices defined over 2D Chebyshev grids.
"""

import jax.numpy as jnp
import jax


jax.config.update("jax_enable_x64", True)

EPS = jnp.finfo(jnp.float64).eps


@jax.jit
def barycentric_lagrange_interpolation_matrix_1D(
    from_pts: jax.Array, to_pts: jax.Array
) -> jax.Array:
    """
    Generates a Lagrange 1D polynomial interpolation matrix, which interpolates
    from the points in from_pts to the points in to_pts.

    This function uses the barycentric formula for Lagrange interpolation, from [1]_

    Args:
        from_pts (jax.Array): Has shape (n,)
        to_pts (jax.Array): Has shape (p,)

    Returns:
        jax.Array: Has shape (p,n)
    """
    p = from_pts.shape[0]
    # n = to_pts.shape[0]

    # Compute the inverses of the Barycentric weights
    # (2025-07-01, OOT) Vectorized version
    from_di = jnp.arange(p)  # for use as diagonal indices for from_pts_x
    tmp_from_dist = from_pts[:, jnp.newaxis] - from_pts[jnp.newaxis, :]
    tmp_from_dist = tmp_from_dist.at[from_di, from_di].set(1)
    w = jnp.prod(tmp_from_dist, axis=0)
    # # Original version
    # w = jnp.ones(p, dtype=jnp.float64)
    # for j in range(p):
    #     for k in range(p):
    #         if j != k:
    #             w = w.at[j].mul(from_pts[j] - from_pts[k])

    # print("barycentric_lagrange_interpolation_matrix: w", w)

    # Normalizing factor is sum_j w_j / (x - x_j)
    # (2025-07-01, OOT) Vectorized version
    norm_factors = jnp.sum(
        1
        / (
            w[:, jnp.newaxis]
            * (to_pts[jnp.newaxis, :] - from_pts[:, jnp.newaxis])
        ),
        axis=0,
    )
    # # Original version
    # norm_factors_ref = jnp.zeros(n, dtype=jnp.float64)
    # for i in range(p):
    #     norm_factors_ref += 1 / (w[i] * (to_pts - from_pts[i]))
    # print(jnp.all(norm_factors_ref==norm_factors))

    # print("barycentric_lagrange_interpolation_matrix: norm_factors", norm_factors)

    # Compute the matrix
    # (2025-07-01, OOT) Vectorized version
    matrix = 1 / (
        (to_pts[:, jnp.newaxis] - from_pts[jnp.newaxis, :])
        * w[jnp.newaxis, :]
        * norm_factors[:, jnp.newaxis]
    )
    # # Original version
    # matrix_ref = jnp.zeros((n, p), dtype=jnp.float64)
    # for i in range(n):
    #     for j in range(p):
    #         matrix_ref = matrix_ref.at[i, j].set(
    #             1 / ((to_pts[i] - from_pts[j]) * w[j] * norm_factors[i])
    #         )
    # print(jnp.all(matrix==matrix_ref))

    # Check if any of the source and target points overlap
    # This code is semantically the same as what comes after.
    # The code below is vectorized and is able to be compiled because it does not
    # use the conditionals on to_pts and from_pts.

    # for i in range(n):
    #     for j in range(p):
    #         if to_pts[i] == from_pts[j]:
    #             matrix = matrix.at[i, :].set(0)
    #             matrix = matrix.at[i, j].set(1)

    # Create a boolean mask for matching points
    matches = to_pts[:, None] == from_pts[None, :]  # Shape: (n, p)

    # Create row masks for any matching points
    has_match = matches.any(axis=1)  # Shape: (n,)

    # Update the matrix
    matrix = jnp.where(
        has_match[:, None],  # Broadcasting to shape (n, p)
        jnp.where(
            matches,
            1.0,  # Where points match
            0.0,  # Where points don't match but row has a match
        ),
        matrix,  # Keep original values where row has no matches
    )

    return matrix


@jax.jit
def barycentric_lagrange_interpolation_matrix_2D(
    from_pts_x: jax.Array,
    from_pts_y: jax.Array,
    to_pts_x: jax.Array,
    to_pts_y: jax.Array,
) -> jax.Array:
    """
    2D Barycentric Lagrange interpolation matrix. A generalization of [1]_,
    modeled after the MATLAB code snippet [2]_.

    The grid of source points is specified by ``from_pts_x`` and ``from_pts_y``. The
    resulting matrix has columns ordered to map from samples on this list of points:

    .. code:: python

       source_X, source_Y = jnp.meshgrid(from_pts_x, from_pts_y, indexing="ij")
       source_pts = jnp.stack((source_X.flatten(), source_Y.flatten()), axis=-1)

    Similarly, the rows are ordered to assume a grid of target points specified by:

    .. code:: python

       target_X, target_Y = jnp.meshgrid(to_pts_x, to_pts_y, indexing="ij")
       target_pts = jnp.stack((target_X.flatten(), target_Y.flatten()), axis=-1)


    Args:
        from_pts_x (jax.Array): Has shape (n_x,)
        from_pts_y (jax.Array): Has shape (n_y,)
        to_pts_x (jax.Array): Has shape (p_x,)
        to_pts_y (jax.Array): Has shape (p_y,)

    Returns:
        jax.Array: Has shape (p_x * p_y, n_x * n_y)
    """
    n_x = from_pts_x.shape[0]
    p_x = to_pts_x.shape[0]
    n_y = from_pts_y.shape[0]
    p_y = to_pts_y.shape[0]
    # print("barycentric_lagrange_2d_interpolation_matrix: n, p", n, p)

    # Compute the inverses of the barycentric weights for x and y dimensions.
    # # w_x[j] = \prod_{k != j} (from_pts_x[j] - from_pts_x[k])
    # (2025-06-20, OOT) Vectorized version
    # Seems to be a bit faster than the original
    from_x_di = jnp.arange(n_x)  # for use as diagonal indices for from_pts_x
    tmp_from_xdist = from_pts_x[:, jnp.newaxis] - from_pts_x[jnp.newaxis, :]
    tmp_from_xdist = tmp_from_xdist.at[from_x_di, from_x_di].set(1)
    w_x = jnp.prod(tmp_from_xdist, axis=0)
    from_y_di = jnp.arange(n_y)
    tmp_from_ydist = from_pts_y[:, jnp.newaxis] - from_pts_y[jnp.newaxis, :]
    tmp_from_ydist = tmp_from_ydist.at[from_y_di, from_y_di].set(1)
    w_y = jnp.prod(tmp_from_ydist, axis=0)

    # # Original version
    # w_x = jnp.ones(n_x, dtype=jnp.float64)
    # w_y = jnp.ones(n_y, dtype=jnp.float64)
    # for j in range(n_x):
    #     for k in range(n_x):
    #         if j != k:
    #             w_x = w_x.at[j].mul(from_pts_x[j] - from_pts_x[k])
    # for j in range(n_y):
    #     for k in range(n_y):
    #         if j != k:
    #             w_y = w_y.at[j].mul(from_pts_y[j] - from_pts_y[k])

    # Compute matrix of distances between x and y points.
    xdist = to_pts_x[None, :] - from_pts_x[:, None]
    ydist = to_pts_y[None, :] - from_pts_y[:, None]
    # print("barycentric_lagrange_2d_interpolation_matrix: xdist", xdist.shape)

    # Replace exact 0's with EPS. This is to avoid division by zero.
    # This is a bit of a hack and the proper way to do this is identifying which
    # rows/cols of the matrix need to be amended using 1D interpolation maps.
    # But this is a good quick fix.
    xdist = jnp.where(xdist == 0, EPS, xdist)
    ydist = jnp.where(ydist == 0, EPS, ydist)

    # Compute the normalization factors for x and y dimensions.
    norm_factors_x = jnp.sum(1 / (w_x[:, None] * (xdist)), axis=0)
    norm_factors_y = jnp.sum(1 / (w_y[:, None] * (ydist)), axis=0)

    # Compute the matrix, iterating over the y_pts first.
    i, j, k, l = jnp.indices((p_x, p_y, n_x, n_y))
    matrix = 1 / (
        xdist[k, i]
        * ydist[l, j]
        * w_x[k]
        * w_y[l]
        * norm_factors_x[i]
        * norm_factors_y[j]
    )

    matrix = matrix.reshape(p_x * p_y, n_x * n_y)

    return matrix


@jax.jit
def barycentric_lagrange_interpolation_matrix_3D(
    from_pts_x: jax.Array,
    from_pts_y: jax.Array,
    from_pts_z: jax.Array,
    to_pts_x: jax.Array,
    to_pts_y: jax.Array,
    to_pts_z: jax.Array,
) -> jax.Array:
    """
    3D Barycentric Lagrange interpolation matrix. A generalization of [1]_.

    The grid of source points is specified by ``from_pts_x``, ``from_pts_y``, and ``from_pts_z``. The
    resulting matrix has columns ordered to map from samples on this list of points:

    .. code:: python

       source_X, source_Y, source_Z = jnp.meshgrid(from_pts_x, from_pts_y, from_pts_z indexing="ij")
       source_pts = jnp.stack((source_X.flatten(), source_Y.flatten(), source_Z.flatten()), axis=-1)

    Similarly, the rows are ordered to assume a grid of target points specified by:

    .. code:: python

       target_X, target_Y, target_Z = jnp.meshgrid(to_pts_x, to_pts_y, to_pts_z, indexing="ij")
       target_pts = jnp.stack((target_X.flatten(), target_Y.flatten(), target_Z.flatten()), axis=-1)

    Args:
        from_pts_x (jax.Array): Has shape (n_x,)
        from_pts_y (jax.Array): Has shape (n_y,)
        from_pts_z (jax.Array): Has shape (n_z,)
        to_pts_x (jax.Array): Has shape (p_x,)
        to_pts_y (jax.Array): Has shape (p_y,)
        to_pts_z (jax.Array): Has shape (p_z,)

    Returns:
        jax.Array: Has shape (p_x * p_y * p_z, n_x * n_y * n_z)
    """

    n_x = from_pts_x.shape[0]
    p_x = to_pts_x.shape[0]

    n_y = from_pts_y.shape[0]
    p_y = to_pts_y.shape[0]

    n_z = from_pts_z.shape[0]
    p_z = to_pts_z.shape[0]

    # Compute the inverses of the barycentric weights for x, y, and z dimensions.
    # (2025-07-01, OOT) Vectorized version
    from_x_di = jnp.arange(n_x)  # for use as diagonal indices for from_pts_x
    tmp_from_xdist = from_pts_x[:, jnp.newaxis] - from_pts_x[jnp.newaxis, :]
    tmp_from_xdist = tmp_from_xdist.at[from_x_di, from_x_di].set(1)
    w_x = jnp.prod(tmp_from_xdist, axis=0)
    from_y_di = jnp.arange(n_y)  # for use as diagonal indices for from_pts_y
    tmp_from_ydist = from_pts_y[:, jnp.newaxis] - from_pts_y[jnp.newaxis, :]
    tmp_from_ydist = tmp_from_ydist.at[from_y_di, from_y_di].set(1)
    w_y = jnp.prod(tmp_from_ydist, axis=0)
    from_z_di = jnp.arange(n_z)  # for use as diagonal indices for from_pts_z
    tmp_from_zdist = from_pts_z[:, jnp.newaxis] - from_pts_z[jnp.newaxis, :]
    tmp_from_zdist = tmp_from_zdist.at[from_z_di, from_z_di].set(1)
    w_z = jnp.prod(tmp_from_zdist, axis=0)
    # # Original version
    # w_x = jnp.ones(n_x, dtype=jnp.float64)
    # for j in range(n_x):
    #     for k in range(n_x):
    #         if j != k:
    #             w_x = w_x.at[j].mul(from_pts_x[j] - from_pts_x[k])
    # w_y = jnp.ones(n_y, dtype=jnp.float64)
    # for j in range(n_y):
    #     for k in range(n_y):
    #         if j != k:
    #             w_y = w_y.at[j].mul(from_pts_y[j] - from_pts_y[k])
    # w_z = jnp.ones(n_z, dtype=jnp.float64)
    # for j in range(n_z):
    #     for k in range(n_z):
    #         if j != k:
    #                 w_z = w_z.at[j].mul(from_pts_z[j] - from_pts_z[k])

    # Compute the normalization factors for x, y, and z dimensions.
    xdist = to_pts_x[None, :] - from_pts_x[:, None]
    ydist = to_pts_y[None, :] - from_pts_y[:, None]
    zdist = to_pts_z[None, :] - from_pts_z[:, None]

    # Replace 0's in the denominator with EPS to avoid division by zero.
    xdist = jnp.where(xdist == 0, EPS, xdist)
    ydist = jnp.where(ydist == 0, EPS, ydist)
    zdist = jnp.where(zdist == 0, EPS, zdist)

    norm_factors_x = jnp.sum(1 / (w_x[:, None] * xdist), axis=0)
    norm_factors_y = jnp.sum(1 / (w_y[:, None] * ydist), axis=0)
    norm_factors_z = jnp.sum(1 / (w_z[:, None] * zdist), axis=0)

    # Compute the matrix, iterating over the z_pts first.
    i, j, k, l, m, o = jnp.indices((p_x, p_y, p_z, n_x, n_y, n_z))

    matrix = 1 / (
        xdist[l, i]
        * ydist[m, j]
        * zdist[o, k]
        * w_x[l]
        * w_y[m]
        * w_z[o]
        * norm_factors_x[i]
        * norm_factors_y[j]
        * norm_factors_z[k]
    )

    mat_shape = (p_x * p_y * p_z, n_x * n_y * n_z)
    matrix = matrix.reshape(mat_shape)
    return matrix
