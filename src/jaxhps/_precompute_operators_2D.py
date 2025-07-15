from typing import Tuple
import jax.numpy as jnp
import jax
from ._grid_creation_2D import rearrange_indices_ext_int
from .quadrature import (
    differentiation_matrix_1D,
    chebyshev_points,
    first_kind_chebyshev_points,
    gauss_points,
    affine_transform,
    barycentric_lagrange_interpolation_matrix_1D,
    barycentric_lagrange_interpolation_matrix_2D,
)
from functools import partial


def precompute_diff_operators_2D(
    p: int, half_side_len: float
) -> Tuple[jax.Array]:
    """
    Returns D_x, D_y, D_xx, D_yy, D_xy
    """
    rearrange_indices = rearrange_indices_ext_int(p)

    pts = chebyshev_points(p)
    cheby_diff_matrix = differentiation_matrix_1D(pts) / half_side_len

    # Precompute du/dx and du/dy
    du_dx = jnp.kron(cheby_diff_matrix, jnp.eye(p))
    du_dy = -1 * jnp.kron(jnp.eye(p), cheby_diff_matrix)

    # Permute the rows and cols of du_dx and du_dy to match the ordering of the Chebyshev points.
    du_dx = du_dx[rearrange_indices, :]
    du_dx = du_dx[:, rearrange_indices]

    du_dy = du_dy[rearrange_indices, :]
    du_dy = du_dy[:, rearrange_indices]

    return (
        du_dx,
        du_dy,
        du_dx @ du_dx,
        du_dy @ du_dy,
        du_dx @ du_dy,
    )


def rectangular_interp_operator(p: int) -> jax.Array:
    """
    Returns an interpolation operator that maps from a grid of p^2 2nd-kind Chebyshev points to a grid of (p-2)^2 1st-kind Chebyshev points.

    Args:
        p (int): Number of points per dimension on the 2nd-kind Chebyshev grid.

    Returns:
        jax.Array: Has shape ((p - 2) ** 2, p ** 2).
    """

    # Precompute a 2D barycentric Lagrange interpolation matrix
    cheby_pts_1d = chebyshev_points(p)
    from_x = cheby_pts_1d
    from_y = jnp.flipud(cheby_pts_1d)
    to_x = first_kind_chebyshev_points(p - 2)
    to_y = jnp.flipud(to_x)
    B = barycentric_lagrange_interpolation_matrix_2D(
        from_x, from_y, to_x, to_y
    )
    # Rearrange the cols of the interpolation matrix to match the expected grid ordering
    rearrange_indices = rearrange_indices_ext_int(p)
    B = B[:, rearrange_indices]
    return B


def precompute_rectangular_diff_operators_2D(
    p: int, half_side_len: float
) -> Tuple[jax.Array]:
    """
    Returns rectangular spectral differentiation operators. That is, each operator maps from a 2D
    grid of p^2 Chebyshev points of the second kind to a grid of (p-2)^2 Chebyshev points of the first kind.

    See this paper for details: "Rectangular spectral collocation", Tobin A. Driscoll and Nicholas Hale. https://doi.org/10.1093/imanum/dru062

    Returns, in order, D_x, D_y, D_xx, D_yy, D_xy, B. The final output, B, is the interpolation matrix from the p^2 grid to the
    (p-2)^2 grid.

    Args:
        p (int): Number of Chebyshev grid points in each dimension.
        half_side_len (float): Half the length of the side of the square domain.

    Returns:
        Tuple[jax.Array]: D_x, D_y, D_xx, D_yy, D_xy, B. Each matrix has shape (p**2 - 4p + 4, p**2)
    """

    D_x, D_y, D_xx, D_yy, D_xy = precompute_diff_operators_2D(p, half_side_len)

    B = rectangular_interp_operator(p)

    # Apply the barycentric Lagrange interpolation matrix to the differentiation operators
    D_x = B @ D_x
    D_y = B @ D_y
    D_xx = B @ D_xx
    D_yy = B @ D_yy
    D_xy = B @ D_xy

    return (D_x, D_y, D_xx, D_yy, D_xy, B)


# Unclear whether this is a win to jit this one.
@partial(jax.jit, static_argnums=(0, 1))
def precompute_P_2D_DtN(p: int, q: int) -> jax.Array:
    """
    Precomputes the function mapping from 4q Gauss points to
    4(p-1) Chebyshev points on the boundary.

    Averages the values at the corners of the boundary.
    """
    gauss_pts = gauss_points(q)
    cheby_pts = chebyshev_points(p)

    n_cheby_bdry_pts = 4 * (p - 1)

    # P = legendre_interpolation_matrix(cheby_pts, node.q)
    P = barycentric_lagrange_interpolation_matrix_1D(gauss_pts, cheby_pts)

    I_P = jnp.zeros((n_cheby_bdry_pts, 4 * q), dtype=jnp.float64)
    # First block: mapping from first q Gauss points to first p Cheby points
    I_P = I_P.at[0:p, 0:q].set(P)
    # Second block: mapping from second q Gauss points to second p Cheby points
    I_P = I_P.at[p - 1 : 2 * p - 1, q : 2 * q].set(P)
    # Third block: mapping from third q Gauss points to third p Cheby points
    I_P = I_P.at[2 * (p - 1) : 3 * p - 2, 2 * q : 3 * q].set(P)
    # Fourth block: mapping from fourth q Gauss points to fourth p Cheby points
    I_P = I_P.at[3 * (p - 1) :, 3 * q :].set(P[:-1])
    I_P = I_P.at[0, 3 * q :].set(P[-1])

    # Take averages of the corners in indices 0, p - 2, 2(p - 1) - 1, 3(p - 1) - 1.
    for row_idx in [0, p - 1, 2 * (p - 1), 3 * (p - 1)]:
        I_P = I_P.at[row_idx, :].set(I_P[row_idx, :] / 2)

    return I_P


# Unclear whether this is a win to jit this one.
@partial(jax.jit, static_argnums=(0, 1))
def precompute_P_2D_ItI(p: int, q: int) -> jnp.array:
    """
    Maps the boundary impedance data to the Chebyshev nodes on the boundary.
    Is formed by taking the kronecker product of I and P_0, which is the standard
    Gauss -> Cheby 1D interp matrix missing the last row.
    Returns:
        I_P_0: has shape (4 * (p - 1), 4 * q).
    """
    gauss_pts = gauss_points(q)
    cheby_pts = chebyshev_points(p)
    P = barycentric_lagrange_interpolation_matrix_1D(gauss_pts, cheby_pts)
    # P = precompute_P_matrix(p, q)
    P_0 = P[:-1]
    I = jnp.eye(4)
    I_P_0 = jnp.kron(I, P_0)
    return I_P_0


@partial(jax.jit, static_argnums=(0, 1))
def precompute_Q_2D_DtN(
    p: int, q: int, du_dx: jax.Array, du_dy: jax.Array
) -> jax.Array:
    # N_dbl = jnp.full((4 * p, p**2), np.nan, dtype=jnp.float64)

    # # S boundary
    # N_dbl = N_dbl.at[:p].set(-1 * du_dy[:p])
    # # E boundary
    # N_dbl = N_dbl.at[p : 2 * p].set(du_dx[p - 1 : 2 * p - 1])
    # # N boundary
    # N_dbl = N_dbl.at[2 * p : 3 * p].set(du_dy[2 * p - 2 : 3 * p - 2])
    # # W boundary
    # N_dbl = N_dbl.at[3 * p :].set(-1 * du_dx[3 * p - 3 : 4 * p - 3])
    # N_dbl = N_dbl.at[-1].set(-1 * du_dx[0])
    N_dbl = precompute_N_matrix_2D(du_dx, du_dy, p)

    cheby_pts = chebyshev_points(p)
    gauss_pts = gauss_points(q)

    # Q_I maps from points on the Chebyshev boundary to points
    # on the Gauss boundary.
    Q = barycentric_lagrange_interpolation_matrix_1D(cheby_pts, gauss_pts)
    Q_I = jnp.kron(jnp.eye(4), Q)

    Q_D = Q_I @ N_dbl

    return Q_D


@partial(jax.jit, static_argnums=(2,))
def precompute_N_matrix_2D(
    du_dx: jnp.array, du_dy: jnp.array, p: int
) -> jnp.array:
    """
    The N matrix is a 4p x p^2 matrix that maps a solution on the
    Cheby points to the outward normal derivatives on the Cheby boundaries.
    This matrix double-counts the corners, i.e. the derivative
    at the corner is evaluated twice, once for each side
    it is on.

    Args:
        du_dx (jnp.array): Has shape (p**2, p**2)
        du_dy (jnp.array): Has shape (p**2, p**2)

    Returns:
        jnp.array: Has shape (4p, p**2)
    """

    N_dbl = jnp.full((4 * p, p**2), 0.0, dtype=jnp.float64)

    # S boundary
    N_dbl = N_dbl.at[:p].set(-1 * du_dy[:p])
    # E boundary
    N_dbl = N_dbl.at[p : 2 * p].set(du_dx[p - 1 : 2 * p - 1])
    # N boundary
    N_dbl = N_dbl.at[2 * p : 3 * p].set(du_dy[2 * p - 2 : 3 * p - 2])
    # W boundary
    N_dbl = N_dbl.at[3 * p :].set(-1 * du_dx[3 * p - 3 : 4 * p - 3])
    N_dbl = N_dbl.at[-1].set(-1 * du_dx[0])

    return N_dbl


@partial(jax.jit, static_argnums=(2,))
def precompute_N_tilde_matrix_2D(
    du_dx: jnp.array, du_dy: jnp.array, p: int
) -> jnp.array:
    """
    Implements an operator mapping from samples on the Chebyshev grid points to normal derivatives at the 4*(p-1) boundary points.

    Args:
        du_dx (jnp.array): Precomputed differential operator for x-direction. Has shape (p**2, p**2)
        du_dy (jnp.array): Precomputed differential operator for y-direction. Has shape (p**2, p**2)
        p (int): Shape parameter.

    Returns:
        jnp.array: Has shape (4*(p-1), p**2)
    """

    N_tilde = jnp.full((4 * (p - 1), p**2), 0.0, dtype=jnp.float64)

    N_tilde = N_tilde.at[: p - 1].set(-1 * du_dy[: p - 1])
    N_tilde = N_tilde.at[p - 1 : 2 * (p - 1)].set(du_dx[p - 1 : 2 * (p - 1)])
    N_tilde = N_tilde.at[2 * (p - 1) : 3 * (p - 1)].set(
        du_dy[2 * (p - 1) : 3 * (p - 1)]
    )
    N_tilde = N_tilde.at[3 * (p - 1) :].set(
        -1 * du_dx[3 * (p - 1) : 4 * (p - 1)]
    )
    return N_tilde


@partial(jax.jit, static_argnums=(1, 2))
def precompute_QH_2D_ItI(
    N: jnp.array, p: int, q: int, eta: float
) -> jnp.array:
    """
    H is the matrix which maps functions on a
    2D Chebyshev grid to outgoing impedance data on the 4p
    Chebyshev boundary points, which include each corner twice.
    It's composed with Q, which is a block-diagonal matrix of shape (4q, 4p) mapping
    from Chebyshev boundary points to Gauss boundary points.

    Args:
        N (jnp.array): Has shape (4p, p**2). Is the result of precompute_N_matrix().
        p (int): Shape parameter.
        q (int): Shape parameter.
        eta (float): Real number

    Returns:
        jnp.array: Has shape (4q, p**2)
    """
    H = N.astype(jnp.complex128)
    # S side rows 0:p and cols 0:p
    H = H.at[:p, :p].set(H[:p, :p] - 1j * eta * jnp.eye(p))
    # E side rows p:2p and cols p-1:2p-1
    H = H.at[p : 2 * p, p - 1 : 2 * p - 1].set(
        H[p : 2 * p, p - 1 : 2 * p - 1] - 1j * eta * jnp.eye(p)
    )
    # N side rows 2p:3p and cols 2p-2:3p-2
    H = H.at[2 * p : 3 * p, 2 * p - 2 : 3 * p - 2].set(
        H[2 * p : 3 * p, 2 * p - 2 : 3 * p - 2] - 1j * eta * jnp.eye(p)
    )
    # W side rows 3p:4p, cols 3p-3:4p-4, 0
    H = H.at[3 * p : 4 * p - 1, 3 * p - 3 : 4 * p - 4].set(
        H[3 * p : 4 * p - 1, 3 * p - 3 : 4 * p - 4] - 1j * eta * jnp.eye(p - 1)
    )
    H = H.at[4 * p - 1, 0].set(H[4 * p - 1, 0] - 1j * eta)

    cheby_pts = chebyshev_points(p)
    gauss_pts = gauss_points(q)

    # Q_I maps from points on the Chebyshev boundary to points
    # on the Gauss boundary.
    Q = barycentric_lagrange_interpolation_matrix_1D(cheby_pts, gauss_pts)
    Q_I = jnp.kron(jnp.eye(4), Q)

    return Q_I @ H


@jax.jit
def precompute_G_2D_ItI(N_tilde: jnp.array, eta: float) -> jnp.array:
    """
    F = N_tilde + i eta I[:4(p-1)] is the matrix which maps functions on a
    2D Chebyshev grid to incoming impedance data on the 4(p - 1)
    Chebyshev boundary points.

    Args:
        N_tilde (jnp.array): Has shape (4(p - 1), p**2). Is the result of precompute_N_tilde_matrix().
        p (int): Shape parameter
        eta (float): Real number

    Returns:
        jnp.array: Has shape (4(p - 1), p**2)
    """
    shape_0 = N_tilde.shape[0]
    F = N_tilde.astype(jnp.complex128)
    # Add i eta I to the top block of F
    F = F.at[:shape_0, :shape_0].set(
        F[:shape_0, :shape_0] + 1j * eta * jnp.eye(shape_0)
    )

    return F


@partial(jax.jit, static_argnums=(0,))
def precompute_projection_ops_2D(
    q: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precomputes a "refining" interpolation operator which maps from one
    Gauss-Legendre panel to two Gauss-Legendre panels, and a "coarsening"
    interpolation operator which maps from two Gauss-Legendre panels to one.

    Args:
        q (int): Number of Gauss-Legendre points on one panel

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The refining and coarsening operators
            refining operator has shape (2*q, q)
            coarsening operator has shape (q, 2*q)
    """
    gauss_pts = gauss_points(q)
    gauss_pts_refined = jnp.concatenate(
        [
            affine_transform(gauss_pts, jnp.array([-1.0, 0.0])),
            affine_transform(gauss_pts, jnp.array([0.0, 1.0])),
        ]
    )

    L_2f1 = barycentric_lagrange_interpolation_matrix_1D(
        gauss_pts, gauss_pts_refined
    )
    L_1f2 = barycentric_lagrange_interpolation_matrix_1D(
        gauss_pts_refined, gauss_pts
    )

    return L_2f1, L_1f2


def precompute_L_4f1(p: int) -> jnp.array:
    """This is an interpolation matrix that maps from a pxp Chebyshev grid points to
    4 copies of a pxp Chebyshev grid. i.e. the refinement of the grid.

    Args:
        p (int): Number of Chebyshev grid points in one direction
    Returns:
        jnp.array: Interpolation matrix with shape (4*p**2, p**2)
    """
    cheby_pts_1d = chebyshev_points(p)
    cheby_pts_refined = jnp.concatenate(
        [
            affine_transform(cheby_pts_1d, jnp.array([-1, 0])),
            affine_transform(cheby_pts_1d, jnp.array([0, 1])),
        ]
    )

    I_refined = barycentric_lagrange_interpolation_matrix_2D(
        cheby_pts_1d, cheby_pts_1d, cheby_pts_refined, cheby_pts_refined
    )

    r, c = indexing_for_refinement_operator(p)

    I_refined = I_refined[r, :]
    I_refined = I_refined[:, c]
    return I_refined


def indexing_for_refinement_operator(p: int) -> jnp.array:
    """Returns row and column indexing to rearrange the refinement_operator matrix.

    Before reordering, that matrix has rows corresponding to a meshgrid of (cheby_pts_1d, cheby_pts_1d)
    and cols corresponding to a meshgrid of (cheby_pts_refined, cheby_pts_refined). After reordering,
    we want the rows to be ordered in the standard way, putting the exterior points first and then the
    interior points. We also want the columns to be ordered in the standard way, putting each of the
    four blocks of the meshgrid together and then ordering the points in each block so that the exterior
    points come first.

    Returns:
        jnp.array: r: row indices to rearrange the rows of the matrix
                   c: column indices to rearrange the columns of the matrix
    """
    col_idxes = rearrange_indices_ext_int(p)

    ii = jnp.arange(4 * p**2)

    # a is where x is in the first half and y is in the first half
    a_bools = jnp.logical_and(ii % (2 * p) >= p, (ii // (2 * p)) % (2 * p) < p)
    a_idxes = ii[a_bools]
    a_idxes = a_idxes[col_idxes]

    # b is where x is in the second half and y is in the first half
    b_bools = jnp.logical_and(
        ii % (2 * p) >= p, (ii // (2 * p)) % (2 * p) >= p
    )
    b_idxes = ii[b_bools]
    b_idxes = b_idxes[col_idxes]

    # c is where x is in the second half and y is in the second half
    c_bools = jnp.logical_and(ii % (2 * p) < p, (ii // (2 * p)) % (2 * p) >= p)
    c_idxes = ii[c_bools]
    c_idxes = c_idxes[col_idxes]

    # d is where x is in the first half and y is in the second half
    d_bools = jnp.logical_and(ii % (2 * p) < p, (ii // (2 * p)) % (2 * p) < p)
    d_idxes = ii[d_bools]
    d_idxes = d_idxes[col_idxes]

    row_idxes = jnp.concatenate(
        [
            a_idxes,
            b_idxes,
            c_idxes,
            d_idxes,
        ]
    )
    return row_idxes, col_idxes
