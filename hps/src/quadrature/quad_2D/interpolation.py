from typing import Tuple
from functools import partial
import jax.numpy as jnp
import jax
import numpy as np
from hps.src.quadrature.quad_2D.indexing import (
    _rearrange_indices,
    indexing_for_refinement_operator,
)
from hps.src.quadrature.quadrature_utils import (
    chebyshev_points,
    differentiation_matrix_1d,
    barycentric_lagrange_interpolation_matrix,
    barycentric_lagrange_2d_interpolation_matrix,
    affine_transform,
)
from hps.src.quadrature.quad_2D.differentiation import precompute_N_matrix
from hps.src.quadrature.quad_2D.grid_creation import vmapped_corners
from hps.src.quadrature.trees import Node, get_all_leaves
from hps.src.utils import meshgrid_to_lst_of_pts


def refinement_operator(p: int) -> jnp.array:
    """This is an interpolation matrix that maps from a pxp Chebyshev grid points to
    4 copies of a pxp Chebyshev grid. i.e. the refinement of the grid.

    Args:
        p (int): Number of Chebyshev grid points in one direction
    Returns:
        jnp.array: Interpolation matrix with shape (4*p**2, p**2)
    """
    cheby_pts_1d = chebyshev_points(p)[0]
    cheby_pts_refined = jnp.concatenate(
        [
            affine_transform(cheby_pts_1d, jnp.array([-1, 0])),
            affine_transform(cheby_pts_1d, jnp.array([0, 1])),
        ]
    )

    I_refined = barycentric_lagrange_2d_interpolation_matrix(
        cheby_pts_1d, cheby_pts_1d, cheby_pts_refined, cheby_pts_refined
    )

    r, c = indexing_for_refinement_operator(p)

    I_refined = I_refined[r, :]
    I_refined = I_refined[:, c]
    return I_refined


def precompute_P_matrix(p: int, q: int) -> jnp.ndarray:
    """Precomputes the function mapping from 4q Gauss points to
    4(p-1) Chebyshev points on the boundary.

    Averages the values at the corners of the boundary.
    """
    gauss_pts = np.polynomial.legendre.leggauss(q)[0]
    cheby_pts = chebyshev_points(p)[0]

    n_cheby_bdry_pts = 4 * (p - 1)

    # P = legendre_interpolation_matrix(cheby_pts, node.q)
    P = barycentric_lagrange_interpolation_matrix(gauss_pts, cheby_pts)

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


@partial(jax.jit, static_argnums=(0, 1))
def precompute_Q_D_matrix(
    p: int, q: int, du_dx: jnp.ndarray, du_dy: jnp.ndarray
) -> jnp.ndarray:
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
    N_dbl = precompute_N_matrix(du_dx, du_dy, p)

    # Q_I maps from points on the Chebyshev boundary to points
    # on the Gauss boundary.
    Q_I = precompute_Q_I_matrix(p, q)

    Q_D = Q_I @ N_dbl

    return Q_D


def precompute_I_P_0_matrix(p: int, q: int) -> jnp.array:
    """
    Maps the boundary impedance data to the Chebyshev nodes on the boundary.
    Is formed by taking the kronecker product of I and P_0, which is the standard
    Gauss -> Cheby 1D interp matrix missing the last row.
    Returns:
        I_P_0: has shape (4 * (p - 1), 4 * q).
    """
    gauss_pts = np.polynomial.legendre.leggauss(q)[0]
    cheby_pts = chebyshev_points(p)[0]
    P = barycentric_lagrange_interpolation_matrix(gauss_pts, cheby_pts)
    # P = precompute_P_matrix(p, q)
    P_0 = P[:-1]
    I = jnp.eye(4)
    I_P_0 = jnp.kron(I, P_0)
    return I_P_0


def precompute_Q_I_matrix(p: int, q: int) -> jnp.array:
    gauss_pts = np.polynomial.legendre.leggauss(q)[0]
    cheby_pts = chebyshev_points(p)[0]

    # Q maps from p Cheby points to q Gauss points
    # Q = chebyshev_interpolation_matrix(node.p - 1, cheby_pts, gauss_pts)
    Q = barycentric_lagrange_interpolation_matrix(cheby_pts, gauss_pts)
    Q_I = jnp.kron(jnp.eye(4), Q)
    return Q_I

    return Q_D


@partial(jax.jit, static_argnums=(0,))
def precompute_refining_coarsening_ops(q: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    gauss_pts = np.polynomial.legendre.leggauss(q)[0]
    gauss_pts_refined = jnp.concatenate(
        [
            affine_transform(gauss_pts, jnp.array([-1.0, 0.0])),
            affine_transform(gauss_pts, jnp.array([0.0, 1.0])),
        ]
    )

    refining_op = barycentric_lagrange_interpolation_matrix(
        gauss_pts, gauss_pts_refined
    )
    coarsening_op = barycentric_lagrange_interpolation_matrix(
        gauss_pts_refined, gauss_pts
    )

    return refining_op, coarsening_op


def interp_operator_to_regular(
    root: Node, from_grid: jnp.array, to_x: jnp.array, to_y: jnp.array, p: int
) -> jnp.array:
    """
    Given a non-uniform quadrature, specified by the tree root and the array of
    quadrature points from_grid, this function returns a matrix that interpolates
    the function values from the non-uniform grid to a regular grid specified by
    the arrays to_x and to_y.

    The interpolation is local, i.e. the for a given point in the regular grid,
    the interpolation matrix only interpolates from one of the leaves of the tree.
    There will be some edge cases where one of the output points is on the boundary
    of two or more leaves, in which case the output is the average of the interpolations
    from the two leaves.

    Args:
        root (Node): Describes the non-uniform quadtree structure
        from_grid (jnp.array): Has shape (n_leaves, p**2, 2)
        to_x (jnp.array): Has shape (m,)
        to_y (jnp.array): Has shape (n,)
        p (int): The number of points per dimension in each leaf of the non-uniform grid

    Returns:
        jnp.array: Has shape (m * n, n_leaves * p**2)
    """

    m = to_x.shape[0]
    n = to_y.shape[0]

    leaves_iter = get_all_leaves(root)
    n_leaves = len(leaves_iter)

    # Chebyshev panel with p points
    cheby_pts_1d = chebyshev_points(p)[0]

    # When we create the local interpolation matrices, we will need to
    # rearrange the columns because we order the points on each patch
    # to be exterior then interior, rather than the standard meshgrid
    # ordering.
    rearrange_idxes = _rearrange_indices(p)

    assert from_grid.shape[0] == n_leaves

    I = jnp.zeros((m * n, n_leaves * p**2), dtype=from_grid.dtype)

    # Grid of output points is being made here.
    X, Y = jnp.meshgrid(to_x, to_y, indexing="ij")
    pts = jnp.stack((X.flatten(), Y.flatten()), axis=-1)
    # print("interp_operator_to_uniform: pts = ", pts)

    # Loop through pts and find the leaves that contains each point
    for i in range(pts.shape[0]):
        leaves_containing_i = []
        to_x_i = pts[i, 0].reshape((1,))
        to_y_i = pts[i, 1].reshape((1,))

        # Make a list of the leaves that contain the point.
        # It's a list because the point could coincide with a boundary or corner
        for leaf_idx, leaf in enumerate(leaves_iter):
            if _node_contains_point(leaf, pts[i]):
                leaves_containing_i.append(leaf_idx)

        n_leaves_containing = len(leaves_containing_i)

        # print("interp_operator_to_uniform: i = ", i)
        # print("interp_operator_to_uniform: leaves_containing_i = ", leaves_containing_i)
        # print("interp_operator_to_uniform: to_x_i = ", to_x_i)
        # print("interp_operator_to_uniform: to_y_i = ", to_y_i)

        # For this particular point, we will interpolate from each leaf that contains it.
        for leaf_idx in leaves_containing_i:
            leaf = leaves_iter[leaf_idx]

            # Get the 2D Cheby panel for this leaf
            from_x = affine_transform(cheby_pts_1d, jnp.array([leaf.xmin, leaf.xmax]))
            # Have to flip this one because the y-axis is flipped in the standard
            # Chebyshev grid creation routine
            from_y = jnp.flip(
                affine_transform(cheby_pts_1d, jnp.array([leaf.ymin, leaf.ymax]))
            )

            # Form an interpolation matrix from the 2D Cheby panel to the point.
            # Don't forget to rearrange the columns.
            I_local = barycentric_lagrange_2d_interpolation_matrix(
                from_x, from_y, to_x_i, to_y_i
            )[:, rearrange_idxes]

            # Add the values in the appropriate row. Remember to normalize by the
            # number of leaves containing the point.
            I = I.at[i : i + 1, leaf_idx * p**2 : (leaf_idx + 1) * p**2].set(
                I_local / n_leaves_containing
                + I[i : i + 1, leaf_idx * p**2 : (leaf_idx + 1) * p**2]
            )

    return I, pts


@jax.jit
def _node_contains_point(node: Node, pt: jnp.array) -> bool:
    x1 = jnp.logical_and(node.xmin <= pt[0], pt[0] <= node.xmax)
    x2 = jnp.logical_and(node.ymin <= pt[1], pt[1] <= node.ymax)
    return jnp.logical_and(x1, x2)
    # return (node.xmin <= pt[0] <= node.xmax) and (node.ymin <= pt[1] <= node.ymax)


@partial(jax.jit, static_argnums=(1, 3))
def interp_from_nonuniform_hps_to_regular_grid(
    root: Node,
    p: int,
    f_evals: jnp.array,
    n_pts: int,
) -> Tuple[jnp.array]:
    """
    This function takes in evaluations on the HPS discretization and uses local
    high-order polynomial interpolation to evaluate the function on a regular grid.

    First, it creates a regular meshgrid of target points.
    Then, it determines which patch each target point is in.
    Finally, it evaluates the function at the target points using the local
    polynomial interpolant.

    Args:
        l (int): Number of levels of grid refinement used to generate the HPS grid.
        p (int): Number of Chebyshev points used in each dimension on each patch.
        xmin (float): Minimum x-coordinate of the domain.
        xmax (float): Maximum x-coordinate of the domain.
        ymin (float): Minimum y-coordinate of the domain.
        ymax (float): Maximum y-coordinate of the domain.
        f (jnp.array): evaluations of the function we want to interpolate. Expect
            shape (4^l, p^2). The function is evaluated on the HPS grid
            specified by get_all_leaf_2d_cheby_points.
        n_pts (int): Number of points in each dimension on the regular grid.

    Returns:
        Tuple[jnp.array]:
            - Has shape (n_pts, n_pts). Evaluations of the function on the regular grid.
            - Has shape (n_pts, n_pts, 2). Specifies the target points on the regular grid.
    """
    xmin = root.xmin
    xmax = root.xmax
    ymin = root.ymin
    ymax = root.ymax

    # If the input has multiple dimensions, we need to remember the shape
    # so we can reshape the output back to the correct shape.
    if f_evals.ndim > 2:
        f_evals_shape = f_evals.shape
        bool_multi_source = True
    else:
        bool_multi_source = False

    # Create the regular grid
    x = jnp.linspace(xmin, xmax, n_pts, endpoint=False, dtype=jnp.float64)
    y = jnp.linspace(ymin, ymax, n_pts, endpoint=False, dtype=jnp.float64)
    y = jnp.flip(y)
    X, Y = jnp.meshgrid(x, y)
    target_pts = jnp.concatenate((jnp.expand_dims(X, 2), jnp.expand_dims(Y, 2)), axis=2)
    pts_lst = target_pts.reshape(-1, 2)

    corners = jnp.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

    all_leaves = get_all_leaves(root)
    corners_lst = [
        jnp.array(
            [
                [node.xmin, node.ymin],
                [node.xmax, node.ymin],
                [node.xmax, node.ymax],
                [node.xmin, node.ymax],
            ]
        )
        for node in all_leaves
    ]
    corners_iter = jnp.stack(corners_lst)

    # Find which patch the point is in
    # These should have shape (n_pts^2, 4^l)
    satisfies_xmin = pts_lst[:, 0, None] >= corners_iter[None, :, 0, 0]
    satisfies_xmax = pts_lst[:, 0, None] <= corners_iter[None, :, 1, 0]
    satisfies_ymin = pts_lst[:, 1, None] >= corners_iter[None, :, 0, 1]
    satisfies_ymax = pts_lst[:, 1, None] <= corners_iter[None, :, 2, 1]

    x_bools = jnp.logical_and(satisfies_xmin, satisfies_xmax)
    y_bools = jnp.logical_and(satisfies_ymin, satisfies_ymax)

    # Has shape (n_pts^2, 4^l)
    x_and_y = jnp.logical_and(x_bools, y_bools)

    # Find the indexes of the patches that contain each point
    patch_idx = jnp.argmax(x_and_y, axis=1)

    corners_for_vmap = corners_iter[patch_idx]
    f_for_vmap = f_evals[patch_idx]

    xvals_for_vmap = pts_lst[:, 0]
    yvals_for_vmap = pts_lst[:, 1]

    # Interpolate to the target points
    vals = vmapped_interp_to_point(
        xvals_for_vmap, yvals_for_vmap, corners_for_vmap, f_for_vmap, p
    )
    if bool_multi_source:
        vals = vals.reshape(n_pts, n_pts, f_evals_shape[2])
    else:
        vals = vals.reshape(n_pts, n_pts)
    return vals, target_pts


@partial(jax.jit, static_argnums=(4,))
def _interp_to_point(
    xval: jnp.array,
    yval: jnp.array,
    corners: jnp.array,
    f: jnp.array,
    p: int,
) -> jnp.array:
    """
    For a particular point (xval, yval) this function finds the patch
    that contains the point, constructs a polynomial interpolation matrix
    to that point, and evaluates the interpolant at the point.

    Args:
        xval (jnp.array): x coordinate of the target point. Has shape () or (n,)
        yval (jnp.array): y coordinate of the target point. Has shape () or (n,)
        p (int): Polynomial order of interpolation
        cheby_grid (jnp.array): HPS discretization. Has size (n_leaves, p^2, 2).
        all_corners (jnp.array): Corners of the patches. Has size (n_leaves, 4, 2).
        f (jnp.array): Function evals on the Cheby grid. Has size (n_leaves, p^2)

    Returns:
        jnp.array: Has shape (1,) or (n^2,) depending on the size of input xval and yval. The output
        is the interpolated value at the target point(s).
    """
    # If xval is a scalar, need to reshape it to be a 1D array for the barycentric interp function.
    # If xval is a 1D vector, this doesn't change anything.
    xval = xval.reshape(-1)
    yval = yval.reshape(-1)
    # print("_interp_to_point: xval: ", xval, " xval shape: ", xval.shape)
    # print("_interp_to_point: yval: ", yval, " yval shape: ", yval.shape)

    cheby_pts = chebyshev_points(p)[0]

    out = jnp.zeros_like(xval)

    xmin_i, ymin_i = corners[0]
    xmax_i, ymax_i = corners[2]

    from_x = affine_transform(cheby_pts, jnp.array([xmin_i, xmax_i]))
    from_y = affine_transform(cheby_pts, jnp.array([ymin_i, ymax_i]))
    # Annoyingly this is how the y vals are ordered
    from_y = jnp.flip(from_y)

    I = barycentric_lagrange_2d_interpolation_matrix(from_x, from_y, xval, yval)

    rearrange_idxes = _rearrange_indices(p)
    I = I[:, rearrange_idxes]

    out = out + I @ f

    return out


vmapped_interp_to_point = jax.vmap(_interp_to_point, in_axes=(0, 0, 0, 0, None))
