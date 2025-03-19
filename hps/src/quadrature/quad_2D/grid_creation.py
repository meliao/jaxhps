from typing import Tuple, List, Dict
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from hps.src.quadrature.quadrature_utils import chebyshev_points, affine_transform
from hps.src.utils import meshgrid_to_lst_of_pts
from hps.src.quadrature.quad_2D.indexing import _rearrange_indices
from hps.src.quadrature.trees import (
    Node,
    get_all_leaves,
    get_all_leaves_jitted,
    get_nodes_at_level,
    find_node_at_corner,
    vmapped_corners,
)


def get_ordered_lst_of_boundary_nodes(root: Node) -> Tuple[Tuple[Node]]:
    """
    Given the root node of an adaptive quadtree, return a list of the boundary nodes in order.
    Starting from the SW corner and moving counter-clockwise around the boundary.

    Args:
        root (Node): Is the root of the adaptive quadtree.

    Returns:
        Tuple[Node]: All boundary elements, in order.
    """
    s_bdry_nodes = _get_next_S_boundary_node(
        root, (find_node_at_corner(root, xmin=root.xmin, ymin=root.ymin),)
    )
    e_bdry_nodes = _get_next_E_boundary_node(root, (s_bdry_nodes[-1],))
    n_bdry_nodes = _get_next_N_boundary_node(root, (e_bdry_nodes[-1],))
    w_bdry_nodes = _get_next_W_boundary_node(root, (n_bdry_nodes[-1],))
    return (s_bdry_nodes, e_bdry_nodes, n_bdry_nodes, w_bdry_nodes)


def _get_next_S_boundary_node(root: Node, carry: Tuple[Node]) -> Tuple[Node]:
    """Recursively finds the next node on the S boundary of the quadtree."""

    # Base case: start with the child at the SW corner
    last = carry[-1]

    if last.xmax == root.xmax:
        # We've reached the end of the S boundary
        return carry

    next = find_node_at_corner(root, xmin=last.xmax, ymin=root.ymin)
    together = carry + (next,)
    if next.xmax == root.xmax:
        # We've reached the end of the S boundary
        return together
    else:
        return _get_next_S_boundary_node(root, together)


def _get_next_E_boundary_node(root: Node, carry: Tuple[Node]) -> Tuple[Node]:
    """Recursively finds the next node on the E boundary of the quadtree."""

    # Base case: start with the child at the SE corner
    last = carry[-1]

    if last.ymax == root.ymax:
        # We've reached the end of the E boundary
        return carry

    next = find_node_at_corner(root, xmax=root.xmax, ymin=last.ymax)
    together = carry + (next,)
    if next.ymax == root.ymax:
        # We've reached the end of the E boundary
        return together
    else:
        return _get_next_E_boundary_node(root, together)


def _get_next_N_boundary_node(root: Node, carry: Tuple[Node]) -> Tuple[Node]:
    """Recursively finds the next node on the N boundary of the quadtree."""

    # Base case: start with the child at the NE corner
    last = carry[-1]

    if last.xmin == root.xmin:
        # We've reached the end of the N boundary
        return carry

    next = find_node_at_corner(root, xmax=last.xmin, ymax=root.ymax)
    together = carry + (next,)
    if next.xmin == root.xmin:
        # We've reached the end of the N boundary
        return together
    else:
        return _get_next_N_boundary_node(root, together)


def _get_next_W_boundary_node(root: Node, carry: Tuple[Node]) -> Tuple[Node]:
    """Recursively finds the next node on the W boundary of the quadtree."""

    # Base case: start with the child at the NW corner
    last = carry[-1]

    if last.ymin == root.ymin:
        # We've reached the end of the W boundary
        return carry

    next = find_node_at_corner(root, xmin=root.xmin, ymax=last.ymin)
    together = carry + (next,)
    if next.ymin == root.ymin:
        # We've reached the end of the W boundary
        return together
    else:
        return _get_next_W_boundary_node(root, together)


@partial(jax.jit, static_argnums=(0,))
def get_all_leaf_2d_cheby_points(p: int, root: Node) -> jnp.ndarray:
    """Given the parameters that describe a quadtree discretization, return an array
    that gives the locations of all of the Chebyshev points on the leaf nodes.

    Args:
        p (int): Number of Chebyshev nodes in one dimension for each leaf node.
        l (int): Number of quad tree levels. There will be 4**l leaf nodes.
        corners (jnp.ndarray): Spatial domain corners. Has shape (4,2). Lists the corners in the order [SW, SE, NE, NW].

    Returns:
        jnp.ndarray: Has shape (4**l, p**2, 2). Lists the Chebyshev points for each leaf node.
    """
    leaves_iter = get_all_leaves_jitted(root)
    bounds = jnp.array(
        [[leaf.xmin, leaf.xmax, leaf.ymin, leaf.ymax] for leaf in leaves_iter]
    )
    cheby_pts_1d = chebyshev_points(p)[0]
    all_cheby_points = vmapped_bounds_to_cheby_points_lst(bounds, cheby_pts_1d)
    return all_cheby_points


@partial(jax.jit, static_argnums=(0, 1))
def get_all_leaf_2d_cheby_points_uniform_refinement(
    p: int, l: int, corners: jnp.ndarray
) -> jnp.ndarray:
    """Given the parameters that describe a quadtree discretization, return an array
    that gives the locations of all of the Chebyshev points on the leaf nodes.

    Args:
        p (int): Number of Chebyshev nodes in one dimension for each leaf node.
        l (int): Number of quad tree levels. There will be 4**l leaf nodes.
        corners (jnp.ndarray): Spatial domain corners. Has shape (4,2). Lists the corners in the order [SW, SE, NE, NW].

    Returns:
        jnp.ndarray: Has shape (4**l, p**2, 2). Lists the Chebyshev points for each leaf node.
    """
    corners_iter = jnp.expand_dims(corners, axis=0)
    for level in range(l):
        corners_iter = vmapped_corners(corners_iter).reshape(-1, 4, 2)

    cheby_pts_1d = chebyshev_points(p)[0]
    all_cheby_points = vmapped_corners_to_cheby_points_lst(corners_iter, cheby_pts_1d)
    return all_cheby_points


@partial(jax.jit, static_argnums=(0, 1))
def get_all_boundary_gauss_legendre_points_uniform_refinement(
    q: int, l: int, corners: jnp.ndarray
) -> jnp.ndarray:
    gauss_pts_1d = np.polynomial.legendre.leggauss(q)[0]
    n_patches_across_side = 2**l

    west, south = corners[0]
    east, north = corners[2]
    we_breakpoints = jnp.linspace(west, east, n_patches_across_side + 1)
    ns_breakpoints = jnp.linspace(south, north, n_patches_across_side + 1)

    we_gauss_nodes = jnp.concatenate(
        [
            affine_transform(gauss_pts_1d, we_breakpoints[i : i + 2])
            for i in range(n_patches_across_side)
        ]
    )
    ns_gauss_nodes = jnp.concatenate(
        [
            affine_transform(gauss_pts_1d, ns_breakpoints[i : i + 2])
            for i in range(n_patches_across_side)
        ]
    )
    gauss_nodes = jnp.concatenate(
        [
            jnp.column_stack(
                (we_gauss_nodes, jnp.full(we_gauss_nodes.shape[0], south))
            ),
            jnp.column_stack((jnp.full(ns_gauss_nodes.shape[0], east), ns_gauss_nodes)),
            jnp.column_stack(
                (jnp.flipud(we_gauss_nodes), jnp.full(we_gauss_nodes.shape[0], north))
            ),
            jnp.column_stack(
                (jnp.full(ns_gauss_nodes.shape[0], west), jnp.flipud(ns_gauss_nodes))
            ),
        ]
    )
    return gauss_nodes


def get_all_boundary_gauss_legendre_points(q: int, root: Node) -> jnp.ndarray:
    gauss_pts_1d = np.polynomial.legendre.leggauss(q)[0]

    corners = get_ordered_lst_of_boundary_nodes(root)

    west = root.xmin
    east = root.xmax
    south = root.ymin
    north = root.ymax

    south_gauss_nodes = jnp.concatenate(
        [affine_transform(gauss_pts_1d, [node.xmin, node.xmax]) for node in corners[0]]
    )
    east_gauss_nodes = jnp.concatenate(
        [affine_transform(gauss_pts_1d, [node.ymin, node.ymax]) for node in corners[1]]
    )
    north_gauss_nodes = jnp.concatenate(
        [affine_transform(gauss_pts_1d, [node.xmax, node.xmin]) for node in corners[2]]
    )
    west_gauss_nodes = jnp.concatenate(
        [affine_transform(gauss_pts_1d, [node.ymax, node.ymin]) for node in corners[3]]
    )
    gauss_nodes = jnp.concatenate(
        [
            jnp.column_stack(
                (south_gauss_nodes, jnp.full(south_gauss_nodes.shape[0], south))
            ),
            jnp.column_stack(
                (jnp.full(east_gauss_nodes.shape[0], east), east_gauss_nodes)
            ),
            jnp.column_stack(
                (north_gauss_nodes, jnp.full(north_gauss_nodes.shape[0], north))
            ),
            jnp.column_stack(
                (jnp.full(west_gauss_nodes.shape[0], west), west_gauss_nodes)
            ),
        ]
    )
    return gauss_nodes


@jax.jit
def bounds_to_cheby_points_lst(
    bounds: jnp.array, cheby_pts_1d: jnp.ndarray
) -> jnp.ndarray:
    """Given a set of corners and a set of Chebyshev points, return a list of Chebyshev points
    that are the tensor product of the 1D Chebyshev points over the 2D domain defined by the corners.

    Args:
        corners (jnp.ndarray): Has shape (4,2). Lists the corners in the order [SW, SE, NE, NW].
        cheby_pts_1d (jnp.ndarray): Chebyshev nodes of the 2nd kind over the interval [-1, 1]. Has shape (p,).

    Returns:
        jnp.ndarray: Has shape (p**2, 2). Lists the Chebyshev points for the 2D domain.
    """

    west = bounds[0]
    east = bounds[1]
    south = bounds[2]
    north = bounds[3]

    we = jnp.array([west, east])
    ns = jnp.array([south, north])

    x_pts = affine_transform(cheby_pts_1d, we)
    y_pts = affine_transform(cheby_pts_1d, ns)

    X, Y = jnp.meshgrid(x_pts, jnp.flipud(y_pts), indexing="ij")
    cheby_pts = meshgrid_to_lst_of_pts(X, Y)

    r = _rearrange_indices(cheby_pts_1d.shape[0])
    cheby_pts = cheby_pts[r]

    return cheby_pts


vmapped_bounds_to_cheby_points_lst = jax.vmap(
    bounds_to_cheby_points_lst, in_axes=(0, None), out_axes=0
)


@jax.jit
def corners_to_cheby_points_lst(
    corners: jnp.ndarray, cheby_pts_1d: jnp.ndarray
) -> jnp.ndarray:
    """Given a set of corners and a set of Chebyshev points, return a list of Chebyshev points
    that are the tensor product of the 1D Chebyshev points over the 2D domain defined by the corners.

    Args:
        corners (jnp.ndarray): Has shape (4,2). Lists the corners in the order [SW, SE, NE, NW].
        cheby_pts_1d (jnp.ndarray): Chebyshev nodes of the 2nd kind over the interval [-1, 1]. Has shape (p,).

    Returns:
        jnp.ndarray: Has shape (p**2, 2). Lists the Chebyshev points for the 2D domain.
    """

    west, south = corners[0]
    east, north = corners[2]

    we = jnp.array([west, east])
    ns = jnp.array([south, north])

    x_pts = affine_transform(cheby_pts_1d, we)
    y_pts = affine_transform(cheby_pts_1d, ns)

    X, Y = jnp.meshgrid(x_pts, jnp.flipud(y_pts), indexing="ij")
    cheby_pts = meshgrid_to_lst_of_pts(X, Y)

    r = _rearrange_indices(cheby_pts_1d.shape[0])
    cheby_pts = cheby_pts[r]

    return cheby_pts


vmapped_corners_to_cheby_points_lst = jax.vmap(
    corners_to_cheby_points_lst, (0, None), 0
)
