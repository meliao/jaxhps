import logging
from functools import partial
from typing import Tuple, Any, List
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt


class Node:
    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float = None,
        zmax: float = None,
        depth: int = 0,
        children: Tuple["Node"] = (),
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.depth = depth
        self.children = children
        self.l2_nrm: float = 0.0

        # Arrays set by the HPS computation in local solve & merge stages.
        self.DtN: jnp.array = None
        self.v_prime: jnp.array = None
        self.S: jnp.array = None
        self.v_int: jnp.array = None  # v_int is the interface particular soln.
        self.L_2f1: jnp.array = None
        self.L_1f2: jnp.array = None

        # These are only ever computed for leaf nodes. It's the local solution
        # operator and the local particular solution.
        self.Y: jnp.array = None
        self.v: jnp.array = None
        self.soln: jnp.array = None
        self.bdry_data: jnp.array = None

        # Keeping track of indexing
        # in 2D, self.n_i is the number of quadrature points along the boundary of
        # side i of the patch. i = 0, 1, 2, 3 corresponds to the bottom, right,
        # top, and left, or (S, E, N, W), sides of the patch, respectively.
        # in 3D, self.n_i is the number of quadrature points along the boundary of
        # each face of the voxel. See notes for ordering of the faces of voxels.
        self.n_0: int = None
        self.n_1: int = None
        self.n_2: int = None
        self.n_3: int = None
        self.n_4: int = None
        self.n_5: int = None

    def __repr__(self) -> str:
        return "Node(xmin={}, xmax={}, ymin={}, ymax={}, zmin={}, zmax={}, depth={})".format(
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.zmin,
            self.zmax,
            self.depth,
        )


def _node_flatten(node: Node) -> Tuple:
    c = (node.children,)
    d = (
        node.xmin,
        node.xmax,
        node.ymin,
        node.ymax,
        node.zmin,
        node.zmax,
        node.depth,
        node.l2_nrm,
        node.DtN,
        node.v_prime,
        node.S,
        node.v_int,
        node.Y,
        node.v,
        node.soln,
        node.bdry_data,
        node.n_0,
        node.n_1,
        node.n_2,
        node.n_3,
        node.n_4,
        node.n_5,
        node.L_2f1,
        node.L_1f2,
    )
    return (c, d)


def _node_unflatten(
    aux_data: Tuple[float | int],
    children: Tuple[Node],
) -> Node:
    z = object.__new__(Node)

    # Aux data here
    z.xmin = aux_data[0]
    z.xmax = aux_data[1]
    z.ymin = aux_data[2]
    z.ymax = aux_data[3]
    z.zmin = aux_data[4]
    z.zmax = aux_data[5]
    z.depth = aux_data[6]
    z.l2_nrm = aux_data[7]

    # Arrays computed during the HPS algorithm.
    z.DtN = aux_data[8]
    z.v_prime = aux_data[9]
    z.S = aux_data[10]
    z.v_int = aux_data[11]
    z.Y = aux_data[12]
    z.v = aux_data[13]
    z.soln = aux_data[14]
    z.bdry_data = aux_data[15]

    # Indexing information
    z.n_0 = aux_data[16]
    z.n_1 = aux_data[17]
    z.n_2 = aux_data[18]
    z.n_3 = aux_data[19]
    z.n_4 = aux_data[20]
    z.n_5 = aux_data[21]

    # Interpolation operators
    z.L_2f1 = aux_data[22]
    z.L_1f2 = aux_data[23]

    # Children here
    z.children = children[0]

    return z


jax.tree_util.register_pytree_node(Node, _node_flatten, _node_unflatten)


@jax.jit
def get_node_area(node: Node) -> float:
    if node.zmax is not None:
        return (
            (node.xmax - node.xmin) * (node.ymax - node.ymin) * (node.zmax - node.zmin)
        )
    else:
        return (node.xmax - node.xmin) * (node.ymax - node.ymin)


def add_four_children(add_to: Node, root: Node = None, q: int = None) -> None:
    """Splits a 2D node into four children. Updates the .children() attribute of the input Node.
    If the root and q are specified, this function will also update the number of quadrature points
    along intermediate nodes' boundaries. Adding nodes will change these quantities, which is why
    this function handles that logic.

    Args:
        add_to (Node): The node which must be split into four children.
        root (Node, optional): The root of the tree. Specified if we want to count number of quadrature points along node boundaries. Defaults to None.
        q (int, optional): Number of quadrature points along the boundary of leaves of the tree. Specified if we want to count number of quadrature points along node boundaries. Defaults to None.
    """
    if len(add_to.children) == 0:
        children = get_four_children(add_to)
        add_to.children = children

    if q is not None:

        # Set the number of quadrature points along the boundary of each child.
        for child in add_to.children:
            child.n_0 = q
            child.n_1 = q
            child.n_2 = q
            child.n_3 = q

        # Update the number of quadrature points in add_to.
        add_to.n_0 = 2 * q
        add_to.n_1 = 2 * q
        add_to.n_2 = 2 * q
        add_to.n_3 = 2 * q

        # Update the number of quadrature points along the path from add_to to the root. Only do this
        # if the root is not the same as add_to.
        if not tree_equal(add_to, root):
            path_info = find_path_from_root_2D(root, add_to)
            for node, _ in path_info:

                # If they share a boundary, then we need to update the number of quadrature points.
                if node.ymin == add_to.ymin:
                    node.n_0 += q

                if node.xmax == add_to.xmax:
                    node.n_1 += q

                if node.ymax == add_to.ymax:
                    node.n_2 += q

                if node.xmin == add_to.xmin:
                    node.n_3 += q


def add_uniform_levels(root: Node, l: int, q: int = None) -> None:
    """Generate a uniform mesh with l levels.

    Args:
        root (Node): The root node of the tree.
        l (int): The number of levels to generate.
    """
    if l == 0:
        return

    bool_3d = root.zmin is not None

    for _ in range(l):
        leaves = get_all_leaves(root)
        for leaf in leaves:
            if bool_3d:
                add_eight_children(add_to=leaf, root=root, q=q)
            else:
                add_four_children(add_to=leaf, root=root, q=q)


def get_all_uniform_leaves_2D(root: Node, l: int) -> List[Node]:
    """_summary_

    Args:
        root (Node): Defines the 2D domain
        l (int): Number of levels of subdivision. There will be 2^l leaves.

    Returns:
        List[Node]: All of the leaves of the tree. Nodes which are not the leaves will not be returned.
    """

    # Make an array of [SW, SE, NE, NW] corners.
    corners_iter = jnp.array(
        [
            [
                [root.xmin, root.ymin],
                [root.xmax, root.ymin],
                [root.xmax, root.ymax],
                [root.xmin, root.ymax],
            ]
        ]
    )
    for level in range(l):
        corners_iter = vmapped_corners(corners_iter).reshape(-1, 4, 2)

    node_lst = [
        Node(xmin=x[0, 0], ymin=x[0, 1], xmax=x[2, 0], ymax=x[2, 1])
        for x in corners_iter
    ]
    return node_lst


@jax.jit
def _corners_for_quad_subdivision(corners: jnp.ndarray) -> jnp.ndarray:
    """Given a list of corners, return a list of lists of corners corresponding to the four quadrants of the original corners.

    Inputs have shape (4, 2) and outputs have shape (4, 4, 2)."""
    west, south = corners[0]
    east, north = corners[2]
    mid_x = (west + east) / 2
    mid_y = (south + north) / 2

    return jnp.array(
        [
            # SW quadrant
            [(west, south), (mid_x, south), (mid_x, mid_y), (west, mid_y)],
            # SE quadrant
            [(mid_x, south), (east, south), (east, mid_y), (mid_x, mid_y)],
            # NE quadrant
            [(mid_x, mid_y), (east, mid_y), (east, north), (mid_x, north)],
            # NW quadrant
            [(west, mid_y), (mid_x, mid_y), (mid_x, north), (west, north)],
        ]
    )


vmapped_corners = jax.vmap(
    _corners_for_quad_subdivision,
)


@jax.jit
def get_four_children(parent: Node) -> None:
    """
    Splits the node into four children, and returns the four children. Does NOT modify the input node.

    This function is not pure, and modifies the input node in place. Thus, I am not able to use jax.jit on this function.

    Args:
        parent (Node): Parent node being split into 4 children
    """

    # find node in root which matches node

    xmid = (parent.xmin + parent.xmax) / 2
    ymid = (parent.ymin + parent.ymax) / 2

    # jax.debug.print("get_four_children: xmid: {out}", out=xmid)
    # jax.debug.print("get_four_children: ymid: {out}", out=ymid)
    # print("get_four_children: xmid=", xmid, xmid.shape)
    # print("get_four_children: ymid=", ymid, ymid.shape)
    child_a = Node(
        depth=parent.depth + 1,
        xmin=parent.xmin,
        xmax=xmid,
        ymin=parent.ymin,
        ymax=ymid,
        zmin=parent.zmin,
        zmax=parent.zmax,
    )
    child_b = Node(
        depth=parent.depth + 1,
        xmin=xmid,
        xmax=parent.xmax,
        ymin=parent.ymin,
        ymax=ymid,
        zmin=parent.zmin,
        zmax=parent.zmax,
    )
    child_c = Node(
        depth=parent.depth + 1,
        xmin=xmid,
        xmax=parent.xmax,
        ymin=ymid,
        ymax=parent.ymax,
        zmin=parent.zmin,
        zmax=parent.zmax,
    )
    child_d = Node(
        depth=parent.depth + 1,
        xmin=parent.xmin,
        xmax=xmid,
        ymin=ymid,
        ymax=parent.ymax,
        zmin=parent.zmin,
        zmax=parent.zmax,
    )
    return (
        child_a,
        child_b,
        child_c,
        child_d,
    )


def add_eight_children(add_to: Node, root: Node = None, q: int = None) -> None:
    """
    Splits the node's 3D volume into 8 children, and assigns them to the children attribute of the node.

    Lists the children in this order:

    a: xmin...xmid, ymin...ymid, zmid...zmax
    b: xmid...xmax, ymin...ymid, zmid...zmax
    c: xmid...xmax, ymid...ymax, zmid...zmax
    d: xmin...xmid, ymid...ymax, zmid...zmax
    e: xmin...xmid, ymin...ymid, zmin...zmid
    f: xmid...xmax, ymin...ymid, zmin...zmid
    g: xmid...xmax, ymid...ymax, zmin...zmid
    h: xmin...xmid, ymid...ymax, zmin...zmid

    If the root and q qre specified, this function will also update the number of quadrature points
    along intermediate nodes' boundaries. Adding nodes will change these quantities, which is why
    this function handles that logic.

    Args:
        add_to (Node): The node whose volume is being split into 8 children.
    """
    # This if statement meant to guard against adding children to a node that already has children.
    # and messing up the n_i values.
    if len(add_to.children) == 0:
        add_to.children = get_eight_children(add_to)

        if q is not None:
            # Set the number of quadrature points along the boundary of each child.
            q_squared = q**2
            for child in add_to.children:
                child.n_0 = q_squared
                child.n_1 = q_squared
                child.n_2 = q_squared
                child.n_3 = q_squared
                child.n_4 = q_squared
                child.n_5 = q_squared

            # Update the number of quadrature points in add_to.
            add_to.n_0 = 4 * q_squared
            add_to.n_1 = 4 * q_squared
            add_to.n_2 = 4 * q_squared
            add_to.n_3 = 4 * q_squared
            add_to.n_4 = 4 * q_squared
            add_to.n_5 = 4 * q_squared

            # Update the number of quadrature points along the path from add_to to the root. Only do this
            # if the root is not the same as add_to.
            if not tree_equal(add_to, root):
                path_info = find_path_from_root_3D(root, add_to)
                for node in path_info:

                    # If they share a boundary, then we need to update the number of quadrature points.
                    if node.xmin == add_to.xmin:
                        node.n_0 += 3 * q_squared

                    if node.xmax == add_to.xmax:
                        node.n_1 += 3 * q_squared

                    if node.ymin == add_to.ymin:
                        node.n_2 += 3 * q_squared

                    if node.ymax == add_to.ymax:
                        node.n_3 += 3 * q_squared

                    if node.zmin == add_to.zmin:
                        node.n_4 += 3 * q_squared

                    if node.zmax == add_to.zmax:
                        node.n_5 += 3 * q_squared


@jax.jit
def get_eight_children(parent: Node) -> Tuple[Node]:
    """
    Splits the node's 3D volume into 8 children, and returns a tuple of these children.

    Lists the children in this order:

    a: xmin...xmid, ymin...ymid, zmid...zmax
    b: xmid...xmax, ymin...ymid, zmid...zmax
    c: xmid...xmax, ymid...ymax, zmid...zmax
    d: xmin...xmid, ymid...ymax, zmid...zmax
    e: xmin...xmid, ymin...ymid, zmin...zmid
    f: xmid...xmax, ymin...ymid, zmin...zmid
    g: xmid...xmax, ymid...ymax, zmin...zmid
    h: xmin...xmid, ymid...ymax, zmin...zmid


    Args:
        parent (Node): The node whose volume is being split into 8 children.
    """
    xmid = (parent.xmin + parent.xmax) / 2
    ymid = (parent.ymin + parent.ymax) / 2
    zmid = (parent.zmin + parent.zmax) / 2

    child_a = Node(
        xmin=parent.xmin,
        xmax=xmid,
        ymin=parent.ymin,
        ymax=ymid,
        zmin=zmid,
        zmax=parent.zmax,
        depth=parent.depth + 1,
    )
    child_b = Node(
        xmin=xmid,
        xmax=parent.xmax,
        ymin=parent.ymin,
        ymax=ymid,
        zmin=zmid,
        zmax=parent.zmax,
        depth=parent.depth + 1,
    )
    child_c = Node(
        xmin=xmid,
        xmax=parent.xmax,
        ymin=ymid,
        ymax=parent.ymax,
        zmin=zmid,
        zmax=parent.zmax,
        depth=parent.depth + 1,
    )
    child_d = Node(
        xmin=parent.xmin,
        xmax=xmid,
        ymin=ymid,
        ymax=parent.ymax,
        zmin=zmid,
        zmax=parent.zmax,
        depth=parent.depth + 1,
    )
    child_e = Node(
        xmin=parent.xmin,
        xmax=xmid,
        ymin=parent.ymin,
        ymax=ymid,
        zmin=parent.zmin,
        zmax=zmid,
        depth=parent.depth + 1,
    )
    child_f = Node(
        xmin=xmid,
        xmax=parent.xmax,
        ymin=parent.ymin,
        ymax=ymid,
        zmin=parent.zmin,
        zmax=zmid,
        depth=parent.depth + 1,
    )
    child_g = Node(
        xmin=xmid,
        xmax=parent.xmax,
        ymin=ymid,
        ymax=parent.ymax,
        zmin=parent.zmin,
        zmax=zmid,
        depth=parent.depth + 1,
    )
    child_h = Node(
        xmin=parent.xmin,
        xmax=xmid,
        ymin=ymid,
        ymax=parent.ymax,
        zmin=parent.zmin,
        zmax=zmid,
        depth=parent.depth + 1,
    )

    return (child_a, child_b, child_c, child_d, child_e, child_f, child_g, child_h)


def get_all_leaves(node: Node) -> Tuple[Node]:
    if not len(node.children):
        return (node,)
    else:
        leaves = ()
        for child in node.children:
            leaves += get_all_leaves(child)
        return leaves


# The jitted version does not allow for updating the leaves.
get_all_leaves_jitted = jax.jit(get_all_leaves)


def get_all_leaves_special_ordering(
    node: Node, child_traversal_order: jnp.array = None
) -> Tuple[Node]:
    if child_traversal_order is None:
        if node.zmax is not None:
            child_traversal_order = jnp.arange(8, dtype=jnp.int32)
        else:
            child_traversal_order = jnp.arange(4, dtype=jnp.int32)
    if node.children == ():
        return (node,)
    else:
        leaves = ()
        for child_idx in child_traversal_order:
            child = node.children[child_idx]
            leaves += get_all_leaves_special_ordering(
                child, child_traversal_order=child_traversal_order
            )
        return leaves


def get_all_parents_of_leaves(root: Node) -> Tuple[Node]:
    """
    This function traverses the tree and returns all the nodes which are direct parents
    of the leaves.
    """
    # Base case: check whether the children have no children.
    n_children_of_children = [len(child.children) for child in root.children]
    # If all children have no children, then the root is a parent of the leaves.
    if all([n == 0 for n in n_children_of_children]):
        return (root,)
    else:
        parents = ()
        for child in root.children:
            parents += get_all_parents_of_leaves(child)
        return parents


def get_all_nodes(node: Node) -> Tuple[Node]:
    if node.children == ():
        return (node,)
    else:
        nodes = (node,)
        for child in node.children:
            nodes += get_all_nodes(child)
        return nodes


get_all_nodes_jitted = jax.jit(get_all_nodes)


# @jax.jit
def get_depth(node: Node) -> int:
    return jnp.max(jnp.array([child.depth for child in get_all_leaves(node)]))


def get_nodes_at_level(node: Node, level: int) -> Tuple[Node]:
    if node.depth == level:
        return (node,)
    else:
        nodes = ()
        for child in node.children:
            nodes += get_nodes_at_level(child, level)
        return nodes


@jax.jit
def tree_equal(node_a: Node, node_b: Node) -> bool:
    """Checks equality between the metadata of two nodes and the metadata of all of their children."""
    _, a = jax.tree_util.tree_flatten(node_a)
    _, b = jax.tree_util.tree_flatten(node_b)
    return a == b


@jax.jit
def node_at(
    node: Node,
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
    zmin: float = None,
    zmax: float = None,
) -> Node:
    """Tests whether the node is at the specified location. Robust to presence of Nones in the input."""
    bool_xmin = jnp.logical_or((xmin is None), (node.xmin == xmin))
    bool_xmax = jnp.logical_and(
        jnp.logical_or((xmax is None), (node.xmax == xmax)), bool_xmin
    )
    bool_ymin = jnp.logical_and(
        jnp.logical_or((ymin is None), (node.ymin == ymin)), bool_xmax
    )
    bool_ymax = jnp.logical_and(
        jnp.logical_or((ymax is None), (node.ymax == ymax)), bool_ymin
    )
    bool_zmin = jnp.logical_and(
        jnp.logical_or((zmin is None), (node.zmin == zmin)), bool_ymax
    )
    bool_zmax = jnp.logical_and(
        jnp.logical_or((zmax is None), (node.zmax == zmax)), bool_zmin
    )
    return bool_zmax


def find_node_at_corner(
    root: Node,
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
) -> Node:

    at_corner = node_at(root, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    if at_corner:
        # Check whether the root has a child that also matches the specified location.
        # This will happen when searching for a particular corner, i.e. only specifying
        # xmin and ymin, or xmax and ymax.
        if root.children != ():
            for child in root.children:
                if node_at(child, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax):
                    return find_node_at_corner(
                        child, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
                    )
        else:
            # If the root has no children, then the root is the node we are looking for.
            return root

    else:
        node_xmid = (root.xmin + root.xmax) / 2
        node_ymid = (root.ymin + root.ymax) / 2

        x_pred_1 = (xmin is None) or (xmin >= node_xmid)
        x_pred_2 = (xmax is None) or (xmax > node_xmid)
        in_upper_half_x = x_pred_1 and x_pred_2

        y_pred_1 = (ymin is None) or (ymin >= node_ymid)
        y_pred_2 = (ymax is None) or (ymax > node_ymid)
        in_upper_half_y = y_pred_1 and y_pred_2

        if in_upper_half_x:
            if in_upper_half_y:
                return find_node_at_corner(
                    root.children[2], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
                )
            else:
                return find_node_at_corner(
                    root.children[1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
                )
        else:
            if in_upper_half_y:
                return find_node_at_corner(
                    root.children[3], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
                )
            else:
                return find_node_at_corner(
                    root.children[0], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
                )


def plot_tree(
    root: Node,
) -> None:
    """Plot the 2D refinement.

    <corners_adaptive> specifies a list of leaf corners.
    For each leaf, draw a box with the corners specified by the leaf corners.

    Then, sample the function f_fn at a regular grid of points and plot the values.


    Args:
        corners (jnp.array): Has shape (n_leaves, 4, 2) specifying the corners of each of the leaves.
        f_fn (Callable[[jnp.array], jnp.array]): _description_
    """

    fig, ax = plt.subplots()

    # Plot the function
    north_outer = root.ymax
    south_outer = root.ymin
    east_outer = root.xmax
    west_outer = root.xmin

    for leaf in get_all_leaves(root):
        w = leaf.xmin
        e = leaf.xmax
        s = leaf.ymin
        n = leaf.ymax
        ax.hlines(s, w, e, color="black")
        ax.hlines(n, w, e, color="black")
        ax.vlines(w, s, n, color="black")
        ax.vlines(e, s, n, color="black")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    plt.clf()


def find_path_from_root_2D(
    root: Node,
    node: Node,
) -> List[Tuple[Node, int]]:
    """Find the path from the root to the node in a 2D tree.

    Args:
        root (Node): The root of the tree.
        node (Node): The node who is being searched for.

    Returns:
        List[Tuple[Node, int]]: A list of tuples, which provides a traversal from the root to the node.
        Each element
    """
    if root.children == ():
        raise ValueError("Specified root has no children.")

    for i, child in enumerate(root.children):
        if tree_equal(child, node):
            return [
                (root, i),
            ]

    # Find which child we need to continue searching in.
    root_xmid = (root.xmin + root.xmax) / 2
    root_ymid = (root.ymin + root.ymax) / 2

    if node.xmin < root_xmid:
        # Either child 0 or 3
        if node.ymin < root_ymid:
            # Child 0
            return [(root, 0)] + find_path_from_root_2D(root.children[0], node)
        else:
            # Child 3
            return [(root, 3)] + find_path_from_root_2D(root.children[3], node)
    else:
        # Either child 1 or 2
        if node.ymin < root_ymid:
            # Child 1
            return [(root, 1)] + find_path_from_root_2D(root.children[1], node)
        else:
            # Child 2
            return [(root, 2)] + find_path_from_root_2D(root.children[2], node)


def find_path_from_root_3D(root: Node, node: Node) -> List[Node]:
    """Find the path from the root to the node in a 3D tree.

    Args:
        root (Node): The root of the tree.
        node (Node): The node who is being searched for.
    Returns:
        List[Node]: A list of nodes, which provides a traversal from the root to the node.
    """

    if root.children == ():
        raise ValueError("Specified root has no children.")

    for child in root.children:
        if tree_equal(child, node):
            return [root]

    # Find which child we need to continue searching in.
    root_xmid = (root.xmin + root.xmax) / 2
    root_ymid = (root.ymin + root.ymax) / 2
    root_zmid = (root.zmin + root.zmax) / 2

    if node.xmin < root_xmid:
        # Either child 0, 3, 4, or 7
        if node.ymin < root_ymid:
            # Either child 0 or 4
            if node.zmin < root_zmid:
                # Child 4
                return [root] + find_path_from_root_3D(root.children[4], node)
            else:
                # Child 0
                return [root] + find_path_from_root_3D(root.children[0], node)
        else:
            # Either child 3 or 7
            if node.zmin < root_zmid:
                # Child 7
                return [root] + find_path_from_root_3D(root.children[7], node)
            else:
                # Child 3
                return [root] + find_path_from_root_3D(root.children[3], node)
    else:
        # Either child 1, 2, 5, or 6
        if node.ymin < root_ymid:
            # Either child 1 or 5
            if node.zmin < root_zmid:
                # Child 5
                return [root] + find_path_from_root_3D(root.children[5], node)
            else:
                # Child 1
                return [root] + find_path_from_root_3D(root.children[1], node)
        else:
            # Either child 2 or 6
            if node.zmin < root_zmid:
                # Child 6
                return [root] + find_path_from_root_3D(root.children[6], node)
            else:
                # Child 2
                return [root] + find_path_from_root_3D(root.children[2], node)


def find_nodes_along_interface_2D(
    root: Node,
    xval: float = None,
    yval: float = None,
) -> List[List[Node | List[Node]]]:
    """
    For a given xval or yval, find all the leaves of the quadtree are bordered by the line
    x = xval or y = yval. This function expects xval or yval to be specified, but not both.

    This function also assumes that xval or yval falls along leaf boundaries, but does not
    intersect the interior of any leaves. Thus the function returns two different Lists of
    Nodes, one for each side of the interface.

    Args:
        root (Node): The root of the possibly non-uniform quadtree.
        xval (float, optional): x value defining an interface. Defaults to None.
        yval (float, optional): y value defining an interface. Defaults to None.

    Returns:
        List[List[Node | List[Node]]]:
        outer list has n_leaf elements, where n_leaf is the
    """
    if xval is not None and yval is not None:
        raise ValueError("Only one of xval or yval can be specified.")

    if xval is not None:
        # Find the leaf nodes that are bordered by the line x = xval.
        neg_side_lst = []
        pos_side_lst = []
        for leaf in get_all_leaves(root):
            if leaf.xmin == xval:
                pos_side_lst.append(leaf)
            elif leaf.xmax == xval:
                neg_side_lst.append(leaf)

        return neg_side_lst, pos_side_lst

    if yval is not None:
        # Find the leaf nodes that are bordered by the line y = yval.
        neg_side_lst = []
        pos_side_lst = []
        for leaf in get_all_leaves(root):
            if leaf.ymin == yval:
                pos_side_lst.append(leaf)
            elif leaf.ymax == yval:
                neg_side_lst.append(leaf)

        return neg_side_lst, pos_side_lst


def find_nodes_along_interface_3D(
    root: Node,
    xval: float = None,
    yval: float = None,
    zval: float = None,
) -> Tuple[List[Node]]:
    """
    For a given xval, yval, or zval, find all the leaves of the octree are bordered by the plane
    x = xval, y = yval, or z = zval. This function expects only one of xval, yval, or zval to be specified.

    This function also assumes that xval, yval, or zval falls along leaf boundaries, but does not
    intersect the interior of any leaves. Thus the function returns two different Lists of
    Nodes, one for each side of the interface.

    Args:
        root (Node): The root of the possibly non-uniform quadtree.
        xval (float, optional): x value defining an interface. Defaults to None.
        yval (float, optional): y value defining an interface. Defaults to None.
        zval (float, optional): z value defining an interface. Defaults to None.

    Returns:
        Tuple[List[Node]]: neg_side_lst: List of nodes on the negative side of the interface.
                           pos_side_lst: List of nodes on the positive side of the interface.
    """
    # Ensure only one of xval, yval, zval are specified
    if sum([xval is not None, yval is not None, zval is not None]) != 1:
        raise ValueError(
            f"Only one of xval, yval, or zval can be specified. Input args: {xval}, {yval}, {zval}"
        )

    if xval is not None:
        # Find the leaf nodes that are bordered by the plane x = xval.
        neg_side_lst = []
        pos_side_lst = []
        for leaf in get_all_leaves(root):
            if leaf.xmin == xval:
                pos_side_lst.append(leaf)
            elif leaf.xmax == xval:
                neg_side_lst.append(leaf)
        return neg_side_lst, pos_side_lst

    if yval is not None:
        # Find the leaf nodes that are bordered by the plane y = yval.
        neg_side_lst = []
        pos_side_lst = []
        for leaf in get_all_leaves(root):
            if leaf.ymin == yval:
                pos_side_lst.append(leaf)
            elif leaf.ymax == yval:
                neg_side_lst.append(leaf)
        return neg_side_lst, pos_side_lst

    if zval is not None:
        # Find the leaf nodes that are bordered by the plane z = zval.
        neg_side_lst = []
        pos_side_lst = []
        for leaf in get_all_leaves(root):
            if leaf.zmin == zval:
                pos_side_lst.append(leaf)
            elif leaf.zmax == zval:
                neg_side_lst.append(leaf)

        return neg_side_lst, pos_side_lst


def find_nodes_along_boundary_2D(
    root: Node,
    xval: float = None,
    yval: float = None,
) -> List[Node]:
    pass
