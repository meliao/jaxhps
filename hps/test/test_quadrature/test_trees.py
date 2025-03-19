import pytest

import numpy as np
import jax
import jax.numpy as jnp

from hps.src.quadrature.trees import (
    Node,
    get_node_area,
    add_four_children,
    add_eight_children,
    add_uniform_levels,
    get_all_leaves,
    find_node_at_corner,
    tree_equal,
    node_at,
    get_all_nodes,
    get_all_nodes_jitted,
    get_all_leaves_special_ordering,
    find_path_from_root_2D,
    find_nodes_along_interface_2D,
    find_nodes_along_interface_3D,
    find_path_from_root_3D,
    get_all_uniform_leaves_2D,
)


class Test_find_nodes_along_interface_2D:
    def test_0(self) -> None:
        # Uniform case
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        l = 3
        add_uniform_levels(root, l)

        x_interface = 0.5
        nodes = find_nodes_along_interface_2D(root, xval=x_interface)
        assert len(nodes) == 2
        for node_lst in nodes:
            assert len(node_lst) == 2**l

        y_interface = 0.25
        nodes = find_nodes_along_interface_2D(root, yval=y_interface)
        assert len(nodes) == 2
        for node_lst in nodes:
            assert len(node_lst) == 2**l

        for node in nodes[0]:
            assert node.ymax == y_interface
        for node in nodes[1]:
            assert node.ymin == y_interface

    def test_1(self) -> None:
        # Non-uniform case
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        add_four_children(add_to=root, root=root)
        add_four_children(add_to=root.children[0], root=root)

        x_interface = 0.5
        nodes = find_nodes_along_interface_2D(root, xval=x_interface)
        assert len(nodes) == 2
        assert len(nodes[0]) == 3
        assert len(nodes[1]) == 2


class Test_find_nodes_along_interface_3D:
    def test_0(self) -> None:
        # Uniform case
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        l = 3
        add_uniform_levels(root, l)

        x_interface = 0.5
        nodes = find_nodes_along_interface_3D(root, xval=x_interface)
        assert len(nodes) == 2
        for node_lst in nodes:
            assert len(node_lst) == 4**l

        y_interface = 0.25
        nodes = find_nodes_along_interface_3D(root, yval=y_interface)
        assert len(nodes) == 2
        for node_lst in nodes:
            assert len(node_lst) == 4**l

        for node in nodes[0]:
            assert node.ymax == y_interface
        for node in nodes[1]:
            assert node.ymin == y_interface

    def test_1(self) -> None:
        # Non-uniform case
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        add_eight_children(add_to=root)
        add_eight_children(add_to=root.children[0])

        x_interface = 0.5
        nodes = find_nodes_along_interface_3D(root, xval=x_interface)
        print("test_1: nodes lengths: ", [len(x) for x in nodes])
        assert len(nodes) == 2
        assert len(nodes[0]) == 7
        assert len(nodes[1]) == 4


class Test_find_path_from_root_2D:
    def test_0(self) -> None:
        # Non-uniform grid.
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        add_four_children(add_to=root, root=root)
        add_four_children(add_to=root.children[0], root=root)

        node = root.children[0].children[0]

        computed_path = find_path_from_root_2D(root, node)
        path_len = len(computed_path)

        for i, xx in enumerate(computed_path):
            current_node = xx[0]
            current_child_idx = xx[1]

            if i != path_len - 1:
                next_node = computed_path[i + 1][0]
                assert tree_equal(
                    current_node.children[current_child_idx],
                    next_node,
                )
            else:
                assert tree_equal(current_node.children[current_child_idx], node)


class Test_find_path_from_root_3D:
    def test_0(self) -> None:
        # Non-uniform grid.
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        q = 3
        add_eight_children(add_to=root, root=root, q=q)
        add_eight_children(add_to=root.children[0], root=root, q=q)

        node = root.children[0].children[0]

        computed_path = find_path_from_root_2D(root, node)
        path_len = len(computed_path)

        for i, xx in enumerate(computed_path):
            current_node = xx[0]
            current_child_idx = xx[1]

            if i != path_len - 1:
                next_node = computed_path[i + 1][0]
                assert tree_equal(
                    current_node.children[current_child_idx],
                    next_node,
                )
            else:
                assert tree_equal(current_node.children[current_child_idx], node)


class Test_get_node_area:
    def test_0(self) -> None:
        node = Node(
            children=(),
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        child_a = Node(
            children=(),
            depth=1,
            xmin=0.0,
            xmax=0.5,
            ymin=0.0,
            ymax=0.5,
            zmin=None,
            zmax=None,
        )
        node.children = (child_a,)
        assert get_node_area(node) == 1


class Test_add_four_children:
    def test_0(self) -> None:
        node = Node(
            children=(),
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        print("test_0: node = ", id(node))
        print("test_0: node: ", node)
        print("test_0: node.children = ", node.children)
        add_four_children(node)
        print("test_0: node.children = ", node.children)

        assert len(node.children) == 4

    def test_1(self) -> None:
        """Checks two levels works"""
        print("test_1: Started")
        node = Node(
            children=(),
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        q = 3

        print("test_1: Making children")
        add_four_children(root=node, add_to=node, q=q)
        for child in node.children:
            add_four_children(root=node, add_to=child, q=q)
            assert len(child.children) == 4
            for gchild in child.children:
                assert get_node_area(gchild) == 1 / 16

        # Check that counting number of boundary points is performed correctly.
        for child in node.children:
            assert child.n_0 == 2 * q
            assert child.n_1 == 2 * q
            assert child.n_2 == 2 * q
            assert child.n_3 == 2 * q

        print(
            "test_1: root # of quadrature points: ",
            node.n_0,
            node.n_1,
            node.n_2,
            node.n_3,
        )
        print("test_1: Expected # of quadrature points: ", 4 * q, 4 * q, 4 * q, 4 * q)
        assert node.n_0 == 4 * q
        assert node.n_1 == 4 * q
        assert node.n_2 == 4 * q
        assert node.n_3 == 4 * q

    def test_2(self) -> None:
        """Checks discretization point counting in non-uniform grid."""
        root = Node(
            children=(),
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        q = 3
        # Each side of root now has 2 panels.
        add_four_children(add_to=root, root=root, q=q)

        for child in root.children:
            # Each side of root now has 4 panels.
            add_four_children(add_to=child, root=root, q=q)

        # n_0 and n_3 of root now have 5 panels each.
        add_four_children(add_to=root.children[0].children[0], root=root, q=q)

        # Check that counting number of boundary points is performed correctly.
        # Root
        print(
            "test_1: root # of quadrature points: ",
            root.n_0,
            root.n_1,
            root.n_2,
            root.n_3,
        )
        assert root.n_0 == 5 * q
        assert root.n_1 == 4 * q
        assert root.n_2 == 4 * q
        assert root.n_3 == 5 * q

    def test_3(self) -> None:
        """Checks discretization point counting in uniform grid of 3 levels of refinement."""
        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=None, zmax=None, depth=0
        )
        l = 3
        q = 3
        for i in range(l):
            print("test_3: level i = ", i)
            for leaf in get_all_leaves(root):
                print("test_3: splitting leaf = ", leaf)
                add_four_children(add_to=leaf, root=root, q=q)
                print(
                    "test_3: root # of quadrature points: ",
                    root.n_0,
                    root.n_1,
                    root.n_2,
                    root.n_3,
                )

        for i in range(4):
            assert root.children[i].n_0 == (2 ** (l - 1)) * q
            assert root.children[i].n_1 == (2 ** (l - 1)) * q
            assert root.children[i].n_2 == (2 ** (l - 1)) * q
            assert root.children[i].n_3 == (2 ** (l - 1)) * q

        print(
            "test_3: root # of quadrature points: ",
            root.n_0,
            root.n_1,
            root.n_2,
            root.n_3,
        )
        print(
            "test_3: Expected # of quadrature points: ",
            (2**l) * q,
            (2**l) * q,
            (2**l) * q,
            (2**l) * q,
        )
        assert root.n_0 == (2**l) * q
        assert root.n_1 == (2**l) * q
        assert root.n_2 == (2**l) * q
        assert root.n_3 == (2**l) * q


class Test_add_eight_children:
    def test_0(self) -> None:
        """Checks function returns without error with uniform refinement."""
        node = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        add_eight_children(node)
        assert len(node.children) == 8
        for child in node.children:
            assert get_node_area(child) == 1 / 8
            add_eight_children(child)
            for gchild in child.children:
                assert get_node_area(gchild) == 1 / 64

    def test_1(self) -> None:
        """Checks discretization point counting in 3 levels of uniform refinement."""
        q = 4
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        add_eight_children(add_to=root, root=root, q=q)
        for child in root.children:
            add_eight_children(add_to=child, root=root, q=q)
            for gchild in child.children:
                add_eight_children(add_to=gchild, root=root, q=q)
        # for leaf in get_all_leaves(root):
        #     add_eight_children(add_to=leaf, root=root, q=q)
        print("test_1: expected_n: ", 8 * q**2)

        assert root.n_0 == 64 * q**2
        assert root.n_1 == 64 * q**2
        assert root.n_2 == 64 * q**2
        assert root.n_3 == 64 * q**2
        assert root.n_4 == 64 * q**2
        assert root.n_5 == 64 * q**2

    def test_2(self) -> None:
        """Checks discretization point counting in non-uniform refinement."""
        q = 4
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        add_eight_children(add_to=root, root=root, q=q)
        add_eight_children(add_to=root.children[0], root=root, q=q)
        add_eight_children(add_to=root.children[0].children[0], root=root, q=q)

        # Faces 1, 3, 4 have only been refined once, so they each have 4 * q**2 points.
        assert root.n_1 == 4 * q**2
        assert root.n_3 == 4 * q**2
        assert root.n_4 == 4 * q**2

        # The other faces have 10 * q**2 points.
        assert root.n_0 == 10 * q**2
        assert root.n_2 == 10 * q**2
        assert root.n_5 == 10 * q**2


class Test_get_all_leaves:
    def test_0(self) -> None:
        node = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        add_four_children(node)
        add_four_children(node.children[0])
        add_four_children(node.children[0].children[0])

        leaves = get_all_leaves(node)
        assert len(leaves) == 10
        assert (
            get_node_area(leaves[0]) == 1 / 64
        ), f"get_node_area(leaves[0]) = {get_node_area(leaves[0])} vs 1/64 = {1/64}"
        assert get_node_area(leaves[1]) == 1 / 64
        assert get_node_area(leaves[2]) == 1 / 64
        assert get_node_area(leaves[3]) == 1 / 64
        assert get_node_area(leaves[4]) == 1 / 16
        assert get_node_area(leaves[5]) == 1 / 16
        assert get_node_area(leaves[6]) == 1 / 16
        assert get_node_area(leaves[7]) == 1 / 4
        assert get_node_area(leaves[8]) == 1 / 4
        assert get_node_area(leaves[9]) == 1 / 4

        assert leaves[0].depth == 3
        assert leaves[1].depth == 3
        assert leaves[2].depth == 3
        assert leaves[3].depth == 3
        assert leaves[4].depth == 2
        assert leaves[5].depth == 2
        assert leaves[6].depth == 2
        assert leaves[7].depth == 1
        assert leaves[8].depth == 1
        assert leaves[9].depth == 1

    def test_1(self) -> None:
        """Check that I can retrieve DtN arrays from leaves using get_all_leaves composed with a list comp"""
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        add_four_children(root)
        add_four_children(root.children[0])

        a = 3

        # Set arrays for the grandchildren
        for gchild in root.children[0].children:
            gchild.DtN = jnp.zeros((a, a))
            gchild.v_prime = jnp.zeros((a,))

        # Set arrays for the other leaves
        for c in range(1, 4):
            root.children[c].DtN = jnp.zeros((a, a))
            root.children[c].v_prime = jnp.zeros((a,))

        for leaf in get_all_leaves(root):
            assert leaf.DtN.shape == (a, a)
            assert leaf.v_prime.shape == (a,)


class Test_get_all_leaves_special_ordering:
    def test_0(self) -> None:
        """Check that specifying a non-standard child traversal order works as
        expected."""
        node = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        add_four_children(node)
        add_four_children(node.children[0])
        add_four_children(node.children[0].children[0])

        child_traversal_order = [1, 2, 3, 0]
        leaves = get_all_leaves_special_ordering(
            node, child_traversal_order=child_traversal_order
        )
        assert len(leaves) == 10

        assert get_node_area(leaves[0]) == 1 / 4
        assert get_node_area(leaves[1]) == 1 / 4
        assert get_node_area(leaves[2]) == 1 / 4
        assert get_node_area(leaves[3]) == 1 / 16
        assert get_node_area(leaves[4]) == 1 / 16
        assert get_node_area(leaves[5]) == 1 / 16
        assert get_node_area(leaves[6]) == 1 / 64
        assert get_node_area(leaves[7]) == 1 / 64
        assert get_node_area(leaves[8]) == 1 / 64
        assert get_node_area(leaves[9]) == 1 / 64

        assert leaves[0].depth == 1
        assert leaves[1].depth == 1
        assert leaves[2].depth == 1
        assert leaves[3].depth == 2
        assert leaves[4].depth == 2
        assert leaves[5].depth == 2
        assert leaves[6].depth == 3
        assert leaves[7].depth == 3
        assert leaves[8].depth == 3
        assert leaves[9].depth == 3


class Test_find_node_at:
    def test_0(self) -> None:
        """Checks that find_node_at works without error."""
        node = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        add_four_children(node)
        add_four_children(node.children[0])
        add_four_children(node.children[0].children[0])

        # Search on xmin, ymin at the corner of the domain
        found_a = find_node_at_corner(node, xmin=node.xmin, ymin=node.ymin)
        expected_a = node.children[0].children[0].children[0]
        assert tree_equal(found_a, expected_a)

        # Search on xmin, ymin NOT at the corner of the domain
        found_b = find_node_at_corner(
            node, xmin=node.children[0].children[1].xmin, ymin=node.ymin
        )
        expected_b = node.children[0].children[1]
        assert tree_equal(found_b, expected_b)

        # Search on xmin, ymax NOT at the corner of the domain.
        expected_c = node.children[0].children[0].children[1]
        found_c = find_node_at_corner(node, xmin=expected_c.xmin, ymax=expected_c.ymax)
        assert tree_equal(found_c, expected_c)

        # Search on xmax, ymin NOT at the corner of the domain.
        expected_d = node.children[0].children[0].children[1]
        found_d = find_node_at_corner(node, xmax=expected_d.xmax, ymin=expected_d.ymin)
        assert tree_equal(found_d, expected_d)

        # Search on xmax, ymax NOT at the corner of the domain.
        expected_e = node.children[0].children[0].children[1]
        found_e = find_node_at_corner(node, xmax=expected_e.xmax, ymax=expected_e.ymax)
        assert tree_equal(found_e, expected_e)


class Test_node_at:
    def test_0(self) -> None:
        a = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        x = node_at(a, xmin=0.0)
        assert x

        y = node_at(a, ymin=1.0)
        assert not y


class Test_add_uniform_levels:
    def test_0(self) -> None:
        """2D case uniform refinement"""
        node = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        l = 3
        q = 3
        add_uniform_levels(root=node, l=l, q=q)
        leaves = get_all_leaves(node)
        assert len(leaves) == 4**l
        for leaf in leaves:
            assert get_node_area(leaf) == (1 / 4) ** l
            assert leaf.depth == l

        # Check that n_* are set correctly for the root.
        print(
            "test_0: root # of quadrature points: ",
            node.n_0,
            node.n_1,
            node.n_2,
            node.n_3,
        )
        print(
            "test_0: expected # of quadrature points: ",
            q * (2**l),
            q * (2**l),
            q * (2**l),
            q * (2**l),
        )
        assert node.n_0 == q * (2**l)
        assert node.n_1 == q * (2**l)
        assert node.n_2 == q * (2**l)
        assert node.n_3 == q * (2**l)

    def test_1(self) -> None:
        """3D case"""
        node = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        l = 2
        add_uniform_levels(node, l)
        leaves = get_all_leaves(node)
        assert len(leaves) == 8**l
        for leaf in leaves:
            assert get_node_area(leaf) == (1 / 8) ** l
            assert leaf.depth == l


class Test_get_all_uniform_leaves_2D:
    def test_0(self) -> None:
        """2D case uniform refinement"""
        node = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )

        l = 3
        q = 3
        add_uniform_levels(root=node, l=l, q=q)
        leaves = get_all_leaves(node)

        node1 = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        leaves1 = get_all_uniform_leaves_2D(node1, l)
        assert len(leaves) == len(leaves1)
        for i in range(len(leaves)):
            leaf_a = leaves[i]
            leaf_b = leaves1[i]
            assert leaf_a.xmin == leaf_b.xmin
            assert leaf_a.xmax == leaf_b.xmax
            assert leaf_a.ymin == leaf_b.ymin
            assert leaf_a.ymax == leaf_b.ymax


class Test_get_all_nodes:
    def test_0(self) -> None:
        """Uniform refinement 2D case"""
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )

        l = 2
        add_uniform_levels(root, l)
        nodes = get_all_nodes(root)
        expected_n_nodes = 4**l + 4 ** (l - 1) + 4 ** (l - 2)
        print("test_0: expected_n_nodes = ", expected_n_nodes)
        assert len(nodes) == expected_n_nodes

        assert len(get_all_nodes_jitted(root)) == expected_n_nodes

    def test_1(self) -> None:
        """Non-Uniform refinement 2D case"""
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )

        add_four_children(root)  # 5 nodes
        add_four_children(root.children[0])  # 5 + 4 = 9 nodes
        add_four_children(root.children[0].children[0])  # 9 + 4 = 13 nodes
        nodes = get_all_nodes(root)
        assert len(nodes) == 13
        assert len(get_all_nodes_jitted(root)) == 13
