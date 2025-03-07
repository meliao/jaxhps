import pytest
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from hps.src.methods.local_solve_stage import (
    _local_solve_stage_2D,
    _local_solve_stage_2D_ItI,
    _local_solve_stage_3D,
    _gather_coeffs,
    _gather_coeffs_3D,
    assemble_diff_operator,
    _prep_nonuniform_refinement_diff_operators_2D,
    vmapped_prep_nonuniform_refinement_diff_operators_2D,
    get_DtN,
    get_ItI,
)

from hps.src.solver_obj import SolverObj, create_solver_obj_2D, create_solver_obj_3D
from hps.src.quadrature.quad_2D.differentiation import (
    precompute_diff_operators,
    precompute_N_matrix,
    precompute_N_tilde_matrix,
    precompute_F_matrix,
    precompute_G_matrix,
)
from hps.src.quadrature.quad_2D.interpolation import (
    precompute_P_matrix,
    precompute_Q_D_matrix,
    precompute_I_P_0_matrix,
    precompute_Q_I_matrix,
)
from hps.src.quadrature.trees import (
    Node,
    get_all_leaves,
    add_four_children,
    add_uniform_levels,
)
from hps.accuracy_checks.utils import plot_soln_from_cheby_nodes
from hps.src.quadrature.quad_2D.grid_creation import (
    get_all_boundary_gauss_legendre_points,
    get_all_leaf_2d_cheby_points,
)
from hps.accuracy_checks.dirichlet_neumann_data import TEST_CASE_POLY_PART_HOMOG


class Test_gather_coeffs:
    def test_0(self) -> None:
        """Tests things return correct shape."""
        n_leaf_nodes = 30
        p_squared = 17

        xx_coeffs = np.random.normal(size=(n_leaf_nodes, p_squared))
        yy_coeffs = np.random.normal(size=(n_leaf_nodes, p_squared))

        out_coeffs, out_bools = _gather_coeffs(
            D_xx_coeffs=xx_coeffs, D_yy_coeffs=yy_coeffs
        )
        assert out_coeffs.shape == (2, n_leaf_nodes, p_squared)
        assert out_bools.shape == (6,)
        assert out_bools[0] == True
        assert out_bools[1] == False
        assert out_bools[2] == True
        assert out_bools[3] == False
        assert out_bools[4] == False
        assert out_bools[5] == False

    def test_2(self) -> None:
        """Tests things return correct shape."""
        n_leaf_nodes = 30
        p_squared = 17

        xx_coeffs = np.ones((n_leaf_nodes, p_squared))
        yy_coeffs = np.zeros((n_leaf_nodes, p_squared))

        out_coeffs, out_bools = _gather_coeffs(
            D_xx_coeffs=xx_coeffs, D_yy_coeffs=yy_coeffs
        )
        assert jnp.all(out_coeffs[0] == 1)
        assert jnp.all(out_coeffs[1] == 0)


class Test_assemble_diff_operator:
    def test_0(self) -> None:

        p = 8
        q = 6
        l = 2
        n_leaves = 2**l
        half_side_len = 0.25
        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]

        d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators(p, half_side_len)

        stacked_diff_operators = jnp.stack([d_xx, d_xy, d_yy, d_x, d_y, jnp.eye(p**2)])
        coeffs_arr = jnp.ones((3, p**2))
        which_coeffs = jnp.array([True, True, True, False, False, False])
        out = assemble_diff_operator(coeffs_arr, which_coeffs, stacked_diff_operators)
        assert out.shape == (p**2, p**2)


class Test__prep_nonuniform_refinement_diff_operators_2D:
    def test_0(self) -> None:
        """Check accuracy"""
        p = 3
        q = 6
        l = 2
        n_leaves = 2**l
        half_side_len = 0.25
        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]

        # First precompute the diff operators without the sidelen scaling. This is how things
        # work in the local solve stage code
        d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators(p, 1.0)
        coeffs_arr = jnp.ones((2, p**2))
        diff_ops = jnp.stack([d_xx, d_xy, d_yy, d_x, d_y, jnp.eye(p**2)])

        coeffs_arr = coeffs_arr.at[0].set(4 * coeffs_arr[0])
        coeffs_arr = coeffs_arr.at[1].set(3 * coeffs_arr[1])
        which_coeffs = jnp.array([True, False, True, False, False, False])
        out, _ = _prep_nonuniform_refinement_diff_operators_2D(
            2 * half_side_len, coeffs_arr, which_coeffs, diff_ops, p, q
        )

        # Next, precompute diff operators with the correct sidelen scaling.
        d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators(
            p, half_side_len=half_side_len
        )
        expected_out = 4 * d_xx + 3 * d_yy

        print("test_1: out = ", out)
        print("test_1: expected_out = ", expected_out)
        assert jnp.allclose(out, expected_out)

    def test_1(self) -> None:
        """Checks accuracy with non-constant coefficients"""
        p = 3
        q = 6
        l = 2
        n_leaves = 2**l
        half_side_len = 0.25
        # First precompute the diff operators without the sidelen scaling. This is how things
        # work in the local solve stage code
        d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators(p, 1.0)
        coeffs_arr = np.random.normal(size=(2, p**2))
        diff_ops = jnp.stack([d_xx, d_xy, d_yy, d_x, d_y, jnp.eye(p**2)])
        which_coeffs = jnp.array([False, True, False, True, False, False])
        out, _ = _prep_nonuniform_refinement_diff_operators_2D(
            2 * half_side_len, coeffs_arr, which_coeffs, diff_ops, p, q
        )

        # Next, precompute diff operators with the correct sidelen scaling.
        d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators(
            p, half_side_len=half_side_len
        )
        expected_out = jnp.diag(coeffs_arr[0]) @ d_xy + jnp.diag(coeffs_arr[1]) @ d_x

        print("test_2: out = ", out)
        print("test_2: expected_out = ", expected_out)
        assert jnp.allclose(out, expected_out)

    def test_2(self) -> None:
        """Want to implement the operator
        3x * dxx + 4y * dyy

        when this is applied to f(x,y) = x^2 + y^2,
        we should get g(x,y) = 3x * 2 + 4y * 2 = 6x + 8y
        """

        p = 14
        q = 12
        l = 2

        sidelen = 1 / (2**l)

        node = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0, zmin=None, zmax=None
        )
        add_uniform_levels(root=node, l=l, q=q)
        t = create_solver_obj_2D(p, q, node)

        dxx_coeffs = 3 * t.leaf_cheby_points[..., 0]
        dyy_coeffs = 4 * t.leaf_cheby_points[..., 1]
        coeffs_arr = jnp.stack([dxx_coeffs, dyy_coeffs])
        print("test_3: coeffs_arr = ", coeffs_arr.shape)

        stacked_diff_operators = jnp.stack(
            [t.D_xx, t.D_xy, t.D_yy, t.D_x, t.D_y, jnp.eye(p**2, dtype=jnp.float64)]
        )
        which_coeffs = jnp.array([True, False, True, False, False, False])

        f = t.leaf_cheby_points[..., 0] ** 2 + t.leaf_cheby_points[..., 1] ** 2

        expected_g = 6 * t.leaf_cheby_points[..., 0] + 8 * t.leaf_cheby_points[..., 1]

        # Loop over all of the nodes and check that the operator is applied correctly
        for i in range(t.leaf_cheby_points.shape[0]):
            op_i, _ = _prep_nonuniform_refinement_diff_operators_2D(
                sidelen, coeffs_arr[:, i], which_coeffs, stacked_diff_operators, p, q
            )
            print("test_3: i = ", i)
            print("test_3: op_i norm = ", jnp.linalg.norm(op_i))
            prod = op_i @ f[i]
            print("test_3: prod nrm = ", jnp.linalg.norm(prod))
            print("test_3: expected_g[i] nrm = ", jnp.linalg.norm(expected_g[i]))
            # plot_soln_from_cheby_nodes(
            #     t.leaf_cheby_points[i], None, prod, expected_g[i]
            # )
            assert jnp.allclose(op_i @ f[i], expected_g[i])

    def test_3(self) -> None:
        """Want to make sure Helmholtz operator looks good."""
        p = 20
        q = 18
        l = 2
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2
        root = Node(xmin=west, xmax=east, ymin=south, ymax=north, depth=0)

        side_len = jnp.pi / (2**l)
        add_uniform_levels(root=root, l=l, q=q)
        t = create_solver_obj_2D(p, q, root)
        k = jnp.pi

        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        i_coeffs = (k**2) * jnp.ones_like(t.leaf_cheby_points[..., 0])

        coeffs_arr = jnp.stack([d_xx_coeffs, d_yy_coeffs, i_coeffs])
        which_coeffs = jnp.array([True, False, True, False, False, True])
        stacked_diff_operators = jnp.stack(
            [t.D_xx, t.D_xy, t.D_yy, t.D_x, t.D_y, jnp.eye(p**2, dtype=jnp.float64)]
        )

        # Plane wave is e^{ikx}
        f = jnp.exp(1j * k * t.leaf_cheby_points[..., 0])

        for i in range(t.leaf_cheby_points.shape[0]):
            print("test_4: i = ", i)
            op_i, _ = _prep_nonuniform_refinement_diff_operators_2D(
                side_len,
                coeffs_arr[:, i],
                which_coeffs,
                stacked_diff_operators,
                p,
                q,
            )
            resid_i = op_i @ f[i]
            print("test_4: resid_i max = ", jnp.max(jnp.abs(resid_i)))
            assert jnp.allclose(op_i @ f[i], jnp.zeros_like(f[i]))


class Test__local_solve_stage_2D:
    def test_0(self) -> None:
        """Tests the solve_stage function returns without error and returns the correct shape when boundary data are not passed to the function."""

        p = 16
        q = 14
        l = 1
        n_leaves = 4**l

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l)

        # This could be np.random.normal(size=(n_leaves, p**2))
        d_xx_coeffs = TEST_CASE_POLY_PART_HOMOG["d_xx_coeff_fn"](t.leaf_cheby_points)
        d_yy_coeffs = TEST_CASE_POLY_PART_HOMOG["d_yy_coeff_fn"](t.leaf_cheby_points)
        source_term = TEST_CASE_POLY_PART_HOMOG["source_fn"](t.leaf_cheby_points)
        sidelens = jnp.array([l.xmax - l.xmin for l in get_all_leaves(t.root)])

        print("test_0: n_leaves = ", n_leaves)
        print("test_0: n_gauss_bdry = ", 4 * q)

        (
            Y_arr,
            DtN_arr,
            v_arr,
            v_prime_arr,
        ) = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            p=p,
            sidelens=sidelens,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            source_term=source_term,
            uniform_grid=True,
        )

        assert DtN_arr.shape == (n_leaves, 4 * q, 4 * q)
        assert v_arr.shape == (n_leaves, p**2)
        assert v_prime_arr.shape == (n_leaves, 4 * q)

    def test_1(self) -> None:
        p = 8
        q = 6
        l = 2
        n_leaves = 4**l
        n_quads = 4 ** (l - 1)

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l)

        # This could be np.random.normal(size=(n_leaves, p**2))
        d_xx_coeffs = TEST_CASE_POLY_PART_HOMOG["d_xx_coeff_fn"](t.leaf_cheby_points)
        d_yy_coeffs = TEST_CASE_POLY_PART_HOMOG["d_yy_coeff_fn"](t.leaf_cheby_points)
        source_term = TEST_CASE_POLY_PART_HOMOG["source_fn"](t.leaf_cheby_points)
        sidelens = jnp.array([l.xmax - l.xmin for l in get_all_leaves(t.root)])

        print("test_0: n_leaves = ", n_leaves)
        print("test_0: n_gauss_bdry = ", 4 * q)

        Y, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            p=p,
            sidelens=sidelens,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            source_term=source_term,
            uniform_grid=False,
        )
        assert DtN_arr.shape == (n_leaves, 4 * q, 4 * q)
        assert v_arr.shape == (n_leaves, p**2)
        assert v_prime_arr.shape == (n_leaves, 4 * q)


class Test__local_solve_stage_2D_ItI:
    def test_0(self) -> None:
        """Tests the solve_stage function returns without error and returns the correct shape. n_src = 1."""

        p = 16
        q = 14
        l = 3
        eta = 4.0

        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        n_leaves = 4**l

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, use_ItI=True, eta=eta)

        sidelens = jnp.array([leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)])

        d_xx_coeffs = np.random.normal(size=(n_leaves, p**2))
        source_term = np.random.normal(size=(n_leaves, p**2))
        print("test_0: d_xx_coeffs = ", d_xx_coeffs.shape)
        print("test_0: source_term = ", source_term.shape)

        R_arr, Y_arr, g_arr, v_arr = _local_solve_stage_2D_ItI(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            I_P_0=t.I_P_0,
            Q_I=t.Q_I,
            F=t.F,
            G=t.G,
            p=t.p,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_xx_coeffs,
            source_term=source_term,
        )

        assert Y_arr.shape == (n_leaves, p**2, 4 * q)
        assert R_arr.shape == (n_leaves // 4, 4, 4 * q, 4 * q)
        assert v_arr.shape == (n_leaves, p**2, 1)
        assert g_arr.shape == (n_leaves // 4, 4, 4 * q, 1)

    def test_1(self) -> None:
        """Tests the solve_stage function returns without error and returns the correct shape. n_src = 3."""

        p = 16
        q = 14
        l = 3
        n_src = 3
        eta = 4.0

        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        n_leaves = 4**l

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, use_ItI=True, eta=eta)

        sidelens = jnp.array([leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)])

        d_xx_coeffs = np.random.normal(size=(n_leaves, p**2))
        source_term = np.random.normal(size=(n_leaves, p**2, n_src))
        print("test_0: d_xx_coeffs = ", d_xx_coeffs.shape)
        print("test_0: source_term = ", source_term.shape)

        R_arr, Y_arr, g_arr, v_arr = _local_solve_stage_2D_ItI(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            I_P_0=t.I_P_0,
            Q_I=t.Q_I,
            F=t.F,
            G=t.G,
            p=t.p,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_xx_coeffs,
            source_term=source_term,
        )

        assert Y_arr.shape == (n_leaves, p**2, 4 * q)
        assert R_arr.shape == (n_leaves // 4, 4, 4 * q, 4 * q)
        assert v_arr.shape == (n_leaves, p**2, n_src)
        assert g_arr.shape == (n_leaves // 4, 4, 4 * q, n_src)


class Test__local_solve_stage_3D:
    def test_0(self) -> None:
        """Tests the _local_solve_stage_3D function returns without error and returns the correct shape."""
        p = 6
        q = 4
        l = 2
        root = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=0.0,
            zmax=1.0,
            depth=0,
        )
        n_leaves = 8**l

        t = create_solver_obj_3D(p, q, root, uniform_levels=l)

        sidelens = jnp.array([leaf.xmax - leaf.xmin for leaf in get_all_leaves(t.root)])

        d_xx_coeffs = np.random.normal(size=(n_leaves, p**3))
        source_term = np.random.normal(size=(n_leaves, p**3))

        DtN_arr, v_arr, v_prime_arr, Y_arr = _local_solve_stage_3D(
            D_xx=t.D_xx,
            D_yy=t.D_yy,
            D_zz=t.D_zz,
            D_xy=t.D_xy,
            D_xz=t.D_xz,
            D_yz=t.D_yz,
            D_x=t.D_x,
            D_y=t.D_y,
            D_z=t.D_z,
            P=t.P,
            p=t.p,
            q=t.q,
            sidelens=sidelens,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
        )

        n_gauss_bdry = 6 * q**2

        assert Y_arr.shape == (n_leaves, p**3, n_gauss_bdry)
        assert DtN_arr.shape == (n_leaves, n_gauss_bdry, n_gauss_bdry)
        assert v_arr.shape == (n_leaves, p**3)
        assert v_prime_arr.shape == (n_leaves, n_gauss_bdry)


class Test__gather_coeffs_3D:
    def test_0(self) -> None:
        """Make sure the function returns the correct shape."""
        n_leaf_nodes = 30
        p_cubed = 17

        xx_coeffs = np.random.normal(size=(n_leaf_nodes, p_cubed))
        yy_coeffs = np.random.normal(size=(n_leaf_nodes, p_cubed))
        i_coeffs = np.random.normal(size=(n_leaf_nodes, p_cubed))

        out_coeffs, out_bools = _gather_coeffs_3D(
            D_xx_coeffs=xx_coeffs, D_yy_coeffs=yy_coeffs, I_coeffs=i_coeffs
        )
        assert out_coeffs.shape == (3, n_leaf_nodes, p_cubed)
        assert out_bools.shape == (10,)
        assert out_bools[0] == True
        assert out_bools[1] == False
        assert out_bools[2] == True
        assert out_bools[3] == False
        assert out_bools[4] == False
        assert out_bools[5] == False
        assert out_bools[6] == False
        assert out_bools[7] == False
        assert out_bools[8] == False
        assert out_bools[9] == True


class Test_get_DtN:
    def test_0(self) -> None:
        """Asserts that the get_DtN function returns without error and returns the correct shape."""
        p = 16
        q = 14
        n_cheby_bdry = 4 * (p - 1)

        diff_operator = np.random.normal(size=(p**2, p**2)).astype(np.float64)
        I_P = np.random.normal(size=(n_cheby_bdry, 4 * q)).astype(np.float64)
        Q_D = np.random.normal(size=(4 * q, p**2)).astype(np.float64)
        source_term = np.random.normal(size=(p**2,)).astype(np.float64)

        Y, DtN, v, v_prime = get_DtN(
            source_term=source_term,
            diff_operator=diff_operator,
            Q_D=Q_D,
            P=I_P,
        )

        assert Y.shape == (p**2, 4 * q)
        assert DtN.shape == (4 * q, 4 * q)
        assert v.shape == (p**2,)
        assert v_prime.shape == (4 * q,)


class Test_vmapped_prep_nonuniform_refinement_diff_operators_2D:
    def test_0(self) -> None:
        """Tests that the function returns without error"""

        p = 4
        q = 2
        n_cheby_pts = p**2
        n_leaves = 13
        n_diff_terms = 6
        n_cheby_bdry = 4 * (p - 1)
        n_gauss_bdry = 4 * q

        sidelens = np.random.normal(size=(n_leaves,))
        diff_ops_2D = np.random.normal(size=(n_diff_terms, n_cheby_pts, n_cheby_pts))
        coeffs_arr = np.random.normal(size=(3, n_leaves, n_cheby_pts))
        which_coeffs = np.array([True, True, True, False, False, False])

        diff_operators, Q_Ds = vmapped_prep_nonuniform_refinement_diff_operators_2D(
            sidelens, coeffs_arr, which_coeffs, diff_ops_2D, p, q
        )

        assert diff_operators.shape == (n_leaves, n_cheby_pts, n_cheby_pts)
        assert Q_Ds.shape == (n_leaves, n_gauss_bdry, n_cheby_pts)


class Test_get_ItI:
    def test_0(self) -> None:
        """Makes sure shapes are correct."""
        p = 4
        q = 3
        n_cheby = p**2
        n_cheby_bdry = 4 * (p - 1)
        n_cheby_bdry_dbl = 4 * p

        source_term = jnp.array(np.random.normal(size=(n_cheby,)).astype(np.float64))
        diff_operator = jnp.array(
            np.random.normal(size=(n_cheby, n_cheby)).astype(np.float64)
        )
        I_P_0 = jnp.array(
            np.random.normal(size=(n_cheby_bdry, 4 * q)).astype(np.float64)
        )
        Q_I = jnp.array(
            np.random.normal(size=(4 * q, n_cheby_bdry_dbl)).astype(np.float64)
        )

        F = jnp.array(
            np.random.normal(size=(n_cheby_bdry, n_cheby)).astype(np.complex128)
        )
        G = jnp.array(
            np.random.normal(size=(n_cheby_bdry_dbl, n_cheby)).astype(np.complex128)
        )
        eta = 4.0

        R, Y, g_part, part_soln = get_ItI(
            diff_operator=diff_operator,
            source_term=source_term,
            I_P_0=I_P_0,
            F=F,
            G=G,
            Q_I=Q_I,
        )

        assert Y.shape == (n_cheby, 4 * q)
        assert R.shape == (4 * q, 4 * q)
        assert g_part.shape == (4 * q,)
        assert part_soln.shape == (n_cheby,)

    def test_1(self) -> None:
        """Makes sure shapes are correct when using multiple source terms."""
        p = 4
        q = 3
        n_cheby = p**2
        n_cheby_bdry = 4 * (p - 1)
        n_cheby_bdry_dbl = 4 * p
        n_src = 7

        diff_operator = jnp.array(
            np.random.normal(size=(n_cheby, n_cheby)).astype(np.float64)
        )

        source_term = jnp.array(
            np.random.normal(size=(n_cheby, n_src)).astype(np.float64)
        )
        I_P_0 = jnp.array(
            np.random.normal(size=(n_cheby_bdry, 4 * q)).astype(np.float64)
        )
        Q_I = jnp.array(
            np.random.normal(size=(4 * q, n_cheby_bdry_dbl)).astype(np.float64)
        )
        F = jnp.array(
            np.random.normal(size=(n_cheby_bdry, n_cheby)).astype(np.complex128)
        )
        G = jnp.array(
            np.random.normal(size=(n_cheby_bdry_dbl, n_cheby)).astype(np.complex128)
        )
        eta = 4.0

        R, Y, g_part, part_soln = get_ItI(
            diff_operator=diff_operator,
            source_term=source_term,
            I_P_0=I_P_0,
            Q_I=Q_I,
            F=F,
            G=G,
        )

        assert Y.shape == (n_cheby, 4 * q)
        assert R.shape == (4 * q, 4 * q)
        assert g_part.shape == (4 * q, n_src)
        assert part_soln.shape == (n_cheby, n_src)

    def test_2(self) -> None:
        """Checks the accuracy of the ItI operator when solving a Laplace problem with zero source and
        polynomial impedance data.

        In the notes this is test case 2D.ItI.a
        """
        p = 8
        q = 6

        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2
        corners = jnp.array(
            [[west, south], [east, south], [east, north], [west, north]]
        )
        half_side_len = jnp.pi / 2
        root = Node(xmin=west, xmax=east, ymin=south, ymax=north)

        # Set up the GL boundary points
        bdry_pts = get_all_boundary_gauss_legendre_points(q, root)
        cheby_pts = get_all_leaf_2d_cheby_points(p, root)[0]

        # Precompute differential operators
        d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators(p, half_side_len)
        N = precompute_N_matrix(d_x, d_y, p)
        N_tilde = precompute_N_tilde_matrix(d_x, d_y, p)

        eta = 4.0
        F = precompute_F_matrix(N_tilde, p, eta)
        G = precompute_G_matrix(N, p, eta)

        # Precompute interpolation operators
        I_P_0 = precompute_I_P_0_matrix(p, q)
        Q_I = precompute_Q_I_matrix(p, q)

        # Stack the differential operators into a single array
        stacked_diff_ops = jnp.stack([d_xx, d_xy, d_yy, d_x, d_y, jnp.eye(p**2)])

        # Make Laplacian coefficients
        lap_coeffs = jnp.ones((2, p**2), dtype=jnp.float64)
        which_coeffs = jnp.array([True, False, True, False, False, False])
        diff_operator = assemble_diff_operator(
            coeffs_arr=lap_coeffs,
            which_coeffs=which_coeffs,
            diff_ops=stacked_diff_ops,
        )
        source_term = jnp.zeros((p**2,), dtype=jnp.float64)

        # Compute ItI map
        R, Y, g_part, part_soln = get_ItI(
            diff_operator=diff_operator,
            source_term=source_term,
            I_P_0=I_P_0,
            Q_I=Q_I,
            F=F,
            G=G,
        )

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - y^2
            return x[..., 0] ** 2 - x[..., 1] ** 2

        def dfdx(x: jnp.array) -> jnp.array:
            # df/dx = 2x
            return 2 * x[..., 0]

        def dfdy(x: jnp.array) -> jnp.array:
            # df/dy = -2y
            return -2 * x[..., 1]

        # Evaluate f on the boundary points
        f_vals = f(bdry_pts)

        f_normal_vals = jnp.concatenate(
            [
                -dfdy(bdry_pts[0:q]),
                dfdx(bdry_pts[q : 2 * q]),
                dfdy(bdry_pts[2 * q : 3 * q]),
                -dfdx(bdry_pts[3 * q :]),
            ]
        )

        incoming_impedance_data = f_normal_vals + 1j * eta * f_vals
        outgoing_impedance_data = R @ incoming_impedance_data
        outgoing_expected = f_normal_vals - 1j * eta * f_vals

        assert jnp.allclose(outgoing_impedance_data, outgoing_expected)
        assert jnp.allclose(part_soln, jnp.zeros_like(part_soln))
        assert jnp.allclose(g_part, jnp.zeros_like(g_part))

        expected_homog_soln = f(cheby_pts)
        computed_homog_soln = Y @ incoming_impedance_data
        assert jnp.allclose(expected_homog_soln, computed_homog_soln)


if __name__ == "__main__":
    pytest.main()
