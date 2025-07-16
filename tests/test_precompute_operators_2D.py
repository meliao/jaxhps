import jax.numpy as jnp
import jax
import numpy as np
from jaxhps._precompute_operators_2D import (
    precompute_diff_operators_2D,
    precompute_rectangular_diff_operators_2D,
    rectangular_interp_operator,
    precompute_N_matrix_2D,
    precompute_P_2D_ItI,
    precompute_G_2D_ItI,
    precompute_N_tilde_matrix_2D,
    precompute_QH_2D_ItI,
    precompute_projection_ops_2D,
    interpolate_cheby_to_cheby_first_kind,
)
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._grid_creation_2D import (
    compute_interior_Chebyshev_points_adaptive_2D,
    compute_boundary_Gauss_points_adaptive_2D,
)
from jaxhps.quadrature import (
    affine_transform,
    gauss_points,
    first_kind_chebyshev_points,
)

# from jaxhps._utils import plot_soln_from_cheby_nodes
import logging


class Test_precompute_P_2D_ItI:
    def test_0(self) -> None:
        """Makes sure the output is correct shape"""
        p = 8
        q = 6

        I_P = precompute_P_2D_ItI(p, q)

        assert I_P.shape == (4 * (p - 1), 4 * q)

    def test_1(self) -> None:
        """Makes sure low-degree polynomial interpolation is exact."""
        p = 8
        q = 6

        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2

        root = DiscretizationNode2D(
            xmin=west, xmax=east, ymin=south, ymax=north
        )

        bdry_pts = compute_boundary_Gauss_points_adaptive_2D(root, q)
        cheby_pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)
        interior_pts = cheby_pts[0, : 4 * (p - 1)]

        print("test_1: bdry_pts shape: ", bdry_pts.shape)
        print("test_1: cheby_pts shape: ", cheby_pts.shape)
        print("test_1: interior_pts shape: ", interior_pts.shape)

        def f(x):
            # f(x,y) = 3x - y**2
            return 3 * x[..., 0] - x[..., 1] ** 2

        # Compute the function values at the Gauss boundary points
        f_vals = f(bdry_pts)

        I_P_0 = precompute_P_2D_ItI(p, q)

        f_interp = I_P_0 @ f_vals

        f_expected = f(interior_pts)

        print("test_1: f_interp shape: ", f_interp.shape)
        print("test_1: f_expected shape: ", f_expected.shape)
        assert jnp.allclose(f_interp, f_expected)


class Test_precompute_N_matrix_2D:
    def test_0(self) -> None:
        """Check the shape of the output."""
        p = 8
        # q = 6
        du_dx, du_dy, _, _, _ = precompute_diff_operators_2D(p, 1.0)
        out = precompute_N_matrix_2D(du_dx, du_dy, p)
        assert out.shape == (4 * p, p**2)

    def test_1(self) -> None:
        """Check the output is correct on low-degree polynomials."""
        p = 8
        # q = 6
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2

        root = DiscretizationNode2D(
            xmin=west, xmax=east, ymin=south, ymax=north
        )
        half_side_len = jnp.pi / 2
        du_dx, du_dy, _, _, _ = precompute_diff_operators_2D(p, half_side_len)
        out = precompute_N_matrix_2D(du_dx, du_dy, p)

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        def dfdx(x: jnp.array) -> jnp.array:
            # df/dx = 2x
            return 2 * x[..., 0]

        def dfdy(x: jnp.array) -> jnp.array:
            # df/dy = -3
            return -3 * jnp.ones_like(x[..., 1])

        # Set up the Chebyshev points.
        cheby_pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)
        # interior_pts = cheby_pts[0, 4 * (p - 1) :]
        exterior_pts = cheby_pts[0, : 4 * (p - 1)]
        all_pts = cheby_pts[0]

        # Compute the function values at the Chebyshev points.
        f_vals = f(all_pts)

        computed_normals = out @ f_vals

        expected_normals = jnp.concatenate(
            [
                -1 * dfdy(exterior_pts[:p]),
                dfdx(exterior_pts[p - 1 : 2 * p - 1]),
                dfdy(exterior_pts[2 * p - 2 : 3 * p - 2]),
                -1 * dfdx(exterior_pts[3 * p - 3 :]),
                -1 * dfdx(exterior_pts[0]).reshape(1),
            ]
        )
        print("test_1: computed_normals shape = ", computed_normals.shape)
        print("test_1: expected_normals shape = ", expected_normals.shape)

        # Check the computed normals against the expected normals. side-by-side.

        # S side
        print("test_1: computed_normals[:p] = ", computed_normals[:p])
        print("test_1: expected_normals[:p] = ", expected_normals[:p])
        assert jnp.allclose(computed_normals[:p], expected_normals[:p])

        # E side
        print(
            "test_1: computed_normals[p:2*p] = ", computed_normals[p : 2 * p]
        )
        print(
            "test_1: expected_normals[p:2*p] = ", expected_normals[p : 2 * p]
        )
        assert jnp.allclose(
            computed_normals[p : 2 * p], expected_normals[p : 2 * p]
        )

        # N side
        print(
            "test_1: computed_normals[2*p:3*p] = ",
            computed_normals[2 * p : 3 * p],
        )
        print(
            "test_1: expected_normals[2*p:3*p] = ",
            expected_normals[2 * p : 3 * p],
        )
        assert jnp.allclose(
            computed_normals[2 * p : 3 * p], expected_normals[2 * p : 3 * p]
        )

        # W side
        print("test_1: computed_normals[3*p:] = ", computed_normals[3 * p :])
        print("test_1: expected_normals[3*p:] = ", expected_normals[3 * p :])
        assert jnp.allclose(
            computed_normals[3 * p :], expected_normals[3 * p :]
        )

        assert jnp.allclose(computed_normals, expected_normals)


class Test_precompute_N_tilde_matrix_2D:
    def test_0(self) -> None:
        """Check the shape of the output."""
        p = 8
        du_dx, du_dy, _, _, _ = precompute_diff_operators_2D(p, 1.0)
        out = precompute_N_tilde_matrix_2D(du_dx, du_dy, p)
        assert out.shape == (4 * (p - 1), p**2)

    def test_1(self) -> None:
        """Check that low-degree polynomials are handled correctly."""

        p = 8
        # q = 6
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2
        root = DiscretizationNode2D(
            xmin=west, xmax=east, ymin=south, ymax=north
        )

        half_side_len = jnp.pi / 2
        du_dx, du_dy, _, _, _ = precompute_diff_operators_2D(p, half_side_len)
        out = precompute_N_tilde_matrix_2D(du_dx, du_dy, p)

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        def dfdx(x: jnp.array) -> jnp.array:
            # df/dx = 2x
            return 2 * x[..., 0]

        def dfdy(x: jnp.array) -> jnp.array:
            # df/dy = -3
            return -3 * jnp.ones_like(x[..., 1])

        # Set up the Chebyshev points.
        cheby_pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)[0]
        cheby_bdry = cheby_pts[: 4 * (p - 1)]

        f_evals = f(cheby_pts)
        computed_normals = out @ f_evals

        expected_normals = jnp.concatenate(
            [
                -1 * dfdy(cheby_bdry[: p - 1]),
                dfdx(cheby_bdry[p - 1 : 2 * (p - 1)]),
                dfdy(cheby_bdry[2 * (p - 1) : 3 * (p - 1)]),
                -1 * dfdx(cheby_bdry[3 * (p - 1) :]),
            ]
        )

        print("test_1: computed_normals shape = ", computed_normals.shape)
        print("test_1: expected_normals shape = ", expected_normals.shape)
        assert jnp.allclose(computed_normals, expected_normals)


class Test_precompute_QH_2D_ItI:
    def test_0(self) -> None:
        """Check the shape of the output."""
        p = 8
        q = 6
        du_dx, du_dy, _, _, _ = precompute_diff_operators_2D(p, 1.0)
        N = precompute_N_matrix_2D(du_dx, du_dy, p)
        out = precompute_QH_2D_ItI(N, p, q, 4.0)
        assert out.shape == (4 * q, p**2)

    def test_1(self) -> None:
        """Check that low-degree polynomials are handled correctly."""
        p = 8
        q = 6
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2
        root = DiscretizationNode2D(
            xmin=west, xmax=east, ymin=south, ymax=north
        )
        half_side_len = jnp.pi / 2
        du_dx, du_dy, _, _, _ = precompute_diff_operators_2D(p, half_side_len)
        N = precompute_N_matrix_2D(du_dx, du_dy, p)
        eta = 4.0
        out = precompute_QH_2D_ItI(N, p, q, eta)

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        def dfdx(x: jnp.array) -> jnp.array:
            # df/dx = 2x
            return 2 * x[..., 0]

        def dfdy(x: jnp.array) -> jnp.array:
            # df/dy = -3
            return -3 * jnp.ones_like(x[..., 1])

        # Set up the Chebyshev points.
        cheby_pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)
        gauss_pts = compute_boundary_Gauss_points_adaptive_2D(root, q)
        print("gauss_pts.shape: ", gauss_pts.shape)
        all_pts = cheby_pts[0]

        f_evals = f(all_pts)
        computed_out_imp = out @ f_evals

        f_normals = jnp.concatenate(
            [
                -1 * dfdy(gauss_pts[:q]),
                dfdx(gauss_pts[q : 2 * q]),
                dfdy(gauss_pts[2 * q : 3 * q]),
                -1 * dfdx(gauss_pts[3 * q :]),
                # -1 * dfdx(cheby_bdry[0]).reshape(1),
            ]
        )
        f_evals = f(gauss_pts)

        print("f_normals.shape: ", f_normals.shape)
        print("f_evals.shape: ", f_evals.shape)

        expected_out_imp = f_normals - 1j * eta * f_evals

        assert jnp.allclose(computed_out_imp, expected_out_imp)


class Test_precompute_G_2D_ItI:
    def test_0(self) -> None:
        """Check the shape of the output."""
        p = 8
        du_dx, du_dy, _, _, _ = precompute_diff_operators_2D(p, 1.0)
        N_tilde = precompute_N_tilde_matrix_2D(du_dx, du_dy, p)
        out = precompute_G_2D_ItI(N_tilde, 4.0)
        assert out.shape == (4 * (p - 1), p**2)

    def test_1(self) -> None:
        """Checks correctness for low-degree polynomials."""
        p = 8
        print("test_1: p = ", p)
        print("test_1: 4(p-1) = ", 4 * (p - 1))
        du_dx, du_dy, _, _, _ = precompute_diff_operators_2D(p, 0.5)
        N_tilde = precompute_N_tilde_matrix_2D(du_dx, du_dy, p)

        # Corners for a square of side length 1.0
        north = 0.5
        south = -0.5
        east = 0.5
        west = -0.5
        root = DiscretizationNode2D(
            xmin=west, xmax=east, ymin=south, ymax=north
        )
        pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        def dfdx(x: jnp.array) -> jnp.array:
            # df/dx = 2x
            return 2 * x[..., 0]

        def dfdy(x: jnp.array) -> jnp.array:
            # df/dy = -3
            return -3 * jnp.ones_like(x[..., 1])

        eta = 4.0

        f_evals = f(pts[0])
        print("test_1: f_evals shape = ", f_evals.shape)
        F = precompute_G_2D_ItI(N_tilde, eta)
        print("test_1: F shape = ", F.shape)

        computed_inc_imp = F @ f_evals
        expected_bdry_normals = jnp.concatenate(
            [
                -1 * dfdy(pts[0][: p - 1]),
                dfdx(pts[0][p - 1 : 2 * p - 2]),
                dfdy(pts[0][2 * p - 2 : 3 * p - 3]),
                -1 * dfdx(pts[0][3 * p - 3 : 4 * (p - 1)]),
            ]
        )
        expected_bdry_f = f(pts[0][: 4 * (p - 1)])
        print(
            "test_1: expected_bdry_normals shape = ",
            expected_bdry_normals.shape,
        )
        print("test_1: expected_bdry_f shape = ", expected_bdry_f.shape)
        expected_inc_imp = expected_bdry_normals + 1j * eta * expected_bdry_f

        # plt.plot(computed_inc_imp.real, "o-", label="computed_inc_imp.real")
        # plt.plot(expected_inc_imp.real, "x-", label="expected_inc_imp.real")
        # plt.plot(computed_inc_imp.imag, "o-", label="computed_inc_imp.imag")
        # plt.plot(expected_inc_imp.imag, "x-", label="expected_inc_imp.imag")
        # plt.legend()
        # plt.show()

        assert jnp.allclose(computed_inc_imp, expected_inc_imp)


class Test_precompute_projection_ops_2D:
    def test_0(self) -> None:
        q = 4

        ref, coarse = precompute_projection_ops_2D(q)

        assert ref.shape == (2 * q, q)
        assert coarse.shape == (q, 2 * q)

        assert not jnp.any(jnp.isnan(ref))
        assert not jnp.any(jnp.isinf(ref))
        assert not jnp.any(jnp.isnan(coarse))
        assert not jnp.any(jnp.isinf(coarse))

    def test_1(self) -> None:
        q = 12
        ref, coarse = precompute_projection_ops_2D(q)

        def f(x: jnp.array) -> jnp.array:
            """f(x) = 3 + 4x - 5x**2"""
            return 3 + 4 * x - 5 * x**2

        gauss_panel = gauss_points(q)
        double_gauss_panel = jnp.concatenate(
            [
                affine_transform(gauss_panel, [-1.0, 0.0]),
                affine_transform(gauss_panel, [0.0, 1.0]),
            ]
        )
        f_vals = f(gauss_panel)
        f_ref = ref @ f_vals
        f_ref_expected = f(double_gauss_panel)

        assert jnp.allclose(f_ref, f_ref_expected)

        f_coarse = coarse @ f_ref_expected
        f_coarse_expected = f_vals
        assert jnp.allclose(f_coarse, f_coarse_expected)


class Test_precompute_rectangular_diff_operators_2D:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)

        p = 4
        half_side_len = 0.5
        D_x, D_y, D_xx, D_yy, D_xy, B, D = (
            precompute_rectangular_diff_operators_2D(p, half_side_len)
        )
        expected_shape = ((p - 2) ** 2, p**2)
        assert D_x.shape == expected_shape
        assert D_y.shape == expected_shape
        assert D_xx.shape == expected_shape
        assert D_yy.shape == expected_shape
        assert D_xy.shape == expected_shape
        assert not jnp.any(jnp.isnan(D_x))
        assert not jnp.any(jnp.isinf(D_x))
        assert not jnp.any(jnp.isnan(D_y))
        assert not jnp.any(jnp.isinf(D_y))
        assert not jnp.any(jnp.isnan(D_xx))
        assert not jnp.any(jnp.isinf(D_xx))
        assert not jnp.any(jnp.isnan(D_yy))
        assert not jnp.any(jnp.isinf(D_yy))
        assert not jnp.any(jnp.isnan(D_xy))
        assert not jnp.any(jnp.isinf(D_xy))

    def test_1(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        """Check that low-degree polynomials are handled correctly."""
        p = 8
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2

        # north = 1.0
        # south = -1.0
        # east = 1.0
        # west = -1.0

        half_side_len = (east - west) / 2
        root = DiscretizationNode2D(
            xmin=west, xmax=east, ymin=south, ymax=north
        )

        # Compute interpolation operator
        D_x, D_y, _, _, _, B, D = precompute_rectangular_diff_operators_2D(
            p, half_side_len
        )
        to_x = affine_transform(
            first_kind_chebyshev_points(p - 2), jnp.array([west, east])
        )
        to_y = jnp.flipud(to_x)
        # Set up target grid
        target_X, target_Y = jnp.meshgrid(to_x, to_y, indexing="ij")
        target_pts = jnp.stack(
            (target_X.flatten(), target_Y.flatten()), axis=-1
        )

        source_pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)[0]

        logging.info(
            "target_pts shape: %s, and source_pts shape: %s",
            target_pts.shape,
            source_pts.shape,
        )

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        def dfdx(x: jnp.array) -> jnp.array:
            # df/dx = 2x
            return 2 * x[..., 0]

        def dfdy(x: jnp.array) -> jnp.array:
            # df/dy = -3
            return -3 * jnp.ones_like(x[..., 1])

        # First, test Dx

        f_src = f(source_pts)
        dfdx_target = dfdx(target_pts)
        dfdx_computed = D_x @ f_src

        logging.debug("dfdx_target: %s", dfdx_target)
        logging.debug("dfdx_computed: %s", dfdx_computed)
        logging.debug("diffs: %s", dfdx_target - dfdx_computed)

        # plot_soln_from_cheby_nodes(
        #     target_pts,
        #     corners=None,
        #     expected_soln=f_target,
        #     computed_soln=f_interp,
        # )
        assert jnp.allclose(dfdx_computed, dfdx_target)

        # Next, test Dy
        f_src = f(source_pts)
        dfdy_target = dfdy(target_pts)
        dfdy_computed = D_y @ f_src
        logging.debug("dfdy_target: %s", dfdy_target)
        logging.debug("dfdy_computed: %s", dfdy_computed)
        logging.debug("diffs: %s", dfdy_target - dfdy_computed)
        # plot_soln_from_cheby_nodes(
        #     target_pts,
        #     corners=None,
        #     expected_soln=f_target,
        #     computed_soln=f_interp,
        # )
        assert jnp.allclose(dfdy_computed, dfdy_target)


class Test_rectangular_interp_operator:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        p = 4
        B, D = rectangular_interp_operator(p)
        expected_shape = ((p - 2) ** 2, p**2)
        assert B.shape == expected_shape
        assert not jnp.any(jnp.isnan(B))
        assert not jnp.any(jnp.isinf(B))

        expected_shape = (p**2, (p - 2) ** 2)
        assert D.shape == expected_shape
        assert not jnp.any(jnp.isnan(D))
        assert not jnp.any(jnp.isinf(D))

    def test_1(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        """Check that low-degree polynomials are handled correctly."""
        p = 8
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2

        # north = 1.0
        # south = -1.0
        # east = 1.0
        # west = -1.0

        # half_side_len = (east - west) / 2
        root = DiscretizationNode2D(
            xmin=west, xmax=east, ymin=south, ymax=north
        )

        # Compute interpolation operator
        B, D = rectangular_interp_operator(p)
        to_x = affine_transform(
            first_kind_chebyshev_points(p - 2), jnp.array([west, east])
        )
        to_y = jnp.flipud(to_x)
        # Set up target grid
        target_X, target_Y = jnp.meshgrid(to_x, to_y, indexing="ij")
        target_pts = jnp.stack(
            (target_X.flatten(), target_Y.flatten()), axis=-1
        )

        source_pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)[0]

        logging.info(
            "target_pts shape: %s, and source_pts shape: %s",
            target_pts.shape,
            source_pts.shape,
        )

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        f_src = f(source_pts)
        f_target = f(target_pts)
        logging.info(
            "f_src shape: %s, and f_target shape: %s",
            f_src.shape,
            f_target.shape,
        )
        f_interp = B @ f_src

        logging.debug("f_target: %s", f_target)
        logging.debug("f_interp: %s", f_interp)
        logging.debug("diffs: %s", f_target - f_interp)

        # plot_soln_from_cheby_nodes(
        #     target_pts,
        #     corners=None,
        #     expected_soln=f_target,
        #     computed_soln=f_interp,
        # )
        assert jnp.allclose(f_interp, f_target)

        # Check that D interpolates correctly
        f_interp_D = D @ f_target
        logging.debug("f_interp_D: %s", f_interp_D)
        logging.debug("f_src: %s", f_src)
        logging.debug("diffs: %s", f_interp_D - f_src)
        assert jnp.allclose(f_interp_D, f_src)

    def test_2(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        """Check that low-degree polynomials are handled correctly."""
        p = 8
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2

        # north = 1.0
        # south = -1.0
        # east = 1.0
        # west = -1.0

        # half_side_len = (east - west) / 2
        root = DiscretizationNode2D(
            xmin=west, xmax=east, ymin=south, ymax=north
        )

        # Compute interpolation operator
        B, D = rectangular_interp_operator(p)
        to_x = affine_transform(
            first_kind_chebyshev_points(p - 2), jnp.array([west, east])
        )
        to_y = jnp.flipud(to_x)
        # Set up target grid
        target_X, target_Y = jnp.meshgrid(to_x, to_y, indexing="ij")
        target_pts = jnp.stack(
            (target_X.flatten(), target_Y.flatten()), axis=-1
        )

        source_pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)[0]

        logging.info(
            "target_pts shape: %s, and source_pts shape: %s",
            target_pts.shape,
            source_pts.shape,
        )

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = 2x - 3y^2
            return 2 * x[..., 0] - 3 * x[..., 1] ** 2

        f_src = f(source_pts)
        f_target = f(target_pts)
        logging.info(
            "f_src shape: %s, and f_target shape: %s",
            f_src.shape,
            f_target.shape,
        )
        f_interp = B @ f_src

        logging.debug("f_target: %s", f_target)
        logging.debug("f_interp: %s", f_interp)
        logging.debug("diffs: %s", f_target - f_interp)

        # plot_soln_from_cheby_nodes(
        #     target_pts,
        #     corners=None,
        #     expected_soln=f_target,
        #     computed_soln=f_interp,
        # )
        assert jnp.allclose(f_interp, f_target)
        # Check that D interpolates correctly
        f_interp_D = D @ f_target
        logging.debug("f_interp_D: %s", f_interp_D)
        logging.debug("f_src: %s", f_src)
        logging.debug("diffs: %s", f_interp_D - f_src)
        assert jnp.allclose(f_interp_D, f_src)


class Test_interpolate_cheby_to_cheby_first_kind:
    def test_0(self) -> None:
        """Tests the interpolate_cheby_to_cheby_first_kind function returns without error and returns the correct shape."""
        p = 8

        B = jnp.array(np.random.normal(size=((p - 2) ** 2, p**2)))
        D = jnp.array(np.random.normal(size=(p**2, (p - 2) ** 2)))

        v = jnp.array(np.random.normal(size=(p**2,)))

        out = interpolate_cheby_to_cheby_first_kind(B, D, v)

        assert out.shape == ((p - 2) ** 2,)
        jax.clear_caches()

    def test_1(self, caplog) -> None:
        """Try out the VJP rule."""
        caplog.set_level(logging.DEBUG)
        p = 8

        B = jnp.array(np.random.normal(size=((p - 2) ** 2, p**2)))
        D = jnp.array(np.random.normal(size=(p**2, (p - 2) ** 2)))

        v = jnp.array(np.random.normal(size=(p**2,)))

        v_dot = jnp.array(np.random.normal(size=((p - 2) ** 2,)))

        out, vjp_fn = jax.vjp(interpolate_cheby_to_cheby_first_kind, B, D.T, v)
        out_jvp = vjp_fn(v_dot)[2]
        assert out.shape == ((p - 2) ** 2,)
        assert out_jvp.shape == (p**2,)
        jax.clear_caches()

    def test_2(self, caplog) -> None:
        """Try out the accuracy of the VJP rule."""
        caplog.set_level(logging.DEBUG)
        p = 16

        B, D = rectangular_interp_operator(p)
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2

        root = DiscretizationNode2D(
            xmin=west, xmax=east, ymin=south, ymax=north
        )
        source_pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)[0]
        to_x = affine_transform(
            first_kind_chebyshev_points(p - 2), jnp.array([west, east])
        )
        to_y = jnp.flipud(to_x)
        # Set up target grid
        target_X, target_Y = jnp.meshgrid(to_x, to_y, indexing="ij")
        target_pts = jnp.stack(
            (target_X.flatten(), target_Y.flatten()), axis=-1
        )

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = 2x - 3y^2
            return 2 * x[..., 0] - 3 * x[..., 1] ** 2

        f_src = f(source_pts)
        f_target = f(target_pts)

        out, vjp_fn = jax.vjp(
            interpolate_cheby_to_cheby_first_kind, B, D.T, f_src
        )
        assert jnp.allclose(out, f_target)

        # Compute the vJp of f_target
        out_vjp = vjp_fn(f_target)[2]
        assert jnp.allclose(out_vjp, f_src)

        jax.clear_caches()
