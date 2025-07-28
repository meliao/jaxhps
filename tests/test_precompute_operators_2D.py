import jax.numpy as jnp

from jaxhps._precompute_operators_2D import (
    precompute_diff_operators_2D,
    precompute_N_matrix_2D,
    precompute_P_2D_ItI,
    precompute_G_2D_ItI,
    precompute_N_tilde_matrix_2D,
    precompute_QH_2D_ItI,
    precompute_QG_2D_ItI,
    precompute_projection_ops_2D,
)
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._grid_creation_2D import (
    compute_interior_Chebyshev_points_adaptive_2D,
    compute_boundary_Gauss_points_adaptive_2D,
)
from jaxhps.quadrature import affine_transform, gauss_points


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


class Test_precompute_QG_2D_ItI:
    def test_0(self) -> None:
        """Check the shape of the output."""
        p = 8
        q = 6
        du_dx, du_dy, _, _, _ = precompute_diff_operators_2D(p, 1.0)
        N = precompute_N_matrix_2D(du_dx, du_dy, p)
        out = precompute_QG_2D_ItI(N, p, q, 4.0)
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
        out = precompute_QG_2D_ItI(N, p, q, eta)

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

        expected_out_imp = f_normals + 1j * eta * f_evals

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
