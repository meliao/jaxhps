import logging
import pytest
import numpy as np
import jax.numpy as jnp

from hps.src.solver_obj import SolverObj, create_solver_obj_2D, create_solver_obj_3D
from hps.src.methods.uniform_down_pass import (
    _uniform_down_pass_2D_DtN,
    _propogate_down_quad,
    _propogate_down_oct,
    _propogate_down_quad_ItI,
    _uniform_down_pass_2D_ItI,
    _uniform_down_pass_3D_DtN,
)
from hps.src.methods.local_solve_stage import (
    _local_solve_stage_2D,
    _local_solve_stage_2D_ItI,
    _local_solve_stage_3D,
)
from hps.src.methods.uniform_build_stage import (
    _uniform_build_stage_2D_DtN,
    _uniform_build_stage_3D_DtN,
    _uniform_build_stage_2D_ItI,
)
from hps.src.quadrature.trees import Node, get_all_leaves, add_uniform_levels


class Test__uniform_down_pass_2D_DtN:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        p = 16
        q = 14
        l = 3
        num_leaves = 4**l
        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l)
        d_xx_coeffs = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        sidelens = jnp.array([l.xmax - l.xmin for l in get_all_leaves(t.root)])
        Y_arr, DtN_arr, v_arr, v_prime_arr = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            p=p,
            sidelens=sidelens,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
        )
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        print("test_0: DtN_arr.shape = ", DtN_arr.shape)
        print("test_0: v_prime_arr.shape = ", v_prime_arr.shape)
        S_arr_lst, DtN_arr_lst, v_int_lst = _uniform_build_stage_2D_DtN(
            DtN_maps=DtN_arr, v_prime_arr=v_prime_arr, l=l
        )

        boundary_data = jnp.ones_like(t.root_boundary_points[..., 0])

        leaf_solns = _uniform_down_pass_2D_DtN(
            boundary_data,
            S_arr_lst,
            v_int_lst,
            leaf_Y_maps=Y_arr,
            v_array=v_arr,
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
        )
        assert leaf_solns.shape == (num_leaves, p**2)


class Test__uniform_down_pass_2D_ItI:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        p = 16
        q = 14
        l = 2
        n_src = 1
        num_leaves = 4**l
        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, eta=4.0, use_ItI=True)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = np.random.normal(size=(num_leaves, p**2))
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
            source_term=source_term,
        )
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        S_arr_lst, R_arr_lst, f_arr_lst = _uniform_build_stage_2D_ItI(
            R_maps=R_arr, h_arr=g_arr, l=l
        )

        logging.debug("test_0: f_arr_lst shapes = %s", [f.shape for f in f_arr_lst])

        boundary_data = jnp.ones(
            (t.root_boundary_points.shape[0], n_src), dtype=jnp.float64
        )

        leaf_solns = _uniform_down_pass_2D_ItI(
            boundary_data,
            S_arr_lst,
            f_arr_lst,
            Y_arr,
            v_arr,
        )
        assert leaf_solns.shape == (num_leaves, p**2, n_src)

    def test_1(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        p = 16
        q = 14
        l = 2
        n_src = 3
        num_leaves = 4**l
        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, eta=4.0, use_ItI=True)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = np.random.normal(size=(num_leaves, p**2, n_src))
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
            source_term=source_term,
        )
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        S_arr_lst, R_arr_lst, f_arr_lst = _uniform_build_stage_2D_ItI(
            R_maps=R_arr, h_arr=g_arr, l=l
        )

        logging.debug("test_0: f_arr_lst shapes = %s", [f.shape for f in f_arr_lst])

        boundary_data = jnp.ones(
            (t.root_boundary_points.shape[0], n_src), dtype=jnp.float64
        )

        leaf_solns = _uniform_down_pass_2D_ItI(
            boundary_data,
            S_arr_lst,
            f_arr_lst,
            Y_arr,
            v_arr,
        )
        assert leaf_solns.shape == (num_leaves, p**2, n_src)


class Test__propogate_down_quad:
    def test_0(self) -> None:
        """Tests to make sure returns without error."""
        n_child = 8
        S_arr = np.random.normal(size=(4 * n_child, 8 * n_child))
        bdry_data = np.random.normal(size=(8 * n_child))
        v_int = np.random.normal(size=(4 * n_child))

        out = _propogate_down_quad(S_arr, bdry_data, v_int)
        expected_out_shape = (4, 4 * n_child)
        assert out.shape == expected_out_shape

    def test_1(self) -> None:
        n_child = 8
        S_arr = np.random.normal(size=(4 * n_child, 8 * n_child))
        bdry_data = np.random.normal(size=(8 * n_child))
        v_int = np.random.normal(size=(4 * n_child))

        out = _propogate_down_quad(S_arr, bdry_data, v_int)

        g_a = out[0]
        g_b = out[1]
        g_c = out[2]
        g_d = out[3]

        # Check the interfaces match up

        # Edge 5
        assert jnp.allclose(g_a[n_child : 2 * n_child], jnp.flipud(g_b[3 * n_child :]))
        # Edge 6
        assert jnp.allclose(g_b[2 * n_child : 3 * n_child], jnp.flipud(g_c[:n_child]))
        # Edge 7
        assert jnp.allclose(g_c[3 * n_child :], jnp.flipud(g_d[n_child : 2 * n_child]))
        # Edge 8
        assert jnp.allclose(g_d[:n_child], jnp.flipud(g_a[2 * n_child : 3 * n_child]))


class Test__propogate_down_oct:
    def test_0(self) -> None:
        n_per_face = 3
        S_arr = np.random.normal(size=(12 * n_per_face, 24 * n_per_face))
        bdry_data = np.random.normal(size=(24 * n_per_face))
        v_int_data = np.random.normal(size=(12 * n_per_face))

        out = _propogate_down_oct(S_arr, bdry_data, v_int_data)
        expected_out_shape = (8, 6 * n_per_face)
        assert out.shape == expected_out_shape


class Test__propogate_down_quad_ItI:
    def test_0(self) -> None:
        """Tests to make sure returns without error."""
        n_child = 8
        n_src = 3
        S_arr = np.random.normal(size=(8 * n_child, 8 * n_child))
        bdry_data = np.random.normal(size=(8 * n_child, n_src))
        f = np.random.normal(size=(8 * n_child, n_src))

        out = _propogate_down_quad_ItI(S_arr, bdry_data, f)
        expected_out_shape = (4, 4 * n_child, n_src)
        assert out.shape == expected_out_shape


if __name__ == "__main__":
    pytest.main()
