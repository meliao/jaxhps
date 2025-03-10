import logging
import pytest
import numpy as np
import jax.numpy as jnp

from hps.src.solver_obj import SolverObj, create_solver_obj_2D, create_solver_obj_3D

from hps.src.methods.uniform_build_stage import (
    _uniform_build_stage_2D_DtN,
    _uniform_quad_merge,
    _uniform_build_stage_3D_DtN,
    _uniform_oct_merge,
    _uniform_quad_merge_ItI,
    _uniform_build_stage_2D_ItI,
)
from hps.src.methods.local_solve_stage import (
    _local_solve_stage_2D,
    _local_solve_stage_3D,
    _local_solve_stage_2D_ItI,
)
from hps.src.quadrature.trees import Node, get_all_leaves


class Test__uniform_build_stage_2D_DtN:
    def test_0(self, caplog) -> None:
        """Tests the _uniform_build_stage_2D_DtN function returns without error."""
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
            sidelens=sidelens,
            p=p,
            D_xx_coeffs=d_xx_coeffs,
            source_term=source_term,
        )

        # assert Y_arr.shape == (num_leaves, p**2, 4 * q)
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        print("test_0: DtN_arr.shape = ", DtN_arr.shape)
        print("test_0: v_prime_arr.shape = ", v_prime_arr.shape)
        S_arr_lst, DtN_arr_lst, v_arr_lst = _uniform_build_stage_2D_DtN(
            DtN_maps=DtN_arr, v_prime_arr=v_prime_arr, l=l
        )

        assert len(S_arr_lst) == l
        # assert len(DtN_arr_lst) == l
        assert len(v_arr_lst) == l

        # Check to make sure the arrays haven't been deleted
        print("test_0: S_arr_lst sums: ", [S_arr.sum() for S_arr in S_arr_lst])
        print("test_0: v_arr_lst sums: ", [v_arr.sum() for v_arr in v_arr_lst])
        print("test_0: DtN_arr_lst sums: ", [DtN_arr.sum() for DtN_arr in DtN_arr_lst])

        for i in range(l):
            assert S_arr_lst[i].shape[-2] == v_arr_lst[i].shape[-1]
            # assert S_arr_lst[i].shape[-1] == DtN_arr_lst[i].shape[-1]

        # Check the shapes of the bottom-level output arrays
        n_quads = (num_leaves // 4) // 4
        assert S_arr_lst[0].shape == (4 * n_quads, 4 * q, 8 * q)
        # assert DtN_arr_lst[0].shape == (n_quads, 4, 8 * q, 8 * q)
        assert v_arr_lst[0].shape == (4 * n_quads, 4 * q)

        # Check the shapes of the middle-level output arrays.
        n_bdry = 16 * q
        n_interface = 8 * q
        assert S_arr_lst[1].shape == (4, n_interface, n_bdry)
        # assert DtN_arr_lst[1].shape == (1, 4, n_bdry, n_bdry)
        assert v_arr_lst[1].shape == (4, n_interface)

        # Check the shapes of the top-level output arrays.
        n_root_bdry = t.root_boundary_points.shape[0]
        n_root_interface = n_root_bdry // 2
        assert S_arr_lst[2].shape == (1, n_root_interface, n_root_bdry)
        # assert DtN_arr_lst[2].shape == (n_root_bdry, n_root_bdry)
        assert v_arr_lst[2].shape == (
            1,
            n_root_interface,
        )


class Test__uniform_build_stage_3D_DtN:
    def test_0(self, caplog) -> None:
        # Make sure it runs with correct inputs and outputs.
        caplog.set_level(logging.DEBUG)
        p = 8
        q = 6
        l = 2
        num_leaves = 8**l
        root = Node(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, depth=0)
        t = create_solver_obj_3D(p=p, q=q, root=root, uniform_levels=l)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**3))
        source_term = np.random.normal(size=(num_leaves, p**3))
        sidelens = jnp.array([l.xmax - l.xmin for l in get_all_leaves(t.root)])
        print("test_0: d_xx_coeffs shape: ", d_xx_coeffs.shape)
        print("test_0: source_term shape: ", source_term.shape)
        print("test_0: sidelens: ", len(sidelens))
        DtN_arr, v, v_prime_arr, Y_arr = _local_solve_stage_3D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_xz=t.D_xz,
            D_yz=t.D_yz,
            D_zz=t.D_zz,
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
        assert Y_arr.shape == (num_leaves, p**3, 6 * q**2)
        S_arr_lst, DtN_arr_lst, v_arr_lst, D_shape = _uniform_build_stage_3D_DtN(
            root=root,
            DtN_maps=DtN_arr,
            v_prime_arr=v_prime_arr,
        )
        assert len(S_arr_lst) == l
        assert len(DtN_arr_lst) == l
        assert len(v_arr_lst) == l
        for i in range(l):
            assert S_arr_lst[i].shape[-2] == v_arr_lst[i].shape[-1]
            assert S_arr_lst[i].shape[-1] == DtN_arr_lst[i].shape[-1]


class Test__uniform_quad_merge:
    def test_0(self) -> None:

        n_bdry = 28
        n_bdry_int = n_bdry // 4
        n_bdry_ext = 2 * (n_bdry // 4)
        T_a = np.random.normal(size=(n_bdry, n_bdry))
        T_b = np.random.normal(size=(n_bdry, n_bdry))
        T_c = np.random.normal(size=(n_bdry, n_bdry))
        T_d = np.random.normal(size=(n_bdry, n_bdry))
        v_prime_a = np.random.normal(size=(n_bdry))
        v_prime_b = np.random.normal(size=(n_bdry))
        v_prime_c = np.random.normal(size=(n_bdry))
        v_prime_d = np.random.normal(size=(n_bdry))

        S, T, v_prime_ext, v_int = _uniform_quad_merge(
            T_a, T_b, T_c, T_d, v_prime_a, v_prime_b, v_prime_c, v_prime_d
        )

        assert S.shape == (4 * n_bdry_int, 4 * n_bdry_ext)
        assert T.shape == (4 * n_bdry_ext, 4 * n_bdry_ext)
        assert v_prime_ext.shape == (4 * n_bdry_ext,)
        assert v_int.shape == (4 * n_bdry_int,)


class Test__uniform_quad_merge_ItI:
    def test_0(self) -> None:

        n_bdry = 28
        n_bdry_int = n_bdry // 4
        n_bdry_ext = 2 * (n_bdry // 4)
        n_src = 1
        T_a = np.random.normal(size=(n_bdry, n_bdry))
        T_b = np.random.normal(size=(n_bdry, n_bdry))
        T_c = np.random.normal(size=(n_bdry, n_bdry))
        T_d = np.random.normal(size=(n_bdry, n_bdry))
        v_prime_a = np.random.normal(size=(n_bdry, n_src))
        v_prime_b = np.random.normal(size=(n_bdry, n_src))
        v_prime_c = np.random.normal(size=(n_bdry, n_src))
        v_prime_d = np.random.normal(size=(n_bdry, n_src))
        print("test_0: T_a shape: ", T_a.shape)
        print("test_0: v_prime_a shape: ", v_prime_a.shape)
        S, R, h, f = _uniform_quad_merge_ItI(
            T_a, T_b, T_c, T_d, v_prime_a, v_prime_b, v_prime_c, v_prime_d
        )

        assert S.shape == (8 * n_bdry_int, 4 * n_bdry_ext)
        assert R.shape == (4 * n_bdry_ext, 4 * n_bdry_ext)
        assert h.shape == (4 * n_bdry_ext, n_src)
        assert f.shape == (8 * n_bdry_int, n_src)

    def test_1(self) -> None:
        """Multiple sources"""

        n_bdry = 28
        n_src = 3
        n_bdry_int = n_bdry // 4
        n_bdry_ext = 2 * (n_bdry // 4)
        T_a = np.random.normal(size=(n_bdry, n_bdry))
        T_b = np.random.normal(size=(n_bdry, n_bdry))
        T_c = np.random.normal(size=(n_bdry, n_bdry))
        T_d = np.random.normal(size=(n_bdry, n_bdry))
        v_prime_a = np.random.normal(size=(n_bdry, n_src))
        v_prime_b = np.random.normal(size=(n_bdry, n_src))
        v_prime_c = np.random.normal(size=(n_bdry, n_src))
        v_prime_d = np.random.normal(size=(n_bdry, n_src))
        print("test_0: T_a shape: ", T_a.shape)
        print("test_0: v_prime_a shape: ", v_prime_a.shape)
        S, R, h, f = _uniform_quad_merge_ItI(
            T_a, T_b, T_c, T_d, v_prime_a, v_prime_b, v_prime_c, v_prime_d
        )

        assert S.shape == (8 * n_bdry_int, 4 * n_bdry_ext)
        assert R.shape == (4 * n_bdry_ext, 4 * n_bdry_ext)
        assert h.shape == (4 * n_bdry_ext, n_src)
        assert f.shape == (8 * n_bdry_int, n_src)


class Test__uniform_oct_merge:
    def test_0(self):
        q = 2
        n_gauss_bdry = 6 * q**2
        T_a = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_b = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_c = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_d = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_e = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_f = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_g = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        T_h = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        v_prime_a = np.random.normal(size=(n_gauss_bdry))
        v_prime_b = np.random.normal(size=(n_gauss_bdry))
        v_prime_c = np.random.normal(size=(n_gauss_bdry))
        v_prime_d = np.random.normal(size=(n_gauss_bdry))
        v_prime_e = np.random.normal(size=(n_gauss_bdry))
        v_prime_f = np.random.normal(size=(n_gauss_bdry))
        v_prime_g = np.random.normal(size=(n_gauss_bdry))
        v_prime_h = np.random.normal(size=(n_gauss_bdry))
        q_idxes = np.arange(q)
        S, T, v_prime_ext, v_int = _uniform_oct_merge(
            q_idxes=q_idxes,
            T_a=T_a,
            T_b=T_b,
            T_c=T_c,
            T_d=T_d,
            T_e=T_e,
            T_f=T_f,
            T_g=T_g,
            T_h=T_h,
            v_prime_a=v_prime_a,
            v_prime_b=v_prime_b,
            v_prime_c=v_prime_c,
            v_prime_d=v_prime_d,
            v_prime_e=v_prime_e,
            v_prime_f=v_prime_f,
            v_prime_g=v_prime_g,
            v_prime_h=v_prime_h,
        )

        assert S.shape == (12 * q**2, 24 * q**2)
        assert T.shape == (24 * q**2, 24 * q**2)
        assert v_prime_ext.shape == (24 * q**2,)
        assert v_int.shape == (12 * q**2,)


class Test__uniform_build_stage_2D_ItI:
    def test_0(self, caplog) -> None:
        """Tests the function returns without error. n_src = 1"""
        caplog.set_level(logging.DEBUG)

        p = 16
        q = 14
        l = 3
        eta = 4.0
        num_leaves = 4**l
        n_src = 1
        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, use_ItI=True, eta=eta)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = np.random.normal(size=(num_leaves, p**2))
        R_arr, Y_arr, h_arr, v_arr = _local_solve_stage_2D_ItI(
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

        print("test_0: h_arr.shape = ", h_arr.shape)

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        S_arr_lst, R_arr_lst, f_arr_lst = _uniform_build_stage_2D_ItI(
            R_maps=R_arr, h_arr=h_arr, l=l
        )
        print("test_0: S_arr_lst shapes = ", [S_arr.shape for S_arr in S_arr_lst])

        assert len(S_arr_lst) == l
        assert len(R_arr_lst) == l
        assert len(f_arr_lst) == l

        for i in range(l):
            print("test_0: i=", i)
            print("test_0: S_arr_lst[i].shape = ", S_arr_lst[i].shape)
            print("test_0: f_arr_lst[i].shape = ", f_arr_lst[i].shape)
            print("test_0: R_arr_lst[i].shape = ", R_arr_lst[i].shape)
            assert S_arr_lst[i].shape[-2] == f_arr_lst[i].shape[-2]
            assert S_arr_lst[i].shape[-1] == R_arr_lst[i].shape[-1]

        # Check the shapes of the bottom-level output arrays
        n_quads = (num_leaves // 4) // 4
        assert S_arr_lst[0].shape == (4 * n_quads, 8 * q, 8 * q)
        assert R_arr_lst[0].shape == (n_quads, 4, 8 * q, 8 * q)
        assert f_arr_lst[0].shape == (4 * n_quads, 8 * q, n_src)

        # Check the shapes of the middle-level output arrays.
        n_bdry = 16 * q
        n_interface = 16 * q
        assert S_arr_lst[1].shape == (4, n_interface, n_bdry)
        assert R_arr_lst[1].shape == (1, 4, n_bdry, n_bdry)
        assert f_arr_lst[1].shape == (4, n_interface, n_src)

        # Check the shapes of the top-level output arrays.
        n_root_bdry = t.root_boundary_points.shape[0]
        n_root_interface = n_root_bdry
        assert S_arr_lst[2].shape == (1, n_root_interface, n_root_bdry)
        assert R_arr_lst[2].shape == (n_root_bdry, n_root_bdry)
        assert f_arr_lst[2].shape == (1, n_root_interface, n_src)

    def test_1(self, caplog) -> None:
        """Tests the function returns without error. n_src = 1"""
        caplog.set_level(logging.DEBUG)
        p = 16
        q = 14
        l = 3
        eta = 4.0
        num_leaves = 4**l
        n_src = 3
        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, use_ItI=True, eta=eta)
        d_xx_coeffs = np.random.normal(size=(num_leaves, p**2))
        source_term = np.random.normal(size=(num_leaves, p**2, n_src))
        R_arr, Y_arr, h_arr, v_arr = _local_solve_stage_2D_ItI(
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

        print("test_0: h_arr.shape = ", h_arr.shape)

        assert Y_arr.shape == (num_leaves, p**2, 4 * q)
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        S_arr_lst, R_arr_lst, f_arr_lst = _uniform_build_stage_2D_ItI(
            R_maps=R_arr, h_arr=h_arr, l=l
        )
        print("test_0: S_arr_lst shapes = ", [S_arr.shape for S_arr in S_arr_lst])

        assert len(S_arr_lst) == l
        assert len(R_arr_lst) == l
        assert len(f_arr_lst) == l

        for i in range(l):
            print("test_0: i=", i)
            print("test_0: S_arr_lst[i].shape = ", S_arr_lst[i].shape)
            print("test_0: f_arr_lst[i].shape = ", f_arr_lst[i].shape)
            print("test_0: R_arr_lst[i].shape = ", R_arr_lst[i].shape)
            assert S_arr_lst[i].shape[-2] == f_arr_lst[i].shape[-2]
            assert S_arr_lst[i].shape[-1] == R_arr_lst[i].shape[-1]

        # Check the shapes of the bottom-level output arrays
        n_quads = (num_leaves // 4) // 4
        assert S_arr_lst[0].shape == (4 * n_quads, 8 * q, 8 * q)
        assert R_arr_lst[0].shape == (n_quads, 4, 8 * q, 8 * q)
        assert f_arr_lst[0].shape == (4 * n_quads, 8 * q, n_src)

        # Check the shapes of the middle-level output arrays.
        n_bdry = 16 * q
        n_interface = 16 * q
        assert S_arr_lst[1].shape == (4, n_interface, n_bdry)
        assert R_arr_lst[1].shape == (1, 4, n_bdry, n_bdry)
        assert f_arr_lst[1].shape == (4, n_interface, n_src)

        # Check the shapes of the top-level output arrays.
        n_root_bdry = t.root_boundary_points.shape[0]
        n_root_interface = n_root_bdry
        assert S_arr_lst[2].shape == (1, n_root_interface, n_root_bdry)
        assert R_arr_lst[2].shape == (n_root_bdry, n_root_bdry)
        assert f_arr_lst[2].shape == (1, n_root_interface, n_src)


if __name__ == "__main__":
    pytest.main()
