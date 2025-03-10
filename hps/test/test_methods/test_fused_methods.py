import pytest
import logging
import numpy as np
import jax.numpy as jnp

from hps.src.solver_obj import create_solver_obj_2D
from hps.src.methods.fused_methods import (
    _fused_local_solve_and_build_2D,
    _fused_local_solve_and_build_2D_ItI,
    _down_pass_from_fused,
    _fused_all_single_chunk,
    _fused_all_single_chunk_ItI,
    _partial_down_pass_ItI,
    _partial_down_pass,
    _down_pass_from_fused_ItI,
)
from hps.src.config import get_fused_chunksize_2D, GPU_AVAILABLE
from hps.src.quadrature.trees import Node, get_all_leaves


class Test_fused_local_solve_and_build:
    @pytest.mark.skipif(
        not GPU_AVAILABLE, reason="Can not run fused methods on CPU-only systems."
    )
    def test_0(self, caplog) -> None:
        """Tests the fused_local_solve_and_build function returns without error when using DtN maps. 2D case."""
        caplog.set_level(logging.DEBUG)
        p = 7
        q = 5
        l = 4

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l)
        logging.debug("test_0: l = %s", l)

        logging.debug("test_0: q = %s", q)
        logging.debug("test_0: Expected number of boundary points: %s", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])
        sidelens = jnp.array([l.xmax - l.xmin for l in get_all_leaves(t.root)])

        S_arr_lst, DtN_arr_lst, v_arr_lst = _fused_local_solve_and_build_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            sidelens=sidelens,
            l=l,
            p=t.p,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )
        n_fused_levels = get_fused_chunksize_2D(p, source_term.dtype, 4**l)[1]

        # assert len(S_arr_lst) == l - n_fused_levels
        # # assert len(DtN_arr_lst) == l - n_fused_levels
        # assert len(v_arr_lst) == l - n_fused_levels
        assert S_arr_lst[-1].shape[-1] == t.root_boundary_points.shape[0]


class Test_fused_local_solve_and_build_ItI:
    def test_0(self) -> None:
        """Tests the fused_local_solve_and_build_ItI function returns without error. 2D case."""
        p = 7
        q = 5
        l = 4
        n_src = 1

        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, use_ItI=True, eta=4.0)
        print("test_0: l = ", l)

        print("test_0: q = ", q)
        print("test_0: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        a, b, _ = t.leaf_cheby_points.shape
        source_term = jnp.zeros((a, b, n_src), dtype=jnp.float64)

        S_arr_lst, DtN_arr_lst, v_arr_lst = _fused_local_solve_and_build_2D_ItI(
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
            l=l,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )
        n_fused_levels = get_fused_chunksize_2D(p, source_term.dtype, 4**l)[1]

        # assert len(S_arr_lst) == l - n_fused_levels
        # # assert len(DtN_arr_lst) == l - n_fused_levels
        # assert len(v_arr_lst) == l - n_fused_levels
        assert S_arr_lst[-1].shape[-1] == t.root_boundary_points.shape[0]

    def test_1(self) -> None:
        """Tests the fused_local_solve_and_build_ItI function returns without error. 2D multi-source case."""
        p = 7
        q = 5
        l = 4
        n_src = 3

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, use_ItI=True, eta=4.0)
        print("test_0: l = ", l)

        print("test_0: q = ", q)
        print("test_0: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        a, b, _ = t.leaf_cheby_points.shape
        source_term = jnp.zeros((a, b, n_src), dtype=jnp.float64)

        S_arr_lst, DtN_arr_lst, v_arr_lst = _fused_local_solve_and_build_2D_ItI(
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
            l=l,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )
        n_fused_levels = get_fused_chunksize_2D(p, source_term.dtype, 4**l)[1]

        # assert len(S_arr_lst) == l - n_fused_levels
        # # assert len(DtN_arr_lst) == l - n_fused_levels
        # assert len(v_arr_lst) == l - n_fused_levels
        assert S_arr_lst[-1].shape[-1] == t.root_boundary_points.shape[0]
        assert v_arr_lst[-1].shape[-1] == n_src


class Test_down_pass_from_fused:
    @pytest.mark.skipif(
        not GPU_AVAILABLE, reason="Can not run fused methods on CPU-only systems."
    )
    def test_0(self, caplog) -> None:
        """Makes sure things run without error."""
        caplog.set_level(logging.INFO)
        p = 7
        q = 5
        l = 4

        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l)
        print("test_0: l = ", l)

        print("test_0: q = ", q)
        print("test_0: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])

        bdry_data = jnp.array(np.random.normal(size=t.root_boundary_points.shape[0]))

        sidelens = jnp.array([l.xmax - l.xmin for l in get_all_leaves(t.root)])

        S_arr_lst, DtN_arr_lst, v_arr_lst = _fused_local_solve_and_build_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            sidelens=sidelens,
            l=l,
            p=t.p,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )
        n_fused_levels = get_fused_chunksize_2D(p, source_term.dtype, 4**l)[1]

        # Print out the shapes of the arrays.
        print("test_0: S_arr_lst shape: ", [S.shape for S in S_arr_lst])
        print("test_0: v_arr_lst shape: ", [v.shape for v in v_arr_lst])
        print("test_0: DtN_arr_lst shape: ", [DtN.shape for DtN in DtN_arr_lst])

        soln = _down_pass_from_fused(
            S_arr_lst=S_arr_lst,
            v_int_lst=v_arr_lst,
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            sidelens=sidelens,
            l=l,
            p=t.p,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            bdry_data=bdry_data,
        )
        print("test_0: soln shape: ", soln.shape)
        print("test_0: expected shape: ", t.leaf_cheby_points[..., 0].shape)

        assert soln.shape == t.leaf_cheby_points[..., 0].shape


class Test_fused_all_single_chunk:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.INFO)

        p = 8
        q = 6
        l = 4

        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l)
        print("test_0: l = ", l)

        print("test_0: q = ", q)
        print("test_0: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        source_term = jnp.zeros_like(t.leaf_cheby_points[..., 0])

        bdry_data = jnp.array(np.random.normal(size=t.root_boundary_points.shape[0]))
        sidelens = jnp.array([l.xmax - l.xmin for l in get_all_leaves(t.root)])

        soln = _fused_all_single_chunk(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            l=l,
            sidelens=sidelens,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            bdry_data=bdry_data,
        )

        print("test_0: soln shape: ", soln.shape)
        assert soln.shape == t.leaf_cheby_points[..., 0].shape


class Test_fused_all_single_chunk_ItI:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        p = 8
        q = 6
        l = 4
        n_src = 3

        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]
        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, use_ItI=True, eta=4.0)
        print("test_0: l = ", l)

        print("test_0: q = ", q)
        print("test_0: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        a, b, _ = t.leaf_cheby_points.shape
        source_term = jnp.zeros((a, b, n_src), dtype=jnp.float64)

        bdry_data = jnp.array(
            np.random.normal(size=(t.root_boundary_points.shape[0], n_src))
        )
        print("test_0: bdry_data shape: ", bdry_data.shape)

        soln = _fused_all_single_chunk_ItI(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            I_P_0=t.I_P_0,
            Q_I=t.Q_I,
            F=t.F,
            G=t.G,
            l=l,
            p=p,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            bdry_data=bdry_data,
        )

        print("test_0: soln shape: ", soln.shape)
        assert soln.shape == (a, b, n_src)


class Test_down_pass_from_fused_ItI:
    def test_0(self) -> None:
        """Makes sure things run without error."""
        p = 7
        q = 5
        l = 4
        n_src = 3

        domain_bounds = [(0, 0), (1, 0), (1, 1), (0, 1)]

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        t = create_solver_obj_2D(p, q, root, uniform_levels=l, use_ItI=True, eta=4.0)
        print("test_0: l = ", l)

        print("test_0: q = ", q)
        print("test_0: Expected number of boundary points: ", 8 * q)

        # Do a local solve with a Laplacian operator.
        d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])
        d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[..., 0])

        a, b, _ = t.leaf_cheby_points.shape
        source_term = jnp.zeros((a, b, n_src), dtype=jnp.float64)

        bdry_data = jnp.array(
            np.random.normal(size=(t.root_boundary_points.shape[0], n_src))
        )

        S_arr_lst, DtN_arr_lst, v_arr_lst = _fused_local_solve_and_build_2D_ItI(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            I_P_0=t.I_P_0,
            Q_I=t.Q_I,
            F=t.F,
            G=t.G,
            l=l,
            p=t.p,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )
        n_fused_levels = get_fused_chunksize_2D(p, source_term.dtype, 4**l)[1]

        # Print out the shapes of the arrays.
        print("test_0: S_arr_lst shape: ", [S.shape for S in S_arr_lst])
        print("test_0: v_arr_lst shape: ", [v.shape for v in v_arr_lst])
        print("test_0: DtN_arr_lst shape: ", [DtN.shape for DtN in DtN_arr_lst])

        soln = _down_pass_from_fused_ItI(
            S_arr_lst=S_arr_lst,
            f_lst=v_arr_lst,
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            I_P_0=t.I_P_0,
            Q_I=t.Q_I,
            F=t.F,
            G=t.G,
            l=l,
            p=t.p,
            source_term=source_term,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            bdry_data=bdry_data,
        )
        print("test_0: soln shape: ", soln.shape)
        print("test_0: expected shape: ", t.leaf_cheby_points[..., 0].shape)

        assert soln.shape == (a, b, n_src)
