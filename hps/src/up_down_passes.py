"""
Defines functions for the upward and downward passes of the HPS algorithm.
These functions are defined over a tree of Node objects.
"""

from functools import partial
import logging
from typing import List, Tuple

import jax.numpy as jnp
import jax

from hps.src.solver_obj import (
    SolverObj,
    create_solver_obj_2D,
    create_solver_obj_3D,
)

from hps.src.methods.local_solve_stage import (
    _local_solve_stage_2D,
    _local_solve_stage_2D_ItI,
    _local_solve_stage_2D_chunked,
    _local_solve_stage_3D,
    _local_solve_stage_3D_chunked,
)
from hps.src.methods.adaptive_build_stage import (
    _build_stage_2D as _adaptive_build_stage_2D,
    _build_stage_3D as _adaptive_build_stage_3D,
)
from hps.src.methods.uniform_build_stage import (
    _uniform_build_stage_3D_DtN,
    _uniform_build_stage_2D_DtN,
    _uniform_build_stage_2D_ItI,
)
from hps.src.methods.adaptive_down_pass import (
    _down_pass_2D as _adaptive_down_pass_2D,
    _down_pass_3D as _adaptive_down_pass_3D,
)

from hps.src.quadrature.trees import get_all_leaves_jitted, get_all_leaves

from hps.src.methods.uniform_down_pass import (
    _uniform_down_pass_2D_DtN,
    _uniform_down_pass_2D_ItI,
    _uniform_down_pass_3D_DtN,
)
from hps.src.methods.fused_methods import (
    _fused_local_solve_and_build_2D,
    _down_pass_from_fused,
    _fused_all_single_chunk,
    _fused_local_solve_and_build_2D_ItI,
    _down_pass_from_fused_ItI,
    _fused_all_single_chunk_ItI,
    _baseline_recomputation_upward_pass,
    _baseline_recomputation_downward_pass,
)
from hps.src.config import (
    DEVICE_ARR,
    get_fused_chunksize_2D,
    get_chunksize_3D,
    HOST_DEVICE,
)


def fused_pde_solve_2D(
    t: SolverObj,
    boundary_data: jax.Array,
    source_term: jax.Array,
    D_xx_coeffs: jax.Array | None = None,
    D_xy_coeffs: jax.Array | None = None,
    D_yy_coeffs: jax.Array | None = None,
    D_x_coeffs: jax.Array | None = None,
    D_y_coeffs: jax.Array | None = None,
    I_coeffs: jax.Array | None = None,
) -> None:
    """_summary_

    Args:
        t (SolverObj): _description_
        source_term (jax.Array): _description_
        D_xx_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_xy_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_yy_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_xz_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_yz_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_zz_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_x_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_y_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_z_coeffs (jax.Array | None, optional): _description_. Defaults to None.
    """
    chunksize, _ = get_fused_chunksize_2D(t.p, source_term.dtype, 4**t.l)
    if chunksize == 4**t.l:
        logging.debug("fused_pde_solve_2D: data is small enough for single chunk.")
        t.interior_solns = _fused_all_single_chunk(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            sidelens=t.sidelens,
            l=t.l,
            source_term=source_term,
            D_xx_coeffs=D_xx_coeffs,
            D_xy_coeffs=D_xy_coeffs,
            D_yy_coeffs=D_yy_coeffs,
            D_x_coeffs=D_x_coeffs,
            D_y_coeffs=D_y_coeffs,
            I_coeffs=I_coeffs,
            bdry_data=boundary_data,
        )

    else:
        S_arr_lst, DtN_arr_lst, v_arr_lst = _fused_local_solve_and_build_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            sidelens=t.sidelens,
            l=t.l,
            p=t.p,
            source_term=source_term,
            D_xx_coeffs=D_xx_coeffs,
            D_xy_coeffs=D_xy_coeffs,
            D_yy_coeffs=D_yy_coeffs,
            D_x_coeffs=D_x_coeffs,
            D_y_coeffs=D_y_coeffs,
            I_coeffs=I_coeffs,
        )

        # Downward pass
        soln = _down_pass_from_fused(
            bdry_data=boundary_data,
            S_arr_lst=S_arr_lst,
            v_int_lst=v_arr_lst,
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            sidelens=t.sidelens,
            l=t.l,
            p=t.p,
            source_term=source_term,
            D_xx_coeffs=D_xx_coeffs,
            D_xy_coeffs=D_xy_coeffs,
            D_yy_coeffs=D_yy_coeffs,
            D_x_coeffs=D_x_coeffs,
            D_y_coeffs=D_y_coeffs,
            I_coeffs=I_coeffs,
        )
        t.interior_solns = soln


def fused_pde_solve_2D_ItI(
    t: SolverObj,
    boundary_data: jax.Array,
    source_term: jax.Array,
    D_xx_coeffs: jax.Array | None = None,
    D_xy_coeffs: jax.Array | None = None,
    D_yy_coeffs: jax.Array | None = None,
    D_x_coeffs: jax.Array | None = None,
    D_y_coeffs: jax.Array | None = None,
    I_coeffs: jax.Array | None = None,
) -> None:
    """_summary_

    Args:
        t (Tree): _description_
        source_term (jax.Array): _description_
        D_xx_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_xy_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_yy_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_xz_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_yz_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_zz_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_x_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_y_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_z_coeffs (jax.Array | None, optional): _description_. Defaults to None.
    """
    chunksize, _ = get_fused_chunksize_2D(t.p, source_term.dtype, 4**t.l)
    if chunksize == 4**t.l:
        logging.debug("fused_pde_solve_2D: data is small enough for single chunk.")
        t.interior_solns = _fused_all_single_chunk_ItI(
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
            l=t.l,
            source_term=source_term,
            D_xx_coeffs=D_xx_coeffs,
            D_xy_coeffs=D_xy_coeffs,
            D_yy_coeffs=D_yy_coeffs,
            D_x_coeffs=D_x_coeffs,
            D_y_coeffs=D_y_coeffs,
            I_coeffs=I_coeffs,
            bdry_data=boundary_data,
        )

    else:
        S_arr_lst, ItI_arr_lst, f_lst = _fused_local_solve_and_build_2D_ItI(
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
            l=t.l,
            source_term=source_term,
            D_xx_coeffs=D_xx_coeffs,
            D_xy_coeffs=D_xy_coeffs,
            D_yy_coeffs=D_yy_coeffs,
            D_x_coeffs=D_x_coeffs,
            D_y_coeffs=D_y_coeffs,
            I_coeffs=I_coeffs,
        )

        # Downward pass
        soln = _down_pass_from_fused_ItI(
            bdry_data=boundary_data,
            S_arr_lst=S_arr_lst,
            f_lst=f_lst,
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
            l=t.l,
            source_term=source_term,
            D_xx_coeffs=D_xx_coeffs,
            D_xy_coeffs=D_xy_coeffs,
            D_yy_coeffs=D_yy_coeffs,
            D_x_coeffs=D_x_coeffs,
            D_y_coeffs=D_y_coeffs,
            I_coeffs=I_coeffs,
        )
        t.interior_solns = soln


def baseline_pde_solve_2D(
    t: SolverObj,
    boundary_data: jax.Array,
    source_term: jax.Array,
    D_xx_coeffs: jax.Array | None = None,
    D_xy_coeffs: jax.Array | None = None,
    D_yy_coeffs: jax.Array | None = None,
    D_x_coeffs: jax.Array | None = None,
    D_y_coeffs: jax.Array | None = None,
    I_coeffs: jax.Array | None = None,
) -> None:
    """_summary_

    Args:
        t (Tree): _description_
        source_term (jax.Array): _description_
        D_xx_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_xy_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_yy_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_xz_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_yz_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_zz_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_x_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_y_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_z_coeffs (jax.Array | None, optional): _description_. Defaults to None.
    """
    chunksize, _ = get_fused_chunksize_2D(t.p, source_term.dtype, 4**t.l)
    if chunksize == 4**t.l:
        logging.debug("baseline_pde_solve_2D: data is small enough for single chunk.")
        t.interior_solns = _fused_all_single_chunk(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            sidelens=t.sidelens,
            l=t.l,
            source_term=source_term,
            D_xx_coeffs=D_xx_coeffs,
            D_xy_coeffs=D_xy_coeffs,
            D_yy_coeffs=D_yy_coeffs,
            D_x_coeffs=D_x_coeffs,
            D_y_coeffs=D_y_coeffs,
            I_coeffs=I_coeffs,
            bdry_data=boundary_data,
        )

    else:
        # Do the upward pass
        S_arr_lst, v_int_lst = _baseline_recomputation_upward_pass(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            l=t.l,
            p=t.p,
            sidelens=t.sidelens,
            source_term=source_term,
            D_xx_coeffs=D_xx_coeffs,
            D_xy_coeffs=D_xy_coeffs,
            D_yy_coeffs=D_yy_coeffs,
            D_x_coeffs=D_x_coeffs,
            D_y_coeffs=D_y_coeffs,
            I_coeffs=I_coeffs,
        )

        # Do the downward pass, which includes re-computing the local solves.
        t.interior_solns = _baseline_recomputation_downward_pass(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            l=t.l,
            p=t.p,
            sidelens=t.sidelens,
            source_term=source_term,
            S_arr_lst=S_arr_lst,
            v_int_lst=v_int_lst,
            bdry_data=boundary_data,
            D_xx_coeffs=D_xx_coeffs,
            D_xy_coeffs=D_xy_coeffs,
            D_yy_coeffs=D_yy_coeffs,
            D_x_coeffs=D_x_coeffs,
            D_y_coeffs=D_y_coeffs,
            I_coeffs=I_coeffs,
        )


def local_solve_stage(
    t: SolverObj,
    source_term: jax.Array,
    D_xx_coeffs: jax.Array | None = None,
    D_xy_coeffs: jax.Array | None = None,
    D_yy_coeffs: jax.Array | None = None,
    D_xz_coeffs: jax.Array | None = None,
    D_yz_coeffs: jax.Array | None = None,
    D_zz_coeffs: jax.Array | None = None,
    D_x_coeffs: jax.Array | None = None,
    D_y_coeffs: jax.Array | None = None,
    D_z_coeffs: jax.Array | None = None,
    I_coeffs: jax.Array | None = None,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> None:
    """Wraps the _local_solve_stage method to fill non-optional arguments with
    the precomputed operators from the tree object. After the _local_solve_stage
    returns, sets the leaf_Y_maps and leaf_DtN_maps attributes of the tree object.

    Args:
        t (Tree): _description_
        D_xx_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_xy_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_yy_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_x_coeffs (jax.Array | None, optional): _description_. Defaults to None.
        D_y_coeffs (jax.Array | None, optional): _description_. Defaults to None.
    """
    bool_3D = t.D_z is not None
    bool_ItI = t.I_P_0 is not None
    n_leaves = source_term.shape[0]

    # Check to see whether we need to chunk the solve stage
    if bool_3D:
        max_chunksize = get_chunksize_3D(p=t.p, n_leaves=source_term.shape[0])
    else:
        max_chunksize = get_fused_chunksize_2D(
            p=t.p, dtype=source_term.dtype, n_leaves=source_term.shape[0]
        )[0]
    # True if we need to chunk the solve stage
    bool_chunked = n_leaves > max_chunksize

    if bool_3D:
        if bool_chunked:
            DtN_arr, v, v_prime, Y_arr = _local_solve_stage_3D_chunked(
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
                sidelens=t.sidelens,
                p=t.p,
                q=t.q,
                source_term=source_term,
                D_xx_coeffs=D_xx_coeffs,
                D_xy_coeffs=D_xy_coeffs,
                D_yy_coeffs=D_yy_coeffs,
                D_xz_coeffs=D_xz_coeffs,
                D_yz_coeffs=D_yz_coeffs,
                D_zz_coeffs=D_zz_coeffs,
                D_x_coeffs=D_x_coeffs,
                D_y_coeffs=D_y_coeffs,
                D_z_coeffs=D_z_coeffs,
                I_coeffs=I_coeffs,
            )
        else:
            DtN_arr, v, v_prime, Y_arr = _local_solve_stage_3D(
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
                sidelens=t.sidelens,
                p=t.p,
                q=t.q,
                source_term=source_term,
                D_xx_coeffs=D_xx_coeffs,
                D_xy_coeffs=D_xy_coeffs,
                D_yy_coeffs=D_yy_coeffs,
                D_xz_coeffs=D_xz_coeffs,
                D_yz_coeffs=D_yz_coeffs,
                D_zz_coeffs=D_zz_coeffs,
                D_x_coeffs=D_x_coeffs,
                D_y_coeffs=D_y_coeffs,
                D_z_coeffs=D_z_coeffs,
                I_coeffs=I_coeffs,
            )
    else:
        # 2D Case
        if bool_ItI:
            R_arr, Y_arr, h_arr, v = _local_solve_stage_2D_ItI(
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
                source_term=source_term,
                D_xx_coeffs=D_xx_coeffs,
                D_xy_coeffs=D_xy_coeffs,
                D_yy_coeffs=D_yy_coeffs,
                D_x_coeffs=D_x_coeffs,
                D_y_coeffs=D_y_coeffs,
                I_coeffs=I_coeffs,
                device=device,
                host_device=host_device,
            )
        else:
            # 2D DtN case
            if bool_chunked:
                DtN_arr, v, v_prime, Y_arr = _local_solve_stage_2D_chunked(
                    D_xx=t.D_xx,
                    D_xy=t.D_xy,
                    D_yy=t.D_yy,
                    D_x=t.D_x,
                    D_y=t.D_y,
                    P=t.P,
                    Q_D=t.Q_D,
                    sidelens=t.sidelens,
                    p=t.p,
                    source_term=source_term,
                    D_xx_coeffs=D_xx_coeffs,
                    D_xy_coeffs=D_xy_coeffs,
                    D_yy_coeffs=D_yy_coeffs,
                    D_x_coeffs=D_x_coeffs,
                    D_y_coeffs=D_y_coeffs,
                    I_coeffs=I_coeffs,
                    uniform_grid=t.uniform_grid,
                )
            else:
                Y_arr, DtN_arr, v, v_prime = _local_solve_stage_2D(
                    D_xx=t.D_xx,
                    D_xy=t.D_xy,
                    D_yy=t.D_yy,
                    D_x=t.D_x,
                    D_y=t.D_y,
                    P=t.P,
                    Q_D=t.Q_D,
                    sidelens=t.sidelens,
                    p=t.p,
                    source_term=source_term,
                    D_xx_coeffs=D_xx_coeffs,
                    D_xy_coeffs=D_xy_coeffs,
                    D_yy_coeffs=D_yy_coeffs,
                    D_x_coeffs=D_x_coeffs,
                    D_y_coeffs=D_y_coeffs,
                    I_coeffs=I_coeffs,
                    uniform_grid=t.uniform_grid,
                )
    # Set the attributes in the tree object.
    if bool_ItI:
        t.interior_node_R_maps.append(R_arr)
        t.leaf_node_h_vecs = h_arr
        t.leaf_Y_maps = Y_arr
        t.leaf_node_v_vecs = v
    elif t.uniform_grid:
        t.interior_node_DtN_maps.append(DtN_arr)
        t.leaf_node_v_prime_vecs = v_prime
        t.leaf_Y_maps = Y_arr
        t.leaf_node_v_vecs = v
    else:
        # Non-uniform case need to set the info for each leaf.
        # We also want the DtN and v_prime in array form to
        # do some of the lowest-level merges
        t.interior_node_DtN_maps.append(DtN_arr)
        t.leaf_node_v_prime_vecs = v_prime
        for i, l in enumerate(get_all_leaves(t.root)):
            l.DtN = DtN_arr[i]
            l.Y = Y_arr[i]
            l.v = v[i]
            l.v_prime = v_prime[i]


def build_stage(
    solver_obj: SolverObj,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
    return_D_size: bool = False,
) -> None:
    """
    This merges the HPS matrices from the leaves all the way up the
    discretization tree up to the root. This is done by calling the relavant
    _build_stage_* method based on the metadata present in the tree object.
    """
    if solver_obj.D_z is not None:
        # 3D case
        if solver_obj.uniform_grid:
            # Outputs of this function are saved in the solver_obj object.
            S_arr_lst, DtN_arr_lst, v_int_arr_lst, D_size = _uniform_build_stage_3D_DtN(
                root=solver_obj.root,
                DtN_maps=solver_obj.interior_node_DtN_maps[0],
                v_prime_arr=solver_obj.leaf_node_v_prime_vecs,
            )
            solver_obj.interior_node_S_maps = S_arr_lst
            solver_obj.interior_node_DtN_maps = DtN_arr_lst
            solver_obj.interior_node_v_int_vecs = v_int_arr_lst
        else:
            # Outputs of this function are saved inside the tree of Node() objects.
            D_size = _adaptive_build_stage_3D(
                root=solver_obj.root,
                refinement_op=solver_obj.refinement_op,
                coarsening_op=solver_obj.coarsening_op,
                DtN_arr=solver_obj.interior_node_DtN_maps[0],
                v_prime_arr=solver_obj.leaf_node_v_prime_vecs,
                q=solver_obj.q,
            )
        if return_D_size:
            return D_size
        else:
            return None

    else:
        if solver_obj.I_P_0 is not None:
            # 2D ItI case
            S_arr_lst, R_arr_lst, f_arr_lst = _uniform_build_stage_2D_ItI(
                solver_obj.interior_node_R_maps[0],
                solver_obj.leaf_node_h_vecs,
                solver_obj.l,
                device=device,
                host_device=host_device,
            )
            logging.debug("build_stage: R_arr_lst len: %s", len(R_arr_lst))
        else:
            if solver_obj.uniform_grid:
                # 2D DtN case
                S_arr_lst, DtN_arr_lst, v_arr_lst = _uniform_build_stage_2D_DtN(
                    solver_obj.interior_node_DtN_maps[0],
                    solver_obj.leaf_node_v_prime_vecs,
                    solver_obj.l,
                )
            else:
                _adaptive_build_stage_2D(
                    solver_obj.root, solver_obj.refinement_op, solver_obj.coarsening_op
                )

    # Set the appropriate attributes in the solver_obj object.
    if solver_obj.I_P_0 is not None:
        solver_obj.interior_node_S_maps = S_arr_lst
        solver_obj.interior_node_R_maps = R_arr_lst
        solver_obj.interior_node_f_vecs = f_arr_lst
    elif solver_obj.uniform_grid:
        solver_obj.interior_node_S_maps = S_arr_lst
        solver_obj.interior_node_DtN_maps = DtN_arr_lst
        solver_obj.interior_node_v_int_vecs = v_arr_lst


def down_pass(
    tree: SolverObj,
    boundary_data_lst: List[jax.Array],
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> None:

    if tree.D_z is not None:
        # 3D case
        if tree.uniform_grid:
            leaf_solns = _uniform_down_pass_3D_DtN(
                boundary_data=jnp.concatenate(boundary_data_lst),
                S_maps_lst=tree.interior_node_S_maps,
                v_int_lst=tree.interior_node_v_int_vecs,
                leaf_Y_maps=tree.leaf_Y_maps,
                v_array=tree.leaf_node_v_vecs,
            )
        else:
            leaf_solns = _adaptive_down_pass_3D(
                root=tree.root,
                boundary_data=boundary_data_lst,
                refinement_op=tree.refinement_op,
            )
    else:
        if tree.I_P_0 is not None:
            # 2D ItI case
            leaf_solns = _uniform_down_pass_2D_ItI(
                boundary_imp_data=jnp.concatenate(boundary_data_lst),
                S_maps_lst=tree.interior_node_S_maps,
                f_lst=tree.interior_node_f_vecs,
                leaf_Y_maps=tree.leaf_Y_maps,
                v_array=tree.leaf_node_v_vecs,
                device=device,
                host_device=host_device,
            )
        elif tree.uniform_grid:
            # 2D DtN Uniform case
            leaf_solns = _uniform_down_pass_2D_DtN(
                boundary_data=jnp.concatenate(boundary_data_lst),
                S_maps_lst=tree.interior_node_S_maps,
                v_int_lst=tree.interior_node_v_int_vecs,
                leaf_Y_maps=tree.leaf_Y_maps,
                v_array=tree.leaf_node_v_vecs,
            )
        else:
            leaf_solns = _adaptive_down_pass_2D(
                root=tree.root,
                boundary_data=boundary_data_lst,
                refinement_op=tree.refinement_op,
            )

    tree.interior_solns = leaf_solns
