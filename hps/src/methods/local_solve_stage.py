"""This file defines the functions for the local solve stage of the HPS algorithm."""

from functools import partial
import logging
from typing import List, Tuple

import numpy as np
import jax.numpy as jnp
import jax


from hps.src.quadrature.quad_2D.interpolation import (
    precompute_Q_D_matrix as precompute_Q_D_matrix_2D,
)
from hps.src.quadrature.quad_3D.interpolation import (
    precompute_Q_D_matrix as precompute_Q_D_matrix_3D,
)


from hps.src.config import (
    DEVICE_MESH,
    DEVICE_ARR,
    HOST_DEVICE,
    get_fused_chunksize_2D,
    get_chunksize_3D,
)


def _local_solve_stage_2D_chunked(
    D_xx: jnp.ndarray,
    D_xy: jnp.ndarray,
    D_yy: jnp.ndarray,
    D_x: jnp.ndarray,
    D_y: jnp.ndarray,
    P: jnp.ndarray,
    Q_D: jnp.ndarray,
    sidelens: jnp.array,
    p: int,
    source_term: jnp.ndarray,
    D_xx_coeffs: jnp.ndarray | None = None,
    D_xy_coeffs: jnp.ndarray | None = None,
    D_yy_coeffs: jnp.ndarray | None = None,
    D_x_coeffs: jnp.ndarray | None = None,
    D_y_coeffs: jnp.ndarray | None = None,
    I_coeffs: jnp.ndarray | None = None,
    uniform_grid: bool = False,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    See the docstring for _local_solve_stage_2D. This function takes the same inputs; just chunks over
    the n_leaves dimension.
    """
    n_leaves = source_term.shape[0]
    n_cheby_bdry = P.shape[0]
    p = (n_cheby_bdry // 4) + 1
    max_chunksizes = get_fused_chunksize_2D(
        p=p, dtype=source_term.dtype, n_leaves=n_leaves
    )
    logging.debug(
        "_local_solve_stage_2D_chunked: p = %s, max_chunksizes = %s, and n_leaves = %s",
        p,
        max_chunksizes,
        n_leaves,
    )

    chunk_start_idx = 0

    # Containers for the output data
    DtN_arr = []
    v_arr = []
    v_prime_arr = []
    Y_arr = []
    solns_arr = []

    while chunk_start_idx < n_leaves:

        # Loop over all available devices
        for device_idx, device in enumerate(DEVICE_ARR):
            logging.debug("_local_solve_stage_2D_chunked: device_idx = %s", device_idx)
            chunk_end_idx = min(chunk_start_idx + max_chunksizes[device_idx], n_leaves)
            logging.debug(
                "_local_solve_stage_2D_chunked: chunk_start_idx = %s, chunk_end_idx = %s",
                chunk_start_idx,
                chunk_end_idx,
            )

            # Index along the n_leaves dimension
            source_term_chunk = source_term[chunk_start_idx:chunk_end_idx]
            sidelens_chunk = sidelens[chunk_start_idx:chunk_end_idx]
            D_xx_coeffs_chunk = (
                D_xx_coeffs[chunk_start_idx:chunk_end_idx]
                if D_xx_coeffs is not None
                else None
            )
            D_xy_coeffs_chunk = (
                D_xy_coeffs[chunk_start_idx:chunk_end_idx]
                if D_xy_coeffs is not None
                else None
            )
            D_yy_coeffs_chunk = (
                D_yy_coeffs[chunk_start_idx:chunk_end_idx]
                if D_yy_coeffs is not None
                else None
            )
            D_x_coeffs_chunk = (
                D_x_coeffs[chunk_start_idx:chunk_end_idx]
                if D_x_coeffs is not None
                else None
            )
            D_y_coeffs_chunk = (
                D_y_coeffs[chunk_start_idx:chunk_end_idx]
                if D_y_coeffs is not None
                else None
            )
            I_coeffs_chunk = (
                I_coeffs[chunk_start_idx:chunk_end_idx]
                if I_coeffs is not None
                else None
            )

            # Call the local solve stage
            out = _local_solve_stage_2D(
                D_xx=D_xx,
                D_xy=D_xy,
                D_yy=D_yy,
                D_x=D_x,
                D_y=D_y,
                P=P,
                Q_D=Q_D,
                p=p,
                sidelens=sidelens,
                source_term=source_term_chunk,
                D_xx_coeffs=D_xx_coeffs_chunk,
                D_xy_coeffs=D_xy_coeffs_chunk,
                D_yy_coeffs=D_yy_coeffs_chunk,
                D_x_coeffs=D_x_coeffs_chunk,
                D_y_coeffs=D_y_coeffs_chunk,
                I_coeffs=I_coeffs_chunk,
                device=device,
                uniform_grid=uniform_grid,
            )

            # if bdry_data_chunk is not None:
            #     solns_chunk = out
            #     solns_arr.append(solns_chunk)
            # else:
            Y_arr_chunk, DtN_arr_chunk, v_chunk, v_prime_chunk = out
            DtN_arr.append(DtN_arr_chunk)
            v_arr.append(v_chunk)
            v_prime_arr.append(v_prime_chunk)
            Y_arr.append(Y_arr_chunk)

            chunk_start_idx = chunk_end_idx

            # Delete the chunks of data
            source_term_chunk.delete()
            if D_xx_coeffs_chunk is not None:
                D_xx_coeffs_chunk.delete()
            if D_xy_coeffs_chunk is not None:
                D_xy_coeffs_chunk.delete()
            if D_yy_coeffs_chunk is not None:
                D_yy_coeffs_chunk.delete()
            if D_x_coeffs_chunk is not None:
                D_x_coeffs_chunk.delete()
            if D_y_coeffs_chunk is not None:
                D_y_coeffs_chunk.delete()
            if I_coeffs_chunk is not None:
                I_coeffs_chunk.delete()

    # if bdry_data is not None:
    #     solns_arr = jnp.concatenate(solns_arr, axis=0)
    #     return solns_arr
    # else:
    DtN_arr_out = jnp.concatenate(DtN_arr, axis=0)
    for D in DtN_arr:
        D.delete()
    v_arr_out = jnp.concatenate(v_arr, axis=0)
    for v in v_arr:
        v.delete()
    v_prime_arr_out = jnp.concatenate(v_prime_arr, axis=0)
    for v_prime in v_prime_arr:
        v_prime.delete()
    Y_arr_out = jnp.concatenate(Y_arr, axis=0)
    for Y in Y_arr:
        Y.delete()

    return DtN_arr_out, v_arr_out, v_prime_arr_out, Y_arr_out


def _local_solve_stage_2D(
    D_xx: jnp.ndarray,
    D_xy: jnp.ndarray,
    D_yy: jnp.ndarray,
    D_x: jnp.ndarray,
    D_y: jnp.ndarray,
    P: jnp.ndarray,
    p: int,
    source_term: jnp.ndarray,
    sidelens: jnp.ndarray,
    D_xx_coeffs: jnp.ndarray | None = None,
    D_xy_coeffs: jnp.ndarray | None = None,
    D_yy_coeffs: jnp.ndarray | None = None,
    D_x_coeffs: jnp.ndarray | None = None,
    D_y_coeffs: jnp.ndarray | None = None,
    I_coeffs: jnp.ndarray | None = None,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
    uniform_grid: bool = False,
    Q_D: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """This function performs the local solve stage of the HPS algorithm.

    Args:
        D_xx (jnp.ndarray): Precomputed differential operator with shape (p**2, p**2).
        D_xy (jnp.ndarray): Precomputed differential operator with shape (p**2, p**2).
        D_yy (jnp.ndarray): Precomputed differential operator with shape (p**2, p**2).
        D_x (jnp.ndarray): Precomputed differential operator with shape (p**2, p**2).
        D_y (jnp.ndarray): Precomputed differential operator with shape (p**2, p**2).
        P (jnp.ndarray): Precomputed interpolation operator with shape (4(p-1), 4q).
            Maps data on the boundary Gauss nodes to data on the boundary Chebyshev nodes.
            Used when computing DtN maps.
        Q_D (jnp.ndarray): Precomputed interpolation + differentiation operator with shape (4q, p**2).
            Maps the solution on the Chebyshev nodes to the normal derivatives on the boundary Gauss nodes.
            Used when computing DtN maps.
        p (int): Shape parameter. Number of Chebyshev nodes along one dimension in a leaf.
        source_term (jnp.ndarray): Has shape (n_leaves, p**2). The right-hand side of the PDE.
        D_xx_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, p**2). Defaults to None, which means zero coeffs.
        D_xy_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, p**2). Defaults to None, which means zero coeffs.
        D_yy_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, p**2). Defaults to None, which means zero coeffs.
        D_x_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, p**2). Defaults to None, which means zero coeffs.
        D_y_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, p**2). Defaults to None, which means zero coeffs.
        I_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, p**2). Defaults to None, which means zero coeffs.

    Returns:
         Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
            Y_arr, DtN_arr, v, v_prime.
            Y_arr is an array of shape (n_leaves, p**2, 4q) containing the Dirichlet-to-Soln maps for each leaf.
            DtN_arr is an array of shape (n_quad_merges, 4, 4q, 4q) containing the DtN maps for each leaf.
            v is an array of shape (n_leaves, p**2) containing the particular solutions for each leaf.
            v_prime is an array of shape (n_quad_merges, 4, 4q) containing the boundary fluxes for each leaf
    """
    logging.debug("_local_solve_stage_2D: started")

    # Gather the coefficients into a single array.
    coeffs_gathered, which_coeffs = _gather_coeffs(
        D_xx_coeffs=D_xx_coeffs,
        D_xy_coeffs=D_xy_coeffs,
        D_yy_coeffs=D_yy_coeffs,
        D_x_coeffs=D_x_coeffs,
        D_y_coeffs=D_y_coeffs,
        I_coeffs=I_coeffs,
    )
    logging.debug(
        "_local_solve_stage_2D: input source_term devices = %s",
        source_term.devices(),
    )
    source_term = jax.device_put(
        source_term,
        device,
    )
    # stack the precomputed differential operators into a single array
    diff_ops = jnp.stack(
        [
            D_xx,
            D_xy,
            D_yy,
            D_x,
            D_y,
            jnp.eye(D_xx.shape[0], dtype=jnp.float64),
        ]
    )

    # Put the input data on the device
    coeffs_gathered = jax.device_put(
        coeffs_gathered,
        device,
    )
    if uniform_grid:
        # Don't have to use the sidelens argument
        diff_operators = vmapped_assemble_diff_operator(
            coeffs_gathered, which_coeffs, diff_ops
        )
        Y_arr, DtN_arr, v, v_prime = vmapped_get_DtN_uniform(
            source_term, diff_operators, Q_D, P
        )
    else:
        # Have to generate Q_D matrices for each leaf
        sidelens = jax.device_put(sidelens, device)
        n_cheby_bdry_pts = 4 * (p - 1)
        q = P.shape[1] // 4

        all_diff_operators, Q_Ds = vmapped_prep_nonuniform_refinement_diff_operators_2D(
            sidelens, coeffs_gathered, which_coeffs, diff_ops, p, q
        )
        Y_arr, DtN_arr, v, v_prime = vmapped_get_DtN(
            source_term, all_diff_operators, Q_Ds, P
        )

    # logging.debug(
    #     "_local_solve_stage_2D: source_term devices = %s", source_term.devices()
    # )
    # logging.debug("_local_solve_stage_2D: source_term shape = %s", source_term.shape)

    # logging.debug(
    #     "_local_solve_stage_2D: all_diff_operators shape = %s", all_diff_operators.shape
    # )

    # Return data to the CPU
    DtN_arr_host = jax.device_put(DtN_arr, host_device)
    del DtN_arr
    v_host = jax.device_put(v, host_device)
    del v
    v_prime_host = jax.device_put(v_prime, host_device)
    del v_prime
    Y_arr_host = jax.device_put(Y_arr, host_device)
    del Y_arr

    # Return the DtN arrays, particular solutions, particular
    # solution fluxes, and the solution operators. The solution
    # operators are not moved to the host.
    return Y_arr_host, DtN_arr_host, v_host, v_prime_host


def _local_solve_stage_2D_ItI(
    D_xx: jnp.ndarray,
    D_xy: jnp.ndarray,
    D_yy: jnp.ndarray,
    D_x: jnp.ndarray,
    D_y: jnp.ndarray,
    I_P_0: jnp.ndarray,
    Q_I: jnp.ndarray,
    F: jnp.ndarray,
    G: jnp.ndarray,
    p: int,
    source_term: jnp.ndarray,
    D_xx_coeffs: jnp.ndarray | None = None,
    D_xy_coeffs: jnp.ndarray | None = None,
    D_yy_coeffs: jnp.ndarray | None = None,
    D_x_coeffs: jnp.ndarray | None = None,
    D_y_coeffs: jnp.ndarray | None = None,
    I_coeffs: jnp.ndarray | None = None,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Performs the local solve stage for the 2D ItI version of the HPS method.

    This function is written to adapt to multiple source terms along a third dimension of the source_term array.
    If source_term has two dimensions, the number of source terms is assumed to be = 1.

    Args:
        D_xx (jnp.ndarray): Precomputed Chebyshev differentiation operator. Expected to be pre-scaled to
            match the sidelength of the leaves. Has shape (p^2, p^2)
        D_xy (jnp.ndarray): Precomputed Chebyshev differentiation operator. Expected to be pre-scaled to
            match the sidelength of the leaves. Has shape (p^2, p^2)
        D_yy (jnp.ndarray): Precomputed Chebyshev differentiation operator. Expected to be pre-scaled to
            match the sidelength of the leaves. Has shape (p^2, p^2)
        D_x (jnp.ndarray): Precomputed Chebyshev differentiation operator. Expected to be pre-scaled to
            match the sidelength of the leaves. Has shape (p^2, p^2)
        D_y (jnp.ndarray): Precomputed Chebyshev differentiation operator. Expected to be pre-scaled to
            match the sidelength of the leaves. Has shape (p^2, p^2)
        I_P_0 (jnp.ndarray): Precomputed interpolation operator from Gauss boundary points to Cheby boundary points.
            Has shape (4(p-1), 4q)
        Q_I (jnp.ndarray): Precomputed interpolation operator from Cheby boundary points to Gauss boundary points.
            Has shape (4q, 4p)
        F (jnp.ndarray): Precomputed interpolation/differentiation operator for constructing incoming impedance.
            Has shape (4p, p^2)
        G (jnp.ndarray): Precomputed interpolation/differentiation operator for constructing outgoing impedance.
            Has shape (4(p-1), p^2)
        p (int): Chebyshev parameter.
        source_term (jnp.ndarray): Has shape (n_leaves, p^2) OR (n_leaves, p^2, n_src).
        D_xx_coeffs (jnp.ndarray | None, optional): Diff operator coefficients. If specified, has shape (n_leaves, p^2). Defaults to None.
        D_xy_coeffs (jnp.ndarray | None, optional): Diff operator coefficients. If specified, has shape (n_leaves, p^2). Defaults to None.
        D_yy_coeffs (jnp.ndarray | None, optional): Diff operator coefficients. If specified, has shape (n_leaves, p^2). Defaults to None.
        D_x_coeffs (jnp.ndarray | None, optional): Diff operator coefficients. If specified, has shape (n_leaves, p^2). Defaults to None.
        D_y_coeffs (jnp.ndarray | None, optional): Diff operator coefficients. If specified, has shape (n_leaves, p^2). Defaults to None.
        I_coeffs (jnp.ndarray | None, optional): Diff operator coefficients. If specified, has shape (n_leaves, p^2). Defaults to None.
        device (jax.Device, optional): Device on which to perform the computation. Defaults to DEVICE_ARR[0].
        host_device (jax.Device, optional): Device on which the output data should be located. Defaults to HOST_DEVICE.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
          - R_arr_host: (jnp.ndarray): The ItI matrices. Has shape (n_leaves // 4, 4, 4q, 4q)
          - Y_arr_host: (jnp.ndarray): The interior solution matrices. Has shape (n_leaves, p^2, 4q)
          - outgoing_part_impedance_arr_host: (jnp.ndarray): The outgoing impedance data. Has shape (n_leaves // 4, 4, 4q, n_src)
          - part_soln_arr_host: (jnp.ndarray): The particular solution(s). Has shape (n_leaves, 4q, n_src)
    """
    logging.debug("_local_solve_stage_2D_ItI: started")

    # Gather the coefficients into a single array.
    coeffs_gathered, which_coeffs = _gather_coeffs(
        D_xx_coeffs=D_xx_coeffs,
        D_xy_coeffs=D_xy_coeffs,
        D_yy_coeffs=D_yy_coeffs,
        D_x_coeffs=D_x_coeffs,
        D_y_coeffs=D_y_coeffs,
        I_coeffs=I_coeffs,
    )

    # stack the precomputed differential operators into a single array

    diff_ops = jnp.stack(
        [
            D_xx,
            D_xy,
            D_yy,
            D_x,
            D_y,
            jnp.eye(D_xx.shape[0], dtype=jnp.float64),
        ]
    )

    coeffs_gathered = jax.device_put(
        coeffs_gathered,
        device,
    )

    all_diff_operators = vmapped_assemble_diff_operator(
        coeffs_gathered, which_coeffs, diff_ops
    )

    # Make sure source term has shape (n_leaves, p**2, n_src)
    if len(source_term.shape) == 2:
        source_term = jnp.expand_dims(source_term, axis=-1)
    logging.debug(
        "_local_solve_stage_2D_ItI: source_term shape = %s", source_term.shape
    )

    R_arr, Y_arr, outgoing_part_impedance_arr, part_soln_arr = (
        vmapped_get_ItI_then_rearrange(
            diff_operator=all_diff_operators,
            source_term=source_term,
            I_P_0=I_P_0,
            Q_I=Q_I,
            F=F,
            G=G,
        )
    )

    R_arr_host = jax.device_put(R_arr, host_device)
    Y_arr_host = jax.device_put(Y_arr, host_device)

    outgoing_part_impedance_arr_host = jax.device_put(
        outgoing_part_impedance_arr, host_device
    )
    part_soln_arr_host = jax.device_put(part_soln_arr, host_device)

    return R_arr_host, Y_arr_host, outgoing_part_impedance_arr_host, part_soln_arr_host


def _local_solve_stage_3D_chunked(
    D_xx: jnp.ndarray,
    D_xy: jnp.ndarray,
    D_yy: jnp.ndarray,
    D_xz: jnp.ndarray,
    D_yz: jnp.ndarray,
    D_zz: jnp.ndarray,
    D_x: jnp.ndarray,
    D_y: jnp.ndarray,
    D_z: jnp.ndarray,
    P: jnp.ndarray,
    p: int,
    q: int,
    source_term: jnp.ndarray,
    sidelens: jnp.ndarray,
    D_xx_coeffs: jnp.ndarray | None = None,
    D_xy_coeffs: jnp.ndarray | None = None,
    D_yy_coeffs: jnp.ndarray | None = None,
    D_xz_coeffs: jnp.ndarray | None = None,
    D_yz_coeffs: jnp.ndarray | None = None,
    D_zz_coeffs: jnp.ndarray | None = None,
    D_x_coeffs: jnp.ndarray | None = None,
    D_y_coeffs: jnp.ndarray | None = None,
    D_z_coeffs: jnp.ndarray | None = None,
    I_coeffs: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    See the docstring for _local_solve_stage_3D. This function takes the same inputs; just chunks over
    the n_leaves dimension.
    """
    n_leaves = source_term.shape[0]
    max_chunksize = get_chunksize_3D(p=p, n_leaves=source_term.shape[0])

    logging.debug(
        "_local_solve_stage_3D_chunked: p = %s, max_chunksizes = %s, and n_leaves = %s",
        p,
        max_chunksize,
        n_leaves,
    )
    chunk_start_idx = 0

    # Containers for the output data
    Y_arr = []
    DtN_arr = []
    v_arr = []
    v_prime_arr = []

    while chunk_start_idx < n_leaves:

        chunk_end_idx = min(chunk_start_idx + max_chunksize, n_leaves)
        logging.debug(
            "_local_solve_stage_3D_chunked: chunk_start_idx = %s, chunk_end_idx = %s",
            chunk_start_idx,
            chunk_end_idx,
        )

        # Index along the n_leaves dimension
        source_term_chunk = source_term[chunk_start_idx:chunk_end_idx]
        sidelens_chunk = sidelens[chunk_start_idx:chunk_end_idx]
        D_xx_coeffs_chunk = (
            D_xx_coeffs[chunk_start_idx:chunk_end_idx]
            if D_xx_coeffs is not None
            else None
        )
        D_xy_coeffs_chunk = (
            D_xy_coeffs[chunk_start_idx:chunk_end_idx]
            if D_xy_coeffs is not None
            else None
        )
        D_yy_coeffs_chunk = (
            D_yy_coeffs[chunk_start_idx:chunk_end_idx]
            if D_yy_coeffs is not None
            else None
        )
        D_xz_coeffs_chunk = (
            D_xz_coeffs[chunk_start_idx:chunk_end_idx]
            if D_xz_coeffs is not None
            else None
        )
        D_yz_coeffs_chunk = (
            D_yz_coeffs[chunk_start_idx:chunk_end_idx]
            if D_yz_coeffs is not None
            else None
        )
        D_zz_coeffs_chunk = (
            D_zz_coeffs[chunk_start_idx:chunk_end_idx]
            if D_zz_coeffs is not None
            else None
        )
        D_x_coeffs_chunk = (
            D_x_coeffs[chunk_start_idx:chunk_end_idx]
            if D_x_coeffs is not None
            else None
        )
        D_y_coeffs_chunk = (
            D_y_coeffs[chunk_start_idx:chunk_end_idx]
            if D_y_coeffs is not None
            else None
        )
        D_z_coeffs_chunk = (
            D_z_coeffs[chunk_start_idx:chunk_end_idx]
            if D_z_coeffs is not None
            else None
        )
        I_coeffs_chunk = (
            I_coeffs[chunk_start_idx:chunk_end_idx] if I_coeffs is not None else None
        )

        # Call the local solve stage
        Y_arr_chunk, DtN_arr_chunk, v_chunk, v_prime_chunk = _local_solve_stage_3D(
            D_xx=D_xx,
            D_xy=D_xy,
            D_yy=D_yy,
            D_xz=D_xz,
            D_yz=D_yz,
            D_zz=D_zz,
            D_x=D_x,
            D_y=D_y,
            D_z=D_z,
            P=P,
            p=p,
            q=q,
            sidelens=sidelens_chunk,
            source_term=source_term_chunk,
            D_xx_coeffs=D_xx_coeffs_chunk,
            D_xy_coeffs=D_xy_coeffs_chunk,
            D_yy_coeffs=D_yy_coeffs_chunk,
            D_xz_coeffs=D_xz_coeffs_chunk,
            D_yz_coeffs=D_yz_coeffs_chunk,
            D_zz_coeffs=D_zz_coeffs_chunk,
            D_x_coeffs=D_x_coeffs_chunk,
            D_y_coeffs=D_y_coeffs_chunk,
            D_z_coeffs=D_z_coeffs_chunk,
            I_coeffs=I_coeffs_chunk,
        )

        Y_arr.append(Y_arr_chunk)
        DtN_arr.append(DtN_arr_chunk)
        v_arr.append(v_chunk)
        v_prime_arr.append(v_prime_chunk)

        # Update the chunk start index
        chunk_start_idx = chunk_end_idx

    # Concatenate all of the chunks
    Y_arr = jnp.concatenate(Y_arr, axis=0)
    DtN_arr = jnp.concatenate(DtN_arr, axis=0)
    v_arr = jnp.concatenate(v_arr, axis=0)
    v_prime_arr = jnp.concatenate(v_prime_arr, axis=0)
    return Y_arr, DtN_arr, v_arr, v_prime_arr


def _local_solve_stage_3D(
    D_xx: jnp.ndarray,
    D_xy: jnp.ndarray,
    D_yy: jnp.ndarray,
    D_xz: jnp.ndarray,
    D_yz: jnp.ndarray,
    D_zz: jnp.ndarray,
    D_x: jnp.ndarray,
    D_y: jnp.ndarray,
    D_z: jnp.ndarray,
    P: jnp.ndarray,
    p: int,
    q: int,
    source_term: jnp.ndarray,
    sidelens: jnp.ndarray,
    D_xx_coeffs: jnp.ndarray | None = None,
    D_xy_coeffs: jnp.ndarray | None = None,
    D_yy_coeffs: jnp.ndarray | None = None,
    D_xz_coeffs: jnp.ndarray | None = None,
    D_yz_coeffs: jnp.ndarray | None = None,
    D_zz_coeffs: jnp.ndarray | None = None,
    D_x_coeffs: jnp.ndarray | None = None,
    D_y_coeffs: jnp.ndarray | None = None,
    D_z_coeffs: jnp.ndarray | None = None,
    I_coeffs: jnp.ndarray | None = None,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This function performs the local solve stage of the HPS algorithm. It produces the
    Dirichlet-to-Neumann (DtN) matrices for each leaf in the domain, as well as the
    solution operators (Y), the leaf particular solutions (v), and the outgoing particular
    solution fluxes (v_prime).

    It expects input data living on the CPU, will transfer it to the specified device, and
    will return the output data to the CPU.


    Shape args:
     - n_cheby_bdry = p**3 - (p-2)**3
     - n_cheby = p**3
     - n_gauss_bdry = 6 * q**2


    Args:
        D_xx (jnp.ndarray): Precomputed differential operator. Has shape (n_cheby, n_cheby).
        D_xy (jnp.ndarray): Precomputed differential operator. Has shape (n_cheby, n_cheby).
        D_yy (jnp.ndarray): Precomputed differential operator. Has shape (n_cheby, n_cheby).
        D_x (jnp.ndarray): Precomputed differential operator. Has shape (n_cheby, n_cheby).
        D_y (jnp.ndarray): Precomputed differential operator. Has shape (n_cheby, n_cheby).
        P (jnp.ndarray): Precomputed interpolation operator. Has shape (n_cheby_bdry, n_gauss_bdry).
        Q_D (jnp.ndarray): Precomputed diff + interp operator. Has shape (n_gauss_bdry, n_cheby).
        source_term (jnp.ndarray): Source term for the PDE. Has shape (n_leaves, n_cheby).
        D_xx_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, n_cheby). Defaults to None.
        D_xy_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, n_cheby). Defaults to None.
        D_yy_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, n_cheby). Defaults to None.
        D_x_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, n_cheby). Defaults to None.
        D_y_coeffs (jnp.ndarray | None, optional): Has shape (n_leaves, n_cheby). Defaults to None.
        device (jax.Device, optional): where to run the computation. Defaults to DEVICE_ARR[0].

    Returns:
         Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
            Y_arr, DtN_arr, v, v_prime.
            Y_arr is an array of shape (n_leaves, n_cheby, n_gauss_bdry) containing the Dirichlet-to-Soln maps for each leaf.
            DtN_arr is an array of shape (n_oct_merges, 8, n_gauss_bdry, n_gauss_bdry) containing the DtN maps for each leaf.
            v is an array of shape (n_leaves, n_cheby) containing the particular solutions for each leaf.
            v_prime is an array of shape (n_oct_merges, 8, n_gauss_bdry) containing the boundary fluxes for each leaf
    """
    logging.debug("_local_solve_stage_3D: started")
    # logging.debug("_local_solve_stage_3D: source_term shape: %s", source_term.shape)
    # logging.debug("_local_solve_stage: D_xx_coeffs shape: %s", D_xx_coeffs.shape)

    # Gather the coefficients into a single array.
    coeffs_gathered, which_coeffs = _gather_coeffs_3D(
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

    # stack the precomputed differential operators into a single array
    diff_ops = jnp.stack(
        [
            D_xx,
            D_xy,
            D_yy,
            D_xz,
            D_yz,
            D_zz,
            D_x,
            D_y,
            D_z,
            jnp.eye(p**3),
        ]
    )
    # Now that arrays are in contiguous blocks of memory, we can move them to the GPU.
    diff_ops = jax.device_put(diff_ops, device)
    coeffs_gathered = jax.device_put(coeffs_gathered, device)
    which_coeffs = jax.device_put(which_coeffs, device)
    source_term = jax.device_put(source_term, device)
    P = jax.device_put(P, device)
    n_cheby_bdry_pts = p**3 - (p - 2) ** 3

    # Prepare the differential operators for non-uniform refined grid
    diff_operators, Q_Ds = vmapped_prep_nonuniform_refinement_diff_operators_3D(
        sidelens,
        coeffs_gathered,
        which_coeffs,
        diff_ops,
        p,
        q,
    )
    logging.debug(
        "_local_solve_stage_3D: diff_operators shape: %s", diff_operators.shape
    )
    logging.debug("_local_solve_stage_3D: Q_Ds shape: %s", Q_Ds.shape)

    Y_arr, DtN_arr, v, v_prime = vmapped_get_DtN(source_term, diff_operators, Q_Ds, P)

    Y_arr = jax.device_put(Y_arr, HOST_DEVICE)
    DtN_arr = jax.device_put(DtN_arr, HOST_DEVICE)
    v = jax.device_put(v, HOST_DEVICE)
    v_prime = jax.device_put(v_prime, HOST_DEVICE)

    return DtN_arr, v, v_prime, Y_arr


@jax.jit
def _gather_coeffs(
    D_xx_coeffs: jnp.ndarray | None = None,
    D_xy_coeffs: jnp.ndarray | None = None,
    D_yy_coeffs: jnp.ndarray | None = None,
    D_x_coeffs: jnp.ndarray | None = None,
    D_y_coeffs: jnp.ndarray | None = None,
    I_coeffs: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """If not None, expects each input to have shape (n_leaf_nodes, p**2).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: coeffs_gathered and which_coeffs
            coeffs_gathered is an array of shape (?, n_leaf_nodes, p**2) containing the non-None coefficients.
            which_coeffs is an array of shape (6) containing boolean values specifying which coefficients were not None.
    """
    coeffs_lst = [
        D_xx_coeffs,
        D_xy_coeffs,
        D_yy_coeffs,
        D_x_coeffs,
        D_y_coeffs,
        I_coeffs,
    ]
    which_coeffs = jnp.array([coeff is not None for coeff in coeffs_lst])
    coeffs_gathered = jnp.array([coeff for coeff in coeffs_lst if coeff is not None])
    return coeffs_gathered, which_coeffs


@jax.jit
def _gather_coeffs_3D(
    D_xx_coeffs: jnp.ndarray | None = None,
    D_xy_coeffs: jnp.ndarray | None = None,
    D_yy_coeffs: jnp.ndarray | None = None,
    D_xz_coeffs: jnp.ndarray | None = None,
    D_yz_coeffs: jnp.ndarray | None = None,
    D_zz_coeffs: jnp.ndarray | None = None,
    D_x_coeffs: jnp.ndarray | None = None,
    D_y_coeffs: jnp.ndarray | None = None,
    D_z_coeffs: jnp.ndarray | None = None,
    I_coeffs: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """If not None, expects each input to have shape (n_leaf_nodes, p**2).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: coeffs_gathered and which_coeffs
            coeffs_gathered is an array of shape (?, n_leaf_nodes, p**3) containing the non-None coefficients.
            which_coeffs is an array of shape (10) containing boolean values specifying which coefficients were not None.
    """
    coeffs_lst = [
        D_xx_coeffs,
        D_xy_coeffs,
        D_yy_coeffs,
        D_xz_coeffs,
        D_yz_coeffs,
        D_zz_coeffs,
        D_x_coeffs,
        D_y_coeffs,
        D_z_coeffs,
        I_coeffs,
    ]
    which_coeffs = jnp.array([coeff is not None for coeff in coeffs_lst])
    coeffs_gathered = jnp.array([coeff for coeff in coeffs_lst if coeff is not None])
    return coeffs_gathered, which_coeffs


@jax.jit
def _add(
    out: jnp.ndarray,
    coeff: jnp.ndarray,
    diff_op: jnp.ndarray,
) -> jnp.ndarray:
    """One branch of add_or_not. Expects out to have shape (p**2, p**2), coeff has shape (p**2), diff_op has shape (p**2, p**2)."""
    # res = out + jnp.diag(coeff) @ diff_op
    res = out + jnp.einsum("ab,a->ab", diff_op, coeff)
    return res


@jax.jit
def _not(out: jnp.ndarray, coeff: jnp.ndarray, diff_op: jnp.ndarray) -> jnp.ndarray:
    return out


@jax.jit
def add_or_not(
    i: int,
    carry_tuple: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """body function for loop in assemble_diff_operator."""
    out = carry_tuple[0]
    coeffs_arr = carry_tuple[1]
    diff_ops = carry_tuple[2]
    which_coeffs = carry_tuple[3]
    counter = carry_tuple[4]

    out = jax.lax.cond(
        which_coeffs[i],
        _add,
        _not,
        out,
        coeffs_arr[counter],
        diff_ops[i],
    )
    counter = jax.lax.cond(
        which_coeffs[i],
        lambda x: x + 1,
        lambda x: x,
        counter,
    )
    return (out, coeffs_arr, diff_ops, which_coeffs, counter)


@jax.jit
def assemble_diff_operator(
    coeffs_arr: jnp.ndarray,
    which_coeffs: jnp.ndarray,
    diff_ops: jnp.ndarray,
) -> jnp.ndarray:
    """Given an array of coefficients, this function assembles the differential operator.

    Args:
        coeffs_arr (jnp.ndarray): Has shape (?, p**2).
        which_coeffs (jnp.ndarray): Has shape (5,) and specifies which coefficients are not None.
        diff_ops (jnp.ndarray): Has shape (6, p**2, p**2). Contains the precomputed differential operators.

    Returns:
        jnp.ndarray: Has shape (p**2, p**2).
    """

    n_loops = which_coeffs.shape[0]

    out = jnp.zeros_like(diff_ops[0])

    # Commenting this out because it is very memory intensive
    counter = 0
    init_val = (out, coeffs_arr, diff_ops, which_coeffs, counter)
    out, _, _, _, _ = jax.lax.fori_loop(0, n_loops, add_or_not, init_val)

    # Semantically the same as this:
    # counter = 0
    # for i in range(n_loops):
    #     if which_coeffs[i]:
    #         # out += jnp.diag(coeffs_arr[counter]) @ diff_ops[i]
    #         out += jnp.einsum("ab,a->ab", diff_ops[i], coeffs_arr[counter])
    #         counter += 1

    return out


vmapped_assemble_diff_operator = jax.vmap(
    assemble_diff_operator,
    in_axes=(1, None, None),
    out_axes=0,
)


@partial(jax.jit, static_argnums=(4, 5))
def _prep_nonuniform_refinement_diff_operators_2D(
    sidelen: jnp.array,
    coeffs_arr: jnp.array,
    which_coeffs: jnp.array,
    diff_ops_2D: jnp.array,
    p: int,
    q: int,
) -> Tuple[jnp.array, jnp.array]:
    """Prepares the differential operators for nonuniform refinement.

    Args:
        sidelen (jnp.array): Array of shape (n_leaves,) containing the sidelengths of each leaf.
        coeffs_arr (jnp.array): Array of shape (?, n_leaves, p**2) containing the PDE coefficients.
        which_coeffs (jnp.array): Array of shape (6,) containing boolean values specifying which coefficients are not None.
        diff_ops_2D (jnp.array): Array of shape (6, p**2, p**2) containing the precomputed differential operators.
        p (int): The number of Chebyshev nodes in each dimension.
        q (int): The number of Gauss nodes in each dimension.

    Returns:
        Tuple[jnp.array, jnp.array]: The precomputed differential operators for nonuniform refinement.
    """
    half_side_len = sidelen / 2
    # Second-order differential operators
    diff_ops_2D = diff_ops_2D.at[:3].set(diff_ops_2D[:3] / (half_side_len**2))
    # First-order differential operators
    diff_ops_2D = diff_ops_2D.at[3:5].set(diff_ops_2D[3:5] / half_side_len)
    diff_operator = assemble_diff_operator(coeffs_arr, which_coeffs, diff_ops_2D)
    Q_D = precompute_Q_D_matrix_2D(p, q, diff_ops_2D[3], diff_ops_2D[4])
    return diff_operator, Q_D


vmapped_prep_nonuniform_refinement_diff_operators_2D = jax.vmap(
    _prep_nonuniform_refinement_diff_operators_2D,
    in_axes=(0, 1, None, None, None, None),
    out_axes=(0, 0),
)


@partial(jax.jit, static_argnums=(4, 5))
def _prep_nonuniform_refinement_diff_operators_3D(
    sidelen: jnp.array,
    coeffs_arr: jnp.array,
    which_coeffs: jnp.array,
    diff_ops_3D: jnp.array,
    p: int,
    q: int,
) -> Tuple[jnp.array, jnp.array]:
    """
    Prepares the differential operators for nonuniform refinement.

    The differential operators are, in order:

    D_xx, D_xy, D_yy, D_xz, D_yz, D_zz, D_x, D_y, D_z, I

    Args:
        sidelen (jnp.array): Array of shape (n_leaves,) containing the sidelengths of each leaf.
        coeffs_arr (jnp.array): Array of shape (?, n_leaves, p**3) containing the PDE coefficients.
        which_coeffs (jnp.array): Array of shape (10,) containing boolean values specifying which coefficients are not None.
        diff_ops_3D (jnp.array): Array of shape (10, p**3, p**3) containing the precomputed differential operators.
        p (int): The number of Chebyshev nodes in each dimension.
        q (int): The number of Gauss nodes in each dimension.

    Returns:
        Tuple[jnp.array, jnp.array]: The precomputed differential operators for nonuniform refinement.
    """
    half_side_len = sidelen / 2
    # Second-order differential operators
    diff_ops_3D = diff_ops_3D.at[:6].set(diff_ops_3D[:6] / (half_side_len**2))
    # First-order differential operators
    diff_ops_3D = diff_ops_3D.at[6:9].set(diff_ops_3D[6:9] / half_side_len)
    diff_operator = assemble_diff_operator(coeffs_arr, which_coeffs, diff_ops_3D)
    Q_D = precompute_Q_D_matrix_3D(p, q, diff_ops_3D[6], diff_ops_3D[7], diff_ops_3D[8])
    return diff_operator, Q_D


vmapped_prep_nonuniform_refinement_diff_operators_3D = jax.vmap(
    _prep_nonuniform_refinement_diff_operators_3D,
    in_axes=(0, 1, None, None, None, None),
    out_axes=(0, 0),
)


@jax.jit
def get_DtN(
    source_term: jnp.ndarray,
    diff_operator: jnp.ndarray,
    Q_D: jnp.ndarray,
    P: jnp.ndarray,
) -> Tuple[jnp.ndarray]:
    """

    Args:
        source_term (jnp.ndarray): Array of size (p**2,) containing the source term.
        diff_operator (jnp.ndarray): Array of size (p**2, p**2) containing the local differential operator defined on the
                    Cheby grid.
        Q_D (jnp.ndarray): Array of size (4q, p**2) containing the matrix interpolating from a soln on the interior
                    to that soln's boundary fluxes on the Gauss boundary.
        P (jnp.ndarray): Array of size (4(p-1), 4q) containing the matrix interpolating from the Gauss to the Cheby boundary.
        n_cheby_bdry (int): The number of Chebyshev nodes on the boundary.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            Y (jnp.ndarray): Matrix of size (p**2, 4q). This is the "DtSoln" map,
                which maps incoming Dirichlet data on the boundary Gauss nodes to the solution on the Chebyshev nodes.
            map_DtN (jnp.ndarray): Matrix of size (4q, 4q). This is the "DtN" map, which maps incoming Dirichlet
                data on the boundary Gauss nodes to the normal derivatives on the boundary Gauss nodes.
    """
    n_cheby_bdry = P.shape[0]

    A_ii = diff_operator[n_cheby_bdry:, n_cheby_bdry:]
    A_ii_inv = jnp.linalg.inv(A_ii)
    # A_ie shape (n_cheby_int, n_cheby_bdry)
    A_ie = diff_operator[n_cheby_bdry:, :n_cheby_bdry]
    L_2 = jnp.zeros((diff_operator.shape[0], n_cheby_bdry), dtype=jnp.float64)
    L_2 = L_2.at[:n_cheby_bdry].set(jnp.eye(n_cheby_bdry))
    soln_operator = -1 * A_ii_inv @ A_ie
    L_2 = L_2.at[n_cheby_bdry:].set(soln_operator)
    Y = L_2 @ P
    map_DtN = Q_D @ Y

    particular_soln = jnp.zeros((diff_operator.shape[0],), dtype=jnp.float64)
    particular_soln = particular_soln.at[n_cheby_bdry:].set(
        A_ii_inv @ source_term[n_cheby_bdry:]
    )
    particular_fluxes = Q_D @ particular_soln

    return Y, map_DtN, particular_soln, particular_fluxes


vmapped_get_DtN = jax.vmap(
    get_DtN,
    in_axes=(0, 0, 0, None),
    out_axes=(0, 0, 0, 0),
)

# In this one, the Q_D is not mapped because we assume the input is on a
# uniform discretization so each leaf has the same size.
vmapped_get_DtN_uniform = jax.vmap(
    get_DtN,
    in_axes=(0, 0, None, None),
    out_axes=(0, 0, 0, 0),
)

# @partial(jax.jit, static_argnums=(6,))
# @jax.jit
# def vmapped_get_DtN_then_rearrange(
#     diff_operators: jnp.ndarray,
#     source_term: jnp.ndarray,
#     P: jnp.ndarray,
#     Q_D: jnp.ndarray,
# ) -> Tuple[jnp.ndarray]:
#     print("vmapped_get_DtN_then_rearrange: Called")
#     print(
#         "vmapped_get_DtN_then_rearrange: diff_operators shape: ", diff_operators.shape
#     )
#     print("vmapped_get_DtN_then_rearrange: source_term shape: ", source_term.shape)

#     Y_arr, DtN_arr, v, v_prime = vmapped_get_DtN(
#         source_term,
#         diff_operators,
#         Q_D,
#         P,
#     )
#     print("vmapped_get_DtN_then_rearrange: DtN_arr shape: ", DtN_arr.shape)
#     # Reshape the DtN and v_prime arrays
#     n_leaves, n_bdry, _ = DtN_arr.shape

#     n_quads = n_leaves // 4
#     DtN_arr = DtN_arr.reshape(n_quads, 4, n_bdry, n_bdry)
#     v_prime = v_prime.reshape(n_quads, 4, n_bdry)

#     return Y_arr, DtN_arr, v, v_prime


# @jax.jit
# def vmapped_get_DtN_then_rearrange_3D(
#     diff_operators: jnp.ndarray,
#     source_term: jnp.ndarray,
#     P: jnp.ndarray,
#     Q_D: jnp.ndarray,
# ) -> Tuple[jnp.ndarray]:
#     Y_arr, DtN_arr, v, v_prime = vmapped_get_DtN(source_term, diff_operators, Q_D, P)

#     # Reshape the DtN and v_prime arrays
#     n_leaves, n_bdry, _ = DtN_arr.shape

#     n_octs = n_leaves // 8
#     DtN_arr = DtN_arr.reshape(n_octs, 8, n_bdry, n_bdry)
#     v_prime = v_prime.reshape(n_octs, 8, n_bdry)

#     return Y_arr, DtN_arr, v, v_prime


@jax.jit
def get_ItI(
    diff_operator: jnp.array,
    source_term: jnp.array,
    I_P_0: jnp.array,
    Q_I: jnp.array,
    F: jnp.array,
    G: jnp.array,
) -> Tuple[jnp.array]:
    """Given the coefficients specifying a partial differential operator on a leaf, this function
    computes the particular solution, particular solution boundary fluxes, the
    impedance to impedance map, and the impedance to solution map.

    Args:
        coeffs_arr (jnp.array): Has shape (?, p**2). Specifies the PDE coefficients.
        source_term (jnp.array): Has shape (p**2, n_sources). Specifies the RHS of the PDE.
        diff_ops (jnp.array): Has shape (5, p**2, p**2). Contains the precomputed differential operators. In 3D,
                                this has shape (9, p**3, p**3).
        which_coeffs (jnp.array): Has shape (5,) and specifies which coefficients are not None.
        I_P_0 (jnp.array): Has shape (4(p-1), 4q). Maps data on the Gauss boundary nodes to data on the Cheby boundary nodes.
                            Is formed by taking the kronecker product of I and P_0, which is the standard
                            Gauss -> Cheby 1D interp matrix missing the last row.
        Q_I (jnp.array): Has shape (4q, 4p). Maps functions on the boundary Cheby nodes (counting corners twice) to functions on the
                    boundary Gauss nodes.
        G (jnp.array): Has shape (4p, p**2). Maps a function on the Chebyshev nodes to the function's outgoing impedance
                    on the boundary Cheby nodes, counting corners twice.
        F (jnp.array): Has shape (4(p-1), p**2). Maps a function on the Chebyshev nodes to the function's incoming
                    impedance on the boundary Cheby nodes, counting corners once.


    Returns:
        Tuple[jnp.array]:
            R (jnp.array): Has shape (4q, 4q). This is the "ItI" operator, which maps incoming impedance data on the
                boundary Gauss nodes to the outgoing impedance data on the boundary Gauss nodes.
            outgoing_part_impedance (jnp.array): Has shape (4q, n_sources). This is the outgoing impedance data on the
                boundary Gauss nodes, due to the particular solution(s).
            part_soln (jnp.array): Has shape (p**2, n_sources). This is the particular solution(s) on the Chebyshev nodes.
    """
    # print("get_ItI: I_P_0 shape: ", I_P_0.shape)
    # print("get_ItI: Q_I shape: ", Q_I.shape)
    n_cheby_pts = diff_operator.shape[-1]
    n_cheby_bdry_pts = I_P_0.shape[0]
    A = diff_operator

    # B has shape (n_cheby_pts, n_cheby_pts). Its top rows are F and its bottom rows are the
    # bottom rows of A.
    B = jnp.zeros((n_cheby_pts, n_cheby_pts), dtype=jnp.complex128)
    B = B.at[:n_cheby_bdry_pts].set(F)
    B = B.at[n_cheby_bdry_pts:].set(A[n_cheby_bdry_pts:])
    B_inv = jnp.linalg.inv(B)

    # Phi has shape (n_cheby_pts, n_cheby_interior_pts). It maps from the source
    # term evaluated on the interior Cheby nodes to the particular soln on all of
    # the Cheby nodes.
    Phi = B_inv[:, n_cheby_bdry_pts:]

    # Y has shape (n_cheby_pts, n_cheby_bdry_pts). It maps from
    # incoming impedance data on the boundary G-L nodes to the
    # homogeneous solution on all of the Cheby nodes.
    Y = B_inv[:, :n_cheby_bdry_pts] @ I_P_0

    source_int = source_term[n_cheby_bdry_pts:]
    part_soln = Phi @ source_int
    # part_soln = part_soln.at[:n_cheby_bdry_pts].set(0.0)
    outgoing_part_impedance = G @ part_soln

    # Interpolate to Gauss nodes
    G_out = Q_I @ G
    R = G_out @ Y
    outgoing_part_impedance = Q_I @ outgoing_part_impedance
    return (R, Y, outgoing_part_impedance, part_soln)


vmapped_get_ItI = jax.vmap(
    get_ItI,
    in_axes=(0, 0, None, None, None, None),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def vmapped_get_ItI_then_rearrange(
    diff_operator: jnp.array,
    source_term: jnp.array,
    I_P_0: jnp.array,
    Q_I: jnp.array,
    F: jnp.array,
    G: jnp.array,
) -> Tuple[jnp.array]:
    """
    See docstring for get_ItI for a more complete description of the inputs. Expects inputs
    coeffs_arr to have shape (?, n_leaves, p**2),
    source_term to have shape (n_leaves, p**2, n_sources)

    Calls a vmapped function over the n_leaves axis.

    Returns:
        Tuple[jnp.array]: R_arr, outgoing_part_impedance_arr, part_soln_arr
            R_arr is an array of shape (n_leaves // 4, 4, 4q, 4q) containing the ItI maps for each leaf.
            outgoing_part_impedance_arr is an array of shape (n_leaves // 4, 4, 4q, n_sources) containing the outgoing impedance data on the
                boundary Gauss nodes, due to the particular solution(s).
            part_soln_arr is an array of shape (n_leaves, p**2, n_sources) containing the particular solution(s) on the Chebyshev
    """

    R_arr, Y_arr, outgoing_part_impedance_arr, part_soln_arr = vmapped_get_ItI(
        diff_operator,
        source_term,
        I_P_0,
        Q_I,
        F,
        G,
    )

    n_leaves, n_bdry, _ = R_arr.shape
    n_src = outgoing_part_impedance_arr.shape[-1]

    n_quads = n_leaves // 4
    R_arr = R_arr.reshape(n_quads, 4, n_bdry, n_bdry)
    outgoing_part_impedance_arr = outgoing_part_impedance_arr.reshape(
        n_quads, 4, n_bdry, n_src
    )

    return R_arr, Y_arr, outgoing_part_impedance_arr, part_soln_arr
