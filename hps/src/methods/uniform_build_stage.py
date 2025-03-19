from functools import partial
import logging
from typing import Tuple, List

import jax
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp

from hps.src.quadrature.quad_2D.uniform_merge_indexing import (
    get_quadmerge_blocks_a,
    get_quadmerge_blocks_b,
    get_quadmerge_blocks_c,
    get_quadmerge_blocks_d,
)

from hps.src.quadrature.quad_3D.uniform_merge_indexing import (
    get_a_submatrices,
    get_b_submatrices,
    get_c_submatrices,
    get_d_submatrices,
    get_e_submatrices,
    get_f_submatrices,
    get_g_submatrices,
    get_h_submatrices,
    get_rearrange_indices,
)
from hps.src.methods.schur_complement import (
    _oct_merge_from_submatrices,
    assemble_merge_outputs_DtN,
    assemble_merge_outputs_ItI,
)
from hps.src.config import DEVICE_ARR, HOST_DEVICE, GPU_AVAILABLE
from hps.src.quadrature.trees import Node, get_nodes_at_level, get_depth


def _uniform_build_stage_3D_DtN(
    root: Node,
    DtN_maps: jnp.ndarray,
    v_prime_arr: jnp.ndarray,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    logging.debug("_uniform_build_stage_3D_DtN: started")

    DtN_arr = DtN_maps
    S_lst = []
    DtN_lst = []
    v_int_lst = []

    logging.debug("_uniform_build_stage_3D_DtN: DtN_maps.device %s", DtN_maps.devices())

    DtN_arr = jax.device_put(DtN_arr, device)
    v_prime_arr = jax.device_put(v_prime_arr, device)
    logging.debug("_uniform_build_stage_3D_DtN: DtN_arr.device %s", DtN_arr.devices())

    q = int(jnp.sqrt(DtN_arr.shape[-1] // 6))
    # print("_build_stage_3D: DtN_arr shape: ", DtN_arr.shape)
    # print("_build_stage_3D: v_prime_arr shape: ", v_prime_arr.shape)

    depth = get_depth(root)

    for i in range(depth - 1, 0, -1):
        # print("_build_stage_3D: starting with depth ", i)
        # print("_build_stage_3D: DtN_arr.shape = ", DtN_arr.shape)

        S_arr, DtN_arr, v_prime_arr, v_int_arr = vmapped_uniform_oct_merge(
            jnp.arange(q), DtN_arr, v_prime_arr
        )
        S_host = jax.device_put(S_arr, host_device)
        v_int_host = jax.device_put(v_int_arr, host_device)
        DtN_host = jax.device_put(DtN_arr, host_device)
        v_prime_host = jax.device_put(v_prime_arr, host_device)

        S_lst.append(S_host)
        DtN_lst.append(DtN_host)
        v_int_lst.append(v_int_host)

        # Do the deletion if there is a GPU available. Otherwise, we need to keep
        # this data
        if GPU_AVAILABLE:
            S_arr.delete()
            v_int_arr.delete()

        # for j, node in enumerate(nodes_level_i):
        #     node.S = S_host[j]
        #     node.DtN = DtN_host[j]
        #     node.v_int = v_int_host[j]
        #     node.v_prime = v_prime_host[j]

    # print("_build_stage_3D: done with the loop. Doing final merge.")
    logging.debug("_build_stage_3D: DtN_arr.shape = %s", DtN_arr.shape)
    # Do the last oct-merge without the reshaping operation.
    S_last, T_last, v_prime_last, v_int_last = _uniform_oct_merge(
        jnp.arange(q),
        DtN_arr[0],
        DtN_arr[1],
        DtN_arr[2],
        DtN_arr[3],
        DtN_arr[4],
        DtN_arr[5],
        DtN_arr[6],
        DtN_arr[7],
        v_prime_arr[0],
        v_prime_arr[1],
        v_prime_arr[2],
        v_prime_arr[3],
        v_prime_arr[4],
        v_prime_arr[5],
        v_prime_arr[6],
        v_prime_arr[7],
    )
    D_shape = S_last.shape[0]
    S_lst.append(jax.device_put(S_last, host_device))
    DtN_lst.append(jax.device_put(T_last, host_device))
    v_int_lst.append(jax.device_put(v_int_last, host_device))

    # logging.debug("_build_stage: done with merging.")
    return S_lst, DtN_lst, v_int_lst, D_shape


def _uniform_build_stage_2D_DtN(
    DtN_maps: jnp.array,
    v_prime_arr: jnp.array,
    l: int,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
    return_fused_info: bool = False,
) -> Tuple[List[jnp.array], List[jnp.array], List[jnp.array]]:
    """
    Implements the build stage of the HPS algorithm for 2D problems. Given a list of
    leaf-level DtN matrices and a list of leaf-level particular solution boundary fluxes,
    this function recursively merges the DtN matrices and the particular solution boundary
    fluxes to produce the DtN matrices and the particular solution boundary fluxes at each
    level of the quadtree.

    Expects input data living on the CPU; will move data to the specified device and return
    data to the CPU.

    Args:
        DtN_maps (jnp.array): Has shape (n_quad_merges, 4, n_ext, n_ext)
        v_prime_arr (jnp.array): Has shape (n_quad_merges, 4, n_ext)
        l (int): Number of levels in the quadtree. 4**(l-1) = n_quad_merges.
        device (jax.Device): Where to perform computation.

    Returns:
        Tuple[List[jnp.array], List[jnp.array], List[jnp.array]]:
            S_lst: List of S matrices at each level of the quadtree. Each S matrix has shape (n_quad_merges, n_int, n_ext).
            DtN_lst: List of DtN matrices at each level of the quadtree. Each DtN matrix has shape (n_quad_merges, 4, n_ext, n_ext).
            v_int_lst: List of particular solution boundary fluxes at each level of the quadtree. Each v_int matrix has shape (n_quad_merges, 4, n_int).
    """
    logging.debug("_uniform_build_stage_2D_DtN: started")
    logging.debug("_uniform_build_stage_2D_DtN: DtN_maps shape: %s", DtN_maps.shape)
    logging.debug(
        "_uniform_build_stage_2D_DtN: input DtN_maps.device %s", DtN_maps.devices()
    )
    logging.debug("_uniform_build_stage_2D_DtN: host_device: %s", host_device)
    # print("_build_stage_2D: device: ", device)
    # print("_build_stage_2D: host_device: ", host_device)

    # Move the data to the compute device if necessary
    DtN_arr = jax.device_put(DtN_maps, device)
    v_prime_arr = jax.device_put(v_prime_arr, device)

    # Reshape the arrays into groups of 4 for merging if necessary
    if len(DtN_arr.shape) < 4:
        n_leaves, n_ext, _ = DtN_arr.shape
        DtN_arr = DtN_arr.reshape(n_leaves // 4, 4, n_ext, n_ext)
        v_prime_arr = v_prime_arr.reshape(n_leaves // 4, 4, n_ext)

    # Start lists to store S, DtN, and v_int arrays
    S_lst = []
    DtN_lst = []
    v_int_lst = []

    # Working on merging the merge pairs at level i
    for i in range(l - 1):
        logging.debug("_build_stage_2D: i: %i", i)

        (
            S_arr,
            DtN_arr_new,
            v_prime_arr_new,
            v_int_arr,
        ) = vmapped_uniform_quad_merge(DtN_arr, v_prime_arr)
        DtN_arr.delete()
        v_prime_arr.delete()
        # Only do these copies and GPU -> CPU moves if necessary
        if host_device != device:
            logging.debug("_build_stage_2D: Moving data to CPU")
            # print("_build_stage_2D: Moving data to CPU")
            # print("_build_stage_2D: S_arr.device: ", S_arr.devices())
            S_host = jax.device_put(S_arr, host_device)
            # print("_build_stage_2D: S_host.device: ", S_host.devices())
            S_lst.append(S_host)
            v_int_host = jax.device_put(v_int_arr, host_device)
            v_int_lst.append(v_int_host)
            # DtN_host = jax.device_put(DtN_arr_new, host_device)
            # DtN_lst.append(DtN_host)

            S_arr.delete()
            v_int_arr.delete()
            DtN_arr.delete()
        else:
            S_lst.append(S_arr)
            v_int_lst.append(v_int_arr)
            # DtN_lst.append(DtN_arr_new)

        DtN_arr = DtN_arr_new
        v_prime_arr = v_prime_arr_new

    if return_fused_info:
        T_last = DtN_arr
        v_prime_last = v_prime_arr
        logging.debug("_uniform_build_stage_2D_DtN: returning fused info")
        return (
            jax.device_put(T_last, host_device),
            jax.device_put(v_prime_last, host_device),
            S_lst,
            v_int_lst,
        )

    S_last, T_last, v_prime_last, v_int_last = _uniform_quad_merge(
        DtN_arr[0, 0],
        DtN_arr[0, 1],
        DtN_arr[0, 2],
        DtN_arr[0, 3],
        v_prime_arr[0, 0],
        v_prime_arr[0, 1],
        v_prime_arr[0, 2],
        v_prime_arr[0, 3],
    )
    # print("_build_stage: T_last shape: ", T_last.shape)
    # print("_build_stage: v_int_last shape: ", v_int_last.shape)

    # DtN_lst.append(jax.device_put(T_last, HOST_DEVICE))
    S_lst.append(jax.device_put(jnp.expand_dims(S_last, axis=0), host_device))
    v_int_lst.append(jax.device_put(jnp.expand_dims(v_int_last, axis=0), host_device))
    # DtN_lst.append(jax.device_put(jnp.expand_dims(T_last, axis=0), host_device))

    S_last.delete()
    T_last.delete()
    v_int_last.delete()
    v_prime_last.delete()

    # logging.debug("_build_stage: done with merging.")
    return S_lst, DtN_lst, v_int_lst


def _uniform_build_stage_2D_ItI(
    R_maps: jnp.ndarray,
    h_arr: jnp.ndarray,
    l: int,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
    return_fused_info: bool = False,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Computes l levels of uniform ItI merging.

    Args:
        R_maps (jnp.ndarray): ItI matrices. Has shape (n_leaves // 4, 4, 4q, 4q)
        h_arr (jnp.ndarray): Outgoing particular impedance data. Has shape (n_leaves // 4, 4, 4q, n_src)
        l (int): Number of levels to merge.
        device (jax.Device, optional): Device on which to run the computation. Defaults to DEVICE_ARR[0].
        host_device (jax.Device, optional): Device on which to return the outputs. Defaults to HOST_DEVICE.
        return_fused_info (bool, optional): If True, return values are different. Defaults to False.

    Returns:
        Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
          - S_lst (List[jnp.ndarray]): List of S matrices, which propagate info from boundary to merge interfaces.
          - R_lst (List[jnp.ndarray]): List of ItI matrices.
          - h_lst (List[jnp.ndarray]): List of outgoing particular impedance vectors.
    """
    logging.debug("_build_stage_2D_ItI: started")
    # logging.debug("_build_stage_2D_ItI: input R_maps.device %s", R_maps.devices())

    R_arr = R_maps

    # Start lists to store S, R, and f arrays
    S_lst = []
    R_lst = []
    f_lst = []

    R_arr = jax.device_put(R_arr, device)
    h_arr = jax.device_put(h_arr, device)

    # Working on merging the merge pairs at level i
    for i in range(l - 1):
        # print("up_pass: merge_EW = ", merge_EW)
        # logging.debug("_build_stage_2D_ItI: starting with level %i", i)
        # logging.debug("_build_stage_2D_ItI: R_arr.shape = %s", R_arr.shape)
        # logging.debug("_build_stage_2D_ItI: h_arr.shape = %s", h_arr.shape)

        S_arr, R_arr, h_arr, f_arr = vmapped_uniform_quad_merge_ItI(R_arr, h_arr)
        if host_device != device:
            logging.debug("_build_stage_2D_ItI: Moving data to host")
            S_host = jax.device_put(S_arr, host_device)
            S_lst.append(S_host)
            R_host = jax.device_put(R_arr, host_device)
            R_lst.append(R_host)
            f_host = jax.device_put(f_arr, host_device)
            f_lst.append(f_host)

            S_arr.delete()
            f_arr.delete()
        else:
            R_lst.append(R_arr)
            S_lst.append(S_arr)
            f_lst.append(f_arr)

    if return_fused_info:
        # Early exit and only return the final merge info
        R_last = R_arr
        h_last = h_arr
        return (
            jax.device_put(R_last, host_device),
            jax.device_put(h_last, host_device),
            S_lst,
            f_lst,
        )

    S_last, R_last, h_last, f_last = _uniform_quad_merge_ItI(
        R_arr[0, 0],
        R_arr[0, 1],
        R_arr[0, 2],
        R_arr[0, 3],
        h_arr[0, 0],
        h_arr[0, 1],
        h_arr[0, 2],
        h_arr[0, 3],
    )

    R_lst.append(jax.device_put(R_last, host_device))
    S_lst.append(jax.device_put(jnp.expand_dims(S_last, axis=0), host_device))
    f_lst.append(jax.device_put(jnp.expand_dims(f_last, axis=0), host_device))

    return S_lst, R_lst, f_lst


@jax.jit
def _uniform_quad_merge(
    T_a: jnp.ndarray,
    T_b: jnp.ndarray,
    T_c: jnp.ndarray,
    T_d: jnp.ndarray,
    v_prime_a: jnp.ndarray,
    v_prime_b: jnp.ndarray,
    v_prime_c: jnp.ndarray,
    v_prime_d: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    (
        v_prime_a_1,
        v_prime_a_5,
        v_prime_a_8,
        T_a_11,
        T_a_15,
        T_a_18,
        T_a_51,
        T_a_55,
        T_a_58,
        T_a_81,
        T_a_85,
        T_a_88,
    ) = get_quadmerge_blocks_a(T_a, v_prime_a)

    (
        v_prime_b_2,
        v_prime_b_6,
        v_prime_b_5,
        T_b_22,
        T_b_26,
        T_b_25,
        T_b_62,
        T_b_66,
        T_b_65,
        T_b_52,
        T_b_56,
        T_b_55,
    ) = get_quadmerge_blocks_b(T_b, v_prime_b)

    (
        v_prime_c_6,
        v_prime_c_3,
        v_prime_c_7,
        T_c_66,
        T_c_63,
        T_c_67,
        T_c_36,
        T_c_33,
        T_c_37,
        T_c_76,
        T_c_73,
        T_c_77,
    ) = get_quadmerge_blocks_c(T_c, v_prime_c)

    (
        v_prime_d_8,
        v_prime_d_7,
        v_prime_d_4,
        T_d_88,
        T_d_87,
        T_d_84,
        T_d_78,
        T_d_77,
        T_d_74,
        T_d_48,
        T_d_47,
        T_d_44,
    ) = get_quadmerge_blocks_d(T_d, v_prime_d)

    n_int, n_ext = T_a_51.shape

    zero_block_ee = jnp.zeros((n_ext, n_ext))
    zero_block_ei = jnp.zeros((n_ext, n_int))
    zero_block_ie = jnp.zeros((n_int, n_ext))
    zero_block_ii = jnp.zeros((n_int, n_int))

    # A = jnp.block(
    #     [
    #         [T_a_11, zero_block_ee, zero_block_ee, zero_block_ee],
    #         [zero_block_ee, T_b_22, zero_block_ee, zero_block_ee],
    #         [zero_block_ee, zero_block_ee, T_c_33, zero_block_ee],
    #         [zero_block_ee, zero_block_ee, zero_block_ee, T_d_44],
    #     ]
    # )

    B = jnp.block(
        [
            [T_a_15, zero_block_ei, zero_block_ei, T_a_18],
            [T_b_25, T_b_26, zero_block_ei, zero_block_ei],
            [zero_block_ei, T_c_36, T_c_37, zero_block_ei],
            [zero_block_ei, zero_block_ei, T_d_47, T_d_48],
        ]
    )
    C = jnp.block(
        [
            [T_a_51, T_b_52, zero_block_ie, zero_block_ie],
            [zero_block_ie, T_b_62, T_c_63, zero_block_ie],
            [zero_block_ie, zero_block_ie, T_c_73, T_d_74],
            [T_a_81, zero_block_ie, zero_block_ie, T_d_84],
        ]
    )

    D = jnp.block(
        [
            [T_a_55 + T_b_55, T_b_56, zero_block_ii, T_a_58],
            [T_b_65, T_b_66 + T_c_66, T_c_67, zero_block_ii],
            [zero_block_ii, T_c_76, T_c_77 + T_d_77, T_d_78],
            [T_a_85, zero_block_ii, T_d_87, T_d_88 + T_a_88],
        ]
    )
    # neg_D_inv = -1 * jnp.linalg.inv(D)
    # S = neg_D_inv @ C

    # T = B @ S
    # Add A to T block-wise
    A_lst = [T_a_11, T_b_22, T_c_33, T_d_44]
    # T = T.at[:n_ext, :n_ext].set(T[:n_ext, :n_ext] + T_a_11)
    # T = T.at[n_ext : 2 * n_ext, n_ext : 2 * n_ext].set(
    #     T[n_ext : 2 * n_ext, n_ext : 2 * n_ext] + T_b_22
    # )
    # T = T.at[2 * n_ext : 3 * n_ext, 2 * n_ext : 3 * n_ext].set(
    #     T[2 * n_ext : 3 * n_ext, 2 * n_ext : 3 * n_ext] + T_c_33
    # )
    # T = T.at[3 * n_ext :, 3 * n_ext :].set(T[3 * n_ext :, 3 * n_ext :] + T_d_44)

    delta_v_prime = jnp.concatenate(
        [
            v_prime_a_5 + v_prime_b_5,
            v_prime_b_6 + v_prime_c_6,
            v_prime_c_7 + v_prime_d_7,
            v_prime_d_8 + v_prime_a_8,
        ]
    )
    # v_int = neg_D_inv @ delta_v_prime

    v_prime_ext = jnp.concatenate([v_prime_a_1, v_prime_b_2, v_prime_c_3, v_prime_d_4])
    # v_prime_ext_out = v_prime_ext + B @ v_int

    T, S, v_prime_ext_out, v_int = assemble_merge_outputs_DtN(
        A_lst, B, C, D, v_prime_ext, delta_v_prime
    )

    # Roll the exterior by n_int to get the correct ordering
    v_prime_ext = jnp.roll(v_prime_ext_out, -n_int, axis=0)
    T = jnp.roll(T, -n_int, axis=0)
    T = jnp.roll(T, -n_int, axis=1)
    S = jnp.roll(S, -n_int, axis=1)

    return S, T, v_prime_ext, v_int


_vmapped_uniform_quad_merge = jax.vmap(
    _uniform_quad_merge, in_axes=(0, 0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0, 0)
)


@jax.jit
def _uniform_oct_merge(
    q_idxes: jnp.array,
    T_a: jnp.ndarray,
    T_b: jnp.ndarray,
    T_c: jnp.ndarray,
    T_d: jnp.ndarray,
    T_e: jnp.ndarray,
    T_f: jnp.ndarray,
    T_g: jnp.ndarray,
    T_h: jnp.ndarray,
    v_prime_a: jnp.ndarray,
    v_prime_b: jnp.ndarray,
    v_prime_c: jnp.ndarray,
    v_prime_d: jnp.ndarray,
    v_prime_e: jnp.ndarray,
    v_prime_f: jnp.ndarray,
    v_prime_g: jnp.ndarray,
    v_prime_h: jnp.ndarray,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:

    a_submatrices_subvecs = get_a_submatrices(T_a, v_prime_a)
    del T_a, v_prime_a

    b_submatrices_subvecs = get_b_submatrices(T_b, v_prime_b)
    del T_b, v_prime_b

    c_submatrices_subvecs = get_c_submatrices(T_c, v_prime_c)
    del T_c, v_prime_c

    d_submatrices_subvecs = get_d_submatrices(T_d, v_prime_d)
    del T_d, v_prime_d

    e_submatrices_subvecs = get_e_submatrices(T_e, v_prime_e)
    del T_e, v_prime_e

    f_submatrices_subvecs = get_f_submatrices(T_f, v_prime_f)
    del T_f, v_prime_f

    g_submatrices_subvecs = get_g_submatrices(T_g, v_prime_g)
    del T_g, v_prime_g

    h_submatrices_subvecs = get_h_submatrices(T_h, v_prime_h)
    del T_h, v_prime_h

    T, S, v_prime_ext_out, v_int = _oct_merge_from_submatrices(
        a_submatrices_subvecs=a_submatrices_subvecs,
        b_submatrices_subvecs=b_submatrices_subvecs,
        c_submatrices_subvecs=c_submatrices_subvecs,
        d_submatrices_subvecs=d_submatrices_subvecs,
        e_submatrices_subvecs=e_submatrices_subvecs,
        f_submatrices_subvecs=f_submatrices_subvecs,
        g_submatrices_subvecs=g_submatrices_subvecs,
        h_submatrices_subvecs=h_submatrices_subvecs,
    )

    r = get_rearrange_indices(jnp.arange(T.shape[0]), q_idxes)
    v_prime_ext_out = v_prime_ext_out[r]
    T = T[r][:, r]
    S = S[:, r]

    return S, T, v_prime_ext_out, v_int


_vmapped_uniform_oct_merge = jax.vmap(
    _uniform_oct_merge,
    in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def vmapped_uniform_quad_merge(
    T_in: jnp.ndarray,
    v_prime: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    S, T, v_prime_ext, v_int = _vmapped_uniform_quad_merge(
        T_in[:, 0],
        T_in[:, 1],
        T_in[:, 2],
        T_in[:, 3],
        v_prime[:, 0],
        v_prime[:, 1],
        v_prime[:, 2],
        v_prime[:, 3],
    )
    n_merges, n_int, n_ext = S.shape
    T_out = T.reshape((n_merges // 4, 4, n_ext, n_ext))
    v_prime_ext_out = v_prime_ext.reshape((n_merges // 4, 4, n_ext))

    return (S, T_out, v_prime_ext_out, v_int)


@jax.jit
def vmapped_uniform_oct_merge(
    q_idxes: jnp.array,
    T_in: jnp.ndarray,
    v_prime_in: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # print("vmapped_uniform_oct_merge: T_in shape: ", T_in.shape)
    # print("vmapped_uniform_oct_merge: v_prime_in shape: ", v_prime_in.shape)
    n_leaves, a, b = T_in.shape
    T_in = T_in.reshape((-1, 8, a, b))
    v_prime_in = v_prime_in.reshape((-1, 8, a))
    # print("vmapped_uniform_oct_merge: T_in shape: ", T_in.shape)
    # print("vmapped_uniform_oct_merge: v_prime_in shape: ", v_prime_in.shape)
    S, T_out, v_prime_ext_out, v_int = _vmapped_uniform_oct_merge(
        q_idxes,
        T_in[:, 0],
        T_in[:, 1],
        T_in[:, 2],
        T_in[:, 3],
        T_in[:, 4],
        T_in[:, 5],
        T_in[:, 6],
        T_in[:, 7],
        v_prime_in[:, 0],
        v_prime_in[:, 1],
        v_prime_in[:, 2],
        v_prime_in[:, 3],
        v_prime_in[:, 4],
        v_prime_in[:, 5],
        v_prime_in[:, 6],
        v_prime_in[:, 7],
    )

    return S, T_out, v_prime_ext_out, v_int


@jax.jit
def _uniform_quad_merge_ItI(
    R_a: jnp.array,
    R_b: jnp.array,
    R_c: jnp.array,
    R_d: jnp.array,
    h_a: jnp.array,
    h_b: jnp.array,
    h_c: jnp.array,
    h_d: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Performs an ItI quad merge. Nodes are arranged like this:

    ----------
    | d | c |
    | a | b |
    ----------

    Assume each child node has n_child discretization points
    around its boundary.

    Args:
        R_a (jnp.array): ItI matrix of node a. Has shape (n_child, n_child).
        R_b (jnp.array): ItI matrix of node b. Has shape (n_child, n_child).
        R_c (jnp.array): ItI matrix of node c. Has shape (n_child, n_child).
        R_d (jnp.array): ItI matrix of node d. Has shape (n_child, n_child).
        h_a (jnp.array): Outgoing particular impedance of node a. Has shape (n_child, n_src).
        h_b (jnp.array): Outgoing particular impedance of node b. Has shape (n_child, n_src).
        h_c (jnp.array): Outgoing particular impedance of node c. Has shape (n_child, n_src).
        h_d (jnp.array): Outgoing particular impedance of node d. Has shape (n_child, n_src).

    Returns:
        Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
         - S (jnp.array): Operator which propagates info from boundary to merge interfaces. Has shape (n_child, 2*n_child)
         - R (jnp.array): ItI matrix of merged node. Has shape (2*n_child, 2*n_child)
         - h (jnp.array): Outgoing part. impedance data of merged nodes. Has shape (2*n_child, n_src)
         - f (jnp.array): Incoming part. impedance data evaluated at merge interfaces. Has shape (n_child, n_src).
    """

    # First, find all of the necessary submatrices and sub-vectors
    (
        h_a_1,
        h_a_5,
        h_a_8,
        R_a_11,
        R_a_15,
        R_a_18,
        R_a_51,
        R_a_55,
        R_a_58,
        R_a_81,
        R_a_85,
        R_a_88,
    ) = get_quadmerge_blocks_a(R_a, h_a)

    (
        h_b_2,
        h_b_6,
        h_b_5,
        R_b_22,
        R_b_26,
        R_b_25,
        R_b_62,
        R_b_66,
        R_b_65,
        R_b_52,
        R_b_56,
        R_b_55,
    ) = get_quadmerge_blocks_b(R_b, h_b)

    (
        h_c_6,
        h_c_3,
        h_c_7,
        R_c_66,
        R_c_63,
        R_c_67,
        R_c_36,
        R_c_33,
        R_c_37,
        R_c_76,
        R_c_73,
        R_c_77,
    ) = get_quadmerge_blocks_c(R_c, h_c)

    (
        h_d_8,
        h_d_7,
        h_d_4,
        R_d_88,
        R_d_87,
        R_d_84,
        R_d_78,
        R_d_77,
        R_d_74,
        R_d_48,
        R_d_47,
        R_d_44,
    ) = get_quadmerge_blocks_d(R_d, h_d)

    n_int, n_ext = R_a_51.shape

    zero_block_ei = jnp.zeros((n_ext, n_int))
    zero_block_ie = jnp.zeros((n_int, n_ext))
    zero_block_ii = jnp.zeros((n_int, n_int))

    # print("_quad_merge_ItI: h_a_1", h_a_1.shape)

    # A t_ext + B t_int = g_ext - h_ext
    B = jnp.block(
        [
            [
                R_a_15,
                R_a_18,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
            ],
            [
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                R_b_25,
                R_b_26,
                zero_block_ei,
                zero_block_ei,
            ],
            [
                zero_block_ei,
                zero_block_ei,
                R_c_36,
                R_c_37,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
            ],
            [
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                zero_block_ei,
                R_d_47,
                R_d_48,
            ],
        ]
    )

    # C t_ext + D t_int + h_int = 0
    C = jnp.block(
        [
            [zero_block_ie, R_b_52, zero_block_ie, zero_block_ie],
            [zero_block_ie, zero_block_ie, zero_block_ie, R_d_84],
            [zero_block_ie, R_b_62, zero_block_ie, zero_block_ie],
            [zero_block_ie, zero_block_ie, zero_block_ie, R_d_74],
            [R_a_51, zero_block_ie, zero_block_ie, zero_block_ie],
            [zero_block_ie, zero_block_ie, R_c_63, zero_block_ie],
            [zero_block_ie, zero_block_ie, R_c_73, zero_block_ie],
            [R_a_81, zero_block_ie, zero_block_ie, zero_block_ie],
        ]
    )

    D_12 = jnp.block(
        [
            [R_b_55, R_b_56, zero_block_ii, zero_block_ii],
            [zero_block_ii, zero_block_ii, R_d_87, R_d_88],
            [R_b_65, R_b_66, zero_block_ii, zero_block_ii],
            [zero_block_ii, zero_block_ii, R_d_77, R_d_78],
        ]
    )

    D_21 = jnp.block(
        [
            [R_a_55, R_a_58, zero_block_ii, zero_block_ii],
            [zero_block_ii, zero_block_ii, R_c_66, R_c_67],
            [zero_block_ii, zero_block_ii, R_c_76, R_c_77],
            [R_a_85, R_a_88, zero_block_ii, zero_block_ii],
        ]
    )

    h_int = jnp.concatenate([h_b_5, h_d_8, h_b_6, h_d_7, h_a_5, h_c_6, h_c_7, h_a_8])
    h_ext = jnp.concatenate([h_a_1, h_b_2, h_c_3, h_d_4])
    A_lst = [R_a_11, R_b_22, R_c_33, R_d_44]

    T, S, h_ext_out, g_tilde_int = assemble_merge_outputs_ItI(
        A_lst, B, C, D_12, D_21, h_ext, h_int
    )

    # Roll the exterior by n_int to get the correct ordering
    h_ext_out = jnp.roll(h_ext_out, -n_int, axis=0)
    T = jnp.roll(T, -n_int, axis=0)
    T = jnp.roll(T, -n_int, axis=1)
    S = jnp.roll(S, -n_int, axis=1)

    # rows of S are ordered like a_5, a_8, c_6, c_7, b_5, b_6, d_7, d_8.
    # Want to rearrange them so they are ordered like
    # a_5, b_5, b_6, c_6, c_7, d_7, d_8, a_8
    r = jnp.concatenate(
        [
            jnp.arange(n_int),  # a5
            jnp.arange(4 * n_int, 5 * n_int),  # b5
            jnp.arange(5 * n_int, 6 * n_int),  # b6
            jnp.arange(2 * n_int, 3 * n_int),  # c6
            jnp.arange(3 * n_int, 4 * n_int),  # c7
            jnp.arange(6 * n_int, 7 * n_int),  # d7
            jnp.arange(7 * n_int, 8 * n_int),  # d8
            jnp.arange(n_int, 2 * n_int),  # a8
        ]
    )
    S = S[r]
    g_tilde_int = g_tilde_int[r]

    return S, T, h_ext_out, g_tilde_int


_vmapped_uniform_quad_merge_ItI = jax.vmap(
    _uniform_quad_merge_ItI,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def vmapped_uniform_quad_merge_ItI(
    R_in: jnp.array,
    h_in: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    n_src = h_in.shape[-1]
    S, R, h, f = _vmapped_uniform_quad_merge_ItI(
        R_in[:, 0],
        R_in[:, 1],
        R_in[:, 2],
        R_in[:, 3],
        h_in[:, 0],
        h_in[:, 1],
        h_in[:, 2],
        h_in[:, 3],
    )

    n_merges, n_int, n_ext = S.shape
    R = R.reshape((n_merges // 4, 4, n_ext, n_ext))
    h = h.reshape((n_merges // 4, 4, n_ext, n_src))
    return S, R, h, f
