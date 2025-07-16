import logging
from typing import Tuple, List

import jax
import jax.numpy as jnp


from ._schur_complement import (
    assemble_merge_outputs_DtN,
)


def merge_stage_uniform_2D_DtN(
    T_arr: jnp.array,
    h_arr: jnp.array,
    l: int,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
    subtree_recomp: bool = False,
    return_T: bool = False,
) -> Tuple[List[jnp.array], List[jnp.array], List[jnp.array]]:
    """
    Implements uniform 2D merges of DtN matrices. Merges the nodes in the quadtree four at a time.
    This function returns lists containing :math:`S` and :math:`\\tilde{g}`, giving enough information
    to propagate boundary data back down the tree in a later part of the algorithm.

    If this function is called with the argument ``return_T=True``, the top-level DtN matrix is also returned.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    T_arr : jax.Array
        Array of DtN matrices from the local solve stage. Has shape (n_leaves, 4q, 4q)

    h_arr : jax.Array
        Array of outgoing boundary data from the local solve stage. Has shape (n_leaves, 4q)

    l : int
        Number of levels to merge.

    device : jax.Device
        Where to perform the computation. Defaults to jax.devices()[0].

    host_device : jax.Device
        Where to place the output. Defaults to jax.devices("cpu")[0].

    subtree_recomp : bool
        A flag for used by the subtree recomputation methods, which triggers an early return with just the top-level T and h.

    return_T : bool
        A flag to return the top-level T matrix. Defaults to False.

    Returns
    -------
    S_lst : List[jax.Array]
        A list of propagation operators. The first element of the list are the propagation operators for the nodes just above the leaves, and the last element of the list is the propagation operator for the root of the quadtree.

    g_tilde_lst: List[jax.Array]
        A list of incoming particular solution data along the merge interfaces. The first element of the list corresponds to the nodes just above the leaves, and the last element of the list corresponds to the root of the quadtree.

    T_last : jax.Array
        The top-level DtN matrix, which is only returned if ``return_T=True``. Has shape (4q, 4q).

    """
    # Move the data to the compute device if necessary
    T_arr = jax.device_put(T_arr, device)
    h_arr = jax.device_put(h_arr, device)

    bool_multi_source = h_arr.ndim == 3

    logging.debug(
        "merge_stage_uniform_2D_DtN: T_arr shape: %s, h_arr shape: %s, l: %d",
        T_arr.shape,
        h_arr.shape,
        l,
    )

    # Reshape the arrays into groups of 4 for merging if necessary
    if len(T_arr.shape) < 4:
        n_leaves, n_ext, _ = T_arr.shape
        T_arr = T_arr.reshape(n_leaves // 4, 4, n_ext, n_ext)
        if bool_multi_source:
            n_src = h_arr.shape[-1]
            h_arr = h_arr.reshape(n_leaves // 4, 4, n_ext, n_src)
        else:
            h_arr = h_arr.reshape(n_leaves // 4, 4, n_ext, 1)

    if not subtree_recomp:
        # Start lists to store S and g_tilde arrays
        S_lst = []
        g_tilde_lst = []

    # Working on merging the merge pairs at level i
    for i in range(l - 1):
        (
            S_arr,
            T_arr_new,
            h_arr_new,
            g_tilde_arr,
        ) = vmapped_uniform_quad_merge_DtN(T_arr, h_arr)
        # T_arr.delete()
        # h_arr.delete()
        logging.debug(
            "merge_stage_uniform_2D_DtN: Merging level %d. S_arr shape: %s, T_arr_new shape: %s, h_arr_new shape: %s, g_tilde_arr shape: %s",
            i,
            S_arr.shape,
            T_arr_new.shape,
            h_arr_new.shape,
            g_tilde_arr.shape,
        )
        # Only do these copies and GPU -> CPU moves if necessary.
        # Necessary when we are not doing subtree recomp and we want
        # the data on the CPU.

        if not bool_multi_source and g_tilde_arr.ndim == 3:
            # Remove source dimension from g_tilde_arr
            g_tilde_arr = jnp.squeeze(g_tilde_arr, axis=-1)

        if host_device != device:
            if not subtree_recomp:
                logging.debug("_merge_stage_2D: Moving data to CPU")
                S_host = jax.device_put(S_arr, host_device)
                S_lst.append(S_host)

                g_tilde_host = jax.device_put(g_tilde_arr, host_device)
                g_tilde_lst.append(g_tilde_host)

            S_arr.delete()
            g_tilde_arr.delete()
        elif not subtree_recomp:
            S_lst.append(S_arr)
            g_tilde_lst.append(g_tilde_arr)

        T_arr = T_arr_new
        h_arr = h_arr_new

    logging.debug(
        "merge_stage_uniform_2D_DtN: Merging final nodes. h_arr shape: %s",
        h_arr.shape,
    )

    S_last, T_last, h_last, g_tilde_last = _uniform_quad_merge_DtN(
        T_arr[0, 0],
        T_arr[0, 1],
        T_arr[0, 2],
        T_arr[0, 3],
        h_arr[0, 0],
        h_arr[0, 1],
        h_arr[0, 2],
        h_arr[0, 3],
    )

    logging.debug(
        "merge_stage_uniform_2D_DtN: g_tilde_last shape: %s",
        g_tilde_last.shape,
    )
    logging.debug("merge_stage_uniform_2D_DtN: h_last shape: %s", h_last.shape)

    if not bool_multi_source:
        # Remove source dimension from g_tilde_last and h_last
        # Need to check whether the last dimension is 1
        if g_tilde_last.ndim > 1 and g_tilde_last.shape[-1] == 1:
            # Remove the last dimension
            # logging.debug(
            #     "merge_stage_uniform_2D_DtN: Removing last dimension from g_tilde_last"
            # )
            g_tilde_last = jnp.squeeze(g_tilde_last, axis=-1)
            h_last = jnp.squeeze(h_last, axis=-1)
        # g_tilde_last = jnp.squeeze(g_tilde_last, axis=-1)
        # h_last = jnp.squeeze(h_last, axis=-1)

    if subtree_recomp:
        # In this branch, we only return T_last and h_last
        S_last.delete()
        g_tilde_last.delete()

        # Expand the dimensions of T_last and h_last so they stack nicely
        T_last = jnp.expand_dims(T_last, 0)
        h_last = jnp.expand_dims(h_last, 0)

        # Move the data to the requested device
        T_last_out = jax.device_put(T_last, host_device)
        h_last_out = jax.device_put(h_last, host_device)

        return (T_last_out, h_last_out)

    else:
        # In this branch, we are returning S_lst, g_tilde_lst, and optionally
        # T_last
        S_lst.append(
            jax.device_put(jnp.expand_dims(S_last, axis=0), host_device)
        )
        g_tilde_lst.append(
            jax.device_put(jnp.expand_dims(g_tilde_last, axis=0), host_device)
        )

        if return_T:
            T_last_out = jax.device_put(T_last, host_device)
            return S_lst, g_tilde_lst, T_last_out
        else:
            return S_lst, g_tilde_lst


@jax.jit
def _uniform_quad_merge_DtN(
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

    zero_block_ei = jnp.zeros((n_ext, n_int))
    zero_block_ie = jnp.zeros((n_int, n_ext))
    zero_block_ii = jnp.zeros((n_int, n_int))

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

    v_prime_ext = jnp.concatenate(
        [v_prime_a_1, v_prime_b_2, v_prime_c_3, v_prime_d_4]
    )
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


_vmapped_uniform_quad_merge_DtN = jax.vmap(
    _uniform_quad_merge_DtN,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def vmapped_uniform_quad_merge_DtN(
    T_in: jnp.ndarray,
    h_in: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    S, T, h_out, g_tilde_out = _vmapped_uniform_quad_merge_DtN(
        T_in[:, 0],
        T_in[:, 1],
        T_in[:, 2],
        T_in[:, 3],
        h_in[:, 0],
        h_in[:, 1],
        h_in[:, 2],
        h_in[:, 3],
    )
    n_merges, n_int, n_ext = S.shape
    T_out = T.reshape((n_merges // 4, 4, n_ext, n_ext))
    if h_in.ndim == 2:
        h_out = h_out.reshape((n_merges // 4, 4, n_ext))
    else:
        nsrc = h_in.shape[-1]
        h_out = h_out.reshape((n_merges // 4, 4, n_ext, nsrc))

    return (S, T_out, h_out, g_tilde_out)


@jax.jit
def get_quadmerge_blocks_a(
    T: jnp.ndarray, v_prime: jnp.ndarray
) -> Tuple[jnp.ndarray]:
    n_per_side = T.shape[0] // 4
    idxes = jnp.arange(T.shape[0])

    idxes_1 = jnp.concatenate([idxes[-n_per_side:], idxes[:n_per_side]])
    idxes_5 = idxes[n_per_side : 2 * n_per_side]
    idxes_8 = jnp.flipud(idxes[2 * n_per_side : 3 * n_per_side])

    return _get_submatrices(T, v_prime, idxes_1, idxes_5, idxes_8)


@jax.jit
def get_quadmerge_blocks_b(
    T: jnp.ndarray, v_prime: jnp.ndarray
) -> Tuple[jnp.ndarray]:
    n_per_side = T.shape[0] // 4
    idxes = jnp.arange(T.shape[0])

    idxes_2 = idxes[: 2 * n_per_side]
    idxes_6 = idxes[2 * n_per_side : 3 * n_per_side]
    idxes_5 = jnp.flipud(idxes[3 * n_per_side :])

    return _get_submatrices(T, v_prime, idxes_2, idxes_6, idxes_5)


@jax.jit
def get_quadmerge_blocks_c(
    T: jnp.ndarray, v_prime: jnp.ndarray
) -> Tuple[jnp.ndarray]:
    n_per_side = T.shape[0] // 4
    idxes = jnp.arange(T.shape[0])

    idxes_6 = jnp.flipud(idxes[:n_per_side])
    idxes_3 = idxes[n_per_side : 3 * n_per_side]
    idxes_7 = idxes[3 * n_per_side :]

    return _get_submatrices(T, v_prime, idxes_6, idxes_3, idxes_7)


@jax.jit
def get_quadmerge_blocks_d(
    T: jnp.ndarray, v_prime: jnp.ndarray
) -> Tuple[jnp.ndarray]:
    n_per_side = T.shape[0] // 4
    idxes = jnp.arange(T.shape[0])

    idxes_8 = idxes[:n_per_side]
    idxes_7 = jnp.flipud(idxes[n_per_side : 2 * n_per_side])
    idxes_4 = idxes[2 * n_per_side :]

    return _get_submatrices(T, v_prime, idxes_8, idxes_7, idxes_4)


@jax.jit
def _get_submatrices(
    T: jnp.ndarray,
    v_prime: jnp.ndarray,
    idxes_0: jnp.ndarray,
    idxes_1: jnp.ndarray,
    idxes_2: jnp.ndarray,
) -> Tuple[jnp.ndarray]:
    v_prime_0 = v_prime[idxes_0]
    v_prime_1 = v_prime[idxes_1]
    v_prime_2 = v_prime[idxes_2]

    T_00 = T[idxes_0][:, idxes_0]
    T_01 = T[idxes_0][:, idxes_1]
    T_02 = T[idxes_0][:, idxes_2]
    T_10 = T[idxes_1][:, idxes_0]
    T_11 = T[idxes_1][:, idxes_1]
    T_12 = T[idxes_1][:, idxes_2]
    T_20 = T[idxes_2][:, idxes_0]
    T_21 = T[idxes_2][:, idxes_1]
    T_22 = T[idxes_2][:, idxes_2]

    return (
        v_prime_0,
        v_prime_1,
        v_prime_2,
        T_00,
        T_01,
        T_02,
        T_10,
        T_11,
        T_12,
        T_20,
        T_21,
        T_22,
    )
