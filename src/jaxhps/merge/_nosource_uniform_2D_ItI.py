from typing import Tuple, List

import jax
import jax.numpy as jnp


from ._schur_complement import (
    nosource_assemble_merge_outputs_ItI,
)
from ._uniform_2D_DtN import (
    get_quadmerge_blocks_a,
    get_quadmerge_blocks_b,
    get_quadmerge_blocks_c,
    get_quadmerge_blocks_d,
)
import logging


def nosource_merge_stage_uniform_2D_ItI(
    T_arr: jnp.array,
    l: int,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
    return_T: bool = False,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Implements uniform 2D merges of ItI matrices. Merges the nodes in the quadtree four at a time.
    This function uses a Schur complement strategy to reduce the size of matrix inverted in each merge operation.
    This function returns lists containing :math:`S` and :math:`\\tilde{g}`, giving enough information
    to propagate boundary data back down the tree in a later part of the algorithm.

    If this function is called with the argument ``return_T=True``, the top-level DtN matrix is also returned.

    Parameters
    ----------

    T_arr : jax.Array
        Array of ItI matrices from the local solve stage. Has shape (n_leaves, 4q, 4q)

    l : int
        Number of levels to merge

    device : jax.Device
        Where to perform the computation. Defaults to jax.devices()[0].

    host_device : jax.Device
        Where to place the output. Defaults to jax.devices("cpu")[0].

    return_T : bool
        A flag to return the top-level T matrix. Defaults to False.

    Returns
    -------
    S_lst : List[jax.Array]
        A list of propagation operators. The first element of the list are the propagation operators for the nodes just above the leaves, and the last element of the list is the propagation operator for the root of the quadtree.

    D_inv_lst : List[jax.Array]
        List of pre-computed D^{-1} matrices for each level of the quadtree.

    BD_inverse_lst : List[jax.Array]
        List of pre-computed BD^{-1} matrices for each level of the quadtree.

    T_last : jax.Array
        The top-level DtN matrix, which is only returned if ``return_T=True``. Has shape (4q, 4q).

    """

    # Start lists to output data
    S_lst = []
    D_inv_lst = []
    BD_inverse_lst = []

    T_arr = jax.device_put(T_arr, device)

    if len(T_arr.shape) < 4:
        logging.debug(
            "merge_stage_uniform_2D_ItI: T_arr.shape = %s", T_arr.shape
        )
        n_leaves, n_ext, _ = T_arr.shape
        T_arr = T_arr.reshape(n_leaves // 4, 4, n_ext, n_ext)

    for i in range(l - 1):
        S_arr, T_arr_new, D_inv_arr, BD_inv_arr = (
            vmapped_nosource_uniform_quad_merge_ItI(T_arr)
        )

        # TODO: Figure out how to safely delete these arrays
        # when using autodiff.

        # T_arr.delete()
        # h_arr.delete()

        if host_device != device:
            S_host = jax.device_put(S_arr, host_device)
            S_lst.append(S_host)
            # S_arr.delete()

            D_inv_host = jax.device_put(D_inv_arr, host_device)
            D_inv_lst.append(D_inv_host)
            # D_inv_arr.delete()

            BD_inv_host = jax.device_put(BD_inv_arr, host_device)
            BD_inverse_lst.append(BD_inv_host)
            # BD_inv_arr.delete()
        else:
            S_lst.append(S_arr)
            D_inv_lst.append(D_inv_arr)
            BD_inverse_lst.append(BD_inv_arr)

        T_arr = T_arr_new

    S_last, T_last, D_inv_last, BD_inv_last = _nosource_uniform_quad_merge_ItI(
        T_arr[0, 0],
        T_arr[0, 1],
        T_arr[0, 2],
        T_arr[0, 3],
    )

    S_lst.append(jax.device_put(jnp.expand_dims(S_last, axis=0), host_device))
    D_inv_lst.append(
        jax.device_put(jnp.expand_dims(D_inv_last, axis=0), host_device)
    )
    BD_inverse_lst.append(
        jax.device_put(jnp.expand_dims(BD_inv_last, axis=0), host_device)
    )

    if return_T:
        T_last_out = jax.device_put(T_last, host_device)
        return (S_lst, D_inv_lst, BD_inverse_lst, T_last_out)
    else:
        return (S_lst, D_inv_lst, BD_inverse_lst)


@jax.jit
def _nosource_uniform_quad_merge_ItI(
    R_a: jnp.array,
    R_b: jnp.array,
    R_c: jnp.array,
    R_d: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    n_a = R_a.shape[0]
    dummy_h = jnp.zeros((n_a,), dtype=R_a.dtype)
    # First, find all of the necessary submatrices and sub-vectors
    (
        _,
        _,
        _,
        R_a_11,
        R_a_15,
        R_a_18,
        R_a_51,
        R_a_55,
        R_a_58,
        R_a_81,
        R_a_85,
        R_a_88,
    ) = get_quadmerge_blocks_a(R_a, dummy_h)

    (
        _,
        _,
        _,
        R_b_22,
        R_b_26,
        R_b_25,
        R_b_62,
        R_b_66,
        R_b_65,
        R_b_52,
        R_b_56,
        R_b_55,
    ) = get_quadmerge_blocks_b(R_b, dummy_h)

    (
        _,
        _,
        _,
        R_c_66,
        R_c_63,
        R_c_67,
        R_c_36,
        R_c_33,
        R_c_37,
        R_c_76,
        R_c_73,
        R_c_77,
    ) = get_quadmerge_blocks_c(R_c, dummy_h)

    (
        _,
        _,
        _,
        R_d_88,
        R_d_87,
        R_d_84,
        R_d_78,
        R_d_77,
        R_d_74,
        R_d_48,
        R_d_47,
        R_d_44,
    ) = get_quadmerge_blocks_d(R_d, dummy_h)

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

    A_lst = [R_a_11, R_b_22, R_c_33, R_d_44]

    T, S, D_inv, BD_inv = nosource_assemble_merge_outputs_ItI(
        A_lst, B, C, D_12, D_21
    )

    # Roll the exterior by n_int to get the correct ordering
    # of the exterior discretization points. Right now, the exterior points are ordered like [a_1, b_2, c_3, d_4]
    # but we want [bottom, left, top, right]. This requires
    # rolling the exterior points by n_int.
    T = jnp.roll(T, -n_int, axis=0)
    T = jnp.roll(T, -n_int, axis=1)
    S = jnp.roll(S, -n_int, axis=1)

    # rows of S and D_inv are ordered like a_5, a_8, c_6, c_7, b_5, b_6, d_7, d_8.
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

    return S, T, D_inv, BD_inv


_vmapped_nosource_uniform_quad_merge_ItI = jax.vmap(
    _nosource_uniform_quad_merge_ItI,
    in_axes=(0, 0, 0, 0),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def vmapped_nosource_uniform_quad_merge_ItI(
    R_in: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    S, R, D_inv, BD_inv = _vmapped_nosource_uniform_quad_merge_ItI(
        R_in[:, 0],
        R_in[:, 1],
        R_in[:, 2],
        R_in[:, 3],
    )

    n_merges, n_int, n_ext = S.shape
    R = R.reshape((n_merges // 4, 4, n_ext, n_ext))
    return S, R, D_inv, BD_inv
