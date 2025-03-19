from typing import List, Tuple
import jax
import jax.numpy as jnp

from hps.src.config import DEVICE_ARR, HOST_DEVICE


@jax.jit
def assemble_merge_outputs_ItI(
    A_lst: List[jnp.array],
    B: jnp.array,
    C: jnp.array,
    D_12: jnp.array,
    D_21: jax.Array,
    h_ext: jnp.array,
    h_int: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Performs a merge for ItI matrices. In the ItI case, the D matrix is structured like:
            ------------
    D = I + | 0    D_12 |
            | D_21 0    |
            -------------
    So we invert it with a Schur complement method and then pass the data to _assemble_merge_outputs()
    Args:
        A_lst (List[jnp.array]): _description_
        B (jnp.array): _description_
        C (jnp.array): _description_
        D (jnp.array): _description_
        h_ext (jnp.array): _description_
        h_int (jnp.array): _description_

    Returns:
        Tuple[jnp.array, jnp.array, jnp.array, jnp.array]

        T (jnp.array): DtN matrix
        S (jnp.array): ext_to_int matrix
        h_ext_out (jnp.array): particular soln outgoing data on the boundary of the merged patches.
        g_tilde_int (jnp.array): particular soln incoming data on the merge interfaces.
    """

    D_inv = _invert_D_ItI(D_12, D_21)
    return _assemble_merge_outputs(A_lst, B, C, D_inv, h_ext, h_int)


@jax.jit
def _invert_D_ItI(D_12: jax.Array, D_21: jax.Array) -> jax.Array:
    """
    In the ItI case, D has this structure:
            ------------
    D = I + | 0    D_12 |
            | D_21 0    |
            -------------

    This function computes D^{-1} by forming a Schur complement W = (I - D_12 D_21). Then D^{-1} is

             --------------------------------------
    D^{-1} = | W^{-1}         -W^{-1} D_12          |
             | -D_21 W^{-1}   I + D_21 W^{-1} D_12  |
             --------------------------------------

    Args:
        D_12 (jax.Array): Upper-right block of D. Is square. Has shape (n, n)
        D_21 (jax.Array): Lower-left block of D. Is square. Has shape (m, m)

    Returns:
        jax.Array: Square matrix that is the inverse of D. Has shape (n+m, n+m)
    """
    n, _ = D_12.shape
    m, _ = D_21.shape
    W = jnp.eye(n) - D_12 @ D_21
    W_inv = jnp.linalg.inv(W)

    D_inv = jnp.zeros((n + m, n + m), dtype=D_12.dtype)
    D_inv = D_inv.at[:n, :n].set(W_inv)
    D_inv = D_inv.at[:n, n:].set(-1 * W_inv @ D_12)
    D_inv = D_inv.at[n:, :n].set(-1 * D_21 @ W_inv)
    D_inv = D_inv.at[n:, n:].set(jnp.eye(m) + D_21 @ W_inv @ D_12)
    return D_inv


@jax.jit
def assemble_merge_outputs_DtN(
    A_lst: List[jnp.array],
    B: jnp.array,
    C: jnp.array,
    D: jnp.array,
    h_ext: jnp.array,
    h_int: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Inverts D, which is dense in the DtN case, and then passes the data to _assemble_merge_outputs()

    Args:
        A_lst (List[jnp.array]): _description_
        B (jnp.array): _description_
        C (jnp.array): _description_
        D (jnp.array): _description_
        h_ext (jnp.array): _description_
        h_int (jnp.array): _description_

    Returns:
        Tuple[jnp.array, jnp.array, jnp.array, jnp.array]

        T (jnp.array): DtN matrix
        S (jnp.array): ext_to_int matrix
        h_ext_out (jnp.array): particular soln outgoing data on the boundary of the merged patches.
        g_tilde_int (jnp.array): particular soln incoming data on the merge interfaces.
    """

    D_inv = jnp.linalg.inv(D)
    return _assemble_merge_outputs(A_lst, B, C, D_inv, h_ext, h_int)


@jax.jit
def _assemble_merge_outputs(
    A_lst: List[jnp.array],
    B: jnp.array,
    C: jnp.array,
    D_inv: jnp.array,
    h_ext: jnp.array,
    h_int: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Computes a Schur complement that shows up in merging
    four patches together. Assumes D is already inverted.

    Given the system

    [ A B ][g_ext] = [u_ext - h_ext]
    [ C D ][g_int]   [-h_int]

    After two steps of block Gaussian elimination, we get

    [A-BD^{-1}C    0][g_ext] = [u_ext - h_ext + BD^{-1} h_int]
    [D^{-1}C       I][g_int]   [-D^{-1} h_int]

    Args:
        A_lst (List[jnp.array]): List of square diagonal blocks which make up A
        B (jnp.array):
        C (jnp.array): _description_
        D_inv (jnp.array): _description_
        h_ext (jnp.array): _description_
        h_int (jnp.array): _description_

    Returns:
        Tuple[jnp.array, jnp.array, jnp.array, jnp.array]

        T (jnp.array): DtN matrix
        S (jnp.array): Propagation matrix
        h_ext_out (jnp.array): particular soln outgoing data on the boundary of the merged patches.
        g_tilde_int (jnp.array): particular soln incoming data on the merge interfaces.
    """

    S = -1 * D_inv @ C
    T = B @ S
    # Need to add A to T block-wise
    counter = 0
    for A in A_lst:
        block_start = counter
        block_end = counter + A.shape[0]
        T = T.at[block_start:block_end, block_start:block_end].set(
            A + T[block_start:block_end, block_start:block_end]
        )
        counter = block_end

    g_tilde_int = -1 * D_inv @ h_int
    h_ext_out = h_ext + B @ g_tilde_int

    return (T, S, h_ext_out, g_tilde_int)


@jax.jit
def _oct_merge_from_submatrices(
    a_submatrices_subvecs: Tuple[jnp.array],
    b_submatrices_subvecs: Tuple[jnp.array],
    c_submatrices_subvecs: Tuple[jnp.array],
    d_submatrices_subvecs: Tuple[jnp.array],
    e_submatrices_subvecs: Tuple[jnp.array],
    f_submatrices_subvecs: Tuple[jnp.array],
    g_submatrices_subvecs: Tuple[jnp.array],
    h_submatrices_subvecs: Tuple[jnp.array],
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    (
        T_a_1_1,
        T_a_1_9,
        T_a_1_12,
        T_a_1_17,
        T_a_9_1,
        T_a_9_9,
        T_a_9_12,
        T_a_9_17,
        T_a_12_1,
        T_a_12_9,
        T_a_12_12,
        T_a_12_17,
        T_a_17_1,
        T_a_17_9,
        T_a_17_12,
        T_a_17_17,
        v_prime_a_1,
        v_prime_a_9,
        v_prime_a_12,
        v_prime_a_17,
    ) = a_submatrices_subvecs

    (
        T_b_2_2,
        T_b_2_9,
        T_b_2_10,
        T_b_2_18,
        T_b_9_2,
        T_b_9_9,
        T_b_9_10,
        T_b_9_18,
        T_b_10_2,
        T_b_10_9,
        T_b_10_10,
        T_b_10_18,
        T_b_18_2,
        T_b_18_9,
        T_b_18_10,
        T_b_18_18,
        v_prime_b_2,
        v_prime_b_9,
        v_prime_b_10,
        v_prime_b_18,
    ) = b_submatrices_subvecs

    (
        T_c_3_3,
        T_c_3_10,
        T_c_3_11,
        T_c_3_19,
        T_c_10_3,
        T_c_10_10,
        T_c_10_11,
        T_c_10_19,
        T_c_11_3,
        T_c_11_10,
        T_c_11_11,
        T_c_11_19,
        T_c_19_3,
        T_c_19_10,
        T_c_19_11,
        T_c_19_19,
        v_prime_c_3,
        v_prime_c_10,
        v_prime_c_11,
        v_prime_c_19,
    ) = c_submatrices_subvecs

    (
        T_d_4_4,
        T_d_4_11,
        T_d_4_12,
        T_d_4_20,
        T_d_11_4,
        T_d_11_11,
        T_d_11_12,
        T_d_11_20,
        T_d_12_4,
        T_d_12_11,
        T_d_12_12,
        T_d_12_20,
        T_d_20_4,
        T_d_20_11,
        T_d_20_12,
        T_d_20_20,
        v_prime_d_4,
        v_prime_d_11,
        v_prime_d_12,
        v_prime_d_20,
    ) = d_submatrices_subvecs

    (
        T_e_5_5,
        T_e_5_13,
        T_e_5_16,
        T_e_5_17,
        T_e_13_5,
        T_e_13_13,
        T_e_13_16,
        T_e_13_17,
        T_e_16_5,
        T_e_16_13,
        T_e_16_16,
        T_e_16_17,
        T_e_17_5,
        T_e_17_13,
        T_e_17_16,
        T_e_17_17,
        v_prime_e_5,
        v_prime_e_13,
        v_prime_e_16,
        v_prime_e_17,
    ) = e_submatrices_subvecs

    (
        T_f_6_6,
        T_f_6_13,
        T_f_6_14,
        T_f_6_18,
        T_f_13_6,
        T_f_13_13,
        T_f_13_14,
        T_f_13_18,
        T_f_14_6,
        T_f_14_13,
        T_f_14_14,
        T_f_14_18,
        T_f_18_6,
        T_f_18_13,
        T_f_18_14,
        T_f_18_18,
        v_prime_f_6,
        v_prime_f_13,
        v_prime_f_14,
        v_prime_f_18,
    ) = f_submatrices_subvecs

    (
        T_g_7_7,
        T_g_7_14,
        T_g_7_15,
        T_g_7_19,
        T_g_14_7,
        T_g_14_14,
        T_g_14_15,
        T_g_14_19,
        T_g_15_7,
        T_g_15_14,
        T_g_15_15,
        T_g_15_19,
        T_g_19_7,
        T_g_19_14,
        T_g_19_15,
        T_g_19_19,
        v_prime_g_7,
        v_prime_g_14,
        v_prime_g_15,
        v_prime_g_19,
    ) = g_submatrices_subvecs

    (
        T_h_8_8,
        T_h_8_15,
        T_h_8_16,
        T_h_8_20,
        T_h_15_8,
        T_h_15_15,
        T_h_15_16,
        T_h_15_20,
        T_h_16_8,
        T_h_16_15,
        T_h_16_16,
        T_h_16_20,
        T_h_20_8,
        T_h_20_15,
        T_h_20_16,
        T_h_20_20,
        v_prime_h_8,
        v_prime_h_15,
        v_prime_h_16,
        v_prime_h_20,
    ) = h_submatrices_subvecs

    n_1, n_9 = T_a_1_9.shape
    n_12, n_17 = T_a_12_17.shape
    n_2, n_10 = T_b_2_10.shape
    n_18 = T_b_18_18.shape[1]
    n_3, n_11 = T_c_3_11.shape
    n_19 = T_c_19_19.shape[1]
    n_4, n_12 = T_d_4_12.shape
    n_20 = T_d_20_20.shape[1]
    n_5, n_13 = T_e_5_13.shape
    n_16, n_17 = T_e_16_17.shape
    n_6, n_14 = T_f_6_14.shape
    n_7, n_15 = T_g_7_15.shape
    n_8, n_16 = T_h_8_16.shape

    # Set up indexes for the blocks; we don't know the shape of the blocks a priori
    n_ext_pts = n_1 + n_2 + n_3 + n_4 + n_5 + n_6 + n_7 + n_8
    n_int_pts = (
        n_9 + n_10 + n_11 + n_12 + n_13 + n_14 + n_15 + n_16 + n_17 + n_18 + n_19 + n_20
    )
    idx_1 = n_1
    idx_2 = n_1 + n_2
    idx_3 = n_1 + n_2 + n_3
    idx_4 = n_1 + n_2 + n_3 + n_4
    idx_5 = n_1 + n_2 + n_3 + n_4 + n_5
    idx_6 = n_1 + n_2 + n_3 + n_4 + n_5 + n_6
    idx_7 = n_1 + n_2 + n_3 + n_4 + n_5 + n_6 + n_7
    idx_8 = n_1 + n_2 + n_3 + n_4 + n_5 + n_6 + n_7 + n_8

    idx_9 = n_9
    idx_10 = n_9 + n_10
    idx_11 = n_9 + n_10 + n_11
    idx_12 = n_9 + n_10 + n_11 + n_12
    idx_13 = n_9 + n_10 + n_11 + n_12 + n_13
    idx_14 = n_9 + n_10 + n_11 + n_12 + n_13 + n_14
    idx_15 = n_9 + n_10 + n_11 + n_12 + n_13 + n_14 + n_15
    idx_16 = n_9 + n_10 + n_11 + n_12 + n_13 + n_14 + n_15 + n_16
    idx_17 = n_9 + n_10 + n_11 + n_12 + n_13 + n_14 + n_15 + n_16 + n_17
    idx_18 = n_9 + n_10 + n_11 + n_12 + n_13 + n_14 + n_15 + n_16 + n_17 + n_18
    idx_19 = n_9 + n_10 + n_11 + n_12 + n_13 + n_14 + n_15 + n_16 + n_17 + n_18 + n_19
    idx_20 = (
        n_9 + n_10 + n_11 + n_12 + n_13 + n_14 + n_15 + n_16 + n_17 + n_18 + n_19 + n_20
    )

    # B is a block matrix with an array of (8x12) blocks.

    B = jnp.zeros((n_ext_pts, n_int_pts), dtype=jnp.float64)

    # First row
    # print("_oct_merge_from_submatrice: B.devices()", B.devices())
    # print("_oct_merge_from_submatrice: T_a_1_9.devices()", T_a_1_9.devices())
    B = B.at[:idx_1, :idx_9].set(T_a_1_9)
    B = B.at[:idx_1, idx_11:idx_12].set(T_a_1_12)
    B = B.at[:idx_1, idx_16:idx_17].set(T_a_1_17)
    # Second row
    B = B.at[idx_1:idx_2, :idx_9].set(T_b_2_9)
    B = B.at[idx_1:idx_2, idx_9:idx_10].set(T_b_2_10)
    B = B.at[idx_1:idx_2, idx_17:idx_18].set(T_b_2_18)
    # Third row
    B = B.at[idx_2:idx_3, idx_9:idx_10].set(T_c_3_10)
    B = B.at[idx_2:idx_3, idx_10:idx_11].set(T_c_3_11)
    B = B.at[idx_2:idx_3, idx_18:idx_19].set(T_c_3_19)
    # Fourth row
    B = B.at[idx_3:idx_4, idx_10:idx_11].set(T_d_4_11)
    B = B.at[idx_3:idx_4, idx_11:idx_12].set(T_d_4_12)
    B = B.at[idx_3:idx_4, idx_19:idx_20].set(T_d_4_20)
    # Fifth row
    B = B.at[idx_4:idx_5, idx_12:idx_13].set(T_e_5_13)
    B = B.at[idx_4:idx_5, idx_15:idx_16].set(T_e_5_16)
    B = B.at[idx_4:idx_5, idx_16:idx_17].set(T_e_5_17)
    # Sixth row
    B = B.at[idx_5:idx_6, idx_12:idx_13].set(T_f_6_13)
    B = B.at[idx_5:idx_6, idx_13:idx_14].set(T_f_6_14)
    B = B.at[idx_5:idx_6, idx_17:idx_18].set(T_f_6_18)
    # Seventh row
    B = B.at[idx_6:idx_7, idx_13:idx_14].set(T_g_7_14)
    B = B.at[idx_6:idx_7, idx_14:idx_15].set(T_g_7_15)
    B = B.at[idx_6:idx_7, idx_18:idx_19].set(T_g_7_19)
    # Eighth row
    B = B.at[idx_7:idx_8, idx_14:idx_15].set(T_h_8_15)
    B = B.at[idx_7:idx_8, idx_15:idx_16].set(T_h_8_16)
    B = B.at[idx_7:idx_8, idx_19:idx_20].set(T_h_8_20)

    # C is a block matrix with an array of (12x8) blocks.
    C = jnp.zeros((n_int_pts, n_ext_pts), dtype=jnp.float64)
    # First row
    C = C.at[:idx_9, :idx_1].set(T_a_9_1)
    C = C.at[:idx_9, idx_1:idx_2].set(T_b_9_2)
    # Second row
    C = C.at[idx_9:idx_10, idx_1:idx_2].set(T_b_10_2)
    C = C.at[idx_9:idx_10, idx_2:idx_3].set(T_c_10_3)
    # Third row
    C = C.at[idx_10:idx_11, idx_2:idx_3].set(T_c_11_3)
    C = C.at[idx_10:idx_11, idx_3:idx_4].set(T_d_11_4)
    # Fourth row
    C = C.at[idx_11:idx_12, :idx_1].set(T_a_12_1)
    C = C.at[idx_11:idx_12, idx_3:idx_4].set(T_d_12_4)
    # Fifth row
    C = C.at[idx_12:idx_13, idx_4:idx_5].set(T_e_13_5)
    C = C.at[idx_12:idx_13, idx_5:idx_6].set(T_f_13_6)
    # Sixth row
    C = C.at[idx_13:idx_14, idx_5:idx_6].set(T_f_14_6)
    C = C.at[idx_13:idx_14, idx_6:idx_7].set(T_g_14_7)
    # Seventh row
    C = C.at[idx_14:idx_15, idx_6:idx_7].set(T_g_15_7)
    C = C.at[idx_14:idx_15, idx_7:idx_8].set(T_h_15_8)
    # Eighth row
    C = C.at[idx_15:idx_16, idx_4:idx_5].set(T_e_16_5)
    C = C.at[idx_15:idx_16, idx_7:idx_8].set(T_h_16_8)
    # Ninth row
    C = C.at[idx_16:idx_17, :idx_1].set(T_a_17_1)
    C = C.at[idx_16:idx_17, idx_4:idx_5].set(T_e_17_5)
    # Tenth row
    C = C.at[idx_17:idx_18, idx_1:idx_2].set(T_b_18_2)
    C = C.at[idx_17:idx_18, idx_5:idx_6].set(T_f_18_6)
    # Eleventh row
    C = C.at[idx_18:idx_19, idx_2:idx_3].set(T_c_19_3)
    C = C.at[idx_18:idx_19, idx_6:idx_7].set(T_g_19_7)
    # Twelfth row
    C = C.at[idx_19:, idx_3:idx_4].set(T_d_20_4)
    C = C.at[idx_19:, idx_7:idx_8].set(T_h_20_8)

    # D is a block matrix with an array of (12x12) blocks.
    D = jnp.zeros((n_int_pts, n_int_pts), dtype=jnp.float64)
    # First row
    D = D.at[:idx_9, :idx_9].set(T_a_9_9 + T_b_9_9)
    D = D.at[:idx_9, idx_9:idx_10].set(T_b_9_10)
    D = D.at[:idx_9, idx_11:idx_12].set(T_a_9_12)
    D = D.at[:idx_9, idx_16:idx_17].set(T_a_9_17)
    D = D.at[:idx_9, idx_17:idx_18].set(T_b_9_18)
    # Second row
    D = D.at[idx_9:idx_10, :idx_9].set(T_b_10_9)
    D = D.at[idx_9:idx_10, idx_9:idx_10].set(T_b_10_10 + T_c_10_10)
    D = D.at[idx_9:idx_10, idx_10:idx_11].set(T_c_10_11)
    D = D.at[idx_9:idx_10, idx_17:idx_18].set(T_b_10_18)
    D = D.at[idx_9:idx_10, idx_18:idx_19].set(T_c_10_19)
    # Third row
    D = D.at[idx_10:idx_11, idx_9:idx_10].set(T_c_11_10)
    D = D.at[idx_10:idx_11, idx_10:idx_11].set(T_c_11_11 + T_d_11_11)
    D = D.at[idx_10:idx_11, idx_11:idx_12].set(T_d_11_12)
    D = D.at[idx_10:idx_11, idx_18:idx_19].set(T_c_11_19)
    D = D.at[idx_10:idx_11, idx_19:idx_20].set(T_d_11_20)
    # Fourth row
    D = D.at[idx_11:idx_12, :idx_9].set(T_a_12_9)
    D = D.at[idx_11:idx_12, idx_10:idx_11].set(T_d_12_11)
    D = D.at[idx_11:idx_12, idx_11:idx_12].set(T_d_12_12 + T_a_12_12)
    D = D.at[idx_11:idx_12, idx_16:idx_17].set(T_a_12_17)
    D = D.at[idx_11:idx_12, idx_19:idx_20].set(T_d_12_20)
    # Fifth row
    D = D.at[idx_12:idx_13, idx_12:idx_13].set(T_e_13_13 + T_f_13_13)
    D = D.at[idx_12:idx_13, idx_13:idx_14].set(T_f_13_14)
    D = D.at[idx_12:idx_13, idx_15:idx_16].set(T_e_13_16)
    D = D.at[idx_12:idx_13, idx_16:idx_17].set(T_e_13_17)
    D = D.at[idx_12:idx_13, idx_17:idx_18].set(T_f_13_18)
    # Sixth row
    D = D.at[idx_13:idx_14, idx_12:idx_13].set(T_f_14_13)
    D = D.at[idx_13:idx_14, idx_13:idx_14].set(T_f_14_14 + T_g_14_14)
    D = D.at[idx_13:idx_14, idx_14:idx_15].set(T_g_14_15)
    D = D.at[idx_13:idx_14, idx_17:idx_18].set(T_f_14_18)
    D = D.at[idx_13:idx_14, idx_18:idx_19].set(T_g_14_19)
    # Seventh row
    D = D.at[idx_14:idx_15, idx_13:idx_14].set(T_g_15_14)
    D = D.at[idx_14:idx_15, idx_14:idx_15].set(T_g_15_15 + T_h_15_15)
    D = D.at[idx_14:idx_15, idx_15:idx_16].set(T_h_15_16)
    D = D.at[idx_14:idx_15, idx_18:idx_19].set(T_g_15_19)
    D = D.at[idx_14:idx_15, idx_19:idx_20].set(T_h_15_20)
    # Eighth row
    D = D.at[idx_15:idx_16, idx_12:idx_13].set(T_e_16_13)
    D = D.at[idx_15:idx_16, idx_14:idx_15].set(T_h_16_15)
    D = D.at[idx_15:idx_16, idx_15:idx_16].set(T_h_16_16 + T_e_16_16)
    D = D.at[idx_15:idx_16, idx_16:idx_17].set(T_e_16_17)
    D = D.at[idx_15:idx_16, idx_19:idx_20].set(T_h_16_20)
    # Ninth row
    D = D.at[idx_16:idx_17, :idx_9].set(T_a_17_9)
    D = D.at[idx_16:idx_17, idx_11:idx_12].set(T_a_17_12)
    D = D.at[idx_16:idx_17, idx_12:idx_13].set(T_e_17_13)
    D = D.at[idx_16:idx_17, idx_15:idx_16].set(T_e_17_16)
    D = D.at[idx_16:idx_17, idx_16:idx_17].set(T_e_17_17 + T_a_17_17)
    # Tenth row
    D = D.at[idx_17:idx_18, :idx_9].set(T_b_18_9)
    D = D.at[idx_17:idx_18, idx_9:idx_10].set(T_b_18_10)
    D = D.at[idx_17:idx_18, idx_12:idx_13].set(T_f_18_13)
    D = D.at[idx_17:idx_18, idx_13:idx_14].set(T_f_18_14)
    D = D.at[idx_17:idx_18, idx_17:idx_18].set(T_f_18_18 + T_b_18_18)
    # Eleventh row
    D = D.at[idx_18:idx_19, idx_9:idx_10].set(T_c_19_10)
    D = D.at[idx_18:idx_19, idx_10:idx_11].set(T_c_19_11)
    D = D.at[idx_18:idx_19, idx_13:idx_14].set(T_g_19_14)
    D = D.at[idx_18:idx_19, idx_14:idx_15].set(T_g_19_15)
    D = D.at[idx_18:idx_19, idx_18:idx_19].set(T_g_19_19 + T_c_19_19)
    # Twelfth row
    D = D.at[idx_19:idx_20, idx_10:idx_11].set(T_d_20_11)
    D = D.at[idx_19:idx_20, idx_11:idx_12].set(T_d_20_12)
    D = D.at[idx_19:idx_20, idx_14:idx_15].set(T_h_20_15)
    D = D.at[idx_19:idx_20, idx_15:idx_16].set(T_h_20_16)
    D = D.at[idx_19:idx_20, idx_19:idx_20].set(T_h_20_20 + T_d_20_20)

    A_lst = [T_a_1_1, T_b_2_2, T_c_3_3, T_d_4_4, T_e_5_5, T_f_6_6, T_g_7_7, T_h_8_8]
    delta_v_prime_int = jnp.concatenate(
        [
            v_prime_a_9 + v_prime_b_9,
            v_prime_b_10 + v_prime_c_10,
            v_prime_c_11 + v_prime_d_11,
            v_prime_d_12 + v_prime_a_12,
            v_prime_e_13 + v_prime_f_13,
            v_prime_f_14 + v_prime_g_14,
            v_prime_g_15 + v_prime_h_15,
            v_prime_h_16 + v_prime_e_16,
            v_prime_a_17 + v_prime_e_17,
            v_prime_b_18 + v_prime_f_18,
            v_prime_c_19 + v_prime_g_19,
            v_prime_d_20 + v_prime_h_20,
        ]
    )
    v_prime_ext = jnp.concatenate(
        [
            v_prime_a_1,
            v_prime_b_2,
            v_prime_c_3,
            v_prime_d_4,
            v_prime_e_5,
            v_prime_f_6,
            v_prime_g_7,
            v_prime_h_8,
        ]
    )

    # A_lst = [jax.device_put(A, DEVICE_ARR[0]) for A in A_lst]
    # B = jax.device_put(B, DEVICE_ARR[0])
    # C = jax.device_put(C, DEVICE_ARR[0])
    # D = jax.device_put(D, DEVICE_ARR[0])
    # v_prime_ext = jax.device_put(v_prime_ext, DEVICE_ARR[0])
    # delta_v_prime_int = jax.device_put(delta_v_prime_int, DEVICE_ARR[0])

    # print("_oct_merge_from_submatrices: D devices: ", D.devices())
    T, S, v_prime_ext_out, v_int = assemble_merge_outputs_DtN(
        A_lst, B, C, D, v_prime_ext, delta_v_prime_int
    )
    # Move outputs back to the CPU
    # T = jax.device_put(T, HOST_DEVICE)
    # S = jax.device_put(S, HOST_DEVICE)
    # v_prime_ext_out = jax.device_put(v_prime_ext_out, HOST_DEVICE)
    # v_int = jax.device_put(v_int, HOST_DEVICE)

    return T, S, v_prime_ext_out, v_int
