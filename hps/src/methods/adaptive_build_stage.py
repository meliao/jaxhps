from functools import partial
import logging
from typing import Tuple, List, Any

import jax
import jax.numpy as jnp
import numpy as np

from hps.src.quadrature.quad_2D.adaptive_merge_indexing import (
    get_quadmerge_blocks_a,
    get_quadmerge_blocks_b,
    get_quadmerge_blocks_c,
    get_quadmerge_blocks_d,
)
from hps.src.quadrature.quad_3D.adaptive_merge_indexing import (
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
from hps.src.quadrature.trees import (
    Node,
    get_all_leaves,
    get_all_leaves_jitted,
    get_nodes_at_level,
    get_depth,
    get_all_nodes,
    get_all_nodes_jitted,
    get_all_parents_of_leaves,
)
from hps.src.methods.uniform_build_stage import vmapped_uniform_oct_merge
from hps.src.methods.schur_complement import (
    assemble_merge_outputs_DtN,
    _oct_merge_from_submatrices,
)
from hps.src.methods.adaptive_merge_utils_2D import find_compression_lists_2D
from hps.src.methods.adaptive_merge_utils_3D import (
    find_compression_lists_3D,
)
from hps.src.config import DEVICE_ARR, HOST_DEVICE


def _build_stage_3D(
    root: Node,
    refinement_op: jnp.array,
    coarsening_op: jnp.array,
    DtN_arr: jnp.array,
    v_prime_arr: jnp.array,
    q: int,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> None:
    """_summary_

    Args:
        root (Node): _description_
        refinement_op (jnp.array): Expect this to have shape (4 q^2, q^2)
        coarsening_op (jnp.array): Expect this to have shpae (q^2, 4 q^2)
        DtN_arr (jnp.array): Expect this to have shape (?, 8, 6 q^2, 6 q^2)
        v_prime_arr (jnp.array): Expect this to have shape (?, 8, 6 q^2)
    """
    logging.debug("_build_stage_3D: started")
    logging.debug("_build_stage_3D: DtN_arr.shape: %s", DtN_arr.shape)

    # # First, do the leaf-level merges using the vmapped code.
    q_idxes = jnp.arange(q)

    # First, find all of the depths of the leaves
    leaves = get_all_leaves(root)
    depths = jnp.array([leaf.depth for leaf in leaves])
    max_depth = jnp.max(depths)
    lowest_level_bools = depths == max_depth

    logging.debug("_build_stage_3D: max_depth = %s", max_depth)

    # We only want to use the DtN arrays that are at the
    # lowest level of the tree
    DtN_lowest_level = DtN_arr[lowest_level_bools]
    v_prime_lowest_level = v_prime_arr[lowest_level_bools]

    # Move these arrays to device
    DtN_lowest_level = jax.device_put(DtN_lowest_level, device)
    v_prime_lowest_level = jax.device_put(v_prime_lowest_level, device)

    # Vmapped code works on the lowest level
    S, DtN, v_prime_ext, v_int = vmapped_uniform_oct_merge(
        q_idxes, DtN_lowest_level, v_prime_lowest_level
    )
    S = jax.device_put(S, host_device)
    DtN = jax.device_put(DtN, host_device)
    v_prime_ext = jax.device_put(v_prime_ext, host_device)
    v_int = jax.device_put(v_int, host_device)

    # Assign the outputs to the nodes at the penuultimate level
    parents_of_leaves = get_nodes_at_level(root, max_depth - 1)
    counter = 0
    for parent in parents_of_leaves:
        # logging.debug("_build_stage_3D: parent = %s", parent)
        if parent.DtN is not None:
            # Filter out the leaves at this level; they already have DtN matrices
            continue
        parent.DtN = DtN[counter]
        parent.v_prime = v_prime_ext[counter]
        parent.S = S[counter]
        parent.v_int = v_int[counter]
        counter += 1

        D_shape = parent.S.shape[0]

    # Perform a merge operation for each level of the tree. Start at the leaves
    # and continue until the root.
    # For comments, suppose the for loop is in iteration j.
    for j, i in enumerate(range(max_depth - 2, -1, -1)):
        # logging.debug("_build_stage_3D: j = %s, i = %s", j, i)

        # Get the inpute to the merge operation from the tree.
        nodes_this_level = get_nodes_at_level(root, i)

        # Filter out the nodes which are leaves.
        nodes_this_level = [node for node in nodes_this_level if len(node.children)]

        # Filter out the nodes which have DtN arrays already set
        nodes_this_level = [node for node in nodes_this_level if node.DtN is None]

        # The leaves in need of refinement are those who have more children than the min number
        # of children among siblings
        # Has shape (m // 8, 8)
        D_shape = oct_merge_nonuniform_whole_level(
            refinement_op,
            coarsening_op,
            nodes_this_level,
        )
    return D_shape


def _build_stage_2D(
    root: Node,
    refinement_op: jnp.array,
    coarsening_op: jnp.array,
) -> None:
    """Implements the upward pass, without the Tree interface.

    Args:
        DtN_maps (jnp.ndarray): Has shape (n_leaves,4*q, 4*q)
        v_prime_arr (jnp.ndarray): Has shape (n_leaves, 4*q)
        root (Node): The root of the tree. Prescribes shape information.

    Returns:
        None. Sets outputs in the tree object.
    """
    logging.debug("_build_stage_2D: started")

    depth = get_depth(root)

    # Perform a merge operation for each level of the tree. Start at the leaves
    # and continue until the root.
    # For comments, suppose the for loop is in iteration j.
    for j, i in enumerate(range(depth, 0, -1)):

        # Get the inputs to the merge operation from the tree.
        nodes_this_level = get_nodes_at_level(root, i)
        DtN_this_level = [n.DtN for n in nodes_this_level]
        v_prime_this_level = [n.v_prime for n in nodes_this_level]

        # The leaves in need of refinement are those who have more children than the min number
        # of children among siblings.
        # Has shape (m // 4, 8)

        # Expect DtN_arr to be a list of length (m // 4) where each element is a matrix.
        S_arr_lst, DtN_arr_lst, v_prime_arr_lst, v_int_arr_lst = (
            quad_merge_nonuniform_whole_level(
                DtN_this_level,
                v_prime_this_level,
                refinement_op,
                coarsening_op,
                nodes_this_level,
            )
        )

        # print("_build_stage_2D: S_arr_lst len = ", len(S_arr_lst))

        # Set the output in the tree object.
        nodes_next_level = get_nodes_at_level(root, i - 1)
        # print("_build_stage_2D: nodes_next_level len = ", len(nodes_next_level))

        # Filter out the nodes which are leaves.
        nodes_next_level = [node for node in nodes_next_level if len(node.children)]
        # print("_build_stage_2D: nodes_next_level len = ", len(nodes_next_level))
        for k, node in enumerate(nodes_next_level):
            node.DtN = DtN_arr_lst[k]
            node.v_prime = v_prime_arr_lst[k]
            node.S = S_arr_lst[k]
            node.v_int = v_int_arr_lst[k]


# @partial(jax.jit, static_argnums=(10, 11, 12, 13))
def _quad_merge(
    T_a: jnp.array,
    T_b: jnp.array,
    T_c: jnp.array,
    T_d: jnp.array,
    v_prime_a: jnp.array,
    v_prime_b: jnp.array,
    v_prime_c: jnp.array,
    v_prime_d: jnp.array,
    L_2f1: jnp.array,
    L_1f2: jnp.array,
    need_interp_lsts: Tuple[jnp.array],
    side_lens_a: jnp.array,
    side_lens_b: jnp.array,
    side_lens_c: jnp.array,
    side_lens_d: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Takes in the DtN matrices and v_prime vectors for a set of four nodes being merged
    together. The function then performs the merge operation and returns the results.

    Args:
        T_a (jnp.array): DtN matrix for node a.
        T_b (jnp.array): DtN matrix for node b.
        T_c (jnp.array): DtN matrix for node c.
        T_d (jnp.array): DtN matrix for node d.
        v_prime_a (jnp.array):
        v_prime_b (jnp.array):
        v_prime_c (jnp.array):
        v_prime_d (jnp.array):
        L_2f1 (jnp.array): Interpolation operator with shape (2q, q)
        L_1f2 (jnp.array): Interpolation operator with shape (q, 2q)
        need_interp_lsts (Tuple[jnp.array]): Tuple of length 8. Each element is a boolean array
            indicating whether particular panels should be interpolated or not. The order is:
            a_5, b_5, b_6, c_6, c_7, d_7, d_8, a_8. Each array is ordered from the outside inward.
        side_lens_a (jnp.array): Length 4 array indicating the number of quadrature points along each side of node a.
        side_lens_b (jnp.array):
        side_lens_c (jnp.array):
        side_lens_d (jnp.array):

    Returns:
        Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
        S: Matrix mapping boundary data to merge interfaces.
        T: DtN matrix for the merged node.
        v_prime_ext: Boundary particular fluxes.
        v_int: Particular solutions evaluated on the merge interfaces.
    """

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
    ) = get_quadmerge_blocks_a(
        T_a,
        v_prime_a,
        L_2f1,
        L_1f2,
        need_interp_lsts[0],
        need_interp_lsts[7],
        side_lens_a[0],
        side_lens_a[1],
        side_lens_a[2],
        side_lens_a[3],
    )
    n_1, n_5 = T_a_15.shape
    n_8 = T_a_18.shape[1]

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
    ) = get_quadmerge_blocks_b(
        T_b,
        v_prime_b,
        L_2f1,
        L_1f2,
        need_interp_lsts[2],
        need_interp_lsts[1],
        side_lens_b[0],
        side_lens_b[1],
        side_lens_b[2],
        side_lens_b[3],
    )
    n_2, n_6 = T_b_26.shape

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
    ) = get_quadmerge_blocks_c(
        T_c,
        v_prime_c,
        L_2f1,
        L_1f2,
        need_interp_lsts[3],
        need_interp_lsts[4],
        side_lens_c[0],
        side_lens_c[1],
        side_lens_c[2],
        side_lens_c[3],
    )
    n_3, n_7 = T_c_37.shape

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
    ) = get_quadmerge_blocks_d(
        T_d,
        v_prime_d,
        L_2f1,
        L_1f2,
        need_interp_lsts[6],
        need_interp_lsts[5],
        side_lens_d[0],
        side_lens_d[1],
        side_lens_d[2],
        side_lens_d[3],
    )
    n_8, n_4 = T_d_84.shape

    # print("_quad_merge: T_a_15 shape: ", T_a_15.shape)
    # print("_quad_merge: n_5 = ", n_5)
    assert T_b_25.shape == (n_2, n_5)
    assert T_b_26.shape == (n_2, n_6)

    B_0 = jnp.block([T_a_15, jnp.zeros((n_1, n_6)), jnp.zeros((n_1, n_7)), T_a_18])
    B_1 = jnp.block([T_b_25, T_b_26, jnp.zeros((n_2, n_7)), jnp.zeros((n_2, n_8))])
    B_2 = jnp.block([jnp.zeros((n_3, n_5)), T_c_36, T_c_37, jnp.zeros((n_3, n_8))])
    B_3 = jnp.block([jnp.zeros((n_4, n_5)), jnp.zeros((n_4, n_6)), T_d_47, T_d_48])
    # print("_quad_merge: B_0.shape = ", B_0.shape)
    # print("_quad_merge: B_1.shape = ", B_1.shape)
    # print("_quad_merge: B_2.shape = ", B_2.shape)
    # print("_quad_merge: B_3.shape = ", B_3.shape)

    B = jnp.concatenate([B_0, B_1, B_2, B_3], axis=0)
    # print("_quad_merge: B.shape: ", B.shape)
    C = jnp.block(
        [
            [T_a_51, T_b_52, jnp.zeros((n_5, n_3)), jnp.zeros((n_5, n_4))],
            [jnp.zeros((n_6, n_1)), T_b_62, T_c_63, jnp.zeros((n_6, n_4))],
            [jnp.zeros((n_7, n_1)), jnp.zeros((n_7, n_2)), T_c_73, T_d_74],
            [T_a_81, jnp.zeros((n_8, n_2)), jnp.zeros((n_8, n_3)), T_d_84],
        ]
    )

    D = jnp.block(
        [
            [T_a_55 + T_b_55, T_b_56, jnp.zeros((n_5, n_7)), T_a_58],
            [T_b_65, T_b_66 + T_c_66, T_c_67, jnp.zeros((n_6, n_8))],
            [jnp.zeros((n_7, n_5)), T_c_76, T_c_77 + T_d_77, T_d_78],
            [T_a_85, jnp.zeros((n_8, n_6)), T_d_87, T_d_88 + T_a_88],
        ]
    )
    A_lst = [T_a_11, T_b_22, T_c_33, T_d_44]
    delta_v_prime_int = jnp.concatenate(
        [
            v_prime_a_5 + v_prime_b_5,
            v_prime_b_6 + v_prime_c_6,
            v_prime_c_7 + v_prime_d_7,
            v_prime_d_8 + v_prime_a_8,
        ]
    )
    v_prime_ext = jnp.concatenate([v_prime_a_1, v_prime_b_2, v_prime_c_3, v_prime_d_4])

    # use the Schur complement code to compute the Schur complement
    T, S, v_prime_ext_out, v_int = assemble_merge_outputs_DtN(
        A_lst, B, C, D, v_prime_ext, delta_v_prime_int
    )

    # Roll the exterior to get the correct ordering. Before rolling
    # The ordering is boundary of A, ..., boundary of D. After rolling
    # The ordering is side 0, side 1, side 2, side 3.
    n_roll = side_lens_a[3]
    v_prime_ext = jnp.roll(v_prime_ext_out, -n_roll, axis=0)
    T = jnp.roll(T, -n_roll, axis=0)
    T = jnp.roll(T, -n_roll, axis=1)
    S = jnp.roll(S, -n_roll, axis=1)

    return S, T, v_prime_ext, v_int


_vmapped_quad_merge = jax.vmap(
    _quad_merge,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, 0, 0, 0, 0),
    out_axes=(0, 0, 0, 0),
)


def quad_merge_nonuniform_whole_level(
    T_in: List[jnp.ndarray],
    v_prime: List[jnp.ndarray],
    L_2f1: jnp.ndarray,
    L_1f2: jnp.ndarray,
    nodes_this_level: List[Node],
) -> Tuple[List[jnp.array], List[jnp.array], List[jnp.array], List[jnp.array]]:
    """
    This function takes in pre-computed DtN matrices and v_prime vectors, as well
    as a list of Nodes, and merges the Nodes 4 at a time. It does the following
    operations:
    1. Splits the input list of nodes into groups of 4 for each merge operation.
    2. Gathers node information about the number of quadrature points along each side
    3. Gathers information about which panels in the nodes need compression.
    4. Calls the _quad_merge function to perform the merge operation.
    5. Returns the results of the merge operation.

    Args:
        T_in (List[jnp.ndarray]): List has length (m,) and each element is a square matrix.
        v_prime (List[jnp.ndarray]): List has length (m,) and each element is a vector. The i'th element of this list
        should have the same shape as the i'th element of T_in.
        L_2f1 (jnp.ndarray): Interpolation operator with shape (2q, q)
        L_1f2 (jnp.ndarray): Interpolation operator with shape (q, 2q)
        nodes_this_level (List[Node]): List of Nodes being merged at this level. Has length (m,)

    Returns:
        Tuple[List[jnp.array], List[jnp.array], List[jnp.array], List[jnp.array]]: In order:
        S_lst: List of matrices mapping boundary data to merge interfaces. Has length (m // 4).
        T_lst: DtN matrices for the merged nodes. Has length (m // 4).
        v_prime_ext_lst: Boundary particular fluxes. Has length (m // 4).
        v_lst: Particular solutions evaluated on the merge interfaces. Has length (m // 4).
    """
    S_lst = []
    T_lst = []
    v_prime_ext_lst = []
    v_lst = []
    n_merges = len(T_in) // 4

    for i in range(n_merges):
        node_a = nodes_this_level[4 * i]
        node_b = nodes_this_level[4 * i + 1]
        node_c = nodes_this_level[4 * i + 2]
        node_d = nodes_this_level[4 * i + 3]

        side_lens_a = jnp.array(
            [node_a.n_0, node_a.n_1, node_a.n_2, node_a.n_3],
            dtype=jnp.int32,
        )
        side_lens_b = jnp.array(
            [node_b.n_0, node_b.n_1, node_b.n_2, node_b.n_3],
            dtype=jnp.int32,
        )
        side_lens_c = jnp.array(
            [node_c.n_0, node_c.n_1, node_c.n_2, node_c.n_3],
            dtype=jnp.int32,
        )
        side_lens_d = jnp.array(
            [node_d.n_0, node_d.n_1, node_d.n_2, node_d.n_3],
            dtype=jnp.int32,
        )

        need_interp_lsts = find_compression_lists_2D(node_a, node_b, node_c, node_d)
        # print(
        #     "quad_merge_nonuniform_whole_level: need_interp_lsts = ", need_interp_lsts
        # )
        # print(
        #     "quad_merge_nonuniform_whole_level: len(need_interp_lsts) = ",
        #     len(need_interp_lsts),
        # )

        S, T, v_prime_ext, v = _quad_merge(
            T_in[4 * i],
            T_in[4 * i + 1],
            T_in[4 * i + 2],
            T_in[4 * i + 3],
            v_prime[4 * i],
            v_prime[4 * i + 1],
            v_prime[4 * i + 2],
            v_prime[4 * i + 3],
            L_2f1,
            L_1f2,
            need_interp_lsts=need_interp_lsts,
            side_lens_a=side_lens_a,
            side_lens_b=side_lens_b,
            side_lens_c=side_lens_c,
            side_lens_d=side_lens_d,
        )
        # print("quad_merge_whole_level: v_prime_ext shape", v_prime_ext.shape)
        S_lst.append(S)
        T_lst.append(T)
        v_prime_ext_lst.append(v_prime_ext)
        v_lst.append(v)

    return S_lst, T_lst, v_prime_ext_lst, v_lst


# @partial(jax.jit, static_argnums=(19, 20, 21, 22, 23, 24, 25, 26))


def _oct_merge(
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
    L_2f1: jnp.ndarray,
    L_1f2: jnp.ndarray,
    need_interp_lsts: Tuple[jnp.array],
    side_lens_a: jnp.array,
    side_lens_b: jnp.array,
    side_lens_c: jnp.array,
    side_lens_d: jnp.array,
    side_lens_e: jnp.array,
    side_lens_f: jnp.array,
    side_lens_g: jnp.array,
    side_lens_h: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:

    a_submatrices_subvecs = get_a_submatrices(
        T_a,
        v_prime_a,
        L_2f1,
        L_1f2,
        need_interp_9=need_interp_lsts[0],
        need_interp_12=need_interp_lsts[7],
        need_interp_17=need_interp_lsts[16],
        n_0=side_lens_a[0],
        n_1=side_lens_a[1],
        n_2=side_lens_a[2],
        n_3=side_lens_a[3],
        n_4=side_lens_a[4],
        n_5=side_lens_a[5],
    )
    a_submatrices_subvecs = [
        jax.device_put(a, DEVICE_ARR[0]) for a in a_submatrices_subvecs
    ]
    b_submatrices_subvecs = get_b_submatrices(
        T_b,
        v_prime_b,
        L_2f1,
        L_1f2,
        need_interp_9=need_interp_lsts[1],
        need_interp_10=need_interp_lsts[2],
        need_interp_18=need_interp_lsts[18],
        n_0=side_lens_b[0],
        n_1=side_lens_b[1],
        n_2=side_lens_b[2],
        n_3=side_lens_b[3],
        n_4=side_lens_b[4],
        n_5=side_lens_b[5],
    )
    b_submatrices_subvecs = [
        jax.device_put(b, DEVICE_ARR[0]) for b in b_submatrices_subvecs
    ]
    c_submatrices_subvecs = get_c_submatrices(
        T_c,
        v_prime_c,
        L_2f1,
        L_1f2,
        need_interp_10=need_interp_lsts[3],
        need_interp_11=need_interp_lsts[4],
        need_interp_19=need_interp_lsts[20],
        n_0=side_lens_c[0],
        n_1=side_lens_c[1],
        n_2=side_lens_c[2],
        n_3=side_lens_c[3],
        n_4=side_lens_c[4],
        n_5=side_lens_c[5],
    )
    c_submatrices_subvecs = [
        jax.device_put(c, DEVICE_ARR[0]) for c in c_submatrices_subvecs
    ]
    d_submatrices_subvecs = get_d_submatrices(
        T_d,
        v_prime_d,
        L_2f1,
        L_1f2,
        need_interp_11=need_interp_lsts[5],
        need_interp_12=need_interp_lsts[6],
        need_interp_20=need_interp_lsts[22],
        n_0=side_lens_d[0],
        n_1=side_lens_d[1],
        n_2=side_lens_d[2],
        n_3=side_lens_d[3],
        n_4=side_lens_d[4],
        n_5=side_lens_d[5],
    )
    d_submatrices_subvecs = [
        jax.device_put(d, DEVICE_ARR[0]) for d in d_submatrices_subvecs
    ]
    e_submatrices_subvecs = get_e_submatrices(
        T_e,
        v_prime_e,
        L_2f1,
        L_1f2,
        need_interp_13=need_interp_lsts[8],
        need_interp_16=need_interp_lsts[15],
        need_interp_17=need_interp_lsts[17],
        n_0=side_lens_e[0],
        n_1=side_lens_e[1],
        n_2=side_lens_e[2],
        n_3=side_lens_e[3],
        n_4=side_lens_e[4],
        n_5=side_lens_e[5],
    )
    e_submatrices_subvecs = [
        jax.device_put(e, DEVICE_ARR[0]) for e in e_submatrices_subvecs
    ]
    f_submatrices_subvecs = get_f_submatrices(
        T_f,
        v_prime_f,
        L_2f1,
        L_1f2,
        need_interp_13=need_interp_lsts[9],
        need_interp_14=need_interp_lsts[10],
        need_interp_18=need_interp_lsts[19],
        n_0=side_lens_f[0],
        n_1=side_lens_f[1],
        n_2=side_lens_f[2],
        n_3=side_lens_f[3],
        n_4=side_lens_f[4],
        n_5=side_lens_f[5],
    )
    f_submatrices_subvecs = [
        jax.device_put(f, DEVICE_ARR[0]) for f in f_submatrices_subvecs
    ]
    g_submatrices_subvecs = get_g_submatrices(
        T_g,
        v_prime_g,
        L_2f1,
        L_1f2,
        need_interp_14=need_interp_lsts[11],
        need_interp_15=need_interp_lsts[12],
        need_interp_19=need_interp_lsts[21],
        n_0=side_lens_g[0],
        n_1=side_lens_g[1],
        n_2=side_lens_g[2],
        n_3=side_lens_g[3],
        n_4=side_lens_g[4],
        n_5=side_lens_g[5],
    )
    g_submatrices_subvecs = [
        jax.device_put(g, DEVICE_ARR[0]) for g in g_submatrices_subvecs
    ]
    h_submatrices_subvecs = get_h_submatrices(
        T_h,
        v_prime_h,
        L_2f1,
        L_1f2,
        need_interp_15=need_interp_lsts[13],
        need_interp_16=need_interp_lsts[14],
        need_interp_20=need_interp_lsts[23],
        n_0=side_lens_h[0],
        n_1=side_lens_h[1],
        n_2=side_lens_h[2],
        n_3=side_lens_h[3],
        n_4=side_lens_h[4],
        n_5=side_lens_h[5],
    )
    h_submatrices_subvecs = [
        jax.device_put(h, DEVICE_ARR[0]) for h in h_submatrices_subvecs
    ]

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

    T = jax.device_put(T, HOST_DEVICE)
    S = jax.device_put(S, HOST_DEVICE)
    v_prime_ext_out = jax.device_put(v_prime_ext_out, HOST_DEVICE)
    v_int = jax.device_put(v_int, HOST_DEVICE)

    r = get_rearrange_indices(
        jnp.arange(T.shape[0]),
        n_a_0=side_lens_a[0],
        n_a_2=side_lens_a[2],
        n_a_5=side_lens_a[5],
        n_b_1=side_lens_b[1],
        n_b_2=side_lens_b[2],
        n_b_5=side_lens_b[5],
        n_c_1=side_lens_c[1],
        n_c_3=side_lens_c[3],
        n_c_5=side_lens_c[5],
        n_d_0=side_lens_d[0],
        n_d_3=side_lens_d[3],
        n_d_5=side_lens_d[5],
        n_e_0=side_lens_e[0],
        n_e_2=side_lens_e[2],
        n_e_4=side_lens_e[4],
        n_f_1=side_lens_f[1],
        n_f_2=side_lens_f[2],
        n_f_4=side_lens_f[4],
        n_g_1=side_lens_g[1],
        n_g_3=side_lens_g[3],
        n_g_4=side_lens_g[4],
        n_h_0=side_lens_h[0],
        n_h_3=side_lens_h[3],
        n_h_4=side_lens_h[4],
    )
    v_prime_ext_out = v_prime_ext_out[r]
    T = T[r][:, r]
    S = S[:, r]

    return S, T, v_prime_ext_out, v_int


def is_node_type(x: Any) -> bool:
    return isinstance(x, Node)


def oct_merge_nonuniform_whole_level(
    L_2f1: jnp.ndarray,
    L_1f2: jnp.ndarray,
    nodes_this_level: List[Node],
) -> Tuple[List[jnp.array], List[jnp.array], List[jnp.array], List[jnp.array]]:
    """
    This function takes in pre-computed DtN matrices and v_prime vectors, as well
    as a list of Nodes, and merges the Nodes 8 at a time. It does the following
    operations:
    1. Splits the input list of nodes into groups of 8 for each merge operation.
    2. Gathers node information about the number of quadrature points along each side
    3. Gathers information about which panels in the nodes need compression.
    4. Calls the _oct_merge function to perform the merge operation.
    5. Returns the results of the merge operation.

    Args:
        T_in (List[jnp.ndarray]): List has length (m,) and each element is a square matrix.
        v_prime (List[jnp.ndarray]): List has length (m,) and each element is a vector. The i'th element of this list
        should have the same shape as the i'th element of T_in.
        L_2f1 (jnp.ndarray): Interpolation operator with shape (4 q^2, q^2)
        L_1f2 (jnp.ndarray): Interpolation operator with shape (q^2, 4q^2)
        nodes_this_level (List[Node]): List of Nodes being merged at this level. Has length (m,)

    Returns:
        Tuple[List[jnp.array], List[jnp.array], List[jnp.array], List[jnp.array]]: In order:
        S_lst: List of matrices mapping boundary data to merge interfaces. Has length (m // 8).
        T_lst: DtN matrices for the merged nodes. Has length (m // 8).
        v_prime_ext_lst: Boundary particular fluxes. Has length (m // 8).
        v_lst: Particular solutions evaluated on the merge interfaces. Has length (m // 8).
    """

    # Set L_2f1 and L_1f2 in all of the Nodes
    for node in nodes_this_level:
        node.L_2f1 = L_2f1
        node.L_1f2 = L_1f2

    map_out = jax.tree.map(
        node_to_oct_merge_outputs,
        nodes_this_level,
        is_leaf=is_node_type,
    )

    for i, node in enumerate(nodes_this_level):
        # Set the output in the tree object.
        node.DtN = map_out[i][1]
        node.v_prime = map_out[i][2]
        node.S = map_out[i][0]
        node.v_int = map_out[i][3]

        D_shape = map_out[i][0].shape[0]
    return D_shape


def node_to_oct_merge_outputs(
    node: Node,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Expect the first 8 entries of data_lst to be the Nodes we want to merge.
    Expect the next 8 entries to be the DtN matrices for these nodes.
    Expect the next 8 entries to be the v_prime vectors for these nodes.
    """

    # Print index and type of each entry in data_lst
    node_a, node_b, node_c, node_d, node_e, node_f, node_g, node_h = node.children

    side_lens_a = jnp.array(
        [node_a.n_0, node_a.n_1, node_a.n_2, node_a.n_3, node_a.n_4, node_a.n_5]
    )
    side_lens_b = jnp.array(
        [node_b.n_0, node_b.n_1, node_b.n_2, node_b.n_3, node_b.n_4, node_b.n_5]
    )
    side_lens_c = jnp.array(
        [node_c.n_0, node_c.n_1, node_c.n_2, node_c.n_3, node_c.n_4, node_c.n_5]
    )
    side_lens_d = jnp.array(
        [node_d.n_0, node_d.n_1, node_d.n_2, node_d.n_3, node_d.n_4, node_d.n_5]
    )
    side_lens_e = jnp.array(
        [node_e.n_0, node_e.n_1, node_e.n_2, node_e.n_3, node_e.n_4, node_e.n_5]
    )
    side_lens_f = jnp.array(
        [node_f.n_0, node_f.n_1, node_f.n_2, node_f.n_3, node_f.n_4, node_f.n_5]
    )
    side_lens_g = jnp.array(
        [node_g.n_0, node_g.n_1, node_g.n_2, node_g.n_3, node_g.n_4, node_g.n_5]
    )
    side_lens_h = jnp.array(
        [node_h.n_0, node_h.n_1, node_h.n_2, node_h.n_3, node_h.n_4, node_h.n_5]
    )
    need_compression_lsts = find_compression_lists_3D(
        node_a, node_b, node_c, node_d, node_e, node_f, node_g, node_h
    )

    S, T, v_prime_ext, v = _oct_merge(
        T_a=node_a.DtN,
        T_b=node_b.DtN,
        T_c=node_c.DtN,
        T_d=node_d.DtN,
        T_e=node_e.DtN,
        T_f=node_f.DtN,
        T_g=node_g.DtN,
        T_h=node_h.DtN,
        v_prime_a=node_a.v_prime,
        v_prime_b=node_b.v_prime,
        v_prime_c=node_c.v_prime,
        v_prime_d=node_d.v_prime,
        v_prime_e=node_e.v_prime,
        v_prime_f=node_f.v_prime,
        v_prime_g=node_g.v_prime,
        v_prime_h=node_h.v_prime,
        L_2f1=node.L_2f1,
        L_1f2=node.L_1f2,
        need_interp_lsts=need_compression_lsts,
        side_lens_a=side_lens_a,
        side_lens_b=side_lens_b,
        side_lens_c=side_lens_c,
        side_lens_d=side_lens_d,
        side_lens_e=side_lens_e,
        side_lens_f=side_lens_f,
        side_lens_g=side_lens_g,
        side_lens_h=side_lens_h,
    )
    return S, T, v_prime_ext, v
