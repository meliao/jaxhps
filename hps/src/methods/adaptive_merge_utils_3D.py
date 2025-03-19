from typing import List, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from hps.src.quadrature.trees import Node, get_all_leaves
from hps.src.quadrature.quad_3D.grid_creation import get_ordered_lst_of_boundary_nodes
# from hps.src.methods.schur_complement import assemble_merge_outputs_DtN
from hps.src.config import DEVICE_ARR, HOST_DEVICE


def _compression_lst(
    lst_0: List[Node], lst_1: List[Node]
) -> Tuple[jnp.array, jnp.array]:
    """
    Repeated logic for all of the _find_compression_list_* functions.
    This function takes two lists of nodes and returns vectors of booleans
    specifying which panels need to be compressed.
    """
    out_0 = []
    out_1 = []

    idx_0 = 0
    idx_1 = 0
    while idx_0 < len(lst_0) and idx_1 < len(lst_1):
        xlen_0 = lst_0[idx_0].xmax - lst_0[idx_0].xmin
        xlen_1 = lst_1[idx_1].xmax - lst_1[idx_1].xmin

        if xlen_0 == xlen_1:
            # These leaves match up.
            out_0.append(False)
            out_1.append(False)
            idx_0 += 1
            idx_1 += 1

        elif xlen_0 < xlen_1:
            # Node 0 needs the next 4 leaves compressed.
            out_0.append(True)
            out_0.append(True)
            out_0.append(True)
            out_0.append(True)

            idx_0 += 4

            out_1.append(False)
            idx_1 += 1
        else:
            # Node 1 needs the next 4 leaves compressed.
            out_1.append(True)
            out_1.append(True)
            out_1.append(True)
            out_1.append(True)

            idx_1 += 4

            out_0.append(False)
            idx_0 += 1
    return jnp.array(out_0), jnp.array(out_1)


def find_compression_lists_3D(
    node_a: Node,
    node_b: Node,
    node_c: Node,
    node_d: Node,
    node_e: Node,
    node_f: Node,
    node_g: Node,
    node_h: Node,
) -> Tuple[jnp.array]:
    """
    Returns a boolean array that indicates which parts of the merge interface need comporession from 2 panels to 1 panel.

    List the merge interface as follows:
    a_9, b_9, b_10, c_10, c_11, d_11, d_12, a_12,
    e_13, f_13, f_14, g_14, g_15, h_15, h_16, e_16,
    a_17, e_17, b_18, f_18, c_19, g_19, d_20, h_20,

    For each of the 8 merge interfaces, we need to determine which pairs of panels need compression. So this function
    should return a list of booleans for each of the 8 merge interfaces. Each boolean indicates whether the corresponding
    panel needs compression.

    Args:
        nodes_this_level (List[Node]): Has length (n,) and contains the nodes at the current merge level.

    Returns:
        jnp.array: Output has shape shape (n // 4, 12)
    """
    face_leaves_a = get_ordered_lst_of_boundary_nodes(node_a)
    leaves_a_9 = face_leaves_a[1]
    leaves_a_12 = face_leaves_a[3]
    leaves_a_17 = face_leaves_a[4]

    face_leaves_b = get_ordered_lst_of_boundary_nodes(node_b)
    leaves_b_9 = face_leaves_b[0]
    leaves_b_10 = face_leaves_b[3]
    leaves_b_18 = face_leaves_b[4]

    face_leaves_c = get_ordered_lst_of_boundary_nodes(node_c)
    leaves_c_10 = face_leaves_c[2]
    leaves_c_11 = face_leaves_c[0]
    leaves_c_19 = face_leaves_c[4]

    face_leaves_d = get_ordered_lst_of_boundary_nodes(node_d)
    leaves_d_11 = face_leaves_d[1]
    leaves_d_12 = face_leaves_d[2]
    leaves_d_20 = face_leaves_d[4]

    face_leaves_e = get_ordered_lst_of_boundary_nodes(node_e)
    leaves_e_13 = face_leaves_e[1]
    leaves_e_16 = face_leaves_e[3]
    leaves_e_17 = face_leaves_e[5]

    face_leaves_f = get_ordered_lst_of_boundary_nodes(node_f)
    leaves_f_13 = face_leaves_f[0]
    leaves_f_14 = face_leaves_f[3]
    leaves_f_18 = face_leaves_f[5]

    face_leaves_g = get_ordered_lst_of_boundary_nodes(node_g)
    leaves_g_14 = face_leaves_g[2]
    leaves_g_15 = face_leaves_g[0]
    leaves_g_19 = face_leaves_g[5]

    face_leaves_h = get_ordered_lst_of_boundary_nodes(node_h)
    leaves_h_15 = face_leaves_h[1]
    leaves_h_16 = face_leaves_h[2]
    leaves_h_20 = face_leaves_h[5]

    lst_a_9, lst_b_9 = _compression_lst(leaves_a_9, leaves_b_9)
    lst_b_10, lst_c_10 = _compression_lst(leaves_b_10, leaves_c_10)
    lst_c_11, lst_d_11 = _compression_lst(leaves_c_11, leaves_d_11)
    lst_d_12, lst_a_12 = _compression_lst(leaves_d_12, leaves_a_12)
    lst_e_13, lst_f_13 = _compression_lst(leaves_e_13, leaves_f_13)
    lst_f_14, lst_g_14 = _compression_lst(leaves_f_14, leaves_g_14)
    lst_g_15, lst_h_15 = _compression_lst(leaves_g_15, leaves_h_15)
    lst_h_16, lst_e_16 = _compression_lst(leaves_h_16, leaves_e_16)
    lst_a_17, lst_e_17 = _compression_lst(leaves_a_17, leaves_e_17)
    lst_b_18, lst_f_18 = _compression_lst(leaves_b_18, leaves_f_18)
    lst_c_19, lst_g_19 = _compression_lst(leaves_c_19, leaves_g_19)
    lst_d_20, lst_h_20 = _compression_lst(leaves_d_20, leaves_h_20)

    return (
        lst_a_9,  # 0. Writing down the indexes to make it easier to read.
        lst_b_9,  # 1
        lst_b_10,  # 2
        lst_c_10,  # 3
        lst_c_11,  # 4
        lst_d_11,  # 5
        lst_d_12,  # 6
        lst_a_12,  # 7
        lst_e_13,  # 8
        lst_f_13,  # 9
        lst_f_14,  # 10
        lst_g_14,  # 11
        lst_g_15,  # 12
        lst_h_15,  # 13
        lst_h_16,  # 14
        lst_e_16,  # 15
        lst_a_17,  # 16
        lst_e_17,  # 17
        lst_b_18,  # 18
        lst_f_18,  # 19
        lst_c_19,  # 20
        lst_g_19,  # 21
        lst_d_20,  # 22
        lst_h_20,  # 23
    )
