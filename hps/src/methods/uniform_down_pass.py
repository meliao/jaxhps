from functools import partial
import logging
from typing import Tuple, List

import jax
import jax.numpy as jnp

from hps.src.config import HOST_DEVICE, DEVICE_ARR


def _uniform_down_pass_3D_DtN(
    boundary_data: jnp.ndarray,
    S_maps_lst: List[jnp.ndarray],
    v_int_lst: List[jnp.ndarray],
    leaf_Y_maps: jnp.ndarray,
    v_array: jnp.ndarray,
    device: jax.Device = HOST_DEVICE,
) -> None:
    logging.debug("_down_pass_3D: started")

    # leaf_Y_maps = jax.device_put(leaf_Y_maps, DEVICE)
    # v_array = jax.device_put(v_array, DEVICE)

    n_levels = len(S_maps_lst)
    bdry_data = jnp.expand_dims(boundary_data, axis=0)

    bdry_data = jax.device_put(bdry_data, device)

    S_maps_lst = [jax.device_put(S_arr, device) for S_arr in S_maps_lst]
    v_int_lst = [jax.device_put(v_int, device) for v_int in v_int_lst]
    leaf_Y_maps = jax.device_put(leaf_Y_maps, device)
    v_array = jax.device_put(v_array, device)

    logging.debug("_down_pass_3D: S_maps_lst[0].device %s", S_maps_lst[0].devices())
    logging.debug("_down_pass_3D: v_array.device %s", v_array.devices())
    logging.debug("_down_pass_3D: v_int_lst[0].device %s", v_int_lst[0].devices())
    logging.debug("_down_pass_3D: leaf_Y_maps.device %s", leaf_Y_maps.devices())

    # Change the last entry of the S_maps_lst and v_int_lst to have batch dimension 1
    S_maps_lst[-1] = jnp.expand_dims(S_maps_lst[-1], axis=0)
    v_int_lst[-1] = jnp.expand_dims(v_int_lst[-1], axis=0)

    # Propogate the Dirichlet data down the tree using the S maps.
    for level in range(n_levels - 1, -1, -1):

        S_arr = S_maps_lst[level]
        v_int = v_int_lst[level]

        # print("_down_pass_3D: S_arr.shape = ", S_arr.shape)
        # print("_down_pass_3D: bdry_data.shape = ", bdry_data.shape)
        # print("_down_pass_3D: v_int.shape = ", v_int.shape)

        bdry_data = vmapped_propogate_down_oct(S_arr, bdry_data, v_int)
        # Reshape from (-1, 4, n_bdry) to (-1, n_bdry)
        n_bdry = bdry_data.shape[-1]
        bdry_data = bdry_data.reshape((-1, n_bdry))

    root_dirichlet_data = bdry_data
    leaf_homog_solns = jnp.einsum("ijk,ik->ij", leaf_Y_maps, root_dirichlet_data)
    leaf_solns = leaf_homog_solns + v_array
    leaf_solns = jax.device_put(leaf_solns, HOST_DEVICE)
    return leaf_solns


def _uniform_down_pass_2D_DtN(
    boundary_data: jnp.ndarray,
    S_maps_lst: List[jnp.ndarray],
    v_int_lst: List[jnp.ndarray],
    leaf_Y_maps: jnp.ndarray,
    v_array: jnp.ndarray,
    # D_xx: jnp.ndarray | None = None,
    # D_xy: jnp.ndarray | None = None,
    # D_yy: jnp.ndarray | None = None,
    # D_x: jnp.ndarray | None = None,
    # D_y: jnp.ndarray | None = None,
    # P: jnp.ndarray | None = None,
    # Q_D: jnp.ndarray | None = None,
    # source_term: jnp.ndarray | None = None,
    # D_xx_coeffs: jnp.ndarray | None = None,
    # D_xy_coeffs: jnp.ndarray | None = None,
    # D_yy_coeffs: jnp.ndarray | None = None,
    # D_x_coeffs: jnp.ndarray | None = None,
    # D_y_coeffs: jnp.ndarray | None = None,
    device: jax.Device = HOST_DEVICE,
) -> jnp.array:
    """
    Computes the downward pass of the HPS algorithm. This function takes the Dirichlet data
    at the boundary of the domain and propogates it down the tree to the leaf nodes.

    Expects all data to be on the CPU. It will be moved to the GPU as needed.

    Args:
        boundary_data (jnp.ndarray): Has shape (n_bdry,)
        S_maps_lst (List[jnp.ndarray]): Matrices mapping data from the patch boundary to the merge interfaces. List has length (l - 1).
        v_int_lst (List[jnp.ndarray]): Vectors specifying the particular solution data at the merge interfaces. List has length (l - 1).
        leaf_Y_maps (jnp.ndarray): Matrices mapping the solution to the interior of the leaf nodes. Has shape (n_leaf, p**2, n_bdry).
        v_array (jnp.ndarray): Particular solutions at the interior of the leaves. Has shape (n_leaf, p**2).
        device (jax.Device, optional): Where to run the computation. Defaults to HOST_DEVICE.

    Returns:
        jnp.ndarray: Has shape (n_leaf, p**2). Interior solution at the leaf nodes.
    """
    logging.debug("_down_pass: started")
    # logging.debug("_down_pass: S_maps_lst[0].device %s", S_maps_lst[0].devices())

    boundary_data = jax.device_put(boundary_data, device)
    leaf_Y_maps = jax.device_put(leaf_Y_maps, device)
    S_maps_lst = [jax.device_put(S_arr, device) for S_arr in S_maps_lst]
    v_int_lst = [jax.device_put(v_int, device) for v_int in v_int_lst]

    n_levels = len(S_maps_lst)
    bdry_data = jnp.expand_dims(boundary_data, axis=0)

    # Propogate the Dirichlet data down the tree using the S maps.
    for level in range(n_levels - 1, -1, -1):

        S_arr = S_maps_lst[level]
        v_int = v_int_lst[level]

        # logging.debug("_down_pass: S_arr.shape: %s", S_arr.shape)
        # logging.debug("_down_pass: v_int.shape: %s", v_int.shape)
        # logging.debug("_down_pass: bdry_data.shape: %s", bdry_data.shape)

        bdry_data = vmapped_propogate_down_quad(S_arr, bdry_data, v_int)
        # Reshape from (-1, 4, n_bdry) to (-1, n_bdry)
        n_bdry = bdry_data.shape[-1]
        bdry_data = bdry_data.reshape((-1, n_bdry))

    if leaf_Y_maps is None:
        # Recompute Y maps
        root_dirichlet_data = bdry_data
        # print(
        #     "_down_pass_2D: root_dirichlet_data.devices() = ",
        #     root_dirichlet_data.devices(),
        # )

        # Call local solve stage to recompute Y matrices and solve for the interior solution.
        return _local_solve_stage_2D_chunked(
            D_xx=D_xx,
            D_xy=D_xy,
            D_yy=D_yy,
            D_x=D_x,
            D_y=D_y,
            P=P,
            Q_D=Q_D,
            source_term=source_term,
            D_xx_coeffs=D_xx_coeffs,
            D_xy_coeffs=D_xy_coeffs,
            D_yy_coeffs=D_yy_coeffs,
            D_x_coeffs=D_x_coeffs,
            D_y_coeffs=D_y_coeffs,
            bdry_data=root_dirichlet_data,
        )

    else:
        root_dirichlet_data = bdry_data
        leaf_homog_solns = jnp.einsum("ijk,ik->ij", leaf_Y_maps, root_dirichlet_data)
        leaf_solns = leaf_homog_solns + v_array
        leaf_solns = jax.device_put(leaf_solns, HOST_DEVICE)
        return leaf_solns


def _uniform_down_pass_2D_ItI(
    boundary_imp_data: jnp.ndarray,
    S_maps_lst: List[jnp.ndarray],
    f_lst: List[jnp.ndarray],
    leaf_Y_maps: jnp.ndarray,
    v_array: jnp.ndarray,
    device: jax.Device = DEVICE_ARR[0],
    host_device: jax.Device = HOST_DEVICE,
) -> None:
    """
    This function performs the downward pass of the HPS algorithm.
    Given the tree, which has S maps for the interior nodes and Y maps for
    the leaf nodes, this function will propogate the Dirichlet boundary data
    down to the leaf nodes via the S maps and then to the interior of each
    leaf node via the Y maps.

    This function doesn't return anything but it does modify the tree object
    by setting the following attributes:
    """
    logging.debug("_down_pass: started")

    n_levels = len(S_maps_lst)
    bdry_data = jnp.expand_dims(
        boundary_imp_data,
        axis=[0],
    )
    logging.debug("_uniform_down_pass_2D_ItI: bdry_data.shape = %s", bdry_data.shape)
    if len(bdry_data.shape) == 2:
        # Add a source dimension
        bdry_data = jnp.expand_dims(bdry_data, axis=-1)
    logging.debug("_uniform_down_pass_2D_ItI: bdry_data.shape = %s", bdry_data.shape)

    leaf_Y_maps = jax.device_put(leaf_Y_maps, device)
    bdry_data = jax.device_put(bdry_data, device)
    v_array = jax.device_put(v_array, device)
    S_maps_lst = [jax.device_put(S_arr, device) for S_arr in S_maps_lst]
    f_lst = [jax.device_put(f, device) for f in f_lst]
    # Propogate the Dirichlet data down the tree using the S maps.
    for level in range(n_levels - 1, -1, -1):

        S_arr = S_maps_lst[level]
        f = f_lst[level]

        bdry_data = vmapped_propogate_down_quad_ItI(S_arr, bdry_data, f)
        _, _, n_bdry, n_src = bdry_data.shape
        bdry_data = bdry_data.reshape((-1, n_bdry, n_src))

    # Once we have the leaf node incoming impedance data, compute solution on the interior
    # of each leaf node using the Y maps.
    root_incoming_imp_data = bdry_data

    # logging.debug(
    #     "_uniform_down_pass_2D_ItI: leaf_Y_maps.shape = %s", leaf_Y_maps.shape
    # )
    # logging.debug(
    #     "_uniform_down_pass_2D_ItI: root_incoming_imp_data.shape = %s",
    #     root_incoming_imp_data.shape,
    # )
    leaf_homog_solns = jnp.einsum("ijk,ikl->ijl", leaf_Y_maps, root_incoming_imp_data)
    leaf_solns = leaf_homog_solns + v_array
    leaf_solns = jax.device_put(leaf_solns, host_device)
    return leaf_solns


@jax.jit
def _propogate_down_quad(
    S_arr: jnp.ndarray,
    bdry_data: jnp.ndarray,
    v_int_data: jnp.ndarray,
) -> jnp.ndarray:
    """
    Given homogeneous data on the boundary, interface homogeneous solution operator S, and
    interface particular solution data, this function returns the solution on the boundaries
    of the four children.

    suppose n_child is the number of quadrature points on EACH SIDE of a child node.

    Args:
        S_arr (jnp.ndarray): Has shape (4 * n_child, 8 * n_child)
        bdry_data (jnp.ndarray): 8 * n_child
        v_int_data (jnp.ndarray): 4 * n_child

    Returns:
        jnp.ndarray: Has shape (4, 4 * n_child)
    """

    n_child = bdry_data.shape[0] // 8

    g_int = S_arr @ bdry_data + v_int_data

    # All of these slices of g_int are propogating from OUTSIDE to INSIDE
    g_int_5 = g_int[:n_child]
    g_int_6 = g_int[n_child : 2 * n_child]
    g_int_7 = g_int[2 * n_child : 3 * n_child]
    g_int_8 = g_int[3 * n_child :]

    g_a = jnp.concatenate(
        [
            bdry_data[:n_child],  # S edge
            g_int_5,  # E edge
            jnp.flipud(g_int_8),  # N edge
            bdry_data[7 * n_child :],  # W edge
        ]
    )

    g_b = jnp.concatenate(
        [
            bdry_data[n_child : 3 * n_child],  # S edge, E edge
            g_int_6,  # N edge
            jnp.flipud(g_int_5),  # W edge
        ]
    )

    g_c = jnp.concatenate(
        [
            jnp.flipud(g_int_6),  # S edge
            bdry_data[3 * n_child : 5 * n_child],  # E edge, N edge
            g_int_7,  # W edge
        ]
    )

    g_d = jnp.concatenate(
        [
            g_int_8,  # S edge
            jnp.flipud(g_int_7),  # E edge
            bdry_data[5 * n_child : 7 * n_child],  # N edge, W edge
        ]
    )
    return jnp.stack([g_a, g_b, g_c, g_d])


vmapped_propogate_down_quad = jax.vmap(
    _propogate_down_quad, in_axes=(0, 0, 0), out_axes=0
)


@jax.jit
def _propogate_down_quad_ItI(
    S_arr: jnp.ndarray,
    bdry_data: jnp.ndarray,
    f_data: jnp.ndarray,
) -> jnp.ndarray:
    """
    Given homogeneous data on the boundary, interface homogeneous solution operator S, and
    interface particular solution data, this function returns the solution on the boundaries
    of the four children.

    suppose n_child is the number of quadrature points on EACH SIDE of a child node.

    Args:
        S_arr (jnp.ndarray): Has shape (8 * n_child, 8 * n_child)
        bdry_data (jnp.ndarray): 8 * n_child
        f_data (jnp.ndarray): 8 * n_child

    Returns:
        jnp.ndarray: Has shape (4, 4 * n_child)
    """

    n_child = bdry_data.shape[0] // 8

    t_int_homog = S_arr @ bdry_data
    # print("_propogate_down_quad_ItI: t_int_homog.shape ", t_int_homog.shape)
    # print("_propogate_down_quad_ItI: f_data.shape ", f_data.shape)
    t_int = t_int_homog + f_data

    # All of these slices of g_int are propogating from OUTSIDE to INSIDE
    t_a_5 = t_int[:n_child]
    t_b_5 = jnp.flipud(t_int[n_child : 2 * n_child])
    t_b_6 = t_int[2 * n_child : 3 * n_child]
    t_c_6 = jnp.flipud(t_int[3 * n_child : 4 * n_child])
    t_c_7 = t_int[4 * n_child : 5 * n_child]
    t_d_7 = jnp.flipud(t_int[5 * n_child : 6 * n_child])
    t_d_8 = t_int[6 * n_child : 7 * n_child]
    t_a_8 = jnp.flipud(t_int[7 * n_child :])

    g_a = jnp.concatenate(
        [
            bdry_data[:n_child],  # S edge
            t_a_5,  # E edge
            t_a_8,  # N edge
            bdry_data[7 * n_child :],  # W edge
        ]
    )

    g_b = jnp.concatenate(
        [
            bdry_data[n_child : 3 * n_child],  # S edge, E edge
            t_b_6,  # N edge
            t_b_5,  # W edge
        ]
    )

    g_c = jnp.concatenate(
        [
            t_c_6,  # S edge
            bdry_data[3 * n_child : 5 * n_child],  # E edge, N edge
            t_c_7,  # W edge
        ]
    )

    g_d = jnp.concatenate(
        [
            t_d_8,  # S edge
            t_d_7,  # E edge
            bdry_data[5 * n_child : 7 * n_child],  # N edge, W edge
        ]
    )
    return jnp.stack([g_a, g_b, g_c, g_d])


vmapped_propogate_down_quad_ItI = jax.vmap(
    _propogate_down_quad_ItI, in_axes=(0, 0, 0), out_axes=0
)


@jax.jit
def _propogate_down_oct(
    S_arr: jnp.ndarray,
    bdry_data: jnp.ndarray,
    v_int_data: jnp.ndarray,
) -> jnp.ndarray:
    """_summary_

    Args:
        S_arr (jnp.ndarray): Has shape (12 * n_per_face, 24 * n_per_face)
        bdry_data (jnp.ndarray): Has shape (24 * n_per_face,)
        v_int_data (jnp.ndarray): Has shape (12 * n_per_face,)

    Returns:
        jnp.ndarray: Has shape (8, 6 * n_per_face)
    """
    n_per_face = bdry_data.shape[0] // 24

    n = 4 * n_per_face

    g_int = S_arr @ bdry_data + v_int_data

    g_int_9 = g_int[:n_per_face]
    g_int_10 = g_int[n_per_face : 2 * n_per_face]
    g_int_11 = g_int[2 * n_per_face : 3 * n_per_face]
    g_int_12 = g_int[3 * n_per_face : 4 * n_per_face]
    g_int_13 = g_int[4 * n_per_face : 5 * n_per_face]
    g_int_14 = g_int[5 * n_per_face : 6 * n_per_face]
    g_int_15 = g_int[6 * n_per_face : 7 * n_per_face]
    g_int_16 = g_int[7 * n_per_face : 8 * n_per_face]
    g_int_17 = g_int[8 * n_per_face : 9 * n_per_face]
    g_int_18 = g_int[9 * n_per_face : 10 * n_per_face]
    g_int_19 = g_int[10 * n_per_face : 11 * n_per_face]
    g_int_20 = g_int[11 * n_per_face :]

    bdry_data_0 = bdry_data[:n]
    bdry_data_1 = bdry_data[n : 2 * n]
    bdry_data_2 = bdry_data[2 * n : 3 * n]
    bdry_data_3 = bdry_data[3 * n : 4 * n]
    bdry_data_4 = bdry_data[4 * n : 5 * n]
    bdry_data_5 = bdry_data[5 * n : 6 * n]

    g_a = jnp.concatenate(
        [
            bdry_data_0[-n_per_face:],
            g_int_9,
            bdry_data_2[-n_per_face:],
            g_int_12,
            g_int_17,
            bdry_data_5[:n_per_face],
        ]
    )

    g_b = jnp.concatenate(
        [
            g_int_9,
            bdry_data_1[-n_per_face:],
            bdry_data_2[2 * n_per_face : 3 * n_per_face],
            g_int_10,
            g_int_18,
            bdry_data_5[n_per_face : 2 * n_per_face],
        ]
    )

    g_c = jnp.concatenate(
        [
            g_int_11,
            bdry_data_1[2 * n_per_face : 3 * n_per_face],
            g_int_10,
            bdry_data_3[2 * n_per_face : 3 * n_per_face],
            g_int_19,
            bdry_data_5[2 * n_per_face : 3 * n_per_face],
        ]
    )

    g_d = jnp.concatenate(
        [
            bdry_data_0[2 * n_per_face : 3 * n_per_face],
            g_int_11,
            g_int_12,
            bdry_data_3[-n_per_face:],
            g_int_20,
            bdry_data_5[3 * n_per_face : 4 * n_per_face],
        ]
    )

    g_e = jnp.concatenate(
        [
            bdry_data_0[:n_per_face],
            g_int_13,
            bdry_data_2[:n_per_face],
            g_int_16,
            bdry_data_4[:n_per_face],
            g_int_17,
        ]
    )

    g_f = jnp.concatenate(
        [
            g_int_13,
            bdry_data_1[:n_per_face],
            bdry_data_2[n_per_face : 2 * n_per_face],
            g_int_14,
            bdry_data_4[n_per_face : 2 * n_per_face],
            g_int_18,
        ]
    )

    g_g = jnp.concatenate(
        [
            g_int_15,
            bdry_data_1[n_per_face : 2 * n_per_face],
            g_int_14,
            bdry_data_3[n_per_face : 2 * n_per_face],
            bdry_data_4[2 * n_per_face : 3 * n_per_face],
            g_int_19,
        ]
    )

    g_h = jnp.concatenate(
        [
            bdry_data_0[n_per_face : 2 * n_per_face],
            g_int_15,
            g_int_16,
            bdry_data_3[:n_per_face],
            bdry_data_4[3 * n_per_face : 4 * n_per_face],
            g_int_20,
        ]
    )

    return jnp.stack([g_a, g_b, g_c, g_d, g_e, g_f, g_g, g_h])


vmapped_propogate_down_oct = jax.vmap(
    _propogate_down_oct, in_axes=(0, 0, 0), out_axes=0
)
