from typing import List, Tuple
import jax
import jax.numpy as jnp
from .._pdeproblem import PDEProblem
import logging


def up_pass_uniform_2D_ItI(
    source: jax.Array,
    pde_problem: PDEProblem,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
    return_bdry_imp_data: bool = False,
) -> Tuple[jax.Array, List[jax.Array], jax.Array]:
    """
    This function performs the upward pass for 2D ItI problems. It recomputes the local solve stage to get
    outgoing impedance data from the particular solution, which is now known because the source is specified.



    Parameters
    ----------
    source : jax.Array
        Specifies the source term(s). Can have shape (nleaves, p^2) or (nleaves, p^2, nsrc).

    pde_probem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.

    device : jax.Device, optional
        Where to perform the computation. Defaults to ``jax.devices()[0]``.

    host_device : jax.Device, optional
        Where to place the output. Defaults to ``jax.devices("cpu")[0]``.

    return_bdry_imp_data : bool, optional
        If True, return outgoing and incoming impedance data for the particular solution evaluated at the domain boundary.

    Returns
    -------
    v : jax.Array
        Leaf-level particular solutions. Has shape (n_leaves, p^2)

    g_tilde_lst : List[jax.Array]
        List of pre-computed g_tilde matrices for each level of the quadtree.

    h_last : jax.Array
        Outgoing impedance data for the particular solution evaluated at the domain boundary. Has shape (nbdry, nsrc). Is only returned if ``return_bdry_imp_data=True``.

    g_last : jax.Array
        Incoming impedance data for the particular solution evaluated at the domain boundary. Has shape (nbdry, nsrc). Is only returned if ``return_bdry_imp_data=True``.
    """

    bool_multi_source = source.ndim == 3

    if not bool_multi_source:
        # Add a source dimension to the source term
        source = jnp.expand_dims(source, axis=-1)
        logging.debug(
            "up_pass_uniform_2D_ItI: new source shape = %s", source.shape
        )

    # Re-do a full local solve.
    # pde_problem.source = source

    # Get the saved D_inv_lst and BD_inv_lst
    D_inv_lst = pde_problem.D_inv_lst
    BD_inv_lst = pde_problem.BD_inv_lst

    n_ext = 4 * pde_problem.domain.p - 4
    source_int = source[:, n_ext:]
    logging.debug(
        "up_pass_uniform_2D_ItI: pde_problem.Phi shape = %s",
        pde_problem.Phi.shape,
    )
    logging.debug(
        "up_pass_uniform_2D_ItI: source shape = %s", source_int.shape
    )
    # Get the particular solution. It's an einsum of the source term with
    # the Phi array stored in the PDEProblem.
    v = jnp.einsum("ijk,ikl->ijl", pde_problem.Phi, source_int)
    logging.debug(
        "up_pass_uniform_2D_ItI: after local solve, v shape = %s", v.shape
    )
    logging.debug(
        "up_pass_uniform_2D_ItI: QH shape = %s",
        pde_problem.QH.shape,
    )

    # Get leaf-level h_in array, which is an einsum between the particular soln and
    # the QH array stored in the PDEProblem.
    h_in = jnp.einsum("ij,kjl->kil", pde_problem.QH, v)
    g_in = jnp.einsum("ij,kjl->kil", pde_problem.QG, v)

    logging.debug(
        "up_pass_uniform_2D_ItI: after local solve, h_in shape = %s",
        h_in.shape,
    )

    g_tilde_lst = []

    for i in range(len(D_inv_lst)):
        # Get h and g_tilde for this level

        nnodes, nbdry, nsrc = h_in.shape
        h_in = h_in.reshape(nnodes // 4, 4, nbdry, nsrc)
        g_in = g_in.reshape(nnodes // 4, 4, nbdry, nsrc)
        D_inv = D_inv_lst[i]
        BD_inv = BD_inv_lst[i]

        h_in, g_in, g_tilde, h_ext_child = vmapped_assemble_boundary_data(
            h_in, g_in, D_inv, BD_inv
        )
        g_tilde_lst.append(g_tilde)

    logging.debug(
        "up_pass_uniform_2D_ItI: g_tilde_lst shapes = %s",
        [g.shape for g in g_tilde_lst],
    )
    logging.debug("up_pass_uniform_2D_ItI: h_in shape = %s", h_in.shape)

    # Remove the source dimension if it's not a multi-source problem
    if not bool_multi_source:
        v = jnp.squeeze(v, axis=-1)
        g_tilde_lst = [
            jnp.squeeze(g_tilde, axis=-1) for g_tilde in g_tilde_lst
        ]
        h_in = jnp.squeeze(h_in, axis=-1)
        g_in = jnp.squeeze(g_in, axis=-1)
        # h_ext_child = jnp.squeeze(h_ext_child, axis=-1)
        logging.debug(
            "up_pass_uniform_2D_ItI: it's not multi source so squeezing h_in to shape %s",
            h_in.shape,
        )

    out = (v, g_tilde_lst)

    if return_bdry_imp_data:
        out = (
            v,
            g_tilde_lst,
            jnp.squeeze(h_in, axis=0),
            jnp.squeeze(g_in, axis=0),
        )

    return out


@jax.jit
def assemble_boundary_data(
    h_in: jax.Array,
    g_in: jax.Array,
    D_inv: jax.Array,
    BD_inv: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """


    Args:
        h_in (jax.Array): Has shape (4, 4 * nside, n_src) where nside is the number of discretization points along each side of the nodes being merged.
        g_in (jax.Array): Has shape (4, 4 * nside, n_src) where nside is the number of discretization points along each side of the nodes being merged.
        D_inv (jax.Array): Has shape (8 * nside, 8 * nside)
        BD_inv (jax.Array): Has shape (8 * nside, 8 * nside)

    Returns:
        h : jax.Array
            Has shape (8 * nside, n_src) and is the outgoing impedance data due to the particular solution on the merged node.
        g : jax.Array
            Has shape (8 * nside, n_src) and is the incoming impedance data due to the particular solution on the merged nodes.
        g_tilde : jax.Array
            Has shape (8 * nside, n_src) and is the incoming impedance data due to the particular solution on the merged node, evaluated along the merge interfaces.
    """

    nside = h_in.shape[1] // 4

    # Remember, the slices along the merge interface go from OUTSIDE to INSIDE
    h_a = h_in[0]
    h_a_1 = jnp.concatenate([h_a[-nside:], h_a[:nside]])
    h_a_5 = h_a[nside : 2 * nside]
    h_a_8 = jnp.flipud(h_a[2 * nside : 3 * nside])

    h_b = h_in[1]
    h_b_2 = h_b[: 2 * nside]
    h_b_6 = h_b[2 * nside : 3 * nside]
    h_b_5 = jnp.flipud(h_b[3 * nside : 4 * nside])

    h_c = h_in[2]
    h_c_6 = jnp.flipud(h_c[:nside])
    h_c_3 = h_c[nside : 3 * nside]
    h_c_7 = h_c[3 * nside : 4 * nside]

    h_d = h_in[3]
    h_d_8 = h_d[:nside]
    h_d_7 = jnp.flipud(h_d[nside : 2 * nside])
    h_d_4 = h_d[2 * nside : 4 * nside]

    h_int_child = jnp.concatenate(
        [h_b_5, h_d_8, h_b_6, h_d_7, h_a_5, h_c_6, h_c_7, h_a_8]
    )

    h_ext_child = jnp.concatenate([h_a_1, h_b_2, h_c_3, h_d_4])

    # Break up the g array in the same way to form g_int_child and g_ext_child
    g_a = g_in[0]
    g_a_1 = jnp.concatenate([g_a[-nside:], g_a[:nside]])
    g_a_5 = g_a[nside : 2 * nside]
    g_a_8 = jnp.flipud(g_a[2 * nside : 3 * nside])
    g_b = g_in[1]
    g_b_2 = g_b[: 2 * nside]
    g_b_6 = g_b[2 * nside : 3 * nside]
    g_b_5 = jnp.flipud(g_b[3 * nside : 4 * nside])
    g_c = g_in[2]
    g_c_6 = jnp.flipud(g_c[:nside])
    g_c_3 = g_c[nside : 3 * nside]
    g_c_7 = g_c[3 * nside : 4 * nside]
    g_d = g_in[3]
    g_d_8 = g_d[:nside]
    g_d_7 = jnp.flipud(g_d[nside : 2 * nside])
    g_d_4 = g_d[2 * nside : 4 * nside]
    g_int_child = jnp.concatenate(
        [g_b_5, g_d_8, g_b_6, g_d_7, g_a_5, g_c_6, g_c_7, g_a_8]
    )
    g_ext_child = jnp.concatenate([g_a_1, g_b_2, g_c_3, g_d_4])

    g_tilde = -1 * D_inv @ h_int_child

    # g_tilde is ordered like a_5, a_8, c_6, c_7, b_5, b_6, d_7, d_8.
    # Want to rearrange it so it's ordered like
    # a_5, b_5, b_6, c_6, c_7, d_7, d_8, a_8
    r = jnp.concatenate(
        [
            jnp.arange(nside),  # a5
            jnp.arange(4 * nside, 5 * nside),  # b5
            jnp.arange(5 * nside, 6 * nside),  # b6
            jnp.arange(2 * nside, 3 * nside),  # c6
            jnp.arange(3 * nside, 4 * nside),  # c7
            jnp.arange(6 * nside, 7 * nside),  # d7
            jnp.arange(7 * nside, 8 * nside),  # d8
            jnp.arange(nside, 2 * nside),  # a8
        ]
    )
    g_tilde = g_tilde[r]

    h = h_ext_child - BD_inv @ h_int_child

    g = g_ext_child - BD_inv @ g_int_child

    # h is ordered like h_a_1, h_b_2, h_c_3, h_d_4. Need to
    # roll it so that it's ordered like [bottom, left, right, top].
    h = jnp.roll(h, -nside, axis=0)
    g = jnp.roll(g, -nside, axis=0)

    return (h, g, g_tilde, h_ext_child)


vmapped_assemble_boundary_data = jax.vmap(
    assemble_boundary_data, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0)
)
