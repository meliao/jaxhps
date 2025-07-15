import jax.numpy as jnp
import jax

from .._pdeproblem import PDEProblem
from typing import Tuple
import logging


def local_solve_stage_uniform_2D_DtN(
    pde_problem: PDEProblem,
    host_device: jax.Device = jax.devices("cpu")[0],
    device: jax.Device = jax.devices()[0],
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    This function performs the local solve stage for 2D problems with a uniform quadtree, creating DtN matrices.

    Parameters
    ----------
    pde_problem : PDEProblem
        Specifies the discretization, differential operator, source function, and keeps track of the pre-computed differentiation and interpolation matrices.
    device : jax.Device
        Where to perform the computation. Defaults to ``jax.devices()[0]``.
    host_device : jax.Device
        Where to place the output. Defaults to ``jax.devices("cpu")[0]``.

    Returns
    -------
    Y : jax.Array
        Solution operators mapping from Dirichlet boundary data to homogeneous solutions on the leaf interiors. Has shape (n_leaves, p^2, 4q)
    T : jax.Array
        Dirichlet-to-Neumann matrices for each leaf. Has shape (n_leaves, 4q, 4q)
    v : jax.Array
        Leaf-level particular solutions. Has shape (n_leaves, p^2) or (n_leaves, p**2, n_src) if multi-source.
    h : jax.Array
        Outgoing boundary data. This is the outward-pointing normal derivative of the particular solution. Has shape (n_leaves, 4q) or (n_leaves, 4q, n_src) if multi-source.
    """
    logging.debug("local_solve_stage_uniform_2D_DtN: started")

    # Gather the coefficients into a single array.
    coeffs_gathered, which_coeffs = _gather_coeffs_2D(
        D_xx_coeffs=pde_problem.D_xx_coefficients,
        D_xy_coeffs=pde_problem.D_xy_coefficients,
        D_yy_coeffs=pde_problem.D_yy_coefficients,
        D_x_coeffs=pde_problem.D_x_coefficients,
        D_y_coeffs=pde_problem.D_y_coefficients,
        I_coeffs=pde_problem.I_coefficients,
    )
    source_term = pde_problem.source
    logging.debug(
        "local_solve_stage_uniform_2D_DtN: input source_term devices = %s",
        source_term.devices(),
    )
    source_term = jax.device_put(
        source_term,
        device,
    )
    bool_multi_source = source_term.ndim == 3
    # stack the precomputed differential operators into a single array

    if pde_problem.bool_rectangular_spectral_collocation:
        logging.debug(
            "local_solve_stage_uniform_2D_DtN: using rectangular spectral collocation"
        )
        diff_ops = jnp.stack(
            [
                pde_problem.D_xx,
                pde_problem.D_xy,
                pde_problem.D_yy,
                pde_problem.D_x,
                pde_problem.D_y,
                pde_problem.B,
            ]
        )
    else:
        logging.debug(
            "local_solve_stage_uniform_2D_DtN: using standard operators"
        )
        diff_ops = jnp.stack(
            [
                pde_problem.D_xx,
                pde_problem.D_xy,
                pde_problem.D_yy,
                pde_problem.D_x,
                pde_problem.D_y,
                jnp.eye(pde_problem.D_xx.shape[0], dtype=jnp.float64),
            ]
        )

    # Put the input data on the device
    coeffs_gathered = jax.device_put(
        coeffs_gathered,
        device,
    )
    if pde_problem.bool_rectangular_spectral_collocation:
        diff_operators = vmapped_assemble_diff_operator_rectangular(
            coeffs_gathered, which_coeffs, diff_ops
        )
    else:
        diff_operators = vmapped_assemble_diff_operator(
            coeffs_gathered, which_coeffs, diff_ops
        )
    if not bool_multi_source:
        source_term = jnp.expand_dims(source_term, axis=-1)

    if pde_problem.bool_rectangular_spectral_collocation:
        Y_arr, T_arr, v, h = vmapped_get_DtN_uniform_rectangular(
            source_term,
            diff_operators,
            pde_problem.Q,
            pde_problem.P,
            pde_problem.B,
        )
    else:
        Y_arr, T_arr, v, h = vmapped_get_DtN_uniform(
            source_term, diff_operators, pde_problem.Q, pde_problem.P
        )

    if not bool_multi_source:
        h = h[..., 0]
        v = v[..., 0]

    # Return data to the requested device
    T_arr_host = jax.device_put(T_arr, host_device)
    del T_arr
    v_host = jax.device_put(v, host_device)
    del v
    h_host = jax.device_put(h, host_device)
    del h
    Y_arr_host = jax.device_put(Y_arr, host_device)
    del Y_arr

    # Return the DtN arrays, particular solutions, particular
    # solution fluxes, and the solution operators. The solution
    # operators are not moved to the host.
    return Y_arr_host, T_arr_host, v_host, h_host


@jax.jit
def _gather_coeffs_2D(
    D_xx_coeffs: jax.Array | None = None,
    D_xy_coeffs: jax.Array | None = None,
    D_yy_coeffs: jax.Array | None = None,
    D_x_coeffs: jax.Array | None = None,
    D_y_coeffs: jax.Array | None = None,
    I_coeffs: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """If not None, expects each input to have shape (n_leaf_nodes, p**2).

    Returns:
        Tuple[jax.Array, jax.Array]: coeffs_gathered and which_coeffs
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
    coeffs_gathered = jnp.array(
        [coeff for coeff in coeffs_lst if coeff is not None]
    )
    return coeffs_gathered, which_coeffs


@jax.jit
def _add(
    out: jax.Array,
    coeff: jax.Array,
    diff_op: jax.Array,
) -> jax.Array:
    """One branch of add_or_not. Expects out to have shape (p**2, p**2), coeff has shape (p**2), diff_op has shape (p**2, p**2)."""
    # res = out + jnp.diag(coeff) @ diff_op
    res = out + jnp.einsum("ab,a->ab", diff_op, coeff)
    return res


@jax.jit
def _not(out: jax.Array, coeff: jax.Array, diff_op: jax.Array) -> jax.Array:
    return out


@jax.jit
def add_or_not(
    i: int,
    carry_tuple: Tuple[jax.Array, jax.Array, jax.Array, jax.Array, int],
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
    coeffs_arr: jax.Array,
    which_coeffs: jax.Array,
    diff_ops: jax.Array,
) -> jax.Array:
    """Given an array of coefficients, this function assembles the differential operator.

    Args:
        coeffs_arr (jax.Array): Has shape (?, p**2).
        which_coeffs (jax.Array): Has shape (5,) or (9,) and specifies which coefficients are not None.
        diff_ops (jax.Array): Has shape (6, p**2, p**2). Contains the precomputed differential operators.

    Returns:
        jax.Array: Has shape (p**2, p**2).
    """

    n_loops = which_coeffs.shape[0]

    # Initialize with the shape of diff_ops but the data type of coeffs_arr.
    # This is important because we may want to use complex coefficients.
    out = jnp.zeros_like(diff_ops[0], dtype=coeffs_arr.dtype)

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


@jax.jit
def get_DtN(
    source_term: jax.Array,
    diff_operator: jax.Array,
    Q: jax.Array,
    P: jax.Array,
) -> Tuple[jax.Array]:
    """
    Args:
        source_term (jax.Array): Array of size (p**2, n_src) containing the source term.
        diff_operator (jax.Array): Array of size (p**2, p**2) containing the local differential operator defined on the
                    Cheby grid.
        Q (jax.Array): Array of size (4q, p**2) containing the matrix interpolating from a soln on the interior
                    to that soln's boundary fluxes on the Gauss boundary.
        P (jax.Array): Array of size (4(p-1), 4q) containing the matrix interpolating from the Gauss to the Cheby boundary.

    Returns:
        Tuple[jax.Array, jax.Array]:
            Y (jax.Array): Matrix of size (p**2, 4q). This is the "DtSoln" map,
                which maps incoming Dirichlet data on the boundary Gauss nodes to the solution on the Chebyshev nodes.
            T (jax.Array): Matrix of size (4q, 4q). This is the "DtN" map, which maps incoming Dirichlet
                data on the boundary Gauss nodes to the normal derivatives on the boundary Gauss nodes.
            v (jax.Array): Array of size (p**2,) containing the particular solution.
            h (jax.Array): Array of size (4q,) containing the outgoing boundary normal derivatives of the particular solution.
    """
    n_cheby_bdry = P.shape[0]
    n_src = source_term.shape[-1]

    A_ii = diff_operator[n_cheby_bdry:, n_cheby_bdry:]
    A_ii_inv = jnp.linalg.inv(A_ii)
    # A_ie shape (n_cheby_int, n_cheby_bdry)
    A_ie = diff_operator[n_cheby_bdry:, :n_cheby_bdry]
    L_2 = jnp.zeros((diff_operator.shape[0], n_cheby_bdry), dtype=jnp.float64)
    L_2 = L_2.at[:n_cheby_bdry].set(jnp.eye(n_cheby_bdry))
    soln_operator = -1 * A_ii_inv @ A_ie
    L_2 = L_2.at[n_cheby_bdry:].set(soln_operator)
    Y = L_2 @ P
    T = Q @ Y

    v = jnp.zeros((diff_operator.shape[0], n_src), dtype=jnp.float64)
    v = v.at[n_cheby_bdry:].set(A_ii_inv @ source_term[n_cheby_bdry:])
    h = Q @ v

    return Y, T, v, h


vmapped_get_DtN_uniform = jax.vmap(
    get_DtN,
    in_axes=(0, 0, None, None),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def get_DtN_rectangular(
    source_term: jax.Array,
    diff_operator: jax.Array,
    Q: jax.Array,
    P: jax.Array,
    B: jax.Array,
) -> Tuple[jax.Array]:
    """
    Args:
        source_term (jax.Array): Array of size (p**2, n_src) containing the source term.
        diff_operator (jax.Array): Array of size ((p-2)**2, p**2) containing the local differential operator defined on the
                    Cheby grid.
        Q (jax.Array): Array of size (4q, p**2) containing the matrix interpolating from a soln on the interior
                    to that soln's boundary fluxes on the Gauss boundary.
        P (jax.Array): Array of size (4(p-1), 4q) containing the matrix interpolating from the Gauss to the Cheby boundary.
        B (jax.Array): Array of size ((p-2)**2, p**2) containing the operator interpolating from the Chebyshev grid of the 2nd kind
                    to the Chebyshev grid of the 1st kind.

    Returns:
        Tuple[jax.Array, jax.Array]:
            Y (jax.Array): Matrix of size (p**2, 4q). This is the "DtSoln" map,
                which maps incoming Dirichlet data on the boundary Gauss nodes to the solution on the Chebyshev nodes.
            T (jax.Array): Matrix of size (4q, 4q). This is the "DtN" map, which maps incoming Dirichlet
                data on the boundary Gauss nodes to the normal derivatives on the boundary Gauss nodes.
            v (jax.Array): Array of size (p**2,) containing the particular solution.
            h (jax.Array): Array of size (4q,) containing the outgoing boundary normal derivatives of the particular solution.
    """
    n_cheby_bdry = P.shape[0]
    n_src = source_term.shape[-1]

    A_i = diff_operator[:, n_cheby_bdry:]
    A_i_inv = jnp.linalg.inv(A_i)
    # A_ie shape (n_cheby_int, n_cheby_bdry)
    A_e = diff_operator[:, :n_cheby_bdry]
    L_2 = jnp.zeros((diff_operator.shape[1], n_cheby_bdry), dtype=jnp.float64)
    L_2 = L_2.at[:n_cheby_bdry].set(jnp.eye(n_cheby_bdry))
    soln_operator = -1 * A_i_inv @ A_e
    L_2 = L_2.at[n_cheby_bdry:].set(soln_operator)
    Y = L_2 @ P
    T = Q @ Y

    v = jnp.zeros((diff_operator.shape[1], n_src), dtype=jnp.float64)
    proj_src = B @ source_term
    v = v.at[n_cheby_bdry:].set(A_i_inv @ proj_src)
    h = Q @ v

    return Y, T, v, h


vmapped_get_DtN_uniform_rectangular = jax.vmap(
    get_DtN_rectangular,
    in_axes=(0, 0, None, None, None),
    out_axes=(0, 0, 0, 0),
)


@jax.jit
def _add_rectangular(
    out: jax.Array,
    coeff: jax.Array,
    diff_op: jax.Array,
    B: jax.Array,
) -> jax.Array:
    """One branch of add_or_not. Expects out to have shape ((p-2)**2, p**2),
    coeff has shape (p**2), diff_op has shape ((p-2)**2, p**2)."""
    # res = out + jnp.diag(coeff) @ diff_op
    coeff_proj = B @ coeff

    res = out + jnp.einsum("ab,a->ab", diff_op, coeff_proj)
    return res


@jax.jit
def _not_rectangular(
    out: jax.Array, coeff: jax.Array, diff_op: jax.Array, B: jax.Array
) -> jax.Array:
    return out


@jax.jit
def add_or_not_rectangular(
    i: int,
    carry_tuple: Tuple[jax.Array, jax.Array, jax.Array, jax.Array, int],
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """body function for loop in assemble_diff_operator."""
    out = carry_tuple[0]
    coeffs_arr = carry_tuple[1]
    diff_ops = carry_tuple[2]
    which_coeffs = carry_tuple[3]
    counter = carry_tuple[4]

    out = jax.lax.cond(
        which_coeffs[i],
        _add_rectangular,
        _not_rectangular,
        out,
        coeffs_arr[counter],
        diff_ops[i],
        diff_ops[-1],  # B is the last operator in diff_ops
    )
    counter = jax.lax.cond(
        which_coeffs[i],
        lambda x: x + 1,
        lambda x: x,
        counter,
    )
    return (out, coeffs_arr, diff_ops, which_coeffs, counter)


@jax.jit
def assemble_diff_operator_rectangular(
    coeffs_arr: jax.Array,
    which_coeffs: jax.Array,
    diff_ops: jax.Array,
) -> jax.Array:
    """Given an array of coefficients, this function assembles the differential operator.

    Args:
        coeffs_arr (jax.Array): Has shape (?, p**2).
        which_coeffs (jax.Array): Has shape (5,) or (9,) and specifies which coefficients are not None.
        diff_ops (jax.Array): Has shape (6, (p-2)**2, p**2). Contains the precomputed differential operators.

    Returns:
        jax.Array: Has shape ((p-2)**2, p**2).
    """

    n_loops = which_coeffs.shape[0]

    # Initialize with the shape of diff_ops but the data type of coeffs_arr.
    # This is important because we may want to use complex coefficients.
    out = jnp.zeros_like(diff_ops[0], dtype=coeffs_arr.dtype)

    # Commenting this out because it is very memory intensive
    counter = 0
    init_val = (out, coeffs_arr, diff_ops, which_coeffs, counter)
    out, _, _, _, _ = jax.lax.fori_loop(
        0, n_loops, add_or_not_rectangular, init_val
    )

    # Semantically the same as this:
    # counter = 0
    # for i in range(n_loops):
    #     if which_coeffs[i]:
    #         # out += jnp.diag(coeffs_arr[counter]) @ diff_ops[i]
    #         out += jnp.einsum("ab,a->ab", diff_ops[i], coeffs_arr[counter])
    #         counter += 1

    return out


vmapped_assemble_diff_operator_rectangular = jax.vmap(
    assemble_diff_operator_rectangular,
    in_axes=(1, None, None),
    out_axes=0,
)
