import jax.numpy as jnp
import jax

from .._pdeproblem import PDEProblem
from ._uniform_2D_DtN import _gather_coeffs_2D, vmapped_assemble_diff_operator
from typing import Tuple


def nosource_local_solve_stage_uniform_2D_DtN(
    pde_problem: PDEProblem,
    device: jax.Device = jax.devices()[0],
    host_device: jax.Device = jax.devices("cpu")[0],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
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
    Phi : jax.Array
        Particular solution operators mapping from the source term evaluated on the interior Cheby nodes to the particular solution on all of the Cheby nodes. Has shape (n_leaves, p^2, (p-1)^2).
    """

    # Gather the coefficients into a single array.
    coeffs_gathered, which_coeffs = _gather_coeffs_2D(
        D_xx_coeffs=pde_problem.D_xx_coefficients,
        D_xy_coeffs=pde_problem.D_xy_coefficients,
        D_yy_coeffs=pde_problem.D_yy_coefficients,
        D_x_coeffs=pde_problem.D_x_coefficients,
        D_y_coeffs=pde_problem.D_y_coefficients,
        I_coeffs=pde_problem.I_coefficients,
    )
    # stack the precomputed differential operators into a single array
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

    coeffs_gathered = jax.device_put(
        coeffs_gathered,
        device,
    )

    all_diff_operators = vmapped_assemble_diff_operator(
        coeffs_gathered, which_coeffs, diff_ops
    )

    Y_arr, T_arr, Phi_arr = vmapped_get_DtN_nosource(
        all_diff_operators, pde_problem.Q, pde_problem.P
    )

    T_arr_host = jax.device_put(T_arr, host_device)
    Y_arr_host = jax.device_put(Y_arr, host_device)
    Phi_arr_host = jax.device_put(Phi_arr, host_device)
    return Y_arr_host, T_arr_host, Phi_arr_host


@jax.jit
def get_DtN_nosource(
    diff_operator: jax.Array,
    Q: jax.Array,
    P: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Computes the Dirichlet-to-Neumann operator for a given differential operator and boundary data.

    Parameters
    ----------
    diff_operator : jax.Array
        The differential operator to be applied.
    Q : jax.Array
        The boundary data.
    P : jax.Array
        The interpolation matrix.

    Returns
    -------
    T : jax.Array
        The Dirichlet-to-Neumann matrix.
    Y : jax.Array
        The solution operator mapping from Dirichlet boundary data to homogeneous solutions on the leaf interiors.
    """
    n_cheby_bdry = P.shape[0]

    A_ii = diff_operator[n_cheby_bdry:, n_cheby_bdry:]
    A_ii_inv = jnp.linalg.inv(A_ii)
    # A_ie shape (n_cheby_int, n_cheby_bdry)
    A_ie = diff_operator[n_cheby_bdry:, :n_cheby_bdry]
    L_2 = jnp.zeros(
        (diff_operator.shape[0], n_cheby_bdry), dtype=diff_operator.dtype
    )
    L_2 = L_2.at[:n_cheby_bdry].set(jnp.eye(n_cheby_bdry))
    soln_operator = -1 * A_ii_inv @ A_ie
    L_2 = L_2.at[n_cheby_bdry:].set(soln_operator)
    Y = L_2 @ P
    T = Q @ Y

    return Y, T, A_ii_inv


vmapped_get_DtN_nosource = jax.vmap(
    get_DtN_nosource,
    in_axes=(0, None, None),
    out_axes=(0, 0, 0),
)
