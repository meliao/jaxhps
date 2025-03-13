"""Utility functions for implementing the wave scattering solver presented in the
Gillman, Barnett, Martinsson paper."""

from typing import Tuple, Callable
import logging
from timeit import default_timer

import jax.numpy as jnp
import jax
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt

# Disable logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from hps.src.solver_obj import create_solver_obj_2D
from hps.src.methods.fused_methods import (
    _fused_local_solve_and_build_2D_ItI,
    _down_pass_from_fused_ItI,
)
from hps.src.config import DEVICE_ARR, HOST_DEVICE
from hps.src.quadrature.quad_2D.interpolation import (
    interp_from_nonuniform_hps_to_regular_grid,
)
from hps.src.quadrature.trees import Node


@jax.jit
def get_DtN_from_ItI(R: jnp.array, eta: float) -> jnp.array:
    """
    Given an ItI matrix, generates the corresponding DtN matrix.

    equation 2.17 from the Gillman, Barnett, Martinsson paper.

    Implements the formula: T = -i eta (R - I)^{-1}(R + I)

    Args:
        R (jnp.array): Has shape (n, n)
        eta (float): Real number; parameter used in the ItI map creation.

    Returns:
        jnp.array: Has shape (n, n)
    """
    n = R.shape[0]
    I = jnp.eye(n)
    T = -1j * eta * jnp.linalg.solve(R - I, R + I)
    return T


def load_SD_matrices(fp: str) -> Tuple[jnp.array, jnp.array]:
    """
    Load S and D matrices from a MATLAB v7.3 .mat file using h5py.

    Args:
        fp (str): File path to the .mat file

    Returns:
        Tuple[jnp.array, jnp.array]: A tuple containing (S, D) matrices

    Raises:
        ValueError: If the file doesn't contain required matrices
        FileNotFoundError: If the file doesn't exist
        RuntimeError: If there's an error reading the file
    """
    # try:

    # Open the file in read mode
    with h5py.File(fp, "r") as f:
        # Check if required matrices exist
        if "S" not in f or "D" not in f:
            raise ValueError("File must contain both 'S' and 'D' matrices")

        # Helper function to convert structured array to complex array
        def to_complex(structured_array):
            # Convert structured array to complex numpy array
            complex_array = structured_array["real"] + 1j * structured_array["imag"]
            # Convert to jax array
            return jnp.array(complex_array)

        # Load matrices and convert to complex numbers
        # Note: transpose the arrays as MATLAB stores them in column-major order
        S = to_complex(f["S"][:].T)
        D = to_complex(f["D"][:].T)

        return S, D

    # except:
    #     raise RuntimeError(f"Error reading MATLAB file: {fp}")


@jax.jit
def setup_scattering_lin_system(
    S: jnp.array,
    D: jnp.array,
    T_int: jnp.array,
    gauss_bdry_pts: jnp.array,
    k: float,
    source_directions: jnp.array,
) -> Tuple[jnp.array, jnp.array]:
    """
    Sets up the BIE system in eqn (3.4) of the Gillman, Barnett, Martinsson paper.

    Args:
        S (jnp.array): Single-layer potential matrix. Has shape (n,n)
        D (jnp.array): Double-layer potential matrix. Has shape (n,n)
        T_int (jnp.array): Dirichlet-to-Neumann matrix. Has shape (n,n)
        gauss_bdry_pts (jnp.array): Has shape (n, 2)
        k (float): Frequency of the incoming plane waves
        source_directions (jnp.array): Has shape (n_sources,). Describes the direction of the incoming plane waves in radians.

    Returns:
        Tuple[jnp.array, jnp.array]: A, which has shape (n, n) and b, which has shape (n, n_sources).
    """
    n_bdry_pts = gauss_bdry_pts.shape[0]

    # T_int = get_DtN_from_ItI(R, eta)

    uin, normals = get_uin_and_normals(k, gauss_bdry_pts, source_directions)

    A = 0.5 * jnp.eye(n_bdry_pts) - D + S @ T_int
    b = S @ (normals - T_int @ uin)

    return A, b


@jax.jit
def get_uin_and_normals(
    k: float, bdry_pts: jnp.array, source_directions: jnp.array
) -> Tuple[jnp.array, jnp.array]:
    """
    Given the boundary points and the source directions, computes the incoming wave and the normal vectors.

    uin(x) = exp(i k <x,s>), where s = (cos(theta), sin(theta)) is the direction of the incoming plane wave.

    d uin(x) / dx = ik s_0 uin(x)
    d uin(x) / dy = ik s_1 uin(x)

    Args:
        k (float): Frequency of the incoming plane waves
        bdry_pts (jnp.array): Has shape (n, 2)
        source_directions (jnp.array): Has shape (n_sources,). Describes the direction of the incoming plane waves in radians.

    Returns:
        Tuple[jnp.array, jnp.array]: uin, normals. uin has shape (n,). normals has shape (n, 2).
    """
    n_per_side = bdry_pts.shape[0] // 4

    uin = get_uin(k, bdry_pts, source_directions)
    # print("get_uin_and_normals: uin shape: ", uin.shape)
    source_vecs = jnp.array([jnp.cos(source_directions), jnp.sin(source_directions)]).T
    # print("get_uin_and_normals: source_vecs shape: ", source_vecs.shape)

    normals = jnp.concatenate(
        [
            -1j
            * k
            * jnp.expand_dims(source_vecs[:, 1], axis=0)
            * uin[:n_per_side],  # -1 duin/dy
            1j
            * k
            * jnp.expand_dims(source_vecs[:, 0], axis=0)
            * uin[n_per_side : 2 * n_per_side],  # duin/dx
            1j
            * k
            * jnp.expand_dims(source_vecs[:, 1], axis=0)
            * uin[2 * n_per_side : 3 * n_per_side],  # duin/dy
            -1j
            * k
            * jnp.expand_dims(source_vecs[:, 0], axis=0)
            * uin[3 * n_per_side :],
        ]
    )

    # Plot the normals
    # plt.plot(normals.real, "-o", label="real")
    # plt.plot(normals.imag, "-o", label="imag")
    # plt.legend()
    # plt.show()

    return uin, normals


@jax.jit
def get_uin(k: float, pts: jnp.array, source_directions: jnp.array) -> jnp.array:
    source_vecs = jnp.array([jnp.cos(source_directions), jnp.sin(source_directions)]).T
    # jax.debug.print("get_uin: source_vecs: {x}", x=source_vecs)
    uin = jnp.exp(1j * k * jnp.dot(pts, source_vecs.T))
    return uin


@jax.jit
def get_scattering_uscat_impedance(
    S: jnp.array,
    D: jnp.array,
    T: jnp.array,
    source_dirs: jnp.array,
    bdry_pts: jnp.array,
    k: float,
    eta: float,
) -> jnp.array:
    # h is the outgoing impedance from the particular solution. It needs to be added to the incoming impedace.
    # print("get_scattering_utot_impedance: S shape: ", S.shape)
    # print("get_scattering_utot_impedance: D shape: ", D.shape)
    # print("get_scattering_utot_impedance: R shape: ", R.shape)
    A, b = setup_scattering_lin_system(
        S=S, D=D, T_int=T, gauss_bdry_pts=bdry_pts, k=k, source_directions=source_dirs
    )
    # logging.debug("get_scattering_uscat_impedance: A shape: %s", A.shape)
    # logging.debug("get_scattering_uscat_impedance: b shape: %s", b.shape)
    cond = jnp.linalg.cond(A)
    # b = b.flatten()
    uin, uin_dn = get_uin_and_normals(k, bdry_pts, source_dirs)
    uin = uin
    uin_dn = uin_dn

    # Solve the lin system to get uscat on the boundary
    uscat = jnp.linalg.solve(A, b)

    # logging.debug("get_scattering_uscat_impedance: uscat shape: %s", uscat.shape)
    # logging.debug("get_scattering_uscat_impedance: uin shape: %s", uin.shape)
    # logging.debug("get_scattering_uscat_impedance: uin_dn shape: %s", uin_dn.shape)

    # Eqn 1.12 from the Gillman, Barnett, Martinsson paper
    uscat_dn = T @ (uscat + uin) - uin_dn

    # Assemble incoming impedance for the scattered field
    imp = uscat_dn + 1j * eta * uscat

    return imp, cond

INTERP_BATCH_SIZE = 20

def solve_scattering_problem(
    l: int,
    p: int,
    n: int,
    k: float,
    q_fn: Callable[[jnp.array], jnp.array],
    domain_corners: jnp.array,
    source_dirs: jnp.array,
    S_D_matrices_fp: str,
    zero_impedance: bool = False,
    interp_xmin: float = None,
    interp_xmax: float = None,
    interp_ymin: float = None,
    interp_ymax: float = None,
    return_utot: bool = False,
) -> Tuple[jnp.array, jnp.array, float]:
    """
    Does the following:

    1. Sets up the HPS quadrature for l levels and polynomial order p.
    2. Evaluates the coefficients and source term for the wave scattering problem.
    3. Performs local solves and merges to get a top-level ItI operator.
    4. Uses top-level ItI operator along with loaded S and D matrices to solve the exterior problem.
    5. Propagates the resulting impedance data down to the leaves.
    6. Interpolates the solution onto a regular grid with n points per dimension.

    """

    # These things are fast to precompute
    S, D = load_SD_matrices(S_D_matrices_fp)
    # Set up the HPS quadrature for l levels and polynomial order p
    logging.debug("solve_scattering_problem: Creating tree...")
    xmin, ymin = domain_corners[0]
    xmax, ymax = domain_corners[2]
    root = Node(xmin=float(xmin), xmax=float(xmax), ymin=float(ymin), ymax=float(ymax))

    t = create_solver_obj_2D(
        p=p, q=p - 2, root=root, uniform_levels=l, use_ItI=True, eta=k
    )

    t_0 = default_timer()

    # Evaluate the coefficients and source term for the wave scattering problem
    d_xx_coeffs = jnp.ones_like(t.leaf_cheby_points[:, :, 0])
    d_yy_coeffs = jnp.ones_like(t.leaf_cheby_points[:, :, 0])
    i_term = k**2 * (1 + q_fn(t.leaf_cheby_points))
    logging.debug("solve_scattering_problem: i_term shape: %s", i_term.shape)

    uin_evals = get_uin(k, t.leaf_cheby_points, source_dirs)

    source_term = (
        -1 * (k**2) * jnp.expand_dims(q_fn(t.leaf_cheby_points), -1) * uin_evals
    )
    logging.debug("solve_scattering_problem: source_term shape: %s", source_term.shape)

    logging.debug("solve_scattering_problem: S device: %s", S.devices())

    S_arr_lst, ItI_arr_lst, f_arr_lst = _fused_local_solve_and_build_2D_ItI(
        D_xx=t.D_xx,
        D_xy=t.D_xy,
        D_yy=t.D_yy,
        D_x=t.D_x,
        D_y=t.D_y,
        I_P_0=t.I_P_0,
        Q_I=t.Q_I,
        F=t.F,
        G=t.G,
        p=p,
        l=l,
        source_term=source_term,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
        I_coeffs=i_term,
        host_device=DEVICE_ARR[0],
    )

    R = ItI_arr_lst[-1]

    T = get_DtN_from_ItI(R, t.eta)

    logging.debug("solve_scattering_problem: Solving boundary integral equation...")
    if zero_impedance:
        incoming_imp_data = jnp.zeros_like(
            t.root_boundary_points[..., 0]
        ) + 1j * jnp.zeros_like(t.root_boundary_points[..., 0])
        cond = 1.0
    else:
        S = jax.device_put(S, DEVICE_ARR[0])
        D = jax.device_put(D, DEVICE_ARR[0])
        incoming_imp_data, cond = get_scattering_uscat_impedance(
            S=S,
            D=D,
            T=T,
            source_dirs=source_dirs,
            bdry_pts=t.root_boundary_points,
            k=k,
            eta=k,
        )

    # Delete exterior matrices we no longer need
    R.delete()
    T.delete()
    S.delete()
    D.delete()

    # Propagate the resulting impedance data down to the leaves
    interior_solns = _down_pass_from_fused_ItI(
        bdry_data=incoming_imp_data,
        S_arr_lst=S_arr_lst,
        f_lst=f_arr_lst,
        D_xx=t.D_xx,
        D_xy=t.D_xy,
        D_yy=t.D_yy,
        D_x=t.D_x,
        D_y=t.D_y,
        I_P_0=t.I_P_0,
        Q_I=t.Q_I,
        F=t.F,
        G=t.G,
        p=p,
        l=l,
        source_term=source_term,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
        I_coeffs=i_term,
    )
    uscat_soln = interior_solns.block_until_ready()
    t_1 = default_timer() - t_0

    n_src = uscat_soln.shape[-1]
    logging.debug("solve_scattering_problem: uscat_soln shape: %s", uscat_soln.shape)

    # Measure consistency with the PDE
    # diff_op = t.D_xx + t.D_yy
    # logging.debug("solve_scattering_problem: diff_op shape: %s", diff_op.shape)
    # logging.debug("solve_scattering_problem: uscat_soln shape: %s", uscat_soln.shape)
    # # Measure PDE residual normalized by k**2
    # lap_u = 1 / (k**2) * jnp.einsum("ij,kj->ki", diff_op, uscat_soln)
    # inhomogeneous_term = (1 + q_fn(t.leaf_cheby_points)) * uscat_soln
    # source_term = -1 * q_fn(t.leaf_cheby_points) * uin_evals
    # pde_error = lap_u + inhomogeneous_term - source_term
    # logging.info(
    #     "solve_scattering_problem: Normalized PDE error: %s",
    #     jnp.max(jnp.abs(pde_error)),
    # )

    # # Measure PDE residual un-normalized
    # lap_u = jnp.einsum("ij,kj->ki", diff_op, uscat_soln)
    # inhomogeneous_term = k**2 * (1 + q_fn(t.leaf_cheby_points)) * uscat_soln

    # if return_utot:
    #     uscat_soln += uin_evals
    #     logging.debug(
    #         "solve_scattering_problem: uscat_soln device: %s", uscat_soln.devices()
    #     )
    # source_term = -1 * k**2 * q_fn(t.leaf_cheby_points) * uin_evals
    # pde_error_unnorm = lap_u + inhomogeneous_term - source_term
    # logging.info(
    #     "solve_scattering_problem: Un-normalized PDE error: %s",
    #     jnp.max(jnp.abs(pde_error_unnorm)),
    # )

    # Interpolate the solution onto a regular grid with n points per dimension
    if interp_xmin is None:
        interp_xmin, interp_ymin = domain_corners[0]
        interp_xmax, interp_ymax = domain_corners[2]
    logging.debug(
        "solve_scattering_problem: Interpolating solution onto regular grid..."
    )

    uscat_regular = jnp.zeros((n, n, n_src), dtype=jnp.complex128, device=HOST_DEVICE)

    # Do the interpolation from HPS to regular grid in batches of size INTERP_BATCH_SIZE
    # along the source dimension
    for i in range(0, n_src, INTERP_BATCH_SIZE):
        chunk_start = i 
        chunk_end = min((i + INTERP_BATCH_SIZE), n_src)
        logging.debug("solve_scattering_problem: Interpolating chunk i=%s, %s:%s",i, chunk_start, chunk_end)
        uscat_i = uscat_soln[..., chunk_start:chunk_end]
        logging.debug("solve_scattering_problem: ||uscat_i||=%s", jnp.linalg.norm(uscat_i))
        chunk_i, target_pts = interp_from_nonuniform_hps_to_regular_grid(
            root=root,
            p=p,
            f_evals=uscat_i,
            n_pts=n,
        )
        logging.debug("solve_scattering_problem: ||chunk_i||=%s", jnp.linalg.norm(chunk_i))
        chunk_i = jax.device_put(chunk_i, HOST_DEVICE)
        uscat_regular = uscat_regular.at[..., chunk_start:chunk_end].set(chunk_i)

    # source_regular, _ = interp_from_hps_to_regular_grid(
    #     l=l,
    #     p=p,
    #     corners=domain_corners,
    #     xmin=interp_xmin,
    #     xmax=interp_xmax,
    #     ymin=interp_ymin,
    #     ymax=interp_ymax,
    #     f_evals=source_term,
    #     n_pts=n,
    # )

    return uscat_regular, target_pts, t_1, cond
