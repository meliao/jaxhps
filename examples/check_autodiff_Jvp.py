import argparse
import os
import logging

import jax.numpy as jnp
import jax
import numpy as np

import matplotlib.pyplot as plt

from jaxhps import (
    DiscretizationNode2D,
    Domain,
    PDEProblem,
    build_solver,
    solve,
)
from jaxhps.up_pass import up_pass_uniform_2D_ItI
from scattering_potentials import q_gaussian_bumps
from wave_scattering_utils import (
    setup_scattering_lin_system,
    get_uin_and_normals,
    load_SD_matrices,
    get_DtN_from_ItI,
    get_uin,
)
from sine_transform import (
    nu_sinetransform,
    adjoint_nu_sinetransform,
    get_freqs_up_to_2k,
)
from gen_SD_exterior import gen_S_exterior, gen_D_exterior

XMIN = -1
XMAX = 1
YMIN = -1
YMAX = 1

plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
plt.rc("text", usetex=True)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute reference solution for wave scattering problem."
    )

    parser.add_argument(
        "-l_vals",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Number of levels in the quadtree.",
    )
    parser.add_argument(
        "-p",
        type=int,
        default=16,
        help="Chebyshev polynomial order.",
    )
    parser.add_argument(
        "-n_pixels",
        type=int,
        default=500,
        help="Number of pixels in each dimension to compare solutions.",
    )
    parser.add_argument("-k", type=float, default=20.0, help="Wavenumber.")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--SD_matrix_prefix",
        default="../data/examples/SD_matrices",
    )
    return parser.parse_args()


def coeffs_to_uscat(
    coeffs: jax.Array,
    freqs: jax.Array,
    problem: PDEProblem,
    source_dirs: jax.Array,
    S_int: jax.Array,
    D_int: jax.Array,
    nrec: int = 100,
    rad: float = 5.0,
    return_DtN: bool = False,
    eval_on_interior: bool = False,
) -> jax.Array:
    # First, go from global sine series coefficients to
    # values of the potential on the interior points
    spatial_points = problem.domain.interior_points
    q_hps_grid = adjoint_nu_sinetransform(coeffs, freqs, spatial_points)

    # Next, update the problem's coefficients
    k = problem.eta
    lap_coeffs = jnp.ones_like(problem.domain.interior_points[..., 0])
    i_coeffs = (k**2) * (1 + q_hps_grid)
    problem.update_coefficients(
        D_xx_coefficients=lap_coeffs,
        D_yy_coefficients=lap_coeffs,
        I_coefficients=i_coeffs,
    )
    zero_source = jnp.zeros_like(problem.domain.interior_points[..., 0])

    # Build the solver
    T_ItI = build_solver(
        pde_problem=problem,
        host_device=jax.devices()[0],
        compute_device=jax.devices()[0],
        return_top_T=True,
    )

    T_DtN = get_DtN_from_ItI(T_ItI, problem.eta)

    # compute uin and uin_dn on the boundary
    uin_bdry, uin_dn_bdry = get_uin_and_normals(
        source_directions=source_dirs,
        bdry_pts=problem.domain.boundary_points,
        k=k,
    )

    # Set up the BIE
    A, b = setup_scattering_lin_system(
        S=S_int,
        D=D_int,
        T_int=T_DtN,
        gauss_bdry_pts=problem.domain.boundary_points,
        k=k,
        source_directions=source_dirs,
    )

    uscat = jnp.linalg.solve(A, b)
    uscat_dn = T_DtN @ (uscat + uin_bdry) - uin_dn_bdry
    incoming_imp_data = (uscat_dn + uin_dn_bdry) + 1j * k * (uscat + uin_bdry)

    if eval_on_interior:
        # Compute the solution on the HPS grid
        utot_interior = solve(
            pde_problem=problem,
            source=zero_source,
            boundary_data=incoming_imp_data.flatten(),
            compute_device=jax.devices()[0],
            host_device=jax.devices()[0],
        )
        uin_interior = get_uin(
            k=k,
            pts=problem.domain.interior_points,
            source_directions=source_dirs,
        )[..., 0]
        uscat_interior = utot_interior - uin_interior

        if return_DtN:
            return uscat_interior, T_DtN
        else:
            return uscat_interior
    else:
        # Evaluate on the exterior reciever points. First need to compute
        # Single and double layer potentials mapping from the boundary to
        # the exterior points.

        # First check whether S_ext and D_ext are saved in the problem object
        if not hasattr(problem, "S_ext"):
            problem.S_ext = gen_S_exterior(
                domain=problem.domain, k=k, nrec=nrec, rad=rad
            )
            problem.D_ext = gen_D_exterior(
                domain=problem.domain, k=k, nrec=nrec, rad=rad
            )
        S_ext = problem.S_ext
        D_ext = problem.D_ext

        uscat_at_rec = D_ext @ uscat - S_ext @ uscat_dn

        if return_DtN:
            return uscat_at_rec, T_DtN
        else:
            return uscat_at_rec


def get_exterior_DtN(S: jax.Array, D: jax.Array) -> jax.Array:
    return jnp.linalg.solve(S, (D - 0.5 * jnp.eye(S.shape[0])))


def get_DtI_from_DtN(DtN: jax.Array, eta: jax.Array) -> jax.Array:
    return DtN + 1j * eta * jnp.eye(DtN.shape[0])


def eval_uscat_with_source_on_bdry(
    problem: PDEProblem,
    source: jax.Array,
    q: jax.Array,
    S: jax.Array,
    D: jax.Array,
    DtN: jax.Array,
    adjoint_radiation_condition: bool = False,
    return_v_g_tilde: bool = False,
) -> jax.Array:
    if adjoint_radiation_condition:
        T_ext = get_exterior_DtN(S=S.conj(), D=D.conj())
    else:
        T_ext = get_exterior_DtN(S=S, D=D)

    # h_part is the particular outgoing impedance data on the boundary.
    v, g_tilde_lst, h_part = up_pass_uniform_2D_ItI(
        source=source, pde_problem=problem, return_h_last=True
    )
    # By definition, g_part = 0
    g_part = jnp.zeros_like(h_part)

    # These are the boundary evaluations of the particular solution and its normal derivative.
    un_part = (h_part + g_part) / 2
    u_part = (h_part - un_part) / (-1j * problem.eta)

    G_int = get_DtI_from_DtN(DtN=DtN, eta=problem.eta)
    G_ext = get_DtI_from_DtN(DtN=T_ext, eta=problem.eta)

    lhs = G_ext - G_int
    u_homog = jnp.linalg.solve(lhs, -1 * G_ext @ u_part)
    un_homog = DtN @ u_homog

    u = u_homog + u_part
    un = un_homog + un_part
    if return_v_g_tilde:
        return u, un, v, g_tilde_lst, T_ext
    else:
        return u, un


def analytical_jvp_coeffs_to_uscat(
    coeffs: jax.Array,
    delta_coeffs: jax.Array,
    freqs: jax.Array,
    problem: PDEProblem,
    source_dirs: jax.Array,
    S_int: jax.Array,
    D_int: jax.Array,
    nrec: int = 100,
    rad: float = 5.0,
) -> jax.Array:
    k = problem.eta
    # First, compute uscat.
    uscat, DtN = coeffs_to_uscat(
        coeffs=coeffs,
        freqs=freqs,
        problem=problem,
        source_dirs=source_dirs,
        S_int=S_int,
        D_int=D_int,
        return_DtN=True,
        eval_on_interior=True,
    )

    # Now, get uin on the interior points
    uin_interior = get_uin(
        k=k, pts=problem.domain.interior_points, source_directions=source_dirs
    )
    # get_uin returns shape (..., n_src_dirs), we want (...)
    utot = uscat + uin_interior[..., 0]

    # Compute q and delta on the HPS grid
    q_hps = adjoint_nu_sinetransform(
        coefficients=coeffs,
        freqs=freqs,
        spatial_points=problem.domain.interior_points,
    )
    delta_hps = adjoint_nu_sinetransform(
        coefficients=delta_coeffs,
        freqs=freqs,
        spatial_points=problem.domain.interior_points,
    )

    source = -1 * (k**2) * utot * delta_hps

    # Now, solve the scattering problem on the boundary with this source term
    v_bdry, v_dn_bdry = eval_uscat_with_source_on_bdry(
        problem=problem,
        source=source,
        q=q_hps,
        S=S_int,
        D=D_int,
        DtN=DtN,
    )

    # Now, map to exterior points
    S_ext = gen_S_exterior(domain=problem.domain, k=k, nrec=nrec, rad=rad)
    D_ext = gen_D_exterior(domain=problem.domain, k=k, nrec=nrec, rad=rad)

    v_at_rec = D_ext @ v_bdry - S_ext @ v_dn_bdry

    return v_at_rec


def main(args: argparse.Namespace) -> None:
    """ """
    # Set up output directory
    args.output_dir = "data/examples/autodiff_checks"
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up the geometry of the problem
    root = DiscretizationNode2D(
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
    )
    reg_x_pts = jnp.linspace(XMIN, XMAX, args.n_pixels)
    reg_y_pts = jnp.linspace(YMIN, YMAX, args.n_pixels)
    X, Y = jnp.meshgrid(reg_x_pts, reg_y_pts, indexing="ij")
    points_reg = jnp.concatenate(
        (jnp.expand_dims(X, 2), jnp.expand_dims(Y, 2)), axis=2
    )
    logging.info("points_reg shape: %s", points_reg.shape)

    # Set up a large domain for the reference problem.
    domain_L = 5
    domain_ref = Domain(root=root, p=22, q=20, L=domain_L)
    problem_ref = PDEProblem(domain=domain_ref, eta=args.k, use_ItI=True)
    source_dirs = jnp.array([0.0])

    # Load S and D matrices for the reference problem
    q = 20
    nside = 2**domain_L
    k_str = str(int(args.k))
    S_D_matrices_fp = os.path.join(
        args.SD_matrix_prefix, f"SD_k{k_str}_n{q}_nside{nside}_dom1.mat"
    )
    logging.debug("Loading S and D from disk...")
    S_int_ref, D_int_ref = load_SD_matrices(S_D_matrices_fp)

    # Take the single Gaussian bump scattering potential, apply a sine transform
    # to get the coefficients.
    dx = reg_x_pts[1] - reg_x_pts[0]
    dy = reg_y_pts[1] - reg_y_pts[0]
    quadrature_weights = jnp.ones((args.n_pixels, args.n_pixels)) * (dx * dy)
    q_evals_reg = q_gaussian_bumps(points_reg)
    freqs = get_freqs_up_to_2k(args.k, root=root)
    logging.info("Freqs shape: %s", freqs.shape)
    gamma_val = (2 * args.k * (XMAX - XMIN)) / jnp.pi
    logging.info("gamma_val: %.2f", gamma_val)
    q_coeffs = nu_sinetransform(
        samples=q_evals_reg,
        freqs=freqs,
        spatial_points=points_reg,
        quad_weights=quadrature_weights,
    )

    # Plot the potential
    q_reg_bandlimited = adjoint_nu_sinetransform(
        coefficients=q_coeffs,
        freqs=freqs,
        spatial_points=points_reg,
    )
    q_reg_bandlimited = q_reg_bandlimited.reshape(args.n_pixels, args.n_pixels)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    im_0 = ax[0].imshow(q_evals_reg.reshape(args.n_pixels, args.n_pixels))
    plt.colorbar(im_0, ax=ax[0])
    ax[0].set_title("Original potential (reg grid)")
    im_1 = ax[1].imshow(q_reg_bandlimited)
    plt.colorbar(im_1, ax=ax[1])
    ax[1].set_title("Bandlimited potential (reg grid)")
    im_2 = ax[2].imshow(
        q_evals_reg.reshape(args.n_pixels, args.n_pixels) - q_reg_bandlimited
    )
    plt.colorbar(im_2, ax=ax[2])
    ax[2].set_title("Difference")
    fp = os.path.join(args.output_dir, "potential_bandlimited_comparison.pdf")
    plt.savefig(fp)
    plt.close()

    # Now, choose random coefficients for the perturbation delta
    np.random.seed(0)
    # delta_coeffs = q_coeffs * 0.01
    delta_coeffs = jnp.array(np.random.randn(q_coeffs.shape[0]))  # * 0.1

    # Now, compute the reference Jvp analytically
    jvp_ref = analytical_jvp_coeffs_to_uscat(
        coeffs=q_coeffs,
        delta_coeffs=delta_coeffs,
        freqs=freqs,
        problem=problem_ref,
        source_dirs=source_dirs,
        S_int=S_int_ref,
        D_int=D_int_ref,
    )

    logging.info("jvp_ref shape: %s", jvp_ref.shape)

    logging.info("Computed reference Jvp.")

    # Loop through L levels computing the Jvp via autodiff and comparing to reference
    rel_diffs = []
    for L in args.l_vals:
        logging.info("Checking autodiff Jvp for L=%d...", L)
        domain = Domain(root=root, p=args.p, q=args.p - 2, L=L)
        problem = PDEProblem(domain=domain, eta=args.k, use_ItI=True)

        # Load S and D matrices for the current problem
        nside = 2**L
        q = args.p - 2
        S_D_matrices_fp = os.path.join(
            args.SD_matrix_prefix, f"SD_k{k_str}_n{q}_nside{nside}_dom1.mat"
        )
        S_int, D_int = load_SD_matrices(S_D_matrices_fp)

        # Define a function that goes from coeffs to uscat on the HPS grid
        def coeffs_to_uscat_fun(coeffs: jax.Array) -> jax.Array:
            return coeffs_to_uscat(
                coeffs=coeffs,
                freqs=freqs,
                problem=problem,
                source_dirs=source_dirs,
                S_int=S_int,
                D_int=D_int,
            )

        # Compute the Jvp via autodiff
        jvp_autodiff = jax.jvp(
            coeffs_to_uscat_fun, (q_coeffs,), (delta_coeffs,)
        )[1]

        # Compute the Jvp via finite differences
        eps = 1e-6
        uscat_plus = coeffs_to_uscat_fun(q_coeffs + eps * delta_coeffs)
        uscat_minus = coeffs_to_uscat_fun(q_coeffs - eps * delta_coeffs)
        jvp_fd = (uscat_plus - uscat_minus) / (2 * eps)

        #############################################
        # Compute error
        error = jnp.linalg.norm(
            jvp_autodiff.flatten() - jvp_ref.flatten()
        ) / jnp.linalg.norm(jvp_ref)
        logging.info(
            "L=%d: Relative error between autodiff Jvp and reference: %.3e",
            L,
            error,
        )

        # Compute error between autodiff and finite diff
        error_fd = jnp.linalg.norm(
            jvp_autodiff.flatten() - jvp_fd.flatten()
        ) / jnp.linalg.norm(jvp_fd)
        logging.info(
            "L=%d: Relative error between autodiff Jvp and finite difference Jvp: %.3e",
            L,
            error_fd,
        )

        rel_diffs.append(error)

    # Plot relative differences vs h
    h_vals = 2 / (2 ** jnp.array(args.l_vals))
    one_over_h_vals = 1 / h_vals
    h_vals_linfit = jnp.linspace(0.1, 0.2, 10)
    const_p = 100000.0
    one_over_h_vals_linfit = 1 / h_vals_linfit
    h_tothe_pm2 = const_p * h_vals_linfit ** (args.p - 2)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(one_over_h_vals, rel_diffs, ".-")
    ax.plot(
        one_over_h_vals_linfit,
        h_tothe_pm2,
        "--",
        label="$O(h^{p-2})$",
        color="black",
    )
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$1 / h$")
    ax.set_ylabel("Relative difference")
    ax.set_title("Relative difference between autodiff Jvp and reference")
    ax.grid()
    fp = os.path.join(args.output_dir, "autodiff_Jvp_relative_diff.pdf")
    plt.savefig(fp)
    plt.close()

    # Save data
    fp = os.path.join(args.output_dir, "autodiff_Jvp_relative_diff.npy")
    dd_out = {
        "L_vals": np.array(args.l_vals),
        "rel_diffs": np.array(rel_diffs),
    }
    np.save(fp, dd_out)


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s:jaxhps: %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    main(args)
