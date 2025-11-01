import argparse
import os
import logging
from functools import partial

import jax.numpy as jnp
import jax
import numpy as np

import matplotlib.pyplot as plt

from jaxhps import (
    DiscretizationNode2D,
    Domain,
    PDEProblem,
)
from scipy import special

from jaxhps.down_pass import down_pass_uniform_2D_ItI
from jaxhps.quadrature import chebyshev_weights
from scattering_potentials import q_GBM_1
from wave_scattering_utils import (
    load_SD_matrices,
    get_uin,
)
from sine_transform import (
    nu_sinetransform,
    adjoint_nu_sinetransform,
    get_freqs_up_to_2k,
)
from gen_SD_exterior import get_ring_points

from check_autodiff_Jvp import (
    XMIN,
    XMAX,
    YMIN,
    YMAX,
    coeffs_to_uscat,
    eval_uscat_with_source_on_bdry,
)

plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
plt.rc("text", usetex=True)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute reference solution for wave scattering problem."
    )

    parser.add_argument(
        "-n_pixels",
        type=int,
        default=256,
        help="Number of pixels in the regular grid for the potential.",
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

    parser.add_argument("-k", type=float, default=20.0, help="Wavenumber.")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--SD_matrix_prefix",
        default="../data/examples/SD_matrices",
    )
    return parser.parse_args()


def hankel_evals(
    k: float, rec_pts: jax.Array, eval_pts: jax.Array
) -> jax.Array:
    """
    Computes - H_0^2(k|x-y|) for all x in rec_pts and y in eval_pts.

    Args:
        k (float): Wavenumber
        rec_pts (jax.Array): Has shape (nrec, 2)
        eval_pts (jax.Array): Has shape (n_leaves, p**2, 2)

    Returns:
        jax.Array: Has shape (n_leaves, p**2, nrec)
    """
    # Compute the distances between each eval point and each receiver point
    # Has shape (n_leaves, p**2, nrec, 2)
    diffs = rec_pts[None, None, ...] - eval_pts[:, :, None]
    # Compute the Euclidean distance
    # Has shape (n_leaves, p**2, nrec)
    distances = jnp.linalg.norm(diffs, axis=-1)
    # Compute the Hankel function of the second kind
    # Has shape (n_leaves, p**2, nrec)
    hankel_evals = -1 * special.hankel2(0, k * distances)
    return hankel_evals


def compute_phi_1(
    problem: PDEProblem,
    duscat: jax.Array,
    nrec: int = 100,
    rad: float = 5.0,
) -> jax.Array:
    """

    phi_1(x) = k^2 i / 4 \\sum_{j}^N H_0^2(k |x-x_j|) duscat_j

    where H_0 is the zeroth order Hankel function of the second kind,
    and the sum is over the receiver points x_j.

    Args:
        scattering_problem (ScatteringProblem): We use this to get reciever and interior points.
        duscat (jax.Array): Has shape (nrec, nsrc).

    Returns:
        jax.Array: Has shape (n_leaves, p**2, nsrc)
    """
    k = problem.eta

    # Check to see if the pdeproblem has precomputed hankel_evals.
    # If not, compute and store them. This is expensive so we only want to
    # do it once.
    if not hasattr(problem, "hankel_evals"):
        rec_pts = get_ring_points(nrec, rad)  # Has shape (nrec, 2)
        eval_pts = problem.domain.interior_points
        hankel_arr = hankel_evals(
            k=k,
            rec_pts=rec_pts,
            eval_pts=eval_pts,
        )
        problem.hankel_evals = hankel_arr

    # Has shape (nrec, 2)
    # rec_pts = scattering_problem.rec_points
    # Has shape (n_leaves, p**2, 2)
    # interior_pts = scattering_problem.pde_problem.domain.interior_points
    phi_1 = (k**2 * 1j / 4) * jnp.sum(
        hankel_arr[:, :, :, None] * duscat[None, None, :], axis=-2
    )

    return phi_1


def analytical_vjp_coeffs_to_uscat(
    coeffs: jax.Array,
    f: jax.Array,
    freqs: jax.Array,
    problem: PDEProblem,
    source_dirs: jax.Array,
    S_int: jax.Array,
    D_int: jax.Array,
    quad_weights_single_panel: jax.Array,
) -> jax.Array:
    k = problem.eta
    uscat_int, DtN = coeffs_to_uscat(
        coeffs=coeffs,
        freqs=freqs,
        problem=problem,
        source_dirs=source_dirs,
        S_int=S_int,
        D_int=D_int,
        return_DtN=True,
        eval_on_interior=True,
    )

    uin_int = get_uin(
        k=problem.eta,
        pts=problem.domain.interior_points,
        source_directions=source_dirs,
    )[..., 0]

    q_hps = adjoint_nu_sinetransform(
        coefficients=coeffs,
        freqs=freqs,
        spatial_points=problem.domain.interior_points,
    )

    # compute phi_1 on the HPS grid points.
    phi_1 = compute_phi_1(problem=problem, duscat=jnp.conj(f[..., None]))
    logging.info("analytical_vjp: phi_1 shape: %s", phi_1.shape)
    # Remove the last dimension since nsrc=1
    phi_1 = phi_1[..., 0]

    source = -1 * k**2 * phi_1 * q_hps
    phi2_bdry, phi2_n_bdry, v, g_tilde, T_ext = eval_uscat_with_source_on_bdry(
        problem=problem,
        source=source,
        q=q_hps,
        S=S_int,
        D=D_int,
        DtN=DtN,
        adjoint_radiation_condition=True,
        return_v_g_tilde=True,
    )

    # Construct incoming impedance data from wn_homog and w_homog
    incoming_imp = phi2_n_bdry + 1j * problem.eta * phi2_bdry
    # Do a downward pass to compute w at the interior points.
    phi2_hps = down_pass_uniform_2D_ItI(
        boundary_data=incoming_imp,
        S_lst=problem.S_lst,
        g_tilde_lst=g_tilde,
        Y_arr=problem.Y,
        v_arr=v,
        host_device=jax.devices()[0],
    )
    w_int = phi_1 + phi2_hps

    u_0 = uscat_int + uin_int
    out_hps = (jnp.conj(u_0) * w_int).real

    # Perform a sine transform to get back to regular grid
    # First, repeat the quad weights from size (p**2,) to (n_leaves, p**2)
    n_leaves = out_hps.shape[0]
    quad_weights_all = jnp.repeat(
        quad_weights_single_panel[None, ...], n_leaves, axis=0
    )
    logging.info(
        "analytical_vjp_coeffs_to_uscat: out_hps shape: %s", out_hps.shape
    )
    logging.info(
        "analytical_vjp_coeffs_to_uscat: quad_weights shape: %s",
        quad_weights_all.shape,
    )
    out_coeffs = nu_sinetransform(
        samples=out_hps,
        freqs=freqs,
        spatial_points=problem.domain.interior_points,
        quad_weights=quad_weights_all,
    )

    return out_coeffs


def quad_weights_hps_panel(domain: Domain) -> jax.Array:
    # First, get the size of each panel
    side_len = (domain.root.xmax - domain.root.xmin) / (2**domain.L)
    bounds = jnp.array([0.0, side_len])
    # Then get the 1D Cheby weights on that panel
    cheby_weights_x = chebyshev_weights(domain.p, bounds)
    cheby_weights_y = chebyshev_weights(domain.p, bounds)
    r_idxes = rearrange_indices_ext_int(domain.p)
    # Form a tensor product and then rearrange to get the boundary weights first
    cheby_weights_2d = jnp.outer(cheby_weights_x, cheby_weights_y).reshape(-1)[
        r_idxes
    ]
    return cheby_weights_2d


@partial(jax.jit, static_argnums=(0,))
def rearrange_indices_ext_int(n: int) -> jnp.ndarray:
    """This function gives the array indices to rearrange the 2D Cheby grid so that the
    4(p-1) boundary points are listed first, starting at the SW corner and going clockwise around the
    boundary. The interior points are listed after.
    """

    idxes = np.zeros(n**2, dtype=int)
    # S border
    for i, j in enumerate(range(n - 1, n**2, n)):
        idxes[i] = j
    # W border
    for i, j in enumerate(range(n**2 - 2, n**2 - n - 1, -1)):
        idxes[n + i] = j
    # N border
    for i, j in enumerate(range(n**2 - 2 * n, 0, -n)):
        idxes[2 * n - 1 + i] = j
    # S border
    for i, j in enumerate(range(1, n - 1)):
        idxes[3 * n - 2 + i] = j
    # Loop through the indices in column-rasterized form and fill in the ones from the interior.
    current_idx = 4 * n - 4
    nums = np.arange(n**2)
    for i in nums:
        if i not in idxes:
            idxes[current_idx] = i
            current_idx += 1
        else:
            continue

    return jnp.array(idxes)


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
    ref_L = 5
    domain_ref = Domain(root=root, p=22, q=20, L=ref_L)
    problem_ref = PDEProblem(domain=domain_ref, eta=args.k, use_ItI=True)
    source_dirs = jnp.array([0.0])

    # Load S and D matrices for the reference problem
    q = 20
    nside = 2**ref_L
    k_str = str(int(args.k))
    S_D_matrices_fp = os.path.join(
        args.SD_matrix_prefix, f"SD_k{k_str}_n{q}_nside{nside}_dom1.mat"
    )
    logging.debug("Loading S and D from disk...")
    S_int_ref, D_int_ref = load_SD_matrices(S_D_matrices_fp)

    quad_weights_ref = quad_weights_hps_panel(domain_ref)

    # Take the single Gaussian bump scattering potential, apply a sine transform
    # to get the coefficients.
    dx = reg_x_pts[1] - reg_x_pts[0]
    dy = reg_y_pts[1] - reg_y_pts[0]
    quadrature_weights = jnp.ones((args.n_pixels, args.n_pixels)) * (dx * dy)
    q_evals_reg = q_GBM_1(points_reg)
    freqs = get_freqs_up_to_2k(args.k, root=root)  # [:10]
    logging.info("Freqs shape: %s", freqs.shape)
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
    f = jnp.array(np.random.randn(100) + 1j * np.random.randn(100)) * 0.01
    f = f.astype(jnp.complex128)

    # Now, compute the reference Jvp analytically
    vjp_ref = analytical_vjp_coeffs_to_uscat(
        coeffs=q_coeffs,
        f=f,
        freqs=freqs,
        problem=problem_ref,
        source_dirs=source_dirs,
        S_int=S_int_ref,
        D_int=D_int_ref,
        quad_weights_single_panel=quad_weights_ref,
    )
    logging.info("Computed reference vJp.")
    logging.info("vjp_ref: %s", vjp_ref[:10])

    # Loop through L levels computing the vJp via autodiff and comparing to reference
    rel_diffs = []
    for L in args.l_vals:
        logging.info("Checking autodiff vJp for L=%d...", L)
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

        # Compute the vJp via autodiff
        vjp_autodiff_fn = jax.vjp(
            coeffs_to_uscat_fun,
            q_coeffs,
        )[1]

        vjp_autodiff = vjp_autodiff_fn(f[..., None])[0]
        logging.info("vJp_autodiff: %s", vjp_autodiff[:10])

        # Compute error
        error = jnp.linalg.norm(vjp_autodiff - vjp_ref) / jnp.linalg.norm(
            vjp_ref
        )
        logging.info(
            "L=%d: Relative error between autodiff Jvp and reference: %.3e",
            L,
            error,
        )

        rel_diffs.append(error)

    # Plot relative differences vs h
    rel_diffs = np.array(rel_diffs)
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
    ax.set_title("Relative difference between autodiff vJp and reference")
    ax.grid()
    fp = os.path.join(args.output_dir, "autodiff_vJp_relative_diff.pdf")
    plt.savefig(fp)
    plt.close()

    # Save data
    fp = os.path.join(args.output_dir, "autodiff_vJp_relative_diff.npy")
    dd_out = {
        "L_vals": np.array(args.l_vals),
        "rel_diffs": rel_diffs,
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
