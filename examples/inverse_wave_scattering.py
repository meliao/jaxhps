import argparse
import os
import logging

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr
from scipy.io import savemat

from jaxhps import DiscretizationNode2D, Domain, PDEProblem

# from inverse_scattering_utils import (
#     SAMPLE_DOMAIN,
#     K,
#     XMIN,
#     XMAX,
#     YMIN,
#     YMAX,
#     OBSERVATION_BOOLS,
#     SOURCE_DIRS,
# )
from check_autodiff_Jvp import coeffs_to_uscat
from check_autodiff_vJp import quad_weights_hps_panel
from wave_scattering_utils import load_SD_matrices
from plotting_utils import (
    make_scaled_colorbar,
    TICKSIZE,
    FONTSIZE,
    FIGSIZE,
    get_discrete_cmap,
    parula_cmap,
)
from scattering_potentials import q_gaussian_bumps
from sine_transform import (
    get_freqs_up_to_2k,
    nu_sinetransform,
)

# Disable all matplotlib logging
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True

jax.config.update("jax_default_device", jax.devices("cpu")[0])

plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
plt.rc("text", usetex=True)

# Uncomment for debugging NaNs. Slows code down.
# jax.config.update("jax_debug_nans", True)

XMIN = -1
XMAX = 1
YMIN = -1
YMAX = 1
SOURCE_DIRS = jnp.array([0.0])


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="data/examples/inverse_wave_scattering",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("-L", type=int, default=3)
    parser.add_argument("-p", type=int, default=16)
    parser.add_argument("-k", type=float, default=20.0)
    parser.add_argument(
        "-SD_matrix_fp", type=str, default="../data/examples/SD_matrices"
    )

    return parser.parse_args()


def plot_uscat(
    uscat_regular: jnp.array, observation_pts: jnp.array, plot_fp: str
) -> None:
    uscat_real_fp = plot_fp
    logging.info("plot_uscat: Saving uscat plot to %s", uscat_real_fp)
    fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))
    im = ax.imshow(
        uscat_regular.real,
        cmap="bwr",
        vmin=-1 * uscat_regular.real.max(),
        vmax=uscat_regular.real.max(),
        extent=(XMIN, XMAX, YMIN, YMAX),
    )
    plt.plot(observation_pts[:, 0], observation_pts[:, 1], "x", color="black")

    # plt.plot(observation_pts[:, 0], observation_pts[:, 1], "x", color="black")
    make_scaled_colorbar(im, ax, fontsize=TICKSIZE)
    # Make x and y ticks = [-1, 0, 1]
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    # Make the ticks the correct size
    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE)
    plt.savefig(uscat_real_fp, bbox_inches="tight")
    plt.clf()


def plot_coeffs(
    coeffs_ground_truth: jax.Array,
    coeffs_estimate: jax.Array,
    freqs: jax.Array,
    plot_fp: str,
) -> None:
    """
    Plots the absolute value of the components of the coefficient vector against
    the frequency norm.

    Also plots the absolute error between the ground-truth and estimate coefficients.
    """
    # freq_norms = jnp.linalg.norm(freqs, axis=1)

    fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))

    colors = get_discrete_cmap(2, cmap=parula_cmap)

    diffs = coeffs_ground_truth - coeffs_estimate

    ax.plot(
        jnp.abs(coeffs_ground_truth),
        ".-",
        color=colors[0],
        label="$\\theta^*$",
    )
    ax.plot(
        jnp.abs(diffs),
        ".-",
        color=colors[1],
        label="Estimate",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Frequency norm", fontsize=FONTSIZE)
    ax.set_ylabel("Coefficient magnitude", fontsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE)
    ax.legend(fontsize=TICKSIZE)
    ax.grid()

    fp = plot_fp
    logging.info("plot_coeffs: Saving coeffs plot to %s", fp)
    plt.savefig(fp, bbox_inches="tight")
    plt.clf()


def plot_residuals(residuals: jnp.array, plot_fp: str) -> None:
    fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))

    sqrt_residuals = jnp.sqrt(residuals)

    plt.plot(
        sqrt_residuals,
    )

    plt.yscale("log")
    plt.xlabel("Iteration $t$", fontsize=FONTSIZE)
    plt.ylabel(
        "$ \\| \\mathcal{F}[\\theta^*] - \\mathcal{F}[\\theta_t] \\|_2$",
        fontsize=FONTSIZE,
    )
    # Make the x-ticks integers [0, 5, 10, 15]
    plt.xticks(np.arange(0, 25, 5))
    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE)
    # Turn off the top and right spines
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    plt.grid()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def obj_fn(u_star: jnp.array, u_obs: jnp.array) -> jnp.array:
    diffs = u_star - u_obs
    diffs_conj = jnp.conj(diffs)
    return jnp.sum(diffs * diffs_conj).real


def gauss_newton_iterations(
    u_star: jnp.array,
    x_t: jnp.array,
    niter: int,
    freqs: jax.Array,
    problem: PDEProblem,
    source_dirs: jax.Array,
    S_int: jax.Array,
    D_int: jax.Array,
    reg_lambda: float,
) -> None:
    cond_vals = jnp.zeros((niter,), dtype=jnp.float64)
    resid_norms = jnp.zeros((niter,), dtype=jnp.float64) * jnp.nan

    nobs = u_star.shape[0]
    ntheta = x_t.shape[0]

    iterates = jnp.zeros((niter, ntheta), dtype=jnp.float64) * jnp.nan

    for t in range(niter):
        logging.info("t = %i", t)
        # logging.info("x_t = %s", x_t)
        logging.debug("x_t.devices: %s", x_t.devices())

        def coeffs_to_uscat_fun(coeffs: jax.Array) -> jax.Array:
            return coeffs_to_uscat(
                coeffs=coeffs,
                freqs=freqs,
                problem=problem,
                source_dirs=source_dirs,
                S_int=S_int,
                D_int=D_int,
            )

        u_t, vjp_fn = jax.vjp(coeffs_to_uscat_fun, x_t)

        r_t = u_star - u_t
        logging.debug("r_t has shape: %s", r_t.shape)
        resid_norm = obj_fn(u_star, u_t)

        logging.info("resid norm squared = %s", resid_norm)
        iterates = iterates.at[t].set(x_t.flatten())
        resid_norms = resid_norms.at[t].set(resid_norm)

        def rmatvec_fn(x: jnp.array) -> jnp.array:
            # x has shape (nobs,)
            # Trying to do J^H @ x
            # which is equivalent to (J^T @ x.conj()).conj()
            logging.debug("rmatvec_fn: called")
            x = jax.device_put(x, jax.devices()[0])
            x = jnp.conj(x)
            out = vjp_fn(x[..., None])[0]
            out = jnp.conj(out)

            out = jax.device_put(out, jax.devices("cpu")[0])
            return out

        def matvec_fn(delta: jnp.array) -> jnp.array:
            # Delta has shape (2,)
            logging.debug("matvec_fn: called")
            delta = jax.device_put(delta, jax.devices()[0])
            a, b = jax.jvp(coeffs_to_uscat_fun, (x_t,), (delta,))
            b = jax.device_put(b, jax.devices("cpu")[0])
            return b

        linop = LinearOperator(
            (nobs, ntheta),
            matvec=matvec_fn,
            rmatvec=rmatvec_fn,
            dtype=jnp.complex128,
        )

        lsqr_out = lsqr(linop, r_t, damp=reg_lambda, atol=1e-06, btol=1e-06)
        delta_t = lsqr_out[0]
        cond = lsqr_out[6]
        cond_vals = cond_vals.at[t].set(cond)
        logging.info(
            "LSQR returned after %i iters with cond=%s", lsqr_out[2], cond
        )

        # logging.info("delta_t = %s", delta_t)
        x_t = x_t + delta_t

    return iterates, resid_norms, cond_vals


def main(args: argparse.Namespace) -> None:
    # Set up plotting directory
    os.makedirs(args.plots_dir, exist_ok=True)
    logging.info("Plots will be saved to %s", args.plots_dir)

    # Set up the PDEProblem object
    root = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
    domain = Domain(p=args.p, q=args.p - 2, root=root, L=args.L)
    problem = PDEProblem(domain=domain, use_ItI=True, eta=args.k)
    quad_weights_single_panel = quad_weights_hps_panel(domain)
    n_leaves = domain.n_leaves
    quad_weights_all_panels = jnp.repeat(
        quad_weights_single_panel[None, ...], n_leaves, axis=0
    )
    logging.info(
        "quad_weights_all_panels shape: %s", quad_weights_all_panels.shape
    )

    # Load the SD matrices
    SD_matrix_fp = os.path.join(
        args.SD_matrix_fp,
        f"SD_k{int(args.k)}_n{args.p - 2}_nside{2**args.L}_dom1.mat",
    )
    S_int, D_int = load_SD_matrices(SD_matrix_fp)

    # Set up ground-truth q
    q_hps = 0.5 * q_gaussian_bumps(domain.interior_points)

    # Set up the sine basis
    freqs = get_freqs_up_to_2k(args.k, root)  # [:20]
    logging.info("Number of optimization variables: %s", freqs.shape[0])

    theta_star = nu_sinetransform(
        samples=q_hps,
        spatial_points=domain.interior_points,
        freqs=freqs,
        quad_weights=quad_weights_all_panels,
    )

    theta_t = jnp.zeros_like(theta_star)
    # Initialize with the ground-truth lowest 5 frequencies
    n_init_freqs = 3
    theta_t = theta_t.at[:n_init_freqs].set(theta_star[:n_init_freqs])

    # u_star is the scattered wave field data we get to observe in the inverse problem
    u_star = coeffs_to_uscat(
        theta_star, freqs, problem, SOURCE_DIRS, S_int=S_int, D_int=D_int
    )
    reg_lambda = 0.0

    iterates, resid_norms, cond_vals = gauss_newton_iterations(
        u_star=u_star,
        x_t=theta_t,
        niter=args.n_iter,
        freqs=freqs,
        problem=problem,
        source_dirs=SOURCE_DIRS,
        S_int=S_int,
        D_int=D_int,
        reg_lambda=reg_lambda,
    )

    final_iterate = iterates[-1, :]

    # Plot the ground-truth and estimated coefficients
    coeffs_ground_truth = theta_star
    coeffs_estimate = final_iterate
    coeffs_fp = os.path.join(args.plots_dir, "coeffs.png")
    plot_coeffs(
        coeffs_ground_truth,
        coeffs_estimate,
        freqs,
        coeffs_fp,
    )

    # plot the residuals
    residuals_fp = os.path.join(args.plots_dir, "residuals.png")
    plot_residuals(resid_norms, residuals_fp)

    # Save the data
    out_dd = {
        "iterates": iterates,
        "resid_norms": resid_norms,
        "u_star": u_star,
        "theta_star": theta_star,
        "cond_vals": cond_vals,
    }
    save_fp = os.path.join(args.plots_dir, "iterates_data.mat")
    savemat(save_fp, out_dd)


if __name__ == "__main__":
    args = setup_args()
    # Get the root logger directly
    root_logger = logging.getLogger()

    # Set the level directly on the root logger
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Clear any existing handlers to avoid duplicate log messages
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Configure the logger
    FMT = "%(asctime)s:jaxhps: %(levelname)s - %(message)s"
    TIMEFMT = "%Y-%m-%d %H:%M:%S"
    handler = logging.StreamHandler()
    formatter = logging.Formatter(FMT, datefmt=TIMEFMT)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    main(args)
