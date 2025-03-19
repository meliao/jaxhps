import sys
import argparse
import os
import logging

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from scipy.sparse.linalg import LinearOperator, lsqr, svds
from scipy.io import savemat, loadmat

# Disable all matplorlib logging
# logging.getLogger("matplotlib").setLevel(logging.WARNING)
# logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True
# Uncomment for debugging NaNs. Slows code down.
# jax.config.update("jax_debug_nans", True)

from hps.src.config import DEVICE_ARR, HOST_DEVICE
from hps.src.logging_utils import FMT, TIMEFMT
from hps.src import plotting
from hps.src.quadrature.quad_2D.interpolation import (
    interp_from_nonuniform_hps_to_regular_grid,
)
from hps.src.inverse_scattering_utils import (
    q_point_sources,
    source_locations_to_scattered_field,
    forward_model,
    SAMPLE_TREE,
    L,
    P,
    XMIN,
    XMAX,
    YMIN,
    YMAX,
    OBSERVATION_BOOLS,
    MIN_TOL,
)

N_BUMPS = 4


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plots_dir", type=str, default="data/inverse_wave_scattering")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--plot_objective_fn", action="store_true")
    parser.add_argument("--landweber", action="store_true")
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--n_per_side", type=int, default=10)

    return parser.parse_args()


def plot_uscat(uscat_regular: jnp.array, observation_pts: jnp.array, fp: str) -> None:
    logging.info("plot_uscat: Saving uscat plot to %s", fp)
    plt.imshow(
        uscat_regular,
        cmap="bwr",
        vmin=-1 * uscat_regular.real.max(),
        vmax=uscat_regular.real.max(),
        extent=(XMIN, XMAX, YMIN, YMAX),
    )
    plt.plot(observation_pts[:, 0], observation_pts[:, 1], "x", color="black")

    # plt.plot(observation_pts[:, 0], observation_pts[:, 1], "x", color="black")
    plt.colorbar()
    plt.savefig(fp, bbox_inches="tight")
    plt.clf()

    # uscat_abs_fp = os.path.join(plots_dir, "uscat_ground_truth_abs.png")
    # logging.info("plot_uscat: Saving uscat plot to %s", uscat_abs_fp)
    # plt.imshow(
    #     jnp.abs(uscat_regular),
    #     cmap="hot",
    #     extent=(XMIN, XMAX, YMIN, YMAX),
    # )
    # plt.colorbar()
    # plt.savefig(uscat_abs_fp, bbox_inches="tight")
    # plt.clf()


def plot_iterates(iterates: jnp.array, plots_dir: str, q_evals: jnp.array) -> None:
    """
    Make a plot of the evaluations of the ground-truth q and then draw arrows showing the iterates.
    """

    # plot q
    plt.imshow(q_evals, cmap="plasma", extent=(XMIN, XMAX, YMIN, YMAX))
    plt.colorbar()
    # logging.info("plot_iterates: iterates shape: %s", iterates.shape)

    n_q = iterates.shape[1] // 2

    # plot arrows for iterates
    for i in range(iterates.shape[0] - 1):
        for j in range(n_q):
            current_iterate = iterates[i, j * 2 : j * 2 + 2]
            next_iterate = iterates[i + 1, j * 2 : j * 2 + 2]
            # logging.info("plot_iterates: current_iterate: %s", current_iterate.shape)
            # logging.info("plot_iterates: next_iterate: %s", next_iterate.shape)
            plt.arrow(
                current_iterate[0],
                current_iterate[1],
                next_iterate[0] - current_iterate[0],  # dx
                next_iterate[1] - current_iterate[1],  # dy
                color="green",
                head_width=0.02,
            )
    fp = os.path.join(plots_dir, "iterates.png")
    plt.savefig(fp, bbox_inches="tight")
    plt.clf()


def obj_fn(u_star: jnp.array, u_obs: jnp.array) -> jnp.array:
    diffs = u_star - u_obs
    diffs_conj = jnp.conj(diffs)
    return jnp.sum(diffs * diffs_conj).real


def plot_residuals(residuals: jnp.array, plots_dir: str) -> None:
    plt.plot(residuals)
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.yscale("log")
    fp = os.path.join(plots_dir, "residuals.png")
    plt.savefig(fp, bbox_inches="tight")
    plt.clf()


def plot_objective_function(plots_dir: str, u_star: str, n_per_side: int) -> None:

    xmin = -0.2
    xmax = 0.2
    ymin = -0.4
    ymax = -0.2
    # Set up grid
    x = jnp.linspace(xmin, xmax, n_per_side)
    y = jnp.linspace(ymin, ymax, n_per_side)
    y = jnp.flip(y)
    xx, yy = jnp.meshgrid(x, y)
    grid = jnp.stack([xx, yy], axis=-1)

    # Evaluate the objective function on the grid
    obj_fn_vals = jnp.zeros((n_per_side, n_per_side), dtype=jnp.float64)
    for i in range(n_per_side):
        for j in range(n_per_side):
            x_t = grid[i, j]
            logging.debug("plot_objective_function: x_t shape: %s", x_t.shape)
            u_t = forward_model(x_t)
            resid_norm = obj_fn(u_star, u_t)
            obj_fn_vals = obj_fn_vals.at[i, j].set(resid_norm)
        logging.info("plot_objective_function: Finished row %i / %i", i + 1, n_per_side)

    # Save the values of the objective function
    out_dd = {
        "obj_fn_vals": obj_fn_vals,
        "grid": grid,
    }
    save_fp = os.path.join(plots_dir, "obj_fn_data.mat")
    savemat(save_fp, out_dd)

    # Plot the objective function
    plt.imshow(obj_fn_vals, cmap="plasma", extent=(xmin, xmax, ymin, ymax))
    plt.colorbar()
    fp = os.path.join(plots_dir, "objective_function.png")
    plt.savefig(fp, bbox_inches="tight")
    plt.clf()

    # Plot the x values of the grid to make sure the plotting code is correct
    plt.imshow(grid[:, :, 0], cmap="plasma", extent=(xmin, xmax, ymin, ymax))
    plt.colorbar()
    fp = os.path.join(plots_dir, "grid_x.png")
    plt.savefig(fp, bbox_inches="tight")
    plt.clf()

    # Plot the y values of the grid to make sure the plotting code is correct
    plt.imshow(grid[:, :, 1], cmap="plasma", extent=(xmin, xmax, ymin, ymax))
    plt.colorbar()
    fp = os.path.join(plots_dir, "grid_y.png")
    plt.savefig(fp, bbox_inches="tight")
    plt.clf()


def landweber_iterations(
    u_star: jnp.array, x_t: jnp.array, niter: int, step_size: float
) -> None:
    iterates = jnp.zeros((niter, 2), dtype=jnp.float64)
    resid_norms = jnp.zeros((niter,), dtype=jnp.float64)

    for t in range(niter):
        u_t, vjp_fn = jax.vjp(forward_model, x_t)
        r_t = u_t - u_star
        resid_norm = jnp.linalg.norm(r_t) ** 2

        # Iteration is x_{t+1} = x_t - step_size * grad f(x_t)
        grad_f = vjp_fn(r_t)[0]
        x_t = x_t - step_size * grad_f

        iterates = iterates.at[t].set(x_t.flatten())
        resid_norms = resid_norms.at[t].set(resid_norm)

        # Logging
        logging.info("t = %i", t)
        logging.info("x_t = %s", x_t)
        logging.info("resid norm squared = %s", resid_norm)

    return iterates, resid_norms


def get_singular_vals(x_t: jnp.array) -> None:
    vv = jax.device_put(jnp.eye(2, dtype=jnp.float64), DEVICE_ARR[0])
    x_t = jax.device_put(x_t, DEVICE_ARR[0])
    a, b_0 = jax.jvp(forward_model, (x_t,), (vv[0],))
    a, b_1 = jax.jvp(forward_model, (x_t,), (vv[1],))
    logging.debug("get_singular_vals: b_0.devices() %s", b_0.devices())
    b = jnp.column_stack((b_0, b_1))
    logging.debug("get_singular_vals: b.shape = %s", b.shape)
    logging.debug("get_singular_vals: b.devices() %s", b.devices())
    s = jnp.linalg.svd(b, compute_uv=False)
    logging.debug("get_singular_vals: s = %s", s)
    return s


def gauss_newton_iterations(
    u_star: jnp.array, x_t: jnp.array, niter: int, reg_lambda: float
) -> None:
    cond_vals = jnp.zeros((niter,), dtype=jnp.float64)
    resid_norms = jnp.zeros((niter,), dtype=jnp.float64) * jnp.nan

    nobs = u_star.shape[0]
    ntheta = x_t.shape[0]

    iterates = jnp.zeros((niter, ntheta), dtype=jnp.float64) * jnp.nan

    for t in range(niter):
        logging.info("t = %i", t)
        logging.info("x_t = %s", x_t)
        logging.debug("x_t.devices: %s", x_t.devices())

        u_t, vjp_fn = jax.vjp(forward_model, x_t)

        r_t = u_star - u_t
        logging.debug("r_t has shape: %s", r_t.shape)
        resid_norm = obj_fn(u_star, u_t)

        logging.info("resid norm squared = %s", resid_norm)
        iterates = iterates.at[t].set(x_t.flatten())
        resid_norms = resid_norms.at[t].set(resid_norm)

        def rmatvec_fn(x: jnp.array) -> jnp.array:
            # x has shape (nobs,)
            logging.debug("rmatvec_fn: called")
            x = jax.device_put(x, DEVICE_ARR[0])
            x = jnp.conj(x)
            out = vjp_fn(x)[0]

            out = jax.device_put(out, HOST_DEVICE)
            return out

        def matvec_fn(delta: jnp.array) -> jnp.array:
            # Delta has shape (2,)
            # Check if input contains NaNs
            logging.debug("matvec_fn: called")
            delta = jax.device_put(delta, DEVICE_ARR[0])
            a, b = jax.jvp(forward_model, (x_t,), (delta,))
            b = jax.device_put(b, HOST_DEVICE)
            return b

        linop = LinearOperator(
            (nobs, ntheta),
            matvec=matvec_fn,
            rmatvec=rmatvec_fn,
            dtype=jnp.complex128,
        )

        # s = get_singular_vals(x_t)
        # logging.info("J singular values: %s", s)
        # svd_vals = svd_vals.at[t].set(s)

        lsqr_out = lsqr(linop, r_t, damp=reg_lambda, atol=1e-06, btol=1e-06)
        delta_t = lsqr_out[0]
        cond = lsqr_out[6]
        cond_vals = cond_vals.at[t].set(cond)
        logging.info("LSQR returned after %i iters with cond=%s", lsqr_out[2], cond)

        logging.info("delta_t = %s", delta_t)
        x_t = x_t + delta_t

    return iterates, resid_norms, cond_vals


def main(args: argparse.Namespace) -> None:

    # Set up plotting directory
    os.makedirs(args.plots_dir, exist_ok=True)
    logging.info("Plots will be saved to %s", args.plots_dir)

    # Set up boolean array for observation points

    observation_pts = SAMPLE_TREE.leaf_cheby_points[OBSERVATION_BOOLS].reshape(-1, 2)
    logging.debug("Observation points has shape %s", observation_pts.shape)

    # Set up ground-truth scatterer locations and evaluate the scattering potential by
    # random initialization
    key = jax.random.key(3)
    ground_truth_locations = jax.random.uniform(
        key, minval=-0.5, maxval=0.5, shape=(N_BUMPS, 2)
    )

    logging.info("Ground-truth locations: %s", ground_truth_locations)
    ground_truth_locations = jax.device_put(ground_truth_locations, DEVICE_ARR[0])
    q_evals_hps = q_point_sources(SAMPLE_TREE.leaf_cheby_points, ground_truth_locations)
    n_X = 200
    corners = jnp.array([[XMIN, YMIN], [XMAX, YMIN], [XMAX, YMAX], [XMIN, YMAX]])
    q_evals_regular, regular_grid = interp_from_nonuniform_hps_to_regular_grid(
        root=SAMPLE_TREE.root,
        p=P,
        f_evals=q_evals_hps,
        n_pts=n_X,
    )

    # plot the scattering potential
    q_fp = os.path.join(args.plots_dir, "q_ground_truth.png")
    logging.info("Saving q plot to %s", q_fp)
    plt.imshow(q_evals_regular, cmap="plasma", extent=(XMIN, XMAX, YMIN, YMAX))
    # plt.plot(observation_pts[:, 0], observation_pts[:, 1], "x", color="black")
    plt.colorbar()
    plt.savefig(q_fp, bbox_inches="tight")
    plt.clf()

    uscat_gt, _ = source_locations_to_scattered_field(ground_truth_locations)

    uscat_regular, regular_grid = interp_from_nonuniform_hps_to_regular_grid(
        root=SAMPLE_TREE.root,
        p=P,
        f_evals=uscat_gt,
        n_pts=n_X,
    )

    # plot the scattered field
    uscat_real_fp = os.path.join(args.plots_dir, "uscat_ground_truth_real.png")
    plot_uscat(uscat_regular.real, observation_pts, uscat_real_fp)

    # u_star is the scattered wave field data we get to observe in the inverse problem
    u_star = forward_model(ground_truth_locations)
    nobs = u_star.shape[0]

    if args.plot_objective_fn:
        plot_objective_function(args.plots_dir, u_star, n_per_side=args.n_per_side)
        logging.info("Finished")
        exit(0)

    reg_lambda = 0.0

    # Initialize the optimization variables randomly
    key = jax.random.key(1)
    x_t = jax.random.uniform(key, minval=-0.5, maxval=0.5, shape=(N_BUMPS, 2)).flatten()

    if args.landweber:
        iterates, resid_norms = landweber_iterations(
            u_star, x_t, args.n_iter, step_size=0.5
        )
    else:
        iterates, resid_norms, cond_vals = gauss_newton_iterations(
            u_star, x_t, args.n_iter, reg_lambda
        )

    # plot the iterates
    plot_iterates(iterates, args.plots_dir, q_evals_regular)

    # plot the residuals
    plot_residuals(resid_norms, args.plots_dir)

    # Get final estimate
    x_t = iterates[-1]
    u_scat_est = source_locations_to_scattered_field(x_t)[0]
    u_scat_est_regular, regular_grid = interp_from_nonuniform_hps_to_regular_grid(
        root=SAMPLE_TREE.root,
        p=P,
        f_evals=u_scat_est,
        n_pts=n_X,
    )
    fp = os.path.join(args.plots_dir, "uscat_est.png")
    plot_uscat(u_scat_est_regular.real, observation_pts, fp)

    # Plot the diffs
    diff = uscat_regular.real - u_scat_est_regular.real
    fp = os.path.join(args.plots_dir, "uscat_diff.png")
    plot_uscat(diff, observation_pts, fp)

    # Save the data
    out_dd = {
        "iterates": iterates,
        "resid_norms": resid_norms,
        "u_star": u_star,
        "ground_truth_locations": jax.device_put(ground_truth_locations, HOST_DEVICE),
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
    handler = logging.StreamHandler()
    formatter = logging.Formatter(FMT, datefmt=TIMEFMT)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    main(args)
