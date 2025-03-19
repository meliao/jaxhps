import argparse
import os
import logging

import jax.numpy as jnp
import jax
from scipy.io import savemat
from scipy.io import loadmat

from hps.src.logging_utils import FMT, TIMEFMT
from hps.src.wave_scattering_utils import solve_scattering_problem, load_SD_matrices
from hps.src.plotting import plot_field_for_wave_scattering_experiment
from hps.src.scattering_potentials import (
    q_luneburg,
    q_vertically_graded,
    q_horizontally_graded,
    q_gaussian_bumps,
    q_GBM_1,
)
from hps.src.config import DEVICE_ARR, HOST_DEVICE

# Silence matplotlib debug messages
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute reference solution for wave scattering problem."
    )
    # parser.add_argument(
    #     "--output_dir",
    #     default="data/wave_scattering",
    #     type=str,
    #     help="Directory to save the reference solution.",
    # )
    parser.add_argument(
        "-l",
        type=int,
        default=8,
        help="Number of levels in the quadtree.",
    )
    parser.add_argument(
        "-p",
        type=int,
        default=16,
        help="Chebyshev polynomial order.",
    )
    parser.add_argument("--n_time_samples", default=5, type=int)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--SD_matrix_prefix",
        default="data/wave_scattering/SD_matrices",
    )
    parser.add_argument("-k", type=float, default=100.0, help="wavenumber")
    parser.add_argument(
        "--scattering_potential",
        default="gauss_bumps",
        help="Scattering potential to use.",
        choices=[
            "luneburg",
            "vertically_graded",
            "horizontally_graded",
            "gauss_bumps",
            "GBM_1",
        ],
    )

    return parser.parse_args()


def compute_linf_error(computed_soln: jnp.array, ref_fp: str) -> float:

    ref_soln = loadmat(ref_fp)["uscat"]

    diffs = computed_soln - ref_soln

    logging.debug("compute_linf_error: ref_soln shape: %s", ref_soln.shape)
    logging.debug("compute_linf_error: computed_soln shape: %s", computed_soln.shape)

    linf_error = jnp.max(jnp.abs(diffs))
    rel_linf_error = linf_error / jnp.max(jnp.abs(ref_soln))

    logging.info("compute_linf_error: absolute Linf error: %f", linf_error)
    logging.info("compute_linf_error: relative Linf error: %f", rel_linf_error)
    return rel_linf_error


def main(args: argparse.Namespace) -> None:
    """
    Does the following:

    1. Solves the wave scattering problem on a quadtree with specified parameters:
        - l: number of levels in the quadtree
        - p: Chebyshev polynomial order
        - k: Incident wave frequency
    2. Interpolates the solution onto a regular grid with specified number of points:
        - n: number of points per dimension for the output regular grid
    3. Computes errors against reference solution
    4. Saves the interpolated solution to a file.
        - The file is saved in the specified output directory.
    5. Plots the interpolated solution.
    """

    # Check the scattering potential argument
    if args.scattering_potential == "luneburg":
        args.data_dir = f"data/wave_scattering/luneburg_k_{int(args.k)}"
        q_fn_handle = q_luneburg
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "vertically_graded":
        args.data_dir = f"data/wave_scattering/vertically_graded_k_{int(args.k)}"
        q_fn_handle = q_vertically_graded
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "horizontally_graded":
        args.data_dir = f"data/wave_scattering/horizontally_graded_k_{int(args.k)}"
        q_fn_handle = q_horizontally_graded
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "gauss_bumps":
        args.data_dir = f"data/wave_scattering/gauss_bumps_k_{int(args.k)}"
        q_fn_handle = q_gaussian_bumps
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "GBM_1":
        args.data_dir = f"data/wave_scattering/GBM_1_k_{int(args.k)}"
        q_fn_handle = q_GBM_1
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    else:
        raise ValueError("Invalid scattering potential")

    # Pre-sets determined by reference solution
    args.n = 500

    # Make sure data dir exists
    assert os.path.isdir(
        args.data_dir
    ), f"{args.data_dir} must be created by running wave_scattering_compute_reference_soln.py"

    output_dir = os.path.join(args.data_dir, f"approx_soln_p_{args.p}_l_{args.l}")
    logging.info("Outputs will be saved to %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    reference_fp = os.path.join(args.data_dir, "reference_solution.mat")

    # Find the right S and D file
    q = args.p - 2
    nside = 2**args.l
    k_str = str(int(args.k))
    S_D_matrices_fp = os.path.join(
        args.SD_matrix_prefix, f"SD_k{k_str}_n{q}_nside{nside}_dom1.mat"
    )

    assert os.path.exists(S_D_matrices_fp), f"Can't find file: {S_D_matrices_fp}"

    domain_corners = jnp.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    source_dirs = jnp.array(
        [
            0.0,
        ]
    )

    wave_freq = args.k

    logging.debug("Loading S and D from disk...")
    S, D = load_SD_matrices(S_D_matrices_fp)
    S_gpu = jax.device_put(S, DEVICE_ARR[0])
    D_gpu = jax.device_put(D, DEVICE_ARR[0])

    # First one to JIT-compile the code
    uscat, target_pts, solve_time = solve_scattering_problem(
        l=args.l,
        p=args.p,
        n=args.n,
        k=wave_freq,
        q_fn=q_fn_handle,
        domain_corners=domain_corners,
        source_dirs=source_dirs,
        S=S_gpu,
        D=D_gpu,
    )

    # Measure solution times
    times = jnp.zeros(args.n_time_samples, dtype=jnp.float64)
    for i in range(args.n_time_samples):
        uscat, target_pts, solve_time = solve_scattering_problem(
            l=args.l,
            p=args.p,
            n=args.n,
            k=wave_freq,
            q_fn=q_fn_handle,
            domain_corners=domain_corners,
            source_dirs=source_dirs,
            S=S_gpu,
            D=D_gpu,
        )
        times = times.at[i].set(solve_time)
        logging.info(
            "Time sample %i / %i was %f seconds", i + 1, args.n_time_samples, solve_time
        )

    mean_total = jnp.mean(times)
    stddev_total = jnp.std(times)
    logging.info("Total time results: %f sec +/- %f", mean_total, stddev_total)
    # Step 3: Compute errors against reference. This function
    # also logs the error
    rel_linf_error = compute_linf_error(uscat, reference_fp)

    # Step 4: Save the interpolated solution to a file.
    fp_out = os.path.join(output_dir, "data.mat")
    savemat(
        fp_out,
        {
            "uscat": uscat,
            "target_pts": target_pts,
            "rel_linf_error": rel_linf_error,
            "times": times,
        },
    )

    # Step 5: Plot
    logging.info("Plotting results...")
    # plot q
    q_vals = q_fn_handle(target_pts)
    plot_field_for_wave_scattering_experiment(
        q_vals, target_pts, title="q(x)", save_fp=os.path.join(output_dir, "q.png")
    )

    # plot real part of uscat
    plot_field_for_wave_scattering_experiment(
        jnp.real(uscat),
        target_pts,
        use_bwr_cmap=True,
        title="uscat real",
        save_fp=os.path.join(output_dir, "uscat_real.png"),
    )

    # plot the imaginary part of uscat
    plot_field_for_wave_scattering_experiment(
        jnp.imag(uscat),
        target_pts,
        use_bwr_cmap=True,
        title="uscat imag",
        save_fp=os.path.join(output_dir, "uscat_imag.png"),
    )

    # plot the absolute value of uscat
    plot_field_for_wave_scattering_experiment(
        jnp.abs(uscat),
        target_pts,
        use_bwr_cmap=False,
        cmap_str="hot",
        title="abs uscat",
        save_fp=os.path.join(output_dir, "uscat_abs.png"),
    )

    logging.info("Finished")


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
