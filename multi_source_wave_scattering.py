import argparse
import os
import logging

import jax.numpy as jnp
import jax
from scipy.io import savemat

from hps.src.logging_utils import FMT, TIMEFMT
from hps.src.wave_scattering_utils import solve_scattering_problem, get_uin
from hps.src.scattering_potentials import (
    q_gaussian_bumps,
    q_luneburg,
    q_vertically_graded,
    q_horizontally_graded,
    q_gaussian_bumps,
    q_GBM_1,
)
from hps.src.plotting import plot_field_for_wave_scattering_experiment


# Silence matplotlib debug messages
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make images for a large number of sources."
    )
    parser.add_argument(
        "-l",
        type=int,
        default=8,
        help="Number of levels in the quadtree.",
    )
    parser.add_argument(
        "-p",
        type=int,
        default=22,
        help="Chebyshev polynomial order.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=500,
        help="Number of points per dimension for the output regular grid.",
    )
    parser.add_argument("-k", type=float, default=100.0, help="Wavenumber.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--SD_matrix_prefix",
        default="data/wave_scattering/SD_matrices",
    )
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
    parser.add_argument(
        "--dirs", type=int, default=100, help="Number of source directions."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Does the following:

    1. Solves the wave scattering problem on a quadtree with specified parameters:
        - l: number of levels in the quadtree
        - p: Chebyshev polynomial order
        - k: Incident wave frequency
    2. Interpolates the solution onto a regular grid with specified number of points:
        - n: number of points per dimension for the output regular grid
    3. Saves the interpolated solutions to image files, one
    for each source direction.
    4. Plots the interpolated solution.
    """
    # Check the scattering potential argument
    if args.scattering_potential == "luneburg":
        args.output_dir = f"data/multi_source_wave_scattering/luneburg_k_{int(args.k)}"
        q_fn_handle = q_luneburg
        plot_utot = False
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "vertically_graded":
        args.output_dir = (
            f"data/multi_source_wave_scattering/vertically_graded_k_{int(args.k)}"
        )
        q_fn_handle = q_vertically_graded
        plot_utot = False
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "horizontally_graded":
        args.output_dir = (
            f"data/multi_source_wave_scattering/horizontally_graded_k_{int(args.k)}"
        )
        q_fn_handle = q_horizontally_graded
        plot_utot = False
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "gauss_bumps":
        args.output_dir = (
            f"data/multi_source_wave_scattering/gauss_bumps_k_{int(args.k)}"
        )
        q_fn_handle = q_gaussian_bumps
        plot_utot = False
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    elif args.scattering_potential == "GBM_1":
        args.output_dir = f"data/multi_source_wave_scattering/GBM_1_k_{int(args.k)}"
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
        plot_utot = False
        q_fn_handle = q_GBM_1
    else:
        raise ValueError("Invalid scattering potential")

    output_dir = args.output_dir
    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Outputs will be written to %s", output_dir)

    # Steps 1 and 2 are combined in the solve_scattering_problem function.
    q = args.p - 2
    nside = 2**args.l
    k_str = str(int(args.k))
    S_D_matrices_fp = os.path.join(
        args.SD_matrix_prefix, f"SD_k{k_str}_n{q}_nside{nside}_dom1.mat"
    )

    domain_corners = jnp.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    source_dirs = jnp.linspace(0.0, 2 * jnp.pi, args.dirs, endpoint=False)

    logging.info("Solving wave scattering problem for %i source directions", args.dirs)

    wave_freq = args.k

    uscat, target_pts, solve_time, _ = solve_scattering_problem(
        l=args.l,
        p=args.p,
        n=args.n,
        k=wave_freq,
        q_fn=q_fn_handle,
        domain_corners=domain_corners,
        source_dirs=source_dirs,
        S_D_matrices_fp=S_D_matrices_fp,
        zero_impedance=False,
        return_utot=plot_utot,
    )

    _, _, t, _ = solve_scattering_problem(
        l=args.l,
        p=args.p,
        n=args.n,
        k=wave_freq,
        q_fn=q_fn_handle,
        domain_corners=domain_corners,
        source_dirs=source_dirs,
        S_D_matrices_fp=S_D_matrices_fp,
        zero_impedance=False,
        return_utot=plot_utot,
    )
    logging.info("Solve time = %s", t)

    uin = get_uin(args.k, target_pts, source_dirs)
    utot = uin + uscat
    # Expect utot to have (n, n, dirs) shape
    logging.info("utot shape: %s", utot.shape)

    maxval = jnp.max(jnp.abs(utot))

    # Save the solutions to .png files
    for i in range(args.dirs):
        fp_i = os.path.join(output_dir, f"multi_source_wave_scattering_{i}.png")
        plot_field_for_wave_scattering_experiment(
            utot[..., i].real,
            target_pts,
            cmap_str="parula",
            save_fp=fp_i,
            maxval=maxval,
            minval=-1 * maxval,
            dpi=100,
        )


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
