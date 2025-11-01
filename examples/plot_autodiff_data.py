import argparse
import os
import logging

import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

from plotting_utils import (
    FIGSIZE,
    FONTSIZE,
    TICKSIZE,
    get_discrete_cmap,
    parula_cmap,
)

plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
plt.rc("text", usetex=True)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the autodiff convergence experiment"
    )

    parser.add_argument(
        "-Jvp_fp",
        default="data/examples/autodiff_checks/autodiff_Jvp_relative_diff.npy",
    )
    parser.add_argument(
        "-vJp_fp",
        default="data/examples/autodiff_checks/autodiff_vJp_relative_diff.npy",
    )
    parser.add_argument(
        "-output_dir",
        type=str,
        default="data/examples/autodiff_checks/",
        help="Directory to save plots and data.",
    )
    parser.add_argument(
        "-p",
        type=int,
        default=16,
        help="Order of discretization used in the autodiff experiments.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, turn on debug logging.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Load Jvp data
    dd_jvp = np.load(args.Jvp_fp, allow_pickle=True).item()
    L_vals_jvp = dd_jvp["L_vals"]
    rel_diffs_jvp = dd_jvp["rel_diffs"]

    # Load vJp data
    dd_vJp = np.load(args.vJp_fp, allow_pickle=True).item()
    rel_diffs_vJp = dd_vJp["rel_diffs"]

    h_vals = 2 / (2**L_vals_jvp)
    one_over_h_vals = 1 / h_vals

    h_vals_linfit = jnp.linspace(0.1, 0.25, 10)
    const_p = 100000.0
    one_over_h_vals_linfit = 1 / h_vals_linfit
    h_tothe_pm2 = const_p * h_vals_linfit ** (args.p - 2)

    # Plot the results
    fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
    ax = fig.add_subplot(1, 1, 1)

    cmap = get_discrete_cmap(2, cmap=parula_cmap)
    ax.plot(
        one_over_h_vals,
        rel_diffs_jvp,
        ".-",
        label="$J[\\theta ] v$",
        markersize=10,
        color=cmap[0],
    )
    ax.plot(
        one_over_h_vals,
        rel_diffs_vJp,
        ".-",
        label="$v^{\\top} J[\\theta ]$",
        markersize=10,
        color=cmap[1],
    )
    ax.plot(
        one_over_h_vals_linfit,
        h_tothe_pm2,
        "k--",
        label="$O(h^{p-2})$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1/h$", fontsize=FONTSIZE)
    ax.set_ylabel("Relative $\\ell_\\infty$ error", fontsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICKSIZE)
    ax.legend(fontsize=TICKSIZE)
    ax.grid()

    fp = os.path.join(args.output_dir, "autodiff_Jvp_vJp_convergence.pdf")
    plt.savefig(fp, bbox_inches="tight")
    plt.close()


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
