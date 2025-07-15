import argparse
import os
import logging

import jax.numpy as jnp
import jax
from scipy.io import savemat

from wave_scattering_utils import (
    solve_scattering_problem,
    load_SD_matrices,
    get_uin,
)
from scattering_potentials import (
    q_gaussian_bumps,
    q_GBM_1,
    q_horizontally_graded,
    q_luneburg,
    q_vertically_graded,
)
from plotting_utils import plot_field_for_wave_scattering_experiment
from jaxhps import (
    DiscretizationNode2D,
    Domain,
    PDEProblem,
    solve,
    build_solver,
)

# Silence matplotlib debug messages
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True

jax.config.update("jax_default_device", jax.devices("cpu")[0])


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute reference solution for wave scattering problem."
    )

    parser.add_argument(
        "-l",
        type=int,
        default=2,
        help="Number of levels in the quadtree.",
    )
    parser.add_argument(
        "-p",
        type=int,
        default=16,
        help="Chebyshev polynomial order.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=128,
        help="Number of points per dimension for the output regular grid.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    xmin = -1.0
    xmax = 1.0
    ymin = -1.0
    ymax = 1.0

    # Generate DiscretizationNode2D and Domain objects
    root = DiscretizationNode2D(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )
    domain = Domain(p=args.p, q=args.p - 2, root=root, L=args.l)

    # Get a regular discretization of the domain
    xvals = jnp.linspace(xmin, xmax, args.n, endpoint=False)
    yvals = jnp.linspace(ymin, ymax, args.n, endpoint=False)
    xgrid, ygrid = jnp.meshgrid(xvals, yvals, indexing="ij")
    regular_grid = jnp.stack([xgrid, ygrid], axis=-1)

    # The PDE problem that we want to solve is - \Delta u + q(x) u = 0
    lap_coeffs = -1 * jnp.ones_like(domain.interior_points[..., 0])

    q_0_reg = jnp.exp(
        -((xgrid**2 + ygrid**2) / 0.1**2)
    )  # Gaussian bump potential

    # Plot the potential
    plot_field_for_wave_scattering_experiment(
        q_0_reg, regular_grid, cmap_str="plasma"
    )

    # q_1 is a Gaussian bump potential with a small offset from center
    q_1_reg = jnp.exp(-(((xgrid + 0.1) ** 2 + (ygrid + 0.1) ** 2) / 0.1**2))
    plot_field_for_wave_scattering_experiment(
        q_1_reg, regular_grid, cmap_str="plasma"
    )

    pde_problem_1 = PDEProblem(
        domain=domain,
        source=jnp.zeros_like(domain.interior_points[..., 0]),
        D_xx_coefficients=lap_coeffs,
        D_yy_coefficients=lap_coeffs,
    )


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
