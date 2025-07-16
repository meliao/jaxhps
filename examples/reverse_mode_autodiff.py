import argparse
import os
import logging

import jax.numpy as jnp
import jax

from plotting_utils import plot_field_for_wave_scattering_experiment
from jaxhps import (
    DiscretizationNode2D,
    Domain,
    PDEProblem,
    solve,
    build_solver,
    plot_soln_from_cheby_nodes,
    rearrange_indices_ext_int,
)
from jaxhps.quadrature import (
    first_kind_chebyshev_points,
    chebyshev_points,
    barycentric_lagrange_interpolation_matrix_1D,
)

import sys

sys.path.append("..")
# from src.jaxhps._utils import plot_soln_from_cheby_nodes

# Silence matplotlib debug messages
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True

jax.config.update("jax_default_device", jax.devices("cpu")[0])


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-output_dir",
        type=str,
        default="data/reverse_mode_autodiff",
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


def reg_q_to_soln(
    q_hps: jax.Array,
    pde_problem: PDEProblem,
    xvals: jax.Array,
    yvals: jax.Array,
) -> jax.Array:
    # # Check if the interpolated potential is NaN
    # if jnp.isnan(q_hps).any():
    #     raise ValueError("The interpolated potential contains NaN values.")

    # Update the PDE problem with the interpolated potential
    lap_coeffs = jnp.ones_like(q_hps) + q_hps
    pde_problem.update_coefficients(
        D_xx_coefficients=lap_coeffs, D_yy_coefficients=lap_coeffs
    )

    # Solve the PDE problem
    build_solver(pde_problem)
    logging.info("reg_q_to_soln: Built solver for PDE problem.")
    bdry_data = jnp.zeros_like(pde_problem.domain.boundary_points[..., 0])

    usoln_hps = solve(pde_problem, bdry_data)
    logging.info("reg_q_to_soln: Computed solution on HPS points.")
    # Check if the solution is NaN
    if jnp.isnan(usoln_hps).any():
        raise ValueError(
            "The solution contains NaN values. Check the input data."
        )

    return usoln_hps

    # # Interpolate the solution back to the regular grid
    # usoln_reg, _ = pde_problem.domain.interp_from_interior_points(
    #     usoln_hps, xvals, yvals
    # )
    # logging.info("reg_q_to_soln: Interpolated solution back to regular grid.")
    # return usoln_reg


def autodiff_of_B(B: jax.Array, p: int) -> None:
    cheb_pts = first_kind_chebyshev_points(p - 2)
    yvals = jnp.flipud(cheb_pts)
    xgrid, ygrid = jnp.meshgrid(cheb_pts, yvals, indexing="ij")
    cheb_grid_first = jnp.stack([xgrid.flatten(), ygrid.flatten()], axis=-1)

    cheb_pts_second = chebyshev_points(p)
    yvals_second = jnp.flipud(cheb_pts_second)
    xgrid_second, ygrid_second = jnp.meshgrid(
        cheb_pts_second, yvals_second, indexing="ij"
    )
    cheb_grid_second = jnp.stack(
        [xgrid_second.flatten(), ygrid_second.flatten()], axis=-1
    )
    r = rearrange_indices_ext_int(p)
    cheb_grid_second = cheb_grid_second[r]

    # f(x) = sin(x)
    f = jnp.sin(cheb_grid_first[..., 0])

    f_expected = jnp.sin(cheb_grid_second[..., 0])

    f_computed = B.T @ f

    plot_soln_from_cheby_nodes(cheb_grid_second, None, f_expected, f_computed)


def lagrange_interpolation_firstkind_to_secondkind(p: int) -> jax.Array:
    pts_firstkind = first_kind_chebyshev_points(p - 2)
    pts_secondkind = chebyshev_points(p)
    logging.info(
        f"First-kind Chebyshev points: {pts_firstkind.shape}, Second-kind Chebyshev points: {pts_secondkind.shape}"
    )

    # Compute a barycentric interpolation matrix interpolating from first-kind to second-kind Chebyshev points
    B = barycentric_lagrange_interpolation_matrix_1D(
        pts_firstkind, pts_secondkind
    )
    logging.info(f"Barycentric interpolation matrix shape: {B.shape}")

    def f(x):
        return jnp.sin(x)

    f_first = f(pts_firstkind)
    f_second = f(pts_secondkind)
    f_interp = B @ f_first

    diffs = f_interp - f_second
    logging.info(f"Max difference: {jnp.max(jnp.abs(diffs))}")


def main(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    logging.info(f"Output directory: {args.output_dir}")
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

    # The PDE problem that we want to solve is q_0(x) \Delta u  = f
    q_0_reg = 0.01 * jnp.exp(  # noqa: F841
        -((xgrid**2 + ygrid**2) / 0.1**2)
    )  # Gaussian bump potential
    q_0_hps = 0.01 * jnp.exp(
        -(
            (
                domain.interior_points[..., 0] ** 2
                + domain.interior_points[..., 1] ** 2
            )
            / 0.1**2
        )
    )  # Gaussian bump potential on HPS points

    # f is -2 pi^2 sin(pi x - pi) * sin(pi y - pi)
    f_hps = (
        -2
        * (jnp.pi / 2) ** 2
        * jnp.sin(jnp.pi / 2 * domain.interior_points[..., 0] - jnp.pi / 2)
        * jnp.sin(jnp.pi / 2 * domain.interior_points[..., 1] - jnp.pi / 2)
    )
    f_reg = (  # noqa: F841
        -2
        * jnp.pi**2
        * jnp.sin(jnp.pi / 2 * xgrid - jnp.pi / 2)
        * jnp.sin(jnp.pi / 2 * ygrid - jnp.pi / 2)
    )  # Regular grid version of f

    # Plot the potential
    # plot_field_for_wave_scattering_experiment(
    #     q_0_reg, regular_grid, cmap_str="plasma"
    # )
    # plot_field_for_wave_scattering_experiment(
    #     f_reg, regular_grid, cmap_str="viridis", title="Source term f"
    # )

    # q_1 is a Gaussian bump potential with a small offset from center
    # q_1_reg = 1 + jnp.exp(-(((xgrid + 0.1) ** 2 + (ygrid + 0.1) ** 2) / 0.1**2))
    # plot_field_for_wave_scattering_experiment(
    #     q_1_reg, regular_grid, cmap_str="plasma"
    # )

    pde_problem_1 = PDEProblem(
        domain=domain, source=f_hps, use_rectangular_spectral_collocation=True
    )
    pde_problem_2 = PDEProblem(
        domain=domain, source=f_hps, use_rectangular_spectral_collocation=False
    )

    autodiff_of_B(pde_problem_1.B, args.p)

    # Get the solution for the first potential
    # usoln_1_hps = reg_q_to_soln(q_0_hps, pde_problem_1, xvals, yvals)
    # Interpolate the solution back to the regular grid
    # usoln_1_reg, _ = domain.interp_from_interior_points(
    #     usoln_1_hps, xvals, yvals
    # )
    # Plot the solution
    # plot_field_for_wave_scattering_experiment(
    #     usoln_1_reg, regular_grid, cmap_str="plasma"
    # )

    expected_soln = jnp.sin(
        jnp.pi / 2 * domain.interior_points[..., 0] - jnp.pi / 2
    ) * jnp.sin(
        jnp.pi / 2 * domain.interior_points[..., 1] - jnp.pi / 2
    )  # Expected solution

    # Now, use jax to compute a vJp
    vjp_fn = jax.vjp(
        lambda x: reg_q_to_soln(x, pde_problem_1, xvals, yvals),
        q_0_hps,
    )[1]

    # Compute the vJp for the expected solution
    vjp_q_0_hps = vjp_fn(expected_soln)[0]
    n_bdry = 4 * args.p - 4
    bdry_pts = vjp_q_0_hps[:, :n_bdry]
    logging.info("bdry_pts: %s", bdry_pts)
    logging.info("Computed vJp for the expected solution.")
    # Show the shape of the vJp
    logging.info(f"Shape of vjp_q_0_hps: {vjp_q_0_hps.shape}")

    # Interpolate the vJp back to the regular grid
    vjp_q_0_reg, _ = domain.interp_from_interior_points(
        vjp_q_0_hps, xvals, yvals
    )
    # Plot it
    fp = os.path.join(args.output_dir, "vjp_q_0_rectangular.pdf")
    plot_field_for_wave_scattering_experiment(
        vjp_q_0_reg,
        regular_grid,
        use_bwr_cmap=True,
        title="vJp using rectangular spectral collocation",
        save_fp=fp,
    )

    # Do the same for the second PDE problem
    vjp_fn_2 = jax.vjp(
        lambda x: reg_q_to_soln(x, pde_problem_2, xvals, yvals),
        q_0_hps,
    )[1]
    vjp_q_0_hps_2 = vjp_fn_2(expected_soln)[0]
    logging.info("Computed vJp for the expected solution (non-rectangular).")
    # Plot the solution
    vjp_q_0_reg_2, _ = domain.interp_from_interior_points(
        vjp_q_0_hps_2, xvals, yvals
    )
    bdry_pts = vjp_q_0_hps_2[:, :n_bdry]
    logging.info("bdry_pts: %s", bdry_pts)
    fp = os.path.join(args.output_dir, "vjp_q_0_current_method.pdf")
    plot_field_for_wave_scattering_experiment(
        vjp_q_0_reg_2,
        regular_grid,
        use_bwr_cmap=True,
        title="vJp using current method",
        save_fp=fp,
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
