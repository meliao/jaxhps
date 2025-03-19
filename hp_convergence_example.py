import sys
import argparse
import os
import logging

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse.linalg import LinearOperator, lsqr, svds
from scipy.io import savemat, loadmat

# Disable all matplorlib logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# Uncomment for debugging NaNs. Slows code down.
# jax.config.update("jax_debug_nans", True)

from hps.src.logging_utils import FMT, TIMEFMT
from hps.src import plotting
from hps.src.solver_obj import SolverObj, create_solver_obj_2D
from hps.src.up_down_passes import (
    local_solve_stage,
    build_stage,
    down_pass,
    fused_pde_solve_2D,
    fused_pde_solve_2D_ItI,
)
from hps.accuracy_checks.utils import plot_soln_from_cheby_nodes
from hps.src.quadrature.trees import Node


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots_dir", type=str, default="data/hp_convergence")
    parser.add_argument("--DtN", action="store_true")
    parser.add_argument("--ItI", action="store_true")
    parser.add_argument("--p_vals", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--l_vals", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


XMIN = -1.0
XMAX = 1.0
YMIN = -1.0
YMAX = 1.0
CORNERS = jnp.array([[XMIN, YMIN], [XMAX, YMIN], [XMAX, YMAX], [XMIN, YMAX]])


K = 5
LAMBDA = 10
PI =  jnp.pi

def problem_1_homog_soln(x: jnp.array) -> jnp.array:
    """
    Expect x to have shape (..., 2).
    Output has shape (...)

    w(x,y) = e^{kx} sin(ky)
    """
    return jnp.zeros_like(x[..., 0])
    # return jnp.exp(K * x[..., 0]) * jnp.sin(K * x[..., 1])


def problem_1_part_soln(x: jnp.array) -> jnp.array:
    """
    Expect x to have shape (..., 2).
    Output has shape (...)

    v(x,y) =  sin(2 pi lambda x) sin(2 pi y))
    """
    return jnp.sin(PI * LAMBDA * x[..., 0]) * jnp.sin(PI * x[..., 1])


def problem_1_lap_coeffs(x: jnp.array) -> jnp.array:
    """
    Expect x to have shape (..., 2).
    Output has shape (...)

    c(x,y) = 1
    """
    return jnp.ones_like(x[..., 0])


def problem_1_d_x_coeffs(x: jnp.array) -> jnp.array:
    """
    Expect x to have shape (..., 2)
    Output has shape (...)

    c(x,y) = -cos(ky)
    """
    return -1 * jnp.cos(K * x[..., 1])


def problem_1_d_y_coeffs(x: jnp.array) -> jnp.array:
    """
    Expect x to have shape (..., 2)
    Output has shape (...)

    c(x,y) = sin(ky)
    """
    return jnp.sin(K * x[..., 1])


def problem_1_source(x: jnp.array) -> jnp.array:
    """
    Expect x to have shape (..., 2).
    Output has shape (...)

    f(x,y) = -\lamba^2 sin(\lambda x) - \lambda cos(\lambda x) cos(k y)
    """
    term_1 = -1 * (PI**2) *  (1 + LAMBDA**2) * problem_1_part_soln(x)
    term_2 = -1 * PI * LAMBDA * jnp.cos(PI * LAMBDA * x[..., 0]) * jnp.sin(PI * x[..., 1]) * jnp.cos(K * x[..., 1])
    term_3 = PI * jnp.sin(PI*LAMBDA * x[..., 0])* jnp.cos(PI * x[..., 1]) * jnp.sin(K * x[..., 1])
    return term_1 + term_2 + term_3


def problem_1(l_vals: int, p_vals: int) -> None:

    errors = jnp.zeros((len(l_vals), len(p_vals)), dtype=jnp.float64)

    for i, l in enumerate(l_vals):
        l = int(l)  # Cast from jax array to int
        for j, p in enumerate(p_vals):
            p = int(p)  # Cast from jax array to int
            # Create the tree
            root = Node(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
            tree = create_solver_obj_2D(
                p=p, q=p - 2, root=root, uniform_levels=l, use_ItI=False
            )

            # Create the right-hand side
            source = problem_1_source(tree.leaf_cheby_points)

            # Get the coefficients for the differential operators
            lap_coeffs = problem_1_lap_coeffs(tree.leaf_cheby_points)
            d_x_coeffs = problem_1_d_x_coeffs(tree.leaf_cheby_points)
            d_y_coeffs = problem_1_d_y_coeffs(tree.leaf_cheby_points)

            # Get the boundary data
            g = problem_1_homog_soln(tree.root_boundary_points)

            # Solve the problem
            fused_pde_solve_2D(
                tree,
                source_term=source,
                D_xx_coeffs=lap_coeffs,
                D_yy_coeffs=lap_coeffs,
                D_x_coeffs=d_x_coeffs,
                D_y_coeffs=d_y_coeffs,
                boundary_data=g,
            )

            # Compute the error
            computed_soln = tree.interior_solns
            expected_soln = problem_1_homog_soln(
                tree.leaf_cheby_points
            ) + problem_1_part_soln(tree.leaf_cheby_points)

            # plot soln
            # plot_soln_from_cheby_nodes(
            #     cheby_nodes=tree.leaf_cheby_points.reshape(-1, 2),
            #     corners=CORNERS,
            #     expected_soln=expected_soln.flatten(),
            #     computed_soln=computed_soln.flatten(),
            # )

            # Compute the error
            err = jnp.max(jnp.abs(computed_soln - expected_soln))
            nrm = jnp.max(jnp.abs(expected_soln))
            errors = errors.at[i, j].set(err / nrm)

            # Log progress
            logging.info("problem_1: l=%i, p=%i, err=%s", l, p, errors[i, j])
    return errors


def problem_2_soln(x: jnp.array) -> jnp.array:
    """
    Expect x to have shape (..., 2).
    Output has shape (...)

    u(x,y) = e^{i20x} + e^{i30y}
    """
    return jnp.exp(1j * 20 * x[..., 0]) + jnp.exp(1j * 30 * x[..., 1])


def problem_2_lap_coeffs(x: jnp.array) -> jnp.array:
    return jnp.ones_like(x[..., 0])


def problem_2_I_term(x: jnp.array) -> jnp.array:
    """
    Expect x to have shape (..., 2).
    Output has shape (...)

    I(x,y) = 1 + exp{-||x||^2 * 50}
    """
    return 1 + jnp.exp(-jnp.linalg.norm(x, axis=-1) ** 2 * 50)


def problem_2_source(x: jnp.array) -> jnp.array:
    """
    Expect x to have shape (..., 2).
    Output has shape (...)

    f(x,y) = -1  * (400 e^{i20x} + 900 e^{i30y}) + I_term(x,y) * u(x,y)
    """
    first_term = -1 * (
        400 * jnp.exp(1j * 20 * x[..., 0]) + 900 * jnp.exp(1j * 30 * x[..., 1])
    )
    second_term = problem_2_I_term(x) * problem_2_soln(x)
    return first_term + second_term


def problem_2_boundary_data(x: jnp.array, eta: float) -> jnp.array:
    """
    Expect x to have shape (..., 2).
    Output has shape (...)

    Want to return the incoming impedance data

    g = u + i * eta * du/dn
    """
    n_per_side = x.shape[0] // 4
    south = x[:n_per_side]
    east = x[n_per_side : 2 * n_per_side]
    north = x[2 * n_per_side : 3 * n_per_side]
    west = x[3 * n_per_side :]

    dudn = jnp.concatenate(
        [
            -1j * 30 * jnp.exp(1j * 30 * south[..., 1]),
            1j * 20 * jnp.exp(1j * 20 * east[..., 0]),
            1j * 30 * jnp.exp(1j * 30 * north[..., 1]),
            -1j * 20 * jnp.exp(1j * 20 * west[..., 0]),
        ]
    )
    return dudn + 1j * eta * problem_2_soln(x)
    # return problem_2_soln(x) + 1j * eta * dudn


def problem_2(l_vals: int, p_vals: int) -> None:

    eta = 4.0
    errors = jnp.zeros((len(l_vals), len(p_vals)), dtype=jnp.float64)

    for i, l in enumerate(l_vals):
        l = int(l)  # Cast from jax array to int
        for j, p in enumerate(p_vals):
            p = int(p)  # Cast from jax array to int
            # Create the tree
            root = Node(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)

            tree = create_solver_obj_2D(
                p=p, q=p - 2, root=root, uniform_levels=l, use_ItI=True, eta=eta
            )

            # Create the right-hand side
            source = problem_2_source(tree.leaf_cheby_points)

            # Get the coefficients for the Laplacian
            lap_coeffs = problem_2_lap_coeffs(tree.leaf_cheby_points)

            # Get the I term
            I_term = problem_2_I_term(tree.leaf_cheby_points)

            # Get the boundary data
            g = problem_2_boundary_data(tree.root_boundary_points, eta=eta)

            # Solve the problem
            fused_pde_solve_2D_ItI(
                tree,
                source_term=source,
                D_xx_coeffs=lap_coeffs,
                D_yy_coeffs=lap_coeffs,
                I_coeffs=I_term,
                boundary_data=g,
            )

            # Compute the error
            computed_soln = tree.interior_solns
            expected_soln = problem_2_soln(tree.leaf_cheby_points)

            # plot soln
            # plot_soln_from_cheby_nodes(
            #     cheby_nodes=tree.leaf_cheby_points.reshape(-1, 2),
            #     corners=CORNERS,
            #     expected_soln=expected_soln.real.flatten(),
            #     computed_soln=computed_soln.real.flatten(),
            # )

            # Compute the error
            err = jnp.max(jnp.abs(computed_soln - expected_soln))
            nrm = jnp.max(jnp.abs(expected_soln))
            errors = errors.at[i, j].set(err / nrm)

            # Log progress
            logging.info("problem_2: l=%i, p=%i, err=%s", l, p, errors[i, j])
    return errors


def plot_problem(
    err_vals: jnp.array, l_vals: jnp.array, p_vals: jnp.array, ax: plt.Axes
) -> None:

    # Compute 1 / h values
    n_patches_per_side = 2**l_vals
    h_vals = 2.0 / n_patches_per_side
    h_inv_vals = 1 / h_vals

    # For each value of p, plot the error vs 1 / h
    for i, p in enumerate(p_vals):
        ax.plot(h_inv_vals, err_vals[:, i], label=f"$p={p-1}$")

    ax.set_xscale("log")
    ax.set_yscale("log")


def main(args: argparse.Namespace) -> None:
    """
    1. Sets up the plots_dir.
    2. If args.DtN is True, runs problem 1.
    3. If args.ItI is True, runs problem 2.
    4. Plots results.
    """

    # Set up plots_dir
    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    l_vals = jnp.array(args.l_vals)
    p_vals = jnp.array(args.p_vals)

    # Problem 1: DtN
    if args.DtN:

        error_vals_DtN = problem_1(l_vals, p_vals)

        # Save the error values
        fp = os.path.join(args.plots_dir, "error_vals_DtN.mat")
        out_dd = {
            "error_vals": error_vals_DtN,
            "l_vals": l_vals,
            "p_vals": p_vals,
        }
        savemat(fp, out_dd)

    # Problem 2: ItI
    if args.ItI:

        error_vals_ItI = problem_2(l_vals, p_vals)
        fp = os.path.join(args.plots_dir, "error_vals_ItI.mat")
        out_dd = {
            "error_vals": error_vals_ItI,
            "l_vals": l_vals,
            "p_vals": p_vals,
        }
        savemat(fp, out_dd)


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)
    main(args)
