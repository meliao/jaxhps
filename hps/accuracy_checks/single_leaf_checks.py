import logging
from typing import Tuple, Callable
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

from hps.src.methods.local_solve_stage import _local_solve_stage_2D
from hps.src.solver_obj import SolverObj, create_solver_obj_2D
from hps.src.quadrature.trees import Node
from hps.src.methods.local_solve_stage import (
    _local_solve_stage_2D,
    _local_solve_stage_2D_ItI,
)
from hps.accuracy_checks.dirichlet_neumann_data import (
    TEST_CASE_POISSON_POLY,
    TEST_CASE_POISSON_NONPOLY,
    TEST_CASE_NONCONSTANT_COEFF_POLY,
    TEST_CASE_POLY_PART_HOMOG,
    TEST_CASE_POLY_ITI,
    TEST_CASE_NONPOLY_ITI,
    TEST_CASE_HELMHOLTZ_0,
    TEST_CASE_HELMHOLTZ_1,
    ETA,
)
from hps.accuracy_checks.utils import (
    plot_soln_from_cheby_nodes,
    _distance_around_boundary_nonode,
)
from hps.src.config import DEVICE_ARR, HOST_DEVICE


def check_l_inf_error_convergence(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
    check_dtn: bool = False,
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
) -> None:
    """
    This function solves a Poisson boundary value problem on the domain [-pi/2, pi/2]^2 with
    increasing levels of Chebyshev discretization.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """
    # Set up the domain
    north = jnp.pi / 2
    south = -jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2
    corners = [(west, south), (east, south), (east, north), (west, north)]
    half_side_len = (east - west) / 2

    n_X = 100
    x = jnp.linspace(west, east, n_X)
    X, Y = jnp.meshgrid(x, x)

    error_vals = jnp.zeros_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        p = int(p)
        # print("check_l_inf_error_convergence: p = ", p)
        q = p - 2

        root = Node(
            xmin=west,
            xmax=east,
            ymin=south,
            ymax=north,
            zmin=None,
            zmax=None,
            depth=0,
        )

        t = create_solver_obj_2D(p, q, root)

        # Set up the LeafNode

        cheby_pts = t.leaf_cheby_points

        d_xx_coeffs = d_xx_coeff_fn(cheby_pts)
        d_yy_coeffs = d_yy_coeff_fn(cheby_pts)
        source = source_fn(cheby_pts)

        # d_xx_coeffs = jnp.expand_dims(d_xx_coeffs, axis=0)
        # d_yy_coeffs = jnp.expand_dims(d_yy_coeffs, axis=0)
        # source = jnp.expand_dims(source, axis=0)

        # Put in a second, third, and fourth copy of the data along the 0th axis
        # to simulate the other part of the merge quad so the
        # reshape operation works
        d_xx_coeffs = jnp.stack(
            [d_xx_coeffs, d_xx_coeffs, d_xx_coeffs, d_xx_coeffs], axis=0
        ).squeeze(1)
        d_yy_coeffs = jnp.stack(
            [d_yy_coeffs, d_yy_coeffs, d_yy_coeffs, d_yy_coeffs], axis=0
        ).squeeze(1)
        source = jnp.stack([source, source, source, source], axis=0).squeeze(1)

        sidelens = jnp.array([root.xmax - root.xmin for _ in range(4)])

        Y, DtN, v, v_prime = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            Q_D=t.Q_D,
            p=p,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            sidelens=sidelens,
            uniform_grid=True,
        )
        # only need the first leaf's solution
        Y = Y[0]
        v = v[0]
        DtN = DtN.squeeze()[0]
        v_prime = v_prime.squeeze()[0]

        # print("check_l_inf_error_convergence: Y: ", Y.shape)
        # print("check_l_inf_error_convergence: DtN: ", DtN.shape)
        # print("check_l_inf_error_convergence: v: ", v.shape)
        # print("check_l_inf_error_convergence: v_prime: ", v_prime.shape)

        boundary_data = dirichlet_data_fn(t.root_boundary_points)
        if check_dtn:
            expected_soln = jnp.concatenate(
                [
                    -1 * dudy_fn(t.root_boundary_points[:q]),
                    dudx_fn(t.root_boundary_points[q : 2 * q]),
                    dudy_fn(t.root_boundary_points[2 * q : 3 * q]),
                    -1 * dudx_fn(t.root_boundary_points[3 * q :]),
                ]
            )
            computed_soln_0 = DtN @ boundary_data
            computed_soln = computed_soln_0 + v_prime

        else:
            expected_soln = dirichlet_data_fn(t.leaf_cheby_points).flatten()
            computed_soln = Y @ boundary_data + v

        l_inf_error = jnp.max(jnp.abs(computed_soln - expected_soln))
        error_vals = error_vals.at[i].set(l_inf_error)
    logging.info("check_l_inf_error_convergence: error_vals = %s", error_vals)
    if compute_convergence_rate:
        # Compute convergence on a semilog plot
        p_values = p_values.astype(np.float64)
        log_error_vals = jnp.log(error_vals)
        m, b = jnp.polyfit(p_values, log_error_vals, 1)

        xvals = jnp.linspace(p_values[0], p_values[-1], 100)
        yvals = np.exp(m * xvals + b)

    fig, ax = plt.subplots()
    ax.set_title("Error convergence for Poisson BVP; Single Patch")
    ax.set_xlabel("p = # Chebyshev nodes")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    ax.plot(p_values, error_vals, "o-")
    ax.set_yscale("log")
    ax.grid()

    if compute_convergence_rate:
        ax.plot(xvals, yvals, "--", label=f"Slope = {m:.2f}")
        ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def check_l_inf_error_convergence_particular_homog_solns(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    homogeneous_soln_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_soln_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_dx_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_dy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
) -> None:
    """
    This function solves a Poisson boundary value problem on the domain [-pi/2, pi/2]^2 with
    increasing levels of Chebyshev discretization.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """
    # Set up the domain
    north = jnp.pi / 2
    south = -jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2
    corners = [(west, south), (east, south), (east, north), (west, north)]
    half_side_len = (east - west) / 2

    n_X = 100
    x = jnp.linspace(west, east, n_X)
    X, Y = jnp.meshgrid(x, x)

    error_vals_homog = jnp.zeros_like(p_values, dtype=jnp.float64)
    error_vals_particular = jnp.zeros_like(p_values, dtype=jnp.float64)
    error_vals_particular_flux = jnp.zeros_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        p = int(p)
        # print("check_l_inf_error_convergence: p = ", p)
        q = p - 2

        root = Node(
            xmin=west,
            xmax=east,
            ymin=south,
            ymax=north,
            zmin=None,
            zmax=None,
            depth=0,
        )
        t = create_solver_obj_2D(p, q, root)

        # Set up the LeafNode

        cheby_pts = t.leaf_cheby_points

        d_xx_coeffs = d_xx_coeff_fn(cheby_pts)
        d_yy_coeffs = d_yy_coeff_fn(cheby_pts)
        source = source_fn(cheby_pts)

        # d_xx_coeffs = jnp.expand_dims(d_xx_coeffs, axis=0)
        # d_yy_coeffs = jnp.expand_dims(d_yy_coeffs, axis=0)
        # source = jnp.expand_dims(source, axis=0)

        # Put in a second, third, and fourth copy of the data along the 0th axis
        # to simulate the other part of the merge quad so the
        # reshape operation works
        d_xx_coeffs = jnp.stack(
            [d_xx_coeffs, d_xx_coeffs, d_xx_coeffs, d_xx_coeffs], axis=0
        ).squeeze(1)
        d_yy_coeffs = jnp.stack(
            [d_yy_coeffs, d_yy_coeffs, d_yy_coeffs, d_yy_coeffs], axis=0
        ).squeeze(1)
        source = jnp.stack([source, source, source, source], axis=0).squeeze(1)

        sidelens = jnp.array([root.xmax - root.xmin for _ in range(4)])

        Y, DtN, v, v_prime = _local_solve_stage_2D(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            P=t.P,
            p=p,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            sidelens=sidelens,
            uniform_grid=True,
            Q_D=t.Q_D,
        )

        # logging.debug(
        #     "check_l_inf_error_convergence_particular_homog_solns: v_prime shape = %s",
        #     v_prime.shape,
        # )
        # only need the first leaf's solution
        Y = Y[1]
        v = v[1]
        DtN = DtN.squeeze()[1]
        v_prime = v_prime.squeeze()[1]
        # logging.debug(
        #     "check_l_inf_error_convergence_particular_homog_solns: v_prime shape = %s",
        #     v_prime.shape,
        # )
        # logging.debug(
        #     "check_l_inf_error_convergence_particular_homog_solns: DtN shape = %s",
        #     DtN.shape,
        # )

        boundary_data = dirichlet_data_fn(t.root_boundary_points)

        # Check homogeneous soln
        expected_h_soln = homogeneous_soln_fn(t.leaf_cheby_points)
        computed_h_soln = Y @ boundary_data
        l_inf_error = jnp.max(jnp.abs(expected_h_soln - computed_h_soln))
        error_vals_homog = error_vals_homog.at[i].set(l_inf_error)

        # Check particular soln
        expected_particular_soln = particular_soln_fn(t.leaf_cheby_points)
        l_inf_error = jnp.max(jnp.abs(expected_particular_soln - v))
        error_vals_particular = error_vals_particular.at[i].set(l_inf_error)

        expected_p_bdry = expected_particular_soln[: 4 * (p - 1)]
        logging.debug(
            "check_l_inf_error_convergence_particular_homog_solns: max of expected_p_bdry = %s",
            jnp.max(jnp.abs(expected_p_bdry)),
        )

        # Check particular flux
        expected_particular_flux = jnp.concatenate(
            [
                -1 * particular_dy_fn(t.root_boundary_points[:q]),
                particular_dx_fn(t.root_boundary_points[q : 2 * q]),
                particular_dy_fn(t.root_boundary_points[2 * q : 3 * q]),
                -1 * particular_dx_fn(t.root_boundary_points[3 * q :]),
            ]
        )
        logging.debug(
            "check_l_inf_error_convergence_particular_homog_solns: expected_particular_flux shape = %s",
            expected_particular_flux.shape,
        )
        logging.debug(
            "check_l_inf_error_convergence_particular_homog_solns: v_prime shape = %s",
            v_prime.shape,
        )

        # plt.plot(expected_particular_flux, ".-", label="expected_particular_flux")
        # plt.plot(v_prime, "x-", label="v_prime")
        # plt.legend()
        # plt.show()
        # plt.clf()

        l_inf_error = jnp.max(jnp.abs(expected_particular_flux - v_prime))
        error_vals_particular_flux = error_vals_particular_flux.at[i].set(l_inf_error)

    logging.info(
        "check_l_inf_error_convergence_particular_homog_solns: error_vals_homog = %s",
        error_vals_homog,
    )
    logging.info(
        "check_l_inf_error_convergence_particular_homog_solns: error_vals_particular = %s",
        error_vals_particular,
    )
    logging.info(
        "check_l_inf_error_convergence_particular_homog_solns: error_vals_particular_flux = %s",
        error_vals_particular_flux,
    )
    if compute_convergence_rate:
        # Compute convergence on a semilog plot
        p_values = p_values.astype(np.float64)
        xvals = jnp.linspace(p_values[0], p_values[-1], 100)

        log_h_error_vals = jnp.log(error_vals_homog)
        m_h, b_h = jnp.polyfit(p_values, log_h_error_vals, 1)
        yvals_h = np.exp(m_h * xvals + b_h)

        log_p_error_vals = jnp.log(error_vals_particular)
        m_p, b_p = jnp.polyfit(p_values, log_p_error_vals, 1)
        yvals_p = np.exp(m_p * xvals + b_p)

        log_pf_error_vals = jnp.log(error_vals_particular_flux)
        m_pf, b_pf = jnp.polyfit(p_values, log_pf_error_vals, 1)
        yvals_pf = np.exp(m_pf * xvals + b_pf)

    fig, ax = plt.subplots()
    ax.set_title("Error convergence for Poisson BVP; Single Patch")
    ax.set_xlabel("p = # Chebyshev nodes")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    ax.plot(p_values, error_vals_homog, "o-", label="Homogeneous soln", color="red")
    if compute_convergence_rate:
        ax.plot(xvals, yvals_h, "--", label=f"Slope = {m_h:.2f}", color="red")
    ax.plot(
        p_values, error_vals_particular, "o-", label="Particular soln", color="blue"
    )
    if compute_convergence_rate:
        ax.plot(xvals, yvals_p, "--", label=f"Slope = {m_p:.2f}", color="blue")
    ax.plot(
        p_values,
        error_vals_particular_flux,
        "o-",
        label="Particular flux",
        color="green",
    )
    if compute_convergence_rate:
        ax.plot(xvals, yvals_pf, "--", label=f"Slope = {m_pf:.2f}", color="green")
    ax.set_yscale("log")
    ax.grid()

    ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def check_l_inf_error_convergence_ItI_maps(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    plot_fp: str,
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray],
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eta: float,
    check_ItI: bool = False,
) -> None:
    """
    This function solves a Poisson boundary value problem on the domain [-pi/2, pi/2]^2 with
    increasing levels of Chebyshev discretization.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """
    # Set up the domain
    north = jnp.pi / 2
    south = -jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2
    corners = [(west, south), (east, south), (east, north), (west, north)]
    half_side_len = (east - west) / 2

    n_X = 100
    x = jnp.linspace(west, east, n_X)
    X, Y = jnp.meshgrid(x, x)

    error_vals = jnp.zeros_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        p = int(p)
        # print("check_l_inf_error_convergence: p = ", p)
        q = p - 2

        root = Node(xmin=west, xmax=east, ymin=south, ymax=north, depth=0)
        t = create_solver_obj_2D(p, q, root, use_ItI=True, eta=eta)

        # Set up the LeafNode

        cheby_pts = t.leaf_cheby_points

        d_xx_coeffs = d_xx_coeff_fn(cheby_pts)
        d_yy_coeffs = d_yy_coeff_fn(cheby_pts)
        source = source_fn(cheby_pts)

        # Put in a second, third, and fourth copy of the data along the 0th axis
        # to simulate the other part of the merge quad so the
        # reshape operation works
        d_xx_coeffs = jnp.stack(
            [d_xx_coeffs, d_xx_coeffs, d_xx_coeffs, d_xx_coeffs], axis=0
        ).squeeze(1)
        d_yy_coeffs = jnp.stack(
            [d_yy_coeffs, d_yy_coeffs, d_yy_coeffs, d_yy_coeffs], axis=0
        ).squeeze(1)
        source = jnp.stack([source, source, source, source], axis=0).squeeze(1)

        R_arr, Y_arr, h_arr, v_arr = _local_solve_stage_2D_ItI(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            I_P_0=t.I_P_0,
            Q_I=t.Q_I,
            F=t.F,
            G=t.G,
            p=p,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
        )
        # only need the first leaf's solution
        R = R_arr.squeeze()[0]
        h = h_arr.squeeze()[0]
        v = v_arr[0]
        Y = Y_arr[0]

        logging.debug("check_l_inf_error_convergence: Y shape: %s", Y.shape)
        logging.debug("check_l_inf_error_convergence: R shape: %s", R.shape)
        logging.debug("check_l_inf_error_convergence: h shape: %s", h.shape)
        logging.debug("check_l_inf_error_convergence: v shape: %s", v.shape)
        boundary_f = dirichlet_data_fn(t.root_boundary_points)
        boundary_normals = jnp.concatenate(
            [
                -1 * dudy_fn(t.root_boundary_points[:q]),
                dudx_fn(t.root_boundary_points[q : 2 * q]),
                dudy_fn(t.root_boundary_points[2 * q : 3 * q]),
                -1 * dudx_fn(t.root_boundary_points[3 * q :]),
            ]
        )
        incoming_imp_data = boundary_normals + 1j * eta * boundary_f

        if check_ItI:
            expected_out_imp_data = boundary_normals - 1j * eta * boundary_f

            expected_soln = expected_out_imp_data
            computed_soln = R @ incoming_imp_data + h

            # plt.plot(computed_soln.real, "o-", label="Computed.real")
            # plt.plot(expected_soln.real, "x-", label="Expected.real")
            # plt.plot(computed_soln.imag, "o-", label="Computed.imag")
            # plt.plot(expected_soln.imag, "x-", label="Expected.imag")
            # plt.legend()
            # plt.show()
        else:
            expected_soln = dirichlet_data_fn(t.leaf_cheby_points)
            computed_soln = Y @ incoming_imp_data
            computed_soln = computed_soln + v.flatten()
        # plot_soln_from_cheby_nodes(
        #     t.leaf_cheby_points.reshape(-1, 2),
        #     corners,
        #     expected_soln=expected_soln.real.flatten(),
        #     computed_soln=computed_soln.real.flatten(),
        # )
        l_inf_error = jnp.max(jnp.abs(computed_soln - expected_soln)) / jnp.max(
            jnp.abs(expected_soln)
        )
        error_vals = error_vals.at[i].set(l_inf_error)
    logging.info("check_l_inf_error_convergence: error_vals = %s", error_vals)
    if compute_convergence_rate:
        # Compute convergence on a semilog plot
        p_values = p_values.astype(np.float64)
        log_error_vals = jnp.log(error_vals)
        m, b = jnp.polyfit(p_values, log_error_vals, 1)

        xvals = jnp.linspace(p_values[0], p_values[-1], 100)
        yvals = np.exp(m * xvals + b)

    fig, ax = plt.subplots()
    if check_ItI:
        t = "Convergence of error in ItI mapping; single patch"
    else:
        t = "Convergence of error in interior soln; single patch"
    ax.set_title(t)
    ax.set_xlabel("p = # Chebyshev nodes")
    ax.set_ylabel("$Err(u, \\tilde{u})$")
    ax.plot(p_values, error_vals, "o-")
    ax.set_yscale("log")
    ax.grid()

    if compute_convergence_rate:
        ax.plot(xvals, yvals, "--", label=f"Slope = {m:.2f}")
        ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def check_l_inf_error_convergence_particular_homog_solns_ItI(
    p_values: jnp.ndarray,
    dirichlet_data_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_xx_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    d_yy_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    I_coeff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    homogeneous_soln_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_soln_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_dx_fn: Callable[[jnp.ndarray], jnp.ndarray],
    particular_dy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    dudx_fn: Callable[[jnp.ndarray], jnp.ndarray],
    dudy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray], jnp.ndarray],
    compute_convergence_rate: bool,
    eta: float,
    plot_fp: str,
) -> None:
    """
    This function solves a Poisson boundary value problem on the domain [-pi/2, pi/2]^2 with
    increasing levels of Chebyshev discretization.

    It then computes the L_inf error at the Chebyshev points, and plots the convergence of the error.
    """
    # Set up the domain
    north = jnp.pi / 2
    south = -jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2
    corners = [(west, south), (east, south), (east, north), (west, north)]

    error_vals_iti = jnp.zeros_like(p_values, dtype=jnp.float64)
    error_vals_homog = jnp.zeros_like(p_values, dtype=jnp.float64)
    error_vals_particular = jnp.zeros_like(p_values, dtype=jnp.float64)
    error_vals_particular_flux = jnp.zeros_like(p_values, dtype=jnp.float64)

    for i, p in enumerate(p_values):
        p = int(p)
        # print("check_l_inf_error_convergence: p = ", p)
        q = p - 2

        root = Node(xmin=west, xmax=east, ymin=south, ymax=north, depth=0)

        t = create_solver_obj_2D(p, q, root, use_ItI=True, eta=eta)

        # Set up the LeafNode

        cheby_pts = t.leaf_cheby_points

        d_xx_coeffs = d_xx_coeff_fn(cheby_pts)
        d_yy_coeffs = d_yy_coeff_fn(cheby_pts)
        I_coeffs = I_coeff_fn(cheby_pts)
        source = source_fn(cheby_pts)

        # d_xx_coeffs = jnp.expand_dims(d_xx_coeffs, axis=0)
        # d_yy_coeffs = jnp.expand_dims(d_yy_coeffs, axis=0)
        # source = jnp.expand_dims(source, axis=0)

        # Put in a second, third, and fourth copy of the data along the 0th axis
        # to simulate the other part of the merge quad so the
        # reshape operation works
        d_xx_coeffs = jnp.stack(
            [d_xx_coeffs, d_xx_coeffs, d_xx_coeffs, d_xx_coeffs], axis=0
        ).squeeze(1)
        d_yy_coeffs = jnp.stack(
            [d_yy_coeffs, d_yy_coeffs, d_yy_coeffs, d_yy_coeffs], axis=0
        ).squeeze(1)
        I_coeffs = jnp.stack([I_coeffs, I_coeffs, I_coeffs, I_coeffs], axis=0).squeeze(
            1
        )
        source = jnp.stack([source, source, source, source], axis=0).squeeze(1)

        R_arr, Y_arr, g_arr, v_arr = _local_solve_stage_2D_ItI(
            D_xx=t.D_xx,
            D_xy=t.D_xy,
            D_yy=t.D_yy,
            D_x=t.D_x,
            D_y=t.D_y,
            I_P_0=t.I_P_0,
            Q_I=t.Q_I,
            F=t.F,
            G=t.G,
            p=p,
            source_term=source,
            D_xx_coeffs=d_xx_coeffs,
            D_yy_coeffs=d_yy_coeffs,
            I_coeffs=I_coeffs,
        )
        # print(
        #     "check_l_inf_error_convergence_particular_homog_solns_ItI: R_arr = ",
        #     R_arr.shape,
        # )
        # print(
        #     "check_l_inf_error_convergence_particular_homog_solns_ItI: Y_arr = ",
        #     Y_arr.shape,
        # )
        # print(
        #     "check_l_inf_error_convergence_particular_homog_solns_ItI: g_arr = ",
        #     g_arr.shape,
        # )
        # print(
        #     "check_l_inf_error_convergence_particular_homog_solns_ItI: v_arr = ",
        #     v_arr.shape,
        # )

        # only need the first leaf's solution
        R = R_arr.squeeze()[1]
        g = g_arr.squeeze()[1]
        v = v_arr[0].flatten()
        Y = Y_arr[0]

        # print("check_l_inf_error_convergence_particular_homog_solns_ItI: R = ", R.shape)
        # print("check_l_inf_error_convergence_particular_homog_solns_ItI: Y = ", Y.shape)
        # print("check_l_inf_error_convergence_particular_homog_solns_ItI: g = ", g.shape)
        # print("check_l_inf_error_convergence_particular_homog_solns_ItI: v = ", v.shape)
        # logging.debug(
        #     "check_l_inf_error_convergence_particular_homog_solns_ItI: Y shape: %s",
        #     Y_arr.shape,
        # )

        boundary_homog_f = dirichlet_data_fn(t.root_boundary_points)
        boundary_homog_n = jnp.concatenate(
            [
                -1 * dudy_fn(t.root_boundary_points[:q]),
                dudx_fn(t.root_boundary_points[q : 2 * q]),
                dudy_fn(t.root_boundary_points[2 * q : 3 * q]),
                -1 * dudx_fn(t.root_boundary_points[3 * q :]),
            ]
        )
        boundary_part_f = particular_soln_fn(t.root_boundary_points)
        boundary_part_n = jnp.concatenate(
            [
                -1 * particular_dy_fn(t.root_boundary_points[:q]),
                particular_dx_fn(t.root_boundary_points[q : 2 * q]),
                particular_dy_fn(t.root_boundary_points[2 * q : 3 * q]),
                -1 * particular_dx_fn(t.root_boundary_points[3 * q :]),
            ]
        )
        incoming_imp_data = (boundary_homog_n) + 1j * eta * (boundary_homog_f)
        expected_outgoing_imp_data = (boundary_homog_n) - 1j * eta * (boundary_homog_f)

        # print(
        #     "check_l_inf_error_convergence_particular_homog_solns_ItI: incoming_imp_data = ",
        #     incoming_imp_data.shape,
        # )
        # Check ItI map
        computed_outgoing_imp_data = R @ incoming_imp_data
        l_inf_error = jnp.max(
            jnp.abs(expected_outgoing_imp_data - computed_outgoing_imp_data)
        )
        error_vals_iti = error_vals_iti.at[i].set(l_inf_error)

        # plt.plot(computed_outgoing_imp_data.real, "o-", label="Computed.real")
        # plt.plot(expected_outgoing_imp_data.real, "x-", label="Expected.real")
        # plt.plot(computed_outgoing_imp_data.imag, "o-", label="Computed.imag")
        # plt.plot(expected_outgoing_imp_data.imag, "x-", label="Expected.imag")
        # plt.legend()
        # plt.show()
        # plt.clf()

        # Check homogeneous soln on interior
        expected_h_soln = homogeneous_soln_fn(t.leaf_cheby_points)
        computed_h_soln = Y @ incoming_imp_data
        l_inf_error = jnp.max(jnp.abs(expected_h_soln - computed_h_soln))
        error_vals_homog = error_vals_homog.at[i].set(l_inf_error)

        # Check particular soln on interior
        expected_particular_soln = particular_soln_fn(t.leaf_cheby_points[0])
        logging.debug(
            "check_l_inf_error_convergence_particular_homog_solns_ItI: v shape = %s",
            v.shape,
        )
        logging.debug(
            "check_l_inf_error_convergence_particular_homog_solns_ItI: expected_particular_soln shape = %s",
            expected_particular_soln.shape,
        )
        l_inf_error = jnp.max(jnp.abs(expected_particular_soln - v))
        error_vals_particular = error_vals_particular.at[i].set(l_inf_error)

        # plot_soln_from_cheby_nodes(
        #     t.leaf_cheby_points.reshape(-1, 2),
        #     corners,
        #     expected_particular_soln.real.squeeze(),
        #     v.real,
        # )
        # Check particular soln outgoing impedance data
        expected_outgoing_part_imp = jnp.concatenate(
            [
                -1 * particular_dy_fn(t.root_boundary_points[:q]),
                particular_dx_fn(t.root_boundary_points[q : 2 * q]),
                particular_dy_fn(t.root_boundary_points[2 * q : 3 * q]),
                -1 * particular_dx_fn(t.root_boundary_points[3 * q :]),
            ]
        ) - 1j * eta * particular_soln_fn(t.root_boundary_points)

        l_inf_error = jnp.max(jnp.abs(expected_outgoing_part_imp - g))
        error_vals_particular_flux = error_vals_particular_flux.at[i].set(l_inf_error)

        # Plot the outgoing impedance data
        # plt.plot(g.real, "o-", label="Computed.real")
        # plt.plot(g.imag, "o-", label="Computed.imag")
        # plt.plot(expected_outgoing_part_imp.real, "x-", label="Expected.real")
        # plt.plot(expected_outgoing_part_imp.imag, "x-", label="Expected.imag")
        # plt.legend()
        # plt.grid()
        # plt.show()

    logging.info(
        "check_l_inf_error_convergence_particular_homog_solns_ItI: error_vals_iti = %s",
        error_vals_iti,
    )
    logging.info(
        "check_l_inf_error_convergence_particular_homog_solns: error_vals_homog = %s",
        error_vals_homog,
    )
    logging.info(
        "check_l_inf_error_convergence_particular_homog_solns: error_vals_particular = %s",
        error_vals_particular,
    )
    logging.info(
        "check_l_inf_error_convergence_particular_homog_solns: error_vals_particular_flux = %s",
        error_vals_particular_flux,
    )
    if compute_convergence_rate:
        # Compute convergence on a semilog plot
        p_values = p_values.astype(np.float64)
        xvals = jnp.linspace(p_values[0], p_values[-1], 100)

        log_h_error_vals = jnp.log(error_vals_homog)
        m_h, b_h = jnp.polyfit(p_values, log_h_error_vals, 1)
        yvals_h = np.exp(m_h * xvals + b_h)

        log_p_error_vals = jnp.log(error_vals_particular)
        m_p, b_p = jnp.polyfit(p_values, log_p_error_vals, 1)
        yvals_p = np.exp(m_p * xvals + b_p)

        log_pf_error_vals = jnp.log(error_vals_particular_flux)
        m_pf, b_pf = jnp.polyfit(p_values, log_pf_error_vals, 1)
        yvals_pf = np.exp(m_pf * xvals + b_pf)

    fig, ax = plt.subplots()
    ax.set_title("Error convergence for Poisson BVP; Single Patch")
    ax.set_xlabel("p = # Chebyshev nodes")
    ax.set_ylabel("L_inf error at Chebyshev nodes")
    ax.plot(p_values, error_vals_homog, "o-", label="Homogeneous soln", color="red")
    if compute_convergence_rate:
        ax.plot(xvals, yvals_h, "--", label=f"Slope = {m_h:.2f}", color="red")
    ax.plot(
        p_values, error_vals_particular, "o-", label="Particular soln", color="blue"
    )
    if compute_convergence_rate:
        ax.plot(xvals, yvals_p, "--", label=f"Slope = {m_p:.2f}", color="blue")
    ax.plot(
        p_values,
        error_vals_particular_flux,
        "o-",
        label="Outgoing part. imp.",
        color="green",
    )
    if compute_convergence_rate:
        ax.plot(xvals, yvals_pf, "--", label=f"Slope = {m_pf:.2f}", color="green")
    ax.set_yscale("log")
    ax.grid()

    ax.legend()

    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()


def single_leaf_check_0(plot_fp: str) -> None:
    """
    Poisson boundary problem with dirichlet data f(x, y) = x^2 - y^2
    """
    logging.info("Running single_leaf_check_0")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_POLY["source_fn"],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
    )


def single_leaf_check_1(plot_fp: str) -> None:
    """
    Poisson boundary problem with dirichlet data f(x, y) = e^x sin(y)
    """
    logging.info("Running single_leaf_check_1")
    p_values = jnp.array([4, 6, 8, 10, 12, 14, 16])
    check_l_inf_error_convergence(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_NONPOLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_NONPOLY["source_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
    )


def single_leaf_check_2(plot_fp: str) -> None:
    """
    Operator with non-constant PDE coefficients and nonzero source term.
    All of the coeffs and solutions are polynomial.

    """
    logging.info("Running single_leaf_check_2")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_NONCONSTANT_COEFF_POLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_NONCONSTANT_COEFF_POLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_NONCONSTANT_COEFF_POLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_NONCONSTANT_COEFF_POLY["source_fn"],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
    )


def single_leaf_check_3(plot_fp: str) -> None:
    """
    Measures convergence of particular soln, homogeneous soln, particular soln
    boundary fluxes for a Poisson problem with polynomial source and dirichlet data.
    """
    logging.info("Running single_leaf_check_3")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_particular_homog_solns(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POLY_PART_HOMOG["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POLY_PART_HOMOG["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POLY_PART_HOMOG["d_yy_coeff_fn"],
        homogeneous_soln_fn=TEST_CASE_POLY_PART_HOMOG["homogeneous_soln_fn"],
        particular_soln_fn=TEST_CASE_POLY_PART_HOMOG["particular_soln_fn"],
        particular_dx_fn=TEST_CASE_POLY_PART_HOMOG["particular_dx_fn"],
        particular_dy_fn=TEST_CASE_POLY_PART_HOMOG["particular_dy_fn"],
        source_fn=TEST_CASE_POLY_PART_HOMOG["source_fn"],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
    )


def single_leaf_check_4(plot_fp: str) -> None:
    """Checks the DtN map for a Poisson problem with polynomial source and dirichlet data."""
    logging.info("Running single_leaf_check_4")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_POLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_POLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_POLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_POLY["source_fn"],
        compute_convergence_rate=False,
        plot_fp=plot_fp,
        check_dtn=True,
        dudx_fn=TEST_CASE_POISSON_POLY["du_dx_fn"],
        dudy_fn=TEST_CASE_POISSON_POLY["du_dy_fn"],
    )


def single_leaf_check_5(plot_fp: str) -> None:
    """
    Measures convergence of soln when using ItI maps for poly data.
    Data is polynomial.
    """
    logging.info("Running single_leaf_check_5")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_ItI_maps(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POLY_ITI["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POLY_ITI["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POLY_ITI["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POLY_ITI["source_fn"],
        dudx_fn=TEST_CASE_POLY_ITI["du_dx_fn"],
        dudy_fn=TEST_CASE_POLY_ITI["du_dy_fn"],
        compute_convergence_rate=False,
        eta=4.0,
        plot_fp=plot_fp,
    )


def single_leaf_check_6(plot_fp: str) -> None:
    """
    Measures convergence of soln when using ItI maps for non-poly data.
    Data is non-polynomial.
    """
    logging.info("Running single_leaf_check_6")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_ItI_maps(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_NONPOLY_ITI["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_NONPOLY_ITI["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_NONPOLY_ITI["d_yy_coeff_fn"],
        source_fn=TEST_CASE_NONPOLY_ITI["source_fn"],
        dudx_fn=TEST_CASE_NONPOLY_ITI["du_dx_fn"],
        dudy_fn=TEST_CASE_NONPOLY_ITI["du_dy_fn"],
        plot_fp=plot_fp,
        compute_convergence_rate=True,
        eta=4.0,
    )


def single_leaf_check_7(plot_fp: str) -> None:
    """Measures convergence of ItI mapping when using poly data."""
    logging.info("Running single_leaf_check_7")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_ItI_maps(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POLY_ITI["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POLY_ITI["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POLY_ITI["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POLY_ITI["source_fn"],
        dudx_fn=TEST_CASE_POLY_ITI["du_dx_fn"],
        dudy_fn=TEST_CASE_POLY_ITI["du_dy_fn"],
        compute_convergence_rate=False,
        eta=4.0,
        plot_fp=plot_fp,
        check_ItI=True,
    )


def single_leaf_check_8(plot_fp: str) -> None:
    """Measures convergence of ItI mapping when using non-poly data."""
    logging.info("Running single_leaf_check_8")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_ItI_maps(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_NONPOLY_ITI["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_NONPOLY_ITI["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_NONPOLY_ITI["d_yy_coeff_fn"],
        source_fn=TEST_CASE_NONPOLY_ITI["source_fn"],
        dudx_fn=TEST_CASE_NONPOLY_ITI["du_dx_fn"],
        dudy_fn=TEST_CASE_NONPOLY_ITI["du_dy_fn"],
        compute_convergence_rate=True,
        eta=4.0,
        plot_fp=plot_fp,
        check_ItI=True,
    )


def single_leaf_check_9(plot_fp: str) -> None:
    """Measures convergence of ItI mapping when using non-poly data with a zero-source problem."""
    logging.info("Running single_leaf_check_9")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_ItI_maps(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_NONPOLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_NONPOLY["source_fn"],
        dudx_fn=TEST_CASE_POISSON_NONPOLY["du_dx_fn"],
        dudy_fn=TEST_CASE_POISSON_NONPOLY["du_dy_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        eta=4.0,
        check_ItI=True,
    )


def single_leaf_check_10(plot_fp: str) -> None:
    """Measures convergence of interior solution when using non-poly data with a zero-source problem."""
    logging.info("Running single_leaf_check_10")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_ItI_maps(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_POISSON_NONPOLY["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_POISSON_NONPOLY["d_yy_coeff_fn"],
        source_fn=TEST_CASE_POISSON_NONPOLY["source_fn"],
        dudx_fn=TEST_CASE_POISSON_NONPOLY["du_dx_fn"],
        dudy_fn=TEST_CASE_POISSON_NONPOLY["du_dy_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        eta=4.0,
        check_ItI=False,
    )


def single_leaf_check_11(plot_fp: str) -> None:
    """Measures convergence of interior solution when using non-poly data with a zero-source problem.

    If convergence does not look good on this problem, it is likely that K_1 is set too high in
    dirichelet_neumann_data.py. This parameter controls the number of oscilations in the data.
    """
    logging.info("Running single_leaf_check_11")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_particular_homog_solns_ItI(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_HELMHOLTZ_0["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_HELMHOLTZ_0["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_HELMHOLTZ_0["d_yy_coeff_fn"],
        I_coeff_fn=TEST_CASE_HELMHOLTZ_0["I_coeff_fn"],
        source_fn=TEST_CASE_HELMHOLTZ_0["source_fn"],
        homogeneous_soln_fn=TEST_CASE_HELMHOLTZ_0["dirichlet_data_fn"],
        particular_soln_fn=TEST_CASE_HELMHOLTZ_0["part_soln_fn"],
        particular_dx_fn=TEST_CASE_HELMHOLTZ_0["part_du_dy_fn"],
        particular_dy_fn=TEST_CASE_HELMHOLTZ_0["part_du_dy_fn"],
        dudx_fn=TEST_CASE_HELMHOLTZ_0["du_dx_fn"],
        dudy_fn=TEST_CASE_HELMHOLTZ_0["du_dy_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        eta=ETA,
    )


def single_leaf_check_12(plot_fp: str) -> None:
    """Measures convergence of interior solution when using non-poly data with a nonzero source problem.

    If convergence does not look good on this problem, it is likely that K_1 is set too high in
    dirichelet_neumann_data.py. This parameter controls the number of oscilations in the data.
    """
    logging.info("Running single_leaf_check_12")
    p_values = jnp.array([4, 8, 12, 16])
    check_l_inf_error_convergence_particular_homog_solns_ItI(
        p_values=p_values,
        dirichlet_data_fn=TEST_CASE_HELMHOLTZ_1["dirichlet_data_fn"],
        d_xx_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_xx_coeff_fn"],
        d_yy_coeff_fn=TEST_CASE_HELMHOLTZ_1["d_yy_coeff_fn"],
        I_coeff_fn=TEST_CASE_HELMHOLTZ_1["I_coeff_fn"],
        source_fn=TEST_CASE_HELMHOLTZ_1["source_fn"],
        homogeneous_soln_fn=TEST_CASE_HELMHOLTZ_1["dirichlet_data_fn"],
        particular_soln_fn=TEST_CASE_HELMHOLTZ_1["part_soln_fn"],
        particular_dx_fn=TEST_CASE_HELMHOLTZ_1["part_du_dx_fn"],
        particular_dy_fn=TEST_CASE_HELMHOLTZ_1["part_du_dy_fn"],
        dudx_fn=TEST_CASE_HELMHOLTZ_1["du_dx_fn"],
        dudy_fn=TEST_CASE_HELMHOLTZ_1["du_dy_fn"],
        compute_convergence_rate=True,
        plot_fp=plot_fp,
        eta=ETA,
    )
