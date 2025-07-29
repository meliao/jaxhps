import logging
from typing import Dict
import jax.numpy as jnp
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem
from jaxhps.local_solve import (
    local_solve_stage_uniform_2D_ItI,
    nosource_local_solve_stage_uniform_2D_ItI,
    nosource_local_solve_stage_uniform_2D_DtN,
)
from jaxhps.merge import (
    merge_stage_uniform_2D_ItI,
    nosource_merge_stage_uniform_2D_DtN,
    nosource_merge_stage_uniform_2D_ItI,
)
from jaxhps.up_pass import up_pass_uniform_2D_DtN, up_pass_uniform_2D_ItI
from jaxhps.down_pass import down_pass_uniform_2D_DtN, down_pass_uniform_2D_ItI

from .cases import (
    XMIN,
    XMAX,
    YMIN,
    YMAX,
    ETA,
    TEST_CASE_HELMHOLTZ_ITI,
    TEST_CASE_POLY_PART_HOMOG,
    TEST_CASE_ITI_PART_HOMOG,
    K_HOMOG_SOLN,
    K_DIRICHLET,
    K_PART_SOLN,
    K_XX_COEFF,
    K_YY_COEFF,
    K_SOURCE,
    K_SOLN,
    K_DUDX,
    K_DUDY,
    K_I_COEFF,
)


# Imports necessary for the plotting functions, which are currently disabled.
# These are useful to sniff out bugs.
from jaxhps._utils import plot_soln_from_cheby_nodes
# import matplotlib.pyplot as plt

ATOL_NONPOLY = 1e-8
ATOL_POLY = 1e-12

ATOL = 1e-12
RTOL = 0.0

P = 6
Q = 4

P_NONPOLY = 16
Q_NONPOLY = 14
ROOT_DTN = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
ROOT_ITI = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
DOMAIN_DTN = Domain(p=P, q=Q, root=ROOT_DTN, L=1)
DOMAIN_ITI = Domain(p=P, q=Q, root=ROOT_ITI, L=1)
DOMAIN_ITI_NONPOLY = Domain(p=P_NONPOLY, q=Q_NONPOLY, root=ROOT_ITI, L=2)


def check_merge_accuracy_nosource_2D_ItI_uniform_Helmholtz_like(
    domain: Domain, test_case: Dict
) -> None:
    """This is for ItI problems solving an inhomogeneous Helmholtz equation where the
    solution is specified as one solution, rathern than the sum of homogeneous and particular parts
    """
    d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
    d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)
    i_coeffs = test_case[K_I_COEFF](domain.interior_points)
    source = test_case[K_SOURCE](domain.interior_points)

    logging.debug(
        "check_leaf_accuracy_ItI_Helmholtz_like: source shape: %s",
        source.shape,
    )

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        I_coefficients=i_coeffs,
        use_ItI=True,
        eta=ETA,
    )

    ##############################################################
    # Build the precomputed solution operators
    Y, T, Phi = nosource_local_solve_stage_uniform_2D_ItI(
        pde_problem=pde_problem
    )
    S_lst, D_inv_lst, BD_inv_lst, T = nosource_merge_stage_uniform_2D_ItI(
        T, domain.L, return_T=True
    )

    pde_problem.Y = Y
    pde_problem.Phi = Phi

    pde_problem.D_inv_lst = D_inv_lst
    pde_problem.BD_inv_lst = BD_inv_lst

    logging.debug("D_inv_lst shapes: %s", [d_inv.shape for d_inv in D_inv_lst])
    logging.debug(
        "BD_inv_lst shapes: %s", [bd_inv.shape for bd_inv in BD_inv_lst]
    )

    #############################################################
    # Upward pass
    v, g_tilde_lst = up_pass_uniform_2D_ItI(
        source=source, pde_problem=pde_problem
    )

    ##############################################################
    # Compute the incoming impedance data

    # Assemble incoming impedance data
    q = domain.boundary_points.shape[0] // 4
    boundary_u = test_case[K_SOLN](domain.boundary_points)
    boundary_u_normals = jnp.concatenate(
        [
            -1 * test_case[K_DUDY](domain.boundary_points[:q]),
            test_case[K_DUDX](domain.boundary_points[q : 2 * q]),
            test_case[K_DUDY](domain.boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_DUDX](domain.boundary_points[3 * q :]),
        ]
    )
    incoming_imp_data = boundary_u_normals + 1j * pde_problem.eta * boundary_u

    ##############################################################
    # Check the accuracy of the homogeneous solution
    # Construct computed homogeneous solution

    computed_soln = down_pass_uniform_2D_ItI(
        incoming_imp_data, S_lst, g_tilde_lst, Y, v
    )
    logging.debug(
        "check_leaf_accuracy_ItI_Helmholtz_like: computed_soln shape: %s",
        computed_soln.shape,
    )
    expected_soln = test_case[K_SOLN](domain.interior_points)
    logging.debug(
        "check_leaf_accuracy_ItI_Helmholtz_like: expected_soln shape: %s",
        expected_soln.shape,
    )

    # Plot the solution. This function can be found in src//_utils.py
    plot_soln_from_cheby_nodes(
        cheby_nodes=domain.interior_points.reshape(-1, 2),
        corners=None,
        expected_soln=expected_soln.real.flatten(),
        computed_soln=computed_soln.real.flatten(),
    )

    max_diff = jnp.max(jnp.abs(computed_soln - expected_soln))

    assert jnp.allclose(
        computed_soln,
        expected_soln,
        atol=ATOL_NONPOLY,
        rtol=RTOL,
    ), f"Maximum difference = {max_diff}"


ATOL_DIFFS = 1e-8
RTOL_DIFFS = 0.0


def check_against_standard_2D_ItI_uniform(
    domain: Domain, test_case: Dict
) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
    d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)
    i_coeffs = test_case[K_I_COEFF](domain.interior_points)
    source = test_case[K_SOURCE](domain.interior_points)

    pde_problem_nosource = PDEProblem(
        domain=domain,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        I_coefficients=i_coeffs,
        use_ItI=True,
        eta=ETA,
    )

    pde_problem_source = PDEProblem(
        domain=domain,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        I_coefficients=i_coeffs,
        source=source,
        use_ItI=True,
        eta=ETA,
    )

    ##############################################################
    # Check outputs of local solve stage
    Y_nosource, T_nosource, Phi_nosource = (
        nosource_local_solve_stage_uniform_2D_ItI(
            pde_problem=pde_problem_nosource
        )
    )
    pde_problem_nosource.Y = Y_nosource
    pde_problem_nosource.Phi = Phi_nosource

    Y, T, v, h = local_solve_stage_uniform_2D_ItI(
        pde_problem=pde_problem_source
    )

    assert jnp.allclose(Y_nosource, Y, atol=ATOL_DIFFS, rtol=RTOL_DIFFS), (
        f"Max difference = {jnp.max(jnp.abs(Y_nosource - Y))}"
    )
    assert jnp.allclose(T_nosource, T, atol=ATOL_DIFFS, rtol=RTOL_DIFFS), (
        f"Max difference = {jnp.max(jnp.abs(T_nosource - T))}"
    )

    ##############################################################
    # Check outputs of merge stage
    (
        S_lst_nosource,
        D_inv_lst_nosource,
        BD_inv_lst_nosource,
        T_last_nosource,
    ) = nosource_merge_stage_uniform_2D_ItI(
        T_nosource, domain.L, return_T=True
    )
    pde_problem_nosource.D_inv_lst = D_inv_lst_nosource
    pde_problem_nosource.BD_inv_lst = BD_inv_lst_nosource
    S_lst, g_tilde_lst, T_last_source = merge_stage_uniform_2D_ItI(
        T, h, domain.L, return_T=True
    )

    for i in range(len(S_lst)):
        assert jnp.allclose(
            S_lst_nosource[i], S_lst[i], atol=ATOL_DIFFS, rtol=RTOL_DIFFS
        ), (
            f"Max difference in S_lst[{i}] = {jnp.max(jnp.abs(S_lst_nosource[i] - S_lst[i]))}"
        )

    # Check top-level ItI matrices
    assert jnp.allclose(
        T_last_nosource, T_last_source, atol=ATOL_DIFFS, rtol=RTOL_DIFFS
    ), (
        f"Max difference in T_last_nosource and T_last_source = {jnp.max(jnp.abs(T_last_nosource - T_last_source))}"
    )

    ##################################################################
    # Check outputs of upward pass
    v_nosource, g_tilde_lst_nosource = up_pass_uniform_2D_ItI(
        source=source, pde_problem=pde_problem_nosource
    )

    # Check v
    assert jnp.allclose(v_nosource, v, atol=ATOL_DIFFS, rtol=RTOL_DIFFS), (
        f"Max difference in v = {jnp.max(jnp.abs(v_nosource - v))}"
    )
    logging.debug("g_tilde_lst len: %s", len(g_tilde_lst))
    logging.debug("g_tilde_lst_nosource len: %s", len(g_tilde_lst_nosource))
    for i in range(len(g_tilde_lst)):
        logging.debug(
            "g_tilde_list_nosource[i].shape=%s", g_tilde_lst_nosource[i].shape
        )
        logging.debug("g_tilde_list[i].shape=%s", g_tilde_lst[i].shape)

        # for j in range(1):
        #     g_tilde = g_tilde_lst[i][j]
        #     g_tilde_nosource = g_tilde_lst_nosource[i][j]
        #     plt.plot(g_tilde.real, ".-", label=f"Expected real, j={j}")
        #     plt.plot(
        #         g_tilde_nosource.real, ".-", label=f"Computed real, j={j}"
        #     )
        # plt.legend()
        # plt.show()
        assert jnp.allclose(
            g_tilde_lst_nosource[i],
            g_tilde_lst[i],
            atol=ATOL_DIFFS,
            rtol=RTOL_DIFFS,
        ), (
            f"Max difference in g_tilde_lst_nosource[{i}] = {jnp.max(jnp.abs(g_tilde_lst_nosource[i] - g_tilde_lst[i]))}"
        )


def check_merge_accuracy_nosource_2D_DtN_uniform(
    domain: Domain, test_case: Dict
) -> None:
    """This is for ItI problems solving an inhomogeneous Helmholtz equation where the
    solution is specified as one solution, rathern than the sum of homogeneous and particular parts
    """
    d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
    d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)
    source = test_case[K_SOURCE](domain.interior_points)[..., 0]

    logging.debug(
        "check_leaf_accuracy_ItI_Helmholtz_like: source shape: %s",
        source.shape,
    )

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
    )

    ##############################################################
    # Build the precomputed solution operators
    Y, T, Phi = nosource_local_solve_stage_uniform_2D_DtN(
        pde_problem=pde_problem
    )
    pde_problem.Y = Y
    pde_problem.Phi = Phi
    S_lst, D_inv_lst, BD_inv_lst, T = nosource_merge_stage_uniform_2D_DtN(
        T, domain.L, return_T=True
    )

    pde_problem.D_inv_lst = D_inv_lst
    pde_problem.BD_inv_lst = BD_inv_lst

    logging.debug("D_inv_lst shapes: %s", [d_inv.shape for d_inv in D_inv_lst])
    logging.debug(
        "BD_inv_lst shapes: %s", [bd_inv.shape for bd_inv in BD_inv_lst]
    )
    logging.debug(
        "check_leaf_accuracy_ItI_Helmholtz_like: S_lst shapes: %s",
        [s.shape for s in S_lst],
    )

    #############################################################
    # Upward pass
    v, g_tilde_lst = up_pass_uniform_2D_DtN(
        source=source, pde_problem=pde_problem
    )

    ##############################################################
    # Compute the incoming impedance data

    # Assemble incoming impedance data
    boundary_u = test_case[K_DIRICHLET](domain.boundary_points)[..., None]
    logging.debug(
        "check_leaf_accuracy_ItI_Helmholtz_like: boundary_u shape: %s",
        boundary_u.shape,
    )

    ##############################################################
    # Check the accuracy of the homogeneous solution
    # Construct computed homogeneous solution

    computed_soln = down_pass_uniform_2D_DtN(
        boundary_u, S_lst, g_tilde_lst, Y, v
    ).flatten()
    logging.debug(
        "check_leaf_accuracy_ItI_Helmholtz_like: computed_soln shape: %s",
        computed_soln.shape,
    )

    expected_soln_homog = test_case[K_HOMOG_SOLN](
        domain.interior_points
    ).flatten()
    expected_soln_part = test_case[K_PART_SOLN](
        domain.interior_points
    ).flatten()

    logging.debug(
        "check_leaf_accuracy_ItI_Helmholtz_like: expected_soln_homog shape: %s",
        expected_soln_homog.shape,
    )
    logging.debug(
        "check_leaf_accuracy_ItI_Helmholtz_like: expected_soln_part shape: %s",
        expected_soln_part.shape,
    )
    expected_soln = expected_soln_homog + expected_soln_part

    # Plot the solution. This function can be found in src//_utils.py
    # plot_soln_from_cheby_nodes(
    #     cheby_nodes=domain.interior_points.reshape(-1, 2),
    #     corners=None,
    #     expected_soln=expected_soln.imag.flatten(),
    #     computed_soln=computed_soln.imag.flatten(),
    # )

    max_diff = jnp.max(jnp.abs(computed_soln - expected_soln))

    assert jnp.allclose(
        computed_soln,
        expected_soln,
        atol=ATOL_POLY,
        rtol=RTOL,
    ), f"Maximum difference = {max_diff}"


class Test_accuracy_2D_ItI_uniform:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)

        check_merge_accuracy_nosource_2D_ItI_uniform_Helmholtz_like(
            DOMAIN_ITI_NONPOLY, TEST_CASE_HELMHOLTZ_ITI
        )

    def test_1(self, caplog) -> None:
        """Tests to make sure things run without error."""
        caplog.set_level(logging.DEBUG)
        check_merge_accuracy_nosource_2D_ItI_uniform_Helmholtz_like(
            DOMAIN_ITI, TEST_CASE_ITI_PART_HOMOG
        )


class Test_against_standard_version_ItI:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        check_against_standard_2D_ItI_uniform(
            DOMAIN_ITI_NONPOLY, TEST_CASE_HELMHOLTZ_ITI
        )


class Test_accuracy_2D_DtN_uniform:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        check_merge_accuracy_nosource_2D_DtN_uniform(
            DOMAIN_DTN, TEST_CASE_POLY_PART_HOMOG
        )
