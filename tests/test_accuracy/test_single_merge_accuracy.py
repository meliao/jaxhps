import logging
from typing import Dict
import jax.numpy as jnp
import jax
import pytest
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem
from jaxhps.local_solve._uniform_2D_DtN import local_solve_stage_uniform_2D_DtN
from jaxhps.local_solve._uniform_2D_ItI import local_solve_stage_uniform_2D_ItI
from jaxhps.merge._uniform_2D_DtN import merge_stage_uniform_2D_DtN
from jaxhps.merge._uniform_2D_ItI import merge_stage_uniform_2D_ItI
from jaxhps.down_pass._uniform_2D_DtN import down_pass_uniform_2D_DtN
from jaxhps.down_pass._uniform_2D_ItI import down_pass_uniform_2D_ItI
# from jaxhps._utils import plot_soln_from_cheby_nodes

from .cases import (
    XMIN,
    XMAX,
    YMIN,
    YMAX,
    ETA,
    TEST_CASE_POLY_PART_HOMOG,
    TEST_CASE_POLY_ZERO_SOURCE,
    TEST_CASE_HELMHOLTZ_ITI,
    TEST_CASE_HELMHOLTZ_ITI_COMPLEX_COEFFS,
    K_DIRICHLET,
    K_XX_COEFF,
    K_YY_COEFF,
    K_SOURCE,
    K_DIRICHLET_DUDX,
    K_DIRICHLET_DUDY,
    K_PART_SOLN,
    K_HOMOG_SOLN,
    K_SOLN,
    K_DUDX,
    K_DUDY,
    K_I_COEFF,
)

ATOL_NONPOLY = 1e-8

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
DOMAIN_ITI_NONPOLY = Domain(p=P_NONPOLY, q=Q_NONPOLY, root=ROOT_ITI, L=1)


def check_merge_accuracy_2D_DtN_uniform(
    domain: Domain,
    test_case: Dict,
    use_rectangular_spectral_collocation: bool = False,
) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
    d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)
    source_term = test_case[K_SOURCE](domain.interior_points).squeeze(-1)
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: d_xx_coeffs = %s",
        d_xx_coeffs.shape,
    )
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: d_yy_coeffs = %s",
        d_yy_coeffs.shape,
    )
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: source_term = %s",
        source_term.shape,
    )

    # Create a PDEProblem object
    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        source=source_term,
        use_rectangular_spectral_collocation=use_rectangular_spectral_collocation,
    )

    # Do the local solve and build stages
    Y, T, v, h = local_solve_stage_uniform_2D_DtN(pde_problem=pde_problem)
    S_lst, g_tilde_lst, T_last = merge_stage_uniform_2D_DtN(
        T, h, domain.L - 1, return_T=True
    )

    # Log shapes of g_tilde_lst
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: g_tilde_lst shapes = %s",
        [g_tilde.shape for g_tilde in g_tilde_lst],
    )

    logging.debug("check_merge_accuracy_2D_DtN_uniform: v shape = %s", v.shape)
    bdry_data = test_case[K_DIRICHLET](domain.boundary_points)

    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: bdry_data shape = %s",
        bdry_data.shape,
    )
    # Do the down pass
    computed_soln = down_pass_uniform_2D_DtN(
        bdry_data, S_lst, g_tilde_lst, Y, v
    )

    ##########################################################
    # Check the accuracy of the DtN matrix
    # TODO: Add an option to save the DtN matrix.

    boundary_g = test_case[K_DIRICHLET](domain.boundary_points)
    n_per_side = domain.boundary_points.shape[0] // 4
    boundary_g_normals = jnp.concatenate(
        [
            -1
            * test_case[K_DIRICHLET_DUDY](domain.boundary_points[:n_per_side]),
            test_case[K_DIRICHLET_DUDX](
                domain.boundary_points[n_per_side : 2 * n_per_side]
            ),
            test_case[K_DIRICHLET_DUDY](
                domain.boundary_points[2 * n_per_side : 3 * n_per_side]
            ),
            -1
            * test_case[K_DIRICHLET_DUDX](
                domain.boundary_points[3 * n_per_side :]
            ),
        ]
    )

    computed_h = T_last @ boundary_g

    assert jnp.allclose(boundary_g_normals, computed_h, atol=ATOL, rtol=RTOL)

    ##########################################################
    # Check the accuracy of the computed solution
    homog_soln = test_case[K_HOMOG_SOLN](domain.interior_points)
    part_soln = test_case[K_PART_SOLN](domain.interior_points).reshape(
        homog_soln.shape
    )

    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: part_soln = %s", part_soln.shape
    )
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: homog_soln = %s",
        homog_soln.shape,
    )
    expected_soln = part_soln + homog_soln
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: computed_soln = %s",
        computed_soln.shape,
    )
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: expected_soln = %s",
        expected_soln.shape,
    )

    # plot_soln_from_cheby_nodes(
    #     domain.interior_points.reshape(-1, 2),
    #     None,
    #     expected_soln.reshape(-1),
    #     computed_soln.reshape(-1),
    # )
    assert jnp.allclose(computed_soln, expected_soln, atol=ATOL, rtol=RTOL)


def check_merge_accuracy_2D_ItI_uniform(
    domain: Domain, test_case: Dict
) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
    d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)
    source_term = test_case[K_SOURCE](domain.interior_points).squeeze(-1)
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: d_xx_coeffs = %s",
        d_xx_coeffs.shape,
    )
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: d_yy_coeffs = %s",
        d_yy_coeffs.shape,
    )
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: source_term = %s",
        source_term.shape,
    )

    # Create a PDEProblem object
    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        source=source_term,
        use_ItI=True,
        eta=ETA,
    )
    # Do the local solve and build stages
    Y, T, v, h = local_solve_stage_uniform_2D_ItI(pde_problem=pde_problem)
    S_lst, g_tilde_lst, T_last = merge_stage_uniform_2D_ItI(
        T, h, domain.L - 1, return_T=True
    )

    # Compute incoming impedance data
    n_per_side = domain.boundary_points.shape[0] // 4
    boundary_g_normals = jnp.concatenate(
        [
            -1
            * test_case[K_DIRICHLET_DUDY](domain.boundary_points[:n_per_side]),
            test_case[K_DIRICHLET_DUDX](
                domain.boundary_points[n_per_side : 2 * n_per_side]
            ),
            test_case[K_DIRICHLET_DUDY](
                domain.boundary_points[2 * n_per_side : 3 * n_per_side]
            ),
            -1
            * test_case[K_DIRICHLET_DUDX](
                domain.boundary_points[3 * n_per_side :]
            ),
        ]
    )
    boundary_g_evals = test_case[K_DIRICHLET](domain.boundary_points)
    incoming_imp_data = (
        boundary_g_normals + 1j * pde_problem.eta * boundary_g_evals
    )
    bdry_data = jnp.concatenate(
        [
            incoming_imp_data[:n_per_side],
            incoming_imp_data[n_per_side : 2 * n_per_side],
            incoming_imp_data[2 * n_per_side : 3 * n_per_side],
            incoming_imp_data[3 * n_per_side :],
        ]
    )
    computed_soln = down_pass_uniform_2D_ItI(
        bdry_data, S_lst, g_tilde_lst, Y, v
    )
    ##########################################################
    # Check the accuracy of the ItI matrix

    expected_outgoing_imp_data = (
        boundary_g_normals - 1j * pde_problem.eta * boundary_g_evals
    )
    computed_outgoing_imp_data = T_last @ incoming_imp_data
    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: computed_outgoing_imp_data = %s",
        computed_outgoing_imp_data.shape,
    )
    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: expected_outgoing_imp_data = %s",
        expected_outgoing_imp_data.shape,
    )
    assert jnp.allclose(
        computed_outgoing_imp_data,
        expected_outgoing_imp_data,
        atol=ATOL,
        rtol=RTOL,
    )

    ##########################################################
    # Check the accuracy of the computed solution
    homog_soln = test_case[K_HOMOG_SOLN](domain.interior_points)
    # Part soln function is written to return a third dimension for different sources. This verison of the code
    # does not support multiple sources, so we reshape the part_soln to not have that third axis.
    part_soln = test_case[K_PART_SOLN](domain.interior_points).reshape(
        domain.interior_points.shape[:2]
    )

    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: part_soln = %s", part_soln.shape
    )
    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: homog_soln = %s",
        homog_soln.shape,
    )
    expected_soln = part_soln + homog_soln

    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: computed_soln = %s",
        computed_soln.shape,
    )
    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: expected_soln = %s",
        expected_soln.shape,
    )

    # Uncomment this if you want to plot the solution for debugging.

    # plot_soln_from_cheby_nodes(
    #     cheby_nodes=domain.interior_points.reshape(-1, 2),
    #     computed_soln=computed_soln.reshape(-1).real,
    #     expected_soln=expected_soln.reshape(-1).real,
    #     corners=None,

    # )
    assert jnp.allclose(computed_soln, expected_soln, atol=ATOL, rtol=RTOL)


def check_merge_accuracy_2D_ItI_uniform_Helmholtz_like(
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
        source=source,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        I_coefficients=i_coeffs,
        use_ItI=True,
        eta=ETA,
    )

    ##############################################################
    # Solve the local problem
    Y, T, v, h = local_solve_stage_uniform_2D_ItI(pde_problem=pde_problem)
    S_lst, g_tilde_lst, T_last = merge_stage_uniform_2D_ItI(
        T, h, domain.L - 1, return_T=True
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

    # Plot the solution
    # plot_soln_from_cheby_nodes(
    #     cheby_nodes=domain.interior_points.reshape(-1, 2),
    #     corners=None,
    #     expected_soln=expected_soln.imag.flatten(),
    #     computed_soln=computed_soln.imag.flatten(),
    # )

    assert jnp.allclose(
        computed_soln,
        expected_soln,
        atol=ATOL_NONPOLY,
        rtol=RTOL,
    )


class Test_accuracy_single_merge_2D_DtN_uniform:
    def test_0(self, caplog) -> None:
        """Polynomial data with zero source term."""
        caplog.set_level(logging.DEBUG)
        check_merge_accuracy_2D_DtN_uniform(
            DOMAIN_DTN, TEST_CASE_POLY_ZERO_SOURCE
        )

    def test_1(self, caplog) -> None:
        """Polynomial data with non-zero source term."""
        caplog.set_level(logging.DEBUG)
        check_merge_accuracy_2D_DtN_uniform(
            DOMAIN_DTN, TEST_CASE_POLY_PART_HOMOG
        )
        jax.clear_caches()

    def test_2(self, caplog) -> None:
        """Polynomial data with zero source term. Uses the rectangular spectral collocation method."""
        caplog.set_level(logging.DEBUG)
        check_merge_accuracy_2D_DtN_uniform(
            DOMAIN_DTN,
            TEST_CASE_POLY_ZERO_SOURCE,
            use_rectangular_spectral_collocation=True,
        )

    def test_3(self, caplog) -> None:
        """Polynomial data with non-zero source term. Uses the rectangular spectral collocation methd."""
        caplog.set_level(logging.DEBUG)
        check_merge_accuracy_2D_DtN_uniform(
            DOMAIN_DTN,
            TEST_CASE_POLY_PART_HOMOG,
            use_rectangular_spectral_collocation=True,
        )
        jax.clear_caches()


class Test_accuracy_single_merge_2D_ItI_uniform:
    def test_0(self, caplog) -> None:
        """Polynomial data with zero source term."""
        caplog.set_level(logging.DEBUG)

        check_merge_accuracy_2D_ItI_uniform(
            DOMAIN_ITI, TEST_CASE_POLY_ZERO_SOURCE
        )
        jax.clear_caches()

    def test_1(self, caplog) -> None:
        """Polynomial data with non-zero source term."""
        caplog.set_level(logging.DEBUG)

        check_merge_accuracy_2D_ItI_uniform_Helmholtz_like(
            DOMAIN_ITI_NONPOLY, TEST_CASE_HELMHOLTZ_ITI
        )
        jax.clear_caches()

    def test_2(self, caplog) -> None:
        """Complex coefficients."""
        caplog.set_level(logging.DEBUG)

        check_merge_accuracy_2D_ItI_uniform_Helmholtz_like(
            DOMAIN_ITI_NONPOLY, TEST_CASE_HELMHOLTZ_ITI_COMPLEX_COEFFS
        )
        jax.clear_caches()


if __name__ == "__main__":
    pytest.main()
