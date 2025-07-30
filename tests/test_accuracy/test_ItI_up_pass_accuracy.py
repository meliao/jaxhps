import logging
from typing import Dict
import jax.numpy as jnp
import jax

from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem
from jaxhps._build_solver import build_solver
from jaxhps.up_pass._uniform_2D_ItI import up_pass_uniform_2D_ItI
from jaxhps.local_solve._nosource_uniform_2D_ItI import (
    nosource_local_solve_stage_uniform_2D_ItI,
)
# from jaxhps._utils import plot_soln_from_cheby_nodes

from .cases import (
    XMIN,
    XMAX,
    YMIN,
    YMAX,
    ETA,
    TEST_CASE_ITI_PART_HOMOG,
    K_XX_COEFF,
    K_YY_COEFF,
    K_SOURCE,
    K_I_COEFF,
    K_PART_SOLN_DUDY,
    K_PART_SOLN_DUDX,
    K_PART_SOLN,
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


def check_up_pass_accuracy_2D_ItI_uniform_Helmholtz_like(
    domain: Domain, test_case: Dict
) -> None:
    """This is for ItI problems solving an inhomogeneous Helmholtz equation where the
    solution is specified as one solution, rathern than the sum of homogeneous and particular parts
    """
    d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
    d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)
    i_coeffs = test_case[K_I_COEFF](domain.interior_points)

    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        I_coefficients=i_coeffs,
        use_ItI=True,
        eta=ETA,
    )

    ##############################################################
    # Build the solver.
    build_solver(pde_problem=pde_problem, return_top_T=False)

    ##############################################################
    # Do an up pass to compute the incoming and outgoing impedance data
    source = test_case[K_SOURCE](domain.interior_points)

    _, _, h_last = up_pass_uniform_2D_ItI(
        source=source, pde_problem=pde_problem, return_bdry_imp_data=True
    )

    # Assemble incoming impedance data
    q = domain.boundary_points.shape[0] // 4
    boundary_u = test_case[K_PART_SOLN](domain.boundary_points)
    boundary_u_normals = jnp.concatenate(
        [
            -1 * test_case[K_PART_SOLN_DUDY](domain.boundary_points[:q]),
            test_case[K_PART_SOLN_DUDX](domain.boundary_points[q : 2 * q]),
            test_case[K_PART_SOLN_DUDY](domain.boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_PART_SOLN_DUDX](domain.boundary_points[3 * q :]),
        ]
    )
    outgoing_imp_data = boundary_u_normals - 1j * pde_problem.eta * boundary_u

    # Plot the magnitudes
    # plt.plot(jnp.abs(h_last), ".-", label="computed out")
    # plt.plot(jnp.abs(outgoing_imp_data), "x-", label="expected out")
    # # plt.plot(jnp.abs(incoming_imp_data), label="expected in")
    # plt.legend()
    # plt.show()

    # Check the outgoing impedance data
    assert jnp.allclose(
        h_last, outgoing_imp_data, atol=ATOL_NONPOLY, rtol=RTOL
    )

    # Check the incoming impedance data
    # assert jnp.allclose(g_last, incoming_imp_data, atol=ATOL, rtol=RTOL)


class Test_accuracy_up_pass_2D_ItI_uniform:
    def test_0(self, caplog) -> None:
        """Testing whether the computed outgoing and incoming particular soln impedance data matches against expectatiosn after 1 merge level."""
        caplog.set_level(logging.DEBUG)

        check_up_pass_accuracy_2D_ItI_uniform_Helmholtz_like(
            DOMAIN_ITI_NONPOLY, TEST_CASE_ITI_PART_HOMOG
        )
        jax.clear_caches()

    def test_1(self, caplog) -> None:
        """Testing whether the computed outgoing and incoming particular soln impedance data matches against expectatiosn after the local solve."""
        caplog.set_level(logging.DEBUG)

        domain = Domain(p=P_NONPOLY, q=Q_NONPOLY, root=ROOT_ITI, L=0)

        test_case = TEST_CASE_ITI_PART_HOMOG
        d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
        d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)
        i_coeffs = test_case[K_I_COEFF](domain.interior_points)

        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            I_coefficients=i_coeffs,
            use_ItI=True,
            eta=ETA,
        )

        # Do the local solve

        solve_out = nosource_local_solve_stage_uniform_2D_ItI(
            pde_problem=pde_problem
        )

        Y, R, Phi = solve_out

        # Compute the particular solution
        n_cheby_bdry = 4 * pde_problem.domain.p - 4
        source = test_case[K_SOURCE](pde_problem.domain.interior_points)[
            :, n_cheby_bdry:
        ]
        logging.debug(
            "up_pass_uniform_2D_ItI: source shape = %s", source.shape
        )
        logging.debug("up_pass_uniform_2D_ItI: Phi shape = %s", Phi.shape)
        v = jnp.einsum("ijk,ik->ij", Phi, source)
        logging.debug("up_pass_uniform_2D_ItI: v shape = %s", v.shape)
        logging.debug(
            "up_pass_uniform_2D_ItI: QH shape = %s", pde_problem.QH.shape
        )

        v = v.squeeze()

        outgoing_imp_part = pde_problem.QH @ v
        incoming_imp_part = pde_problem.QG @ v

        # Construct the expected impedance data
        q = pde_problem.domain.boundary_points.shape[0] // 4
        boundary_u = test_case[K_PART_SOLN](pde_problem.domain.boundary_points)
        boundary_u_normals = jnp.concatenate(
            [
                -1
                * test_case[K_PART_SOLN_DUDY](
                    pde_problem.domain.boundary_points[:q]
                ),
                test_case[K_PART_SOLN_DUDX](
                    pde_problem.domain.boundary_points[q : 2 * q]
                ),
                test_case[K_PART_SOLN_DUDY](
                    pde_problem.domain.boundary_points[2 * q : 3 * q]
                ),
                -1
                * test_case[K_PART_SOLN_DUDX](
                    pde_problem.domain.boundary_points[3 * q :]
                ),
            ]
        )

        expected_out = boundary_u_normals - 1j * pde_problem.eta * boundary_u
        expected_in = boundary_u_normals + 1j * pde_problem.eta * boundary_u

        # Plot the outgoing data
        # plt.plot(jnp.abs(outgoing_imp_part), ".-", label="computed out")
        # plt.plot(jnp.abs(incoming_imp_part), "x-", label="computed in")
        # plt.plot(jnp.abs(expected_out), label="expected out")
        # plt.plot(jnp.abs(expected_in), label="expected in")
        # plt.legend()
        # plt.show()

        # Check equality
        assert jnp.allclose(
            outgoing_imp_part, expected_out, atol=ATOL_NONPOLY, rtol=RTOL
        )
        assert jnp.allclose(
            incoming_imp_part, expected_in, atol=ATOL_NONPOLY, rtol=RTOL
        )

        jax.clear_caches()
