import logging
from typing import Dict
import jax.numpy as jnp
import jax
import pytest
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem
from jaxhps.local_solve._uniform_2D_DtN import local_solve_stage_uniform_2D_DtN
from jaxhps.local_solve._nosource_uniform_2D_ItI import (
    nosource_local_solve_stage_uniform_2D_ItI,
)
from jaxhps.local_solve._uniform_2D_ItI import local_solve_stage_uniform_2D_ItI


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
    K_I_COEFF,
    K_SOURCE,
    K_DIRICHLET_DUDX,
    K_DIRICHLET_DUDY,
    K_PART_SOLN,
    K_PART_SOLN_DUDX,
    K_PART_SOLN_DUDY,
    K_HOMOG_SOLN,
    K_SOLN,
    K_DUDX,
    K_DUDY,
)


ATOL = 1e-12
RTOL = 0.0

ATOL_NONPOLY = 1e-5

P = 16
Q = 14
ROOT_DTN = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
ROOT_ITI = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
DOMAIN_DTN = Domain(p=P, q=Q, root=ROOT_DTN, L=0)
DOMAIN_ITI = Domain(p=P, q=Q, root=ROOT_ITI, L=0)


def check_leaf_accuracy_nosource_ItI_uniform(
    domain: Domain, test_case: Dict
) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
    d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)
    pde_problem = PDEProblem(
        domain=domain,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        use_ItI=True,
        eta=ETA,
    )

    ##############################################################
    # Solve the local problem
    Y, T, Phi = nosource_local_solve_stage_uniform_2D_ItI(
        pde_problem=pde_problem
    )

    ##############################################################
    # Check the accuracy of the ItI map

    # Assemble incoming impedance data
    q = domain.q
    boundary_g = test_case[K_DIRICHLET](domain.boundary_points)
    boundary_g_normals = jnp.concatenate(
        [
            -1 * test_case[K_DIRICHLET_DUDY](domain.boundary_points[:q]),
            test_case[K_DIRICHLET_DUDX](domain.boundary_points[q : 2 * q]),
            test_case[K_DIRICHLET_DUDY](domain.boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_DIRICHLET_DUDX](domain.boundary_points[3 * q :]),
        ]
    )
    incoming_imp_data = boundary_g_normals + 1j * pde_problem.eta * boundary_g
    expected_outgoing_imp_data = (
        boundary_g_normals - 1j * pde_problem.eta * boundary_g
    )

    # Construct computed outgoing imp data
    computed_outgoing_imp_data = T @ incoming_imp_data
    logging.debug(
        "check_leaf_accuracy_ItI: computed_outgoing_imp_data shape: %s",
        computed_outgoing_imp_data.shape,
    )
    logging.debug(
        "check_leaf_accuracy_ItI: expected_outgoing_imp_data shape: %s",
        expected_outgoing_imp_data.shape,
    )
    assert jnp.allclose(
        computed_outgoing_imp_data,
        expected_outgoing_imp_data,
        atol=ATOL,
        rtol=RTOL,
    )

    ##############################################################
    # Check the accuracy of the homogeneous solution
    # Construct computed homogeneous solution
    computed_homogeneous_soln = Y @ incoming_imp_data
    expected_homogeneous_soln = test_case[K_HOMOG_SOLN](
        domain.interior_points
    ).flatten()
    logging.debug(
        "check_leaf_accuracy_ItI: computed_homogeneous_soln shape: %s",
        computed_homogeneous_soln.shape,
    )
    logging.debug(
        "check_leaf_accuracy_ItI: expected_homogeneous_soln shape: %s",
        expected_homogeneous_soln.shape,
    )
    assert jnp.allclose(
        computed_homogeneous_soln,
        expected_homogeneous_soln,
        atol=ATOL,
        rtol=RTOL,
    )


def check_leaf_accuracy_DtN(
    domain: Domain,
    test_case: Dict,
    use_rectangular_spectral_collocation: bool = False,
) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
    d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)
    source = test_case[K_SOURCE](domain.interior_points).squeeze(2)

    pde_problem = PDEProblem(
        domain=domain,
        source=source,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        use_rectangular_spectral_collocation=use_rectangular_spectral_collocation,
    )

    logging.debug("d_xx_coeffs shape: %s", d_xx_coeffs.shape)
    logging.debug("d_yy_coeffs shape: %s", d_yy_coeffs.shape)
    logging.debug("source shape: %s", source.shape)

    ##############################################################
    # Solve the local problem
    Y, T, v, h = local_solve_stage_uniform_2D_DtN(pde_problem=pde_problem)
    logging.debug("T shape: %s", T.shape)
    logging.debug("Y shape: %s", Y.shape)
    logging.debug("h shape: %s", h.shape)
    logging.debug("v shape: %s", v.shape)
    # logging.debug("v.imag: %s", v.imag)
    T = T[0]
    h = h[0]
    v = v[0]
    Y = Y[0]

    ##############################################################
    # Check the accuracy of the DtN map

    # Assemble incoming impedance data
    q = domain.q
    boundary_g = test_case[K_DIRICHLET](domain.boundary_points)
    boundary_g_normals = jnp.concatenate(
        [
            -1 * test_case[K_DIRICHLET_DUDY](domain.boundary_points[:q]),
            test_case[K_DIRICHLET_DUDX](domain.boundary_points[q : 2 * q]),
            test_case[K_DIRICHLET_DUDY](domain.boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_DIRICHLET_DUDX](domain.boundary_points[3 * q :]),
        ]
    )
    expected_outgoing_data = boundary_g_normals

    # Construct computed outgoing data
    computed_outgoing_data = T @ boundary_g
    logging.debug(
        "check_leaf_accuracy_DtN: computed_outgoing_data shape: %s",
        computed_outgoing_data.shape,
    )
    logging.debug(
        "check_leaf_accuracy_DtN: expected_outgoing_data shape: %s",
        expected_outgoing_data.shape,
    )

    # Plot the expected and computed outgoing data
    # plt.plot(computed_outgoing_data, label="Computed Outgoing Data")
    # plt.plot(expected_outgoing_data, label="Expected Outgoing Data")
    # plt.legend()
    # plt.show()
    # plt.clf()

    assert jnp.allclose(
        computed_outgoing_data, expected_outgoing_data, atol=ATOL, rtol=RTOL
    )

    ##############################################################
    # Check the accuracy of the homogeneous solution
    # Construct computed homogeneous solution
    computed_homogeneous_soln = Y @ boundary_g
    expected_homogeneous_soln = test_case[K_HOMOG_SOLN](
        domain.interior_points
    ).flatten()
    logging.debug(
        "check_leaf_accuracy_DtN: computed_homogeneous_soln shape: %s",
        computed_homogeneous_soln.shape,
    )
    logging.debug(
        "check_leaf_accuracy_DtN: expected_homogeneous_soln shape: %s",
        expected_homogeneous_soln.shape,
    )

    # plot it
    # plot_soln_from_cheby_nodes(
    #     cheby_nodes=domain.interior_points.reshape(-1, 2),
    #     corners=None,
    #     expected_soln=expected_homogeneous_soln.real.flatten(),
    #     computed_soln=computed_homogeneous_soln.real.flatten(),
    #     t="Homogeneous Solution",
    # )

    assert jnp.allclose(
        computed_homogeneous_soln,
        expected_homogeneous_soln,
        atol=ATOL,
        rtol=RTOL,
    )
    ##############################################################
    # Check the accuracy of the particular solution
    # Construct computed particular solution
    computed_part_soln = v
    expected_part_soln = test_case[K_PART_SOLN](
        domain.interior_points
    ).flatten()
    logging.debug(
        "check_leaf_accuracy_DtN: computed_part_soln shape: %s",
        computed_part_soln.shape,
    )
    logging.debug(
        "check_leaf_accuracy_DtN: expected_part_soln shape: %s",
        expected_part_soln.shape,
    )
    assert jnp.allclose(
        computed_part_soln, expected_part_soln, atol=ATOL, rtol=RTOL
    )

    # Plot it
    # plot_soln_from_cheby_nodes(
    #     cheby_nodes=domain.interior_points.reshape(-1, 2),
    #     corners=None,
    #     expected_soln=expected_part_soln.real.flatten(),
    #     computed_soln=computed_part_soln.real.flatten(),
    #     t="Particular Solution",
    # )

    logging.debug("check_leaf_accuracy_DtN: passed part soln check")
    ##############################################################
    # Check the accuracy of the outgoing particular data
    # Construct expected outgoing particular data
    boundary_part_soln_normals = jnp.concatenate(
        [
            -1 * test_case[K_PART_SOLN_DUDY](domain.boundary_points[:q]),
            test_case[K_PART_SOLN_DUDX](domain.boundary_points[q : 2 * q]),
            test_case[K_PART_SOLN_DUDY](domain.boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_PART_SOLN_DUDX](domain.boundary_points[3 * q :]),
        ]
    )
    # Expect part solution = 0 on the boundary so impedance data is just normals
    expected_outgoing_part_data = (
        boundary_part_soln_normals + 0 * 1j
    ).flatten()
    computed_outgoing_part_data = h
    logging.debug(
        "check_leaf_accuracy_DtN: computed_outgoing_part_data shape: %s",
        computed_outgoing_part_data.shape,
    )
    logging.debug(
        "check_leaf_accuracy_DtN: expected_outgoing_part_data shape: %s",
        expected_outgoing_part_data.shape,
    )

    # Plot the expected and computed outgoing data
    # plt.plot(computed_outgoing_part_data.real, label="Computed Outgoing Data")
    # plt.plot(expected_outgoing_part_data.real, label="Expected Outgoing Data")
    # plt.legend()
    # plt.show()
    # plt.clf()
    assert jnp.allclose(
        computed_outgoing_part_data,
        expected_outgoing_part_data,
        atol=ATOL,
        rtol=RTOL,
    )
    d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
    d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)


def check_leaf_accuracy_ItI(domain: Domain, test_case: Dict) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](domain.interior_points)
    d_yy_coeffs = test_case[K_YY_COEFF](domain.interior_points)
    source = test_case[K_SOURCE](domain.interior_points)

    # Put in a second, third, and fourth copy of the data along the 0th axis
    # to simulate the other part of the merge quad so the
    # reshape operation works
    # d_xx_coeffs = jnp.stack(
    #     [d_xx_coeffs, d_xx_coeffs, d_xx_coeffs, d_xx_coeffs], axis=0
    # ).squeeze(1)
    # d_yy_coeffs = jnp.stack(
    #     [d_yy_coeffs, d_yy_coeffs, d_yy_coeffs, d_yy_coeffs], axis=0
    # ).squeeze(1)
    # source = jnp.stack([source, source, source, source], axis=0).squeeze(1)
    logging.debug("source shape: %s", source.shape)

    pde_problem = PDEProblem(
        domain=domain,
        source=source,
        D_xx_coefficients=d_xx_coeffs,
        D_yy_coefficients=d_yy_coeffs,
        use_ItI=True,
        eta=ETA,
    )

    ##############################################################
    # Solve the local problem
    Y, R, v, h = local_solve_stage_uniform_2D_ItI(pde_problem=pde_problem)
    # logging.debug("v.imag: %s", v.imag)
    R = R[0]
    h = h[0]
    v = v[0]
    Y = Y[0]
    logging.debug("R shape: %s", R.shape)
    logging.debug("Y shape: %s", Y.shape)
    logging.debug("h shape: %s", h.shape)
    logging.debug("v shape: %s", v.shape)

    ##############################################################
    # Check the accuracy of the ItI map

    # Assemble incoming impedance data
    q = domain.q
    boundary_g = test_case[K_DIRICHLET](domain.boundary_points)
    boundary_g_normals = jnp.concatenate(
        [
            -1 * test_case[K_DIRICHLET_DUDY](domain.boundary_points[:q]),
            test_case[K_DIRICHLET_DUDX](domain.boundary_points[q : 2 * q]),
            test_case[K_DIRICHLET_DUDY](domain.boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_DIRICHLET_DUDX](domain.boundary_points[3 * q :]),
        ]
    )
    incoming_imp_data = boundary_g_normals + 1j * pde_problem.eta * boundary_g
    expected_outgoing_imp_data = (
        boundary_g_normals - 1j * pde_problem.eta * boundary_g
    )

    # Construct computed outgoing imp data
    computed_outgoing_imp_data = R @ incoming_imp_data
    logging.debug(
        "check_leaf_accuracy_ItI: computed_outgoing_imp_data shape: %s",
        computed_outgoing_imp_data.shape,
    )
    logging.debug(
        "check_leaf_accuracy_ItI: expected_outgoing_imp_data shape: %s",
        expected_outgoing_imp_data.shape,
    )
    assert jnp.allclose(
        computed_outgoing_imp_data,
        expected_outgoing_imp_data,
        atol=ATOL,
        rtol=RTOL,
    )

    ##############################################################
    # Check the accuracy of the homogeneous solution
    # Construct computed homogeneous solution
    computed_homogeneous_soln = Y @ incoming_imp_data
    expected_homogeneous_soln = test_case[K_HOMOG_SOLN](
        domain.interior_points
    ).flatten()
    logging.debug(
        "check_leaf_accuracy_ItI: computed_homogeneous_soln shape: %s",
        computed_homogeneous_soln.shape,
    )
    logging.debug(
        "check_leaf_accuracy_ItI: expected_homogeneous_soln shape: %s",
        expected_homogeneous_soln.shape,
    )
    assert jnp.allclose(
        computed_homogeneous_soln,
        expected_homogeneous_soln,
        atol=ATOL,
        rtol=RTOL,
    )
    ##############################################################
    # Check the accuracy of the particular solution
    # Construct computed particular solution
    computed_part_soln = v
    expected_part_soln = test_case[K_PART_SOLN](domain.interior_points)
    logging.debug(
        "check_leaf_accuracy_ItI: computed_part_soln shape: %s",
        computed_part_soln.shape,
    )
    logging.debug(
        "check_leaf_accuracy_ItI: expected_part_soln shape: %s",
        expected_part_soln.shape,
    )
    assert jnp.allclose(
        computed_part_soln, expected_part_soln, atol=ATOL, rtol=RTOL
    )

    ##############################################################
    # Check the accuracy of the outgoing particular impedance data
    # Construct expected outgoing particular impedance data
    boundary_part_soln_normals = jnp.concatenate(
        [
            -1 * test_case[K_PART_SOLN_DUDY](domain.boundary_points[:q]),
            test_case[K_PART_SOLN_DUDX](domain.boundary_points[q : 2 * q]),
            test_case[K_PART_SOLN_DUDY](domain.boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_PART_SOLN_DUDX](domain.boundary_points[3 * q :]),
        ]
    )
    # Expect part solution = 0 on the boundary so impedance data is just normals
    expected_outgoing_part_imp_data = boundary_part_soln_normals + 0 * 1j
    computed_outgoing_part_imp_data = h
    logging.debug(
        "check_leaf_accuracy_ItI: computed_outgoing_part_imp_data shape: %s",
        computed_outgoing_part_imp_data.shape,
    )
    logging.debug(
        "check_leaf_accuracy_ItI: expected_outgoing_part_imp_data shape: %s",
        expected_outgoing_part_imp_data.shape,
    )
    assert jnp.allclose(
        computed_outgoing_part_imp_data,
        expected_outgoing_part_imp_data,
        atol=ATOL,
        rtol=RTOL,
    )


def check_leaf_accuracy_ItI_Helmholtz_like(
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
    Y, R, v, h = local_solve_stage_uniform_2D_ItI(pde_problem=pde_problem)
    # logging.debug("v.imag: %s", v.imag)
    R = R[0]
    h = h[0]
    v = v[0]
    Y = Y[0]
    logging.debug("R shape: %s", R.shape)
    logging.debug("Y shape: %s", Y.shape)
    logging.debug("h shape: %s", h.shape)
    logging.debug("v shape: %s", v.shape)

    ##############################################################
    # Compute the incoming impedance data

    # Assemble incoming impedance data
    q = domain.q
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
    computed_homogeneous_soln = Y @ incoming_imp_data
    computed_part_soln = v

    computed_soln = computed_homogeneous_soln + computed_part_soln
    logging.debug(
        "check_leaf_accuracy_ItI_Helmholtz_like: computed_soln shape: %s",
        computed_soln.shape,
    )
    expected_soln = test_case[K_SOLN](domain.interior_points).flatten()
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


class Test_accuracy_local_solve_stage_uniform_2D_ItI:
    def test_0(self, caplog) -> None:
        """
        Uses TEST_CASE_POLY_ZERO_SOURCE. Tests the accuracy of the outputs, in this order:
        1. ItI map.
        2. homogeneous solution.
        3. particular solution.
        4. outgoing particular impedance data
        """
        caplog.set_level(logging.DEBUG)
        ##############################################################
        # Set up the test case
        test_case = TEST_CASE_POLY_ZERO_SOURCE
        domain = DOMAIN_ITI

        check_leaf_accuracy_ItI(domain, test_case)
        jax.clear_caches()

    def test_1(self, caplog) -> None:
        """NOTE this one is a little brittle and will likely fail if gobal parameters P and Q are reduced.

        Unlike the other tests in this script, this one is testing accuracy on a non-polynomial problem
        so we need a larger tolerance, which is a global parameter called ATOL_NONPOLY
        """
        caplog.set_level(logging.DEBUG)
        test_case = TEST_CASE_HELMHOLTZ_ITI
        domain = DOMAIN_ITI
        check_leaf_accuracy_ItI_Helmholtz_like(domain, test_case)
        jax.clear_caches()

    def test_2(self, caplog) -> None:
        """Uses TEST_CASE_HELMHOLTZ_ITI_COMPLEX_COEFFS. Tests the accuracy when the coefficients are complex-valued."""
        caplog.set_level(logging.DEBUG)
        test_case = TEST_CASE_HELMHOLTZ_ITI_COMPLEX_COEFFS
        domain = DOMAIN_ITI
        check_leaf_accuracy_ItI_Helmholtz_like(domain, test_case)
        jax.clear_caches()


class Test_accuracy_local_solve_stage_uniform_2D_DtN:
    def test_0(self, caplog) -> None:
        """
        Uses TEST_CASE_POLY_ZERO_SOURCE. Tests the accuracy of the outputs, in this order:
        1. DtN map.
        2. homogeneous solution.
        3. particular solution.
        4. outgoing particular data
        """
        caplog.set_level(logging.DEBUG)
        ##############################################################
        # Set up the test case
        test_case = TEST_CASE_POLY_ZERO_SOURCE
        domain = DOMAIN_DTN

        check_leaf_accuracy_DtN(domain, test_case)

    def test_1(self, caplog) -> None:
        """
        Uses TEST_CASE_POLY_PART_HOMOG. Tests the accuracy of the outputs, in this order:
        1. DtN map.
        2. homogeneous solution.
        3. particular solution.
        4. outgoing particular data
        """
        caplog.set_level(logging.DEBUG)
        ##############################################################
        # Set up the test case
        test_case = TEST_CASE_POLY_PART_HOMOG
        domain = DOMAIN_DTN

        check_leaf_accuracy_DtN(domain, test_case)

        jax.clear_caches()

    def test_2(self, caplog) -> None:
        """
        Uses TEST_CASE_POLY_ZERO_SOURCE with rectangular spectral collocation.
        """
        caplog.set_level(logging.DEBUG)
        ##############################################################
        # Set up the test case
        test_case = TEST_CASE_POLY_ZERO_SOURCE
        domain = DOMAIN_DTN

        check_leaf_accuracy_DtN(
            domain, test_case, use_rectangular_spectral_collocation=True
        )

    def test_3(self, caplog) -> None:
        """
        Uses TEST_CASE_POLY_PART_HOMOG with rectangular spectral collocation.
        """
        caplog.set_level(logging.DEBUG)
        ##############################################################
        # Set up the test case
        test_case = TEST_CASE_POLY_PART_HOMOG
        domain = DOMAIN_DTN

        check_leaf_accuracy_DtN(
            domain, test_case, use_rectangular_spectral_collocation=True
        )

        jax.clear_caches()


class Test_accuracy_nosource_local_solve_stage_uniform_2D_ItI:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        test_case = TEST_CASE_POLY_ZERO_SOURCE
        domain = DOMAIN_ITI
        check_leaf_accuracy_nosource_ItI_uniform(domain, test_case)
        jax.clear_caches()


class Test_accuracy_local_solutions_2D_DtN_uniform:
    @pytest.mark.skip("Not implemented yet")
    def test_0(self, caplog) -> None:
        """
        Uses TEST_CASE_POLY_ZERO_SOURCE. Tests the accuracy of the outputs.
        """
        caplog.set_level(logging.DEBUG)
        ##############################################################
        # Set up the test case
        test_case = TEST_CASE_POLY_ZERO_SOURCE  # noqa: F841
        domain = DOMAIN_DTN  # noqa: F841

        assert False
        # check_solution_accuracy_DtN(domain, test_case)

    @pytest.mark.skip("Not implemented yet")
    def test_1(self, caplog) -> None:
        """
        Uses TEST_CASE_POLY_PART_HOMOG. Tests the accuracy of the outputs.
        """
        caplog.set_level(logging.DEBUG)
        ##############################################################
        # Set up the test case
        test_case = TEST_CASE_POLY_PART_HOMOG  # noqa: F841
        domain = DOMAIN_DTN  # noqa: F841

        assert False
        # check_solution_accuracy_DtN(domain, test_case)
        jax.clear_caches()


if __name__ == "__main__":
    pytest.main()
