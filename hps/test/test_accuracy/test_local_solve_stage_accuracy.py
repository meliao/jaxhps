import logging
from typing import Tuple, Callable, Dict
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import pytest

from hps.src.solver_obj import SolverObj, create_solver_obj_2D
from hps.src.quadrature.trees import Node
from hps.src.methods.local_solve_stage import (
    _local_solve_stage_2D,
    _local_solve_stage_2D_ItI,
)
from hps.test.test_accuracy.cases import (
    ROOT,
    TEST_CASE_POLY_PART_HOMOG,
    TEST_CASE_POLY_ZERO_SOURCE,
    K_DIRICHLET,
    K_XX_COEFF,
    K_YY_COEFF,
    K_SOURCE,
    K_DIRICHLET_DUDX,
    K_DIRICHLET_DUDY,
    K_PART_SOLN,
    K_PART_SOLN_DUDX,
    K_PART_SOLN_DUDY,
    K_HOMOG_SOLN,
)


ATOL = 1e-12
RTOL = 0.0

SOLVER_LOCAL_SOLVE_DTN = create_solver_obj_2D(p=6, q=4, root=ROOT)

SOLVER_LOCAL_SOLVE_ITI = create_solver_obj_2D(
    p=6, q=4, root=ROOT, use_ItI=True, eta=1.0
)


def check_leaf_accuracy_DtN(solver: SolverObj, test_case: Dict) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](solver.leaf_cheby_points)
    d_yy_coeffs = test_case[K_YY_COEFF](solver.leaf_cheby_points)
    source = test_case[K_SOURCE](solver.leaf_cheby_points).squeeze(2)

    logging.debug("d_xx_coeffs shape: %s", d_xx_coeffs.shape)
    logging.debug("d_yy_coeffs shape: %s", d_yy_coeffs.shape)
    logging.debug("source shape: %s", source.shape)

    ##############################################################
    # Solve the local problem
    Y, T, v, h = _local_solve_stage_2D(
        D_xx=solver.D_xx,
        D_yy=solver.D_yy,
        D_xy=solver.D_xy,
        D_x=solver.D_x,
        D_y=solver.D_y,
        P=solver.P,
        Q_D=solver.Q_D,
        sidelens=solver.sidelens,
        p=solver.p,
        source_term=source,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
        uniform_grid=True,
    )
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
    q = solver.q
    boundary_g = test_case[K_DIRICHLET](solver.root_boundary_points)
    boundary_g_normals = jnp.concatenate(
        [
            -1 * test_case[K_DIRICHLET_DUDY](solver.root_boundary_points[:q]),
            test_case[K_DIRICHLET_DUDX](solver.root_boundary_points[q : 2 * q]),
            test_case[K_DIRICHLET_DUDY](solver.root_boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_DIRICHLET_DUDX](solver.root_boundary_points[3 * q :]),
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
    assert jnp.allclose(
        computed_outgoing_data, expected_outgoing_data, atol=ATOL, rtol=RTOL
    )

    ##############################################################
    # Check the accuracy of the homogeneous solution
    # Construct computed homogeneous solution
    computed_homogeneous_soln = Y @ boundary_g
    expected_homogeneous_soln = test_case[K_HOMOG_SOLN](
        solver.leaf_cheby_points
    ).flatten()
    logging.debug(
        "check_leaf_accuracy_DtN: computed_homogeneous_soln shape: %s",
        computed_homogeneous_soln.shape,
    )
    logging.debug(
        "check_leaf_accuracy_DtN: expected_homogeneous_soln shape: %s",
        expected_homogeneous_soln.shape,
    )
    assert jnp.allclose(
        computed_homogeneous_soln, expected_homogeneous_soln, atol=ATOL, rtol=RTOL
    )
    ##############################################################
    # Check the accuracy of the particular solution
    # Construct computed particular solution
    computed_part_soln = v
    expected_part_soln = test_case[K_PART_SOLN](solver.leaf_cheby_points).flatten()
    logging.debug(
        "check_leaf_accuracy_DtN: computed_part_soln shape: %s",
        computed_part_soln.shape,
    )
    logging.debug(
        "check_leaf_accuracy_DtN: expected_part_soln shape: %s",
        expected_part_soln.shape,
    )
    assert jnp.allclose(computed_part_soln, expected_part_soln, atol=ATOL, rtol=RTOL)

    ##############################################################
    # Check the accuracy of the outgoing particular data
    # Construct expected outgoing particular data
    boundary_part_soln_normals = jnp.concatenate(
        [
            -1 * test_case[K_PART_SOLN_DUDY](solver.root_boundary_points[:q]),
            test_case[K_PART_SOLN_DUDX](solver.root_boundary_points[q : 2 * q]),
            test_case[K_PART_SOLN_DUDY](solver.root_boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_PART_SOLN_DUDX](solver.root_boundary_points[3 * q :]),
        ]
    )
    # Expect part solution = 0 on the boundary so impedance data is just normals
    expected_outgoing_part_data = (boundary_part_soln_normals + 0 * 1j).flatten()
    computed_outgoing_part_data = h
    logging.debug(
        "check_leaf_accuracy_DtN: computed_outgoing_part_data shape: %s",
        computed_outgoing_part_data.shape,
    )
    logging.debug(
        "check_leaf_accuracy_DtN: expected_outgoing_part_data shape: %s",
        expected_outgoing_part_data.shape,
    )
    assert jnp.allclose(
        computed_outgoing_part_data,
        expected_outgoing_part_data,
        atol=ATOL,
        rtol=RTOL,
    )


def check_leaf_accuracy_ItI(solver: SolverObj, test_case: Dict) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](solver.leaf_cheby_points)
    d_yy_coeffs = test_case[K_YY_COEFF](solver.leaf_cheby_points)
    source = test_case[K_SOURCE](solver.leaf_cheby_points)

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
    logging.debug("source shape: %s", source.shape)

    ##############################################################
    # Solve the local problem
    R, Y, h, v = _local_solve_stage_2D_ItI(
        D_xx=solver.D_xx,
        D_yy=solver.D_yy,
        D_xy=solver.D_xy,
        D_x=solver.D_x,
        D_y=solver.D_y,
        I_P_0=solver.I_P_0,
        Q_I=solver.Q_I,
        F=solver.F,
        G=solver.G,
        p=solver.p,
        source_term=source,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
    )
    # logging.debug("v.imag: %s", v.imag)
    R = R.squeeze()[1]
    h = h.squeeze(0)[1]
    v = v[0]
    Y = Y[0]

    ##############################################################
    # Check the accuracy of the ItI map

    # Assemble incoming impedance data
    q = solver.q
    boundary_g = test_case[K_DIRICHLET](solver.root_boundary_points)
    boundary_g_normals = jnp.concatenate(
        [
            -1 * test_case[K_DIRICHLET_DUDY](solver.root_boundary_points[:q]),
            test_case[K_DIRICHLET_DUDX](solver.root_boundary_points[q : 2 * q]),
            test_case[K_DIRICHLET_DUDY](solver.root_boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_DIRICHLET_DUDX](solver.root_boundary_points[3 * q :]),
        ]
    )
    incoming_imp_data = boundary_g_normals + 1j * solver.eta * boundary_g
    expected_outgoing_imp_data = boundary_g_normals - 1j * solver.eta * boundary_g

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
        computed_outgoing_imp_data, expected_outgoing_imp_data, atol=ATOL, rtol=RTOL
    )

    ##############################################################
    # Check the accuracy of the homogeneous solution
    # Construct computed homogeneous solution
    computed_homogeneous_soln = Y @ incoming_imp_data
    expected_homogeneous_soln = test_case[K_HOMOG_SOLN](
        solver.leaf_cheby_points
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
        computed_homogeneous_soln, expected_homogeneous_soln, atol=ATOL, rtol=RTOL
    )
    ##############################################################
    # Check the accuracy of the particular solution
    # Construct computed particular solution
    computed_part_soln = v
    expected_part_soln = test_case[K_PART_SOLN](solver.leaf_cheby_points)
    logging.debug(
        "check_leaf_accuracy_ItI: computed_part_soln shape: %s",
        computed_part_soln.shape,
    )
    logging.debug(
        "check_leaf_accuracy_ItI: expected_part_soln shape: %s",
        expected_part_soln.shape,
    )
    assert jnp.allclose(computed_part_soln, expected_part_soln, atol=ATOL, rtol=RTOL)

    ##############################################################
    # Check the accuracy of the outgoing particular impedance data
    # Construct expected outgoing particular impedance data
    boundary_part_soln_normals = jnp.concatenate(
        [
            -1 * test_case[K_PART_SOLN_DUDY](solver.root_boundary_points[:q]),
            test_case[K_PART_SOLN_DUDX](solver.root_boundary_points[q : 2 * q]),
            test_case[K_PART_SOLN_DUDY](solver.root_boundary_points[2 * q : 3 * q]),
            -1 * test_case[K_PART_SOLN_DUDX](solver.root_boundary_points[3 * q :]),
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


class Test_accuracy_local_solve_stage_2D_ItI:
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
        solver = SOLVER_LOCAL_SOLVE_ITI

        check_leaf_accuracy_ItI(solver, test_case)


class Test_accuracy_local_solve_stage_2D_DtN:
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
        solver = SOLVER_LOCAL_SOLVE_DTN

        check_leaf_accuracy_DtN(solver, test_case)

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
        solver = SOLVER_LOCAL_SOLVE_DTN

        check_leaf_accuracy_DtN(solver, test_case)


if __name__ == "__main__":
    pytest.main()
