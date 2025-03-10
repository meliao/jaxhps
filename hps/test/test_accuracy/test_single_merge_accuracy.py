# from __future__ import annotations
import logging
from typing import Tuple, Callable, Dict
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import pytest

from hps.src.solver_obj import (
    SolverObj,
    create_solver_obj_2D,
    get_bdry_data_evals_lst_2D,
)
from hps.src.quadrature.trees import Node
from hps.src.up_down_passes import local_solve_stage, build_stage, down_pass
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
from hps.accuracy_checks.utils import plot_soln_from_cheby_nodes


ATOL = 1e-12
RTOL = 0.0

SOLVER_LOCAL_SOLVE_DTN = create_solver_obj_2D(
    p=6, q=4, root=ROOT, uniform_levels=1, fill_tree=False
)

SOLVER_LOCAL_SOLVE_ITI = create_solver_obj_2D(
    p=6, q=4, root=ROOT, uniform_levels=1, use_ItI=True, eta=1.0
)


def check_merge_accuracy_2D_DtN_uniform(solver: SolverObj, test_case: Dict) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](solver.leaf_cheby_points)
    d_yy_coeffs = test_case[K_YY_COEFF](solver.leaf_cheby_points)
    source_term = test_case[K_SOURCE](solver.leaf_cheby_points).squeeze()
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: d_xx_coeffs = %s", d_xx_coeffs.shape
    )
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: d_yy_coeffs = %s", d_yy_coeffs.shape
    )
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: source_term = %s", source_term.shape
    )

    # Do the local solve and build stages
    local_solve_stage(
        solver,
        source_term=source_term,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
    )
    build_stage(solver)

    # Do the down pass
    bdry_data_lst = get_bdry_data_evals_lst_2D(solver, test_case[K_DIRICHLET])
    down_pass(solver, bdry_data_lst)

    ##########################################################
    # Check the accuracy of the DtN matrix
    # TODO: Add an option to save the DtN matrix.

    ##########################################################
    # Check the accuracy of the computed solution
    homog_soln = test_case[K_HOMOG_SOLN](solver.leaf_cheby_points)
    part_soln = test_case[K_PART_SOLN](solver.leaf_cheby_points).reshape(
        homog_soln.shape
    )

    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: part_soln = %s", part_soln.shape
    )
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: homog_soln = %s", homog_soln.shape
    )
    expected_soln = part_soln + homog_soln
    computed_soln = solver.interior_solns
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: computed_soln = %s", computed_soln.shape
    )
    logging.debug(
        "check_merge_accuracy_2D_DtN_uniform: expected_soln = %s", expected_soln.shape
    )
    assert jnp.allclose(computed_soln, expected_soln, atol=ATOL, rtol=RTOL)


def check_merge_accuracy_2D_ItI_uniform(solver: SolverObj, test_case: Dict) -> None:
    d_xx_coeffs = test_case[K_XX_COEFF](solver.leaf_cheby_points)
    d_yy_coeffs = test_case[K_YY_COEFF](solver.leaf_cheby_points)
    source_term = test_case[K_SOURCE](solver.leaf_cheby_points).squeeze()
    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: d_xx_coeffs = %s", d_xx_coeffs.shape
    )
    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: d_yy_coeffs = %s", d_yy_coeffs.shape
    )
    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: source_term = %s", source_term.shape
    )
    # Do the local solve and build stages
    local_solve_stage(
        solver,
        source_term=source_term,
        D_xx_coeffs=d_xx_coeffs,
        D_yy_coeffs=d_yy_coeffs,
    )
    build_stage(solver)

    # Do the down pass

    # Compute incoming impedance data
    n_per_side = solver.root_boundary_points.shape[0] // 4
    boundary_g_normals = jnp.concatenate(
        [
            -1 * test_case[K_DIRICHLET_DUDY](solver.root_boundary_points[:n_per_side]),
            test_case[K_DIRICHLET_DUDX](
                solver.root_boundary_points[n_per_side : 2 * n_per_side]
            ),
            test_case[K_DIRICHLET_DUDY](
                solver.root_boundary_points[2 * n_per_side : 3 * n_per_side]
            ),
            -1
            * test_case[K_DIRICHLET_DUDX](
                solver.root_boundary_points[3 * n_per_side :]
            ),
        ]
    )
    boundary_g_evals = test_case[K_DIRICHLET](solver.root_boundary_points)
    incoming_imp_data = boundary_g_normals + 1j * solver.eta * boundary_g_evals
    bdry_data_lst = [
        incoming_imp_data[:n_per_side],
        incoming_imp_data[n_per_side : 2 * n_per_side],
        incoming_imp_data[2 * n_per_side : 3 * n_per_side],
        incoming_imp_data[3 * n_per_side :],
    ]
    down_pass(solver, bdry_data_lst)
    ##########################################################
    # Check the accuracy of the DtN matrix
    # TODO: Add an option to save the DtN matrix.

    ##########################################################
    # Check the accuracy of the computed solution
    homog_soln = jnp.expand_dims(test_case[K_HOMOG_SOLN](solver.leaf_cheby_points), -1)
    part_soln = test_case[K_PART_SOLN](solver.leaf_cheby_points)

    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: part_soln = %s", part_soln.shape
    )
    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: homog_soln = %s", homog_soln.shape
    )
    expected_soln = part_soln + homog_soln
    computed_soln = solver.interior_solns

    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: computed_soln = %s", computed_soln.shape
    )
    logging.debug(
        "check_merge_accuracy_2D_ItI_uniform: expected_soln = %s", expected_soln.shape
    )
    assert jnp.allclose(computed_soln, expected_soln, atol=ATOL, rtol=RTOL)


class Test_accuracy_single_merge_2D_DtN_uniform:
    def test_0(self, caplog) -> None:
        """Polynomial data with zero source term."""
        caplog.set_level(logging.DEBUG)
        check_merge_accuracy_2D_DtN_uniform(
            SOLVER_LOCAL_SOLVE_DTN, TEST_CASE_POLY_ZERO_SOURCE
        )

    def test_1(self, caplog) -> None:
        """Polynomial data with non-zero source term."""
        caplog.set_level(logging.DEBUG)
        check_merge_accuracy_2D_DtN_uniform(
            SOLVER_LOCAL_SOLVE_DTN, TEST_CASE_POLY_PART_HOMOG
        )


class Test_accuracy_single_merge_2D_ItI_uniform:
    def test_0(self, caplog) -> None:
        """Polynomial data with zero source term."""
        caplog.set_level(logging.DEBUG)

        check_merge_accuracy_2D_ItI_uniform(
            SOLVER_LOCAL_SOLVE_ITI, TEST_CASE_POLY_ZERO_SOURCE
        )


if __name__ == "__main__":
    pytest.main()
