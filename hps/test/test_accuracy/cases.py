import logging
from typing import Tuple, Callable
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

from hps.src.methods.local_solve_stage import _local_solve_stage_2D
from hps.src.solver_obj import SolverObj, create_solver_obj_2D
from hps.src.quadrature.trees import Node


####################################################################
# Define the keys that will be used in the test cases
K_DIRICHLET = "dirichlet_data_fn"
K_XX_COEFF = "d_xx_coeff_fn"
K_YY_COEFF = "d_yy_coeff_fn"
K_ZZ_COEFF = "d_zz_coeff_fn"
K_SOURCE = "source_fn"
K_DIRICHLET_DUDX = "d_dx_dirichlet_data_fn"
K_DIRICHLET_DUDY = "d_dy_dirichlet_data_fn"
K_DIRICHLET_DUDZ = "d_dz_dirichlet_data_fn"
K_PART_SOLN = "part_soln_fn"
K_HOMOG_SOLN = "homog_soln_fn"
K_PART_SOLN_DUDX = "d_dx_part_soln_fn"
K_PART_SOLN_DUDY = "d_dy_part_soln_fn"
K_PART_SOLN_DUDZ = "d_dz_part_soln_fn"


######################################################################
# Define the root node that all test cases will use
ROOT_DTN = Node(xmin=-jnp.pi / 2, xmax=jnp.pi / 2, ymin=-jnp.pi / 2, ymax=jnp.pi / 2)
ROOT_ITI = Node(xmin=-jnp.pi / 2, xmax=jnp.pi / 2, ymin=-jnp.pi / 2, ymax=jnp.pi / 2)


######################################################################
# Define coefficient and source functions that will be re-used
def default_lap_coeffs(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.ones_like(x[..., 0])


def default_zero_source(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(x[..., 0])


######################################################################
# Test Case 1: Polynomial Data with One Non-Zero Source Function
# \Delta u = f  in \Omega
# u = g         in \partial \Omega
#
#
#
# f(x,y) = 6y(x^2 - (pi/2)^2) + 2y(y^2 - (pi/2)^2)
# g(x,y) = x^2 - y^2
# Homogeneous solution:
# w(x,y) = x^2 - y^2
# Particular solution:
# v(x,y) = y(x^2 - (pi/2)^2)(y^2 - (pi/2)^2)
# NOTE This does not work for ItI version because the ItI version
# needs a particular solution that has Robin data on the boundary.


def polynomial_dirichlet_data_0(x):
    # Assumes x has shape (n, 2).
    # f(x, y) = x^2 - y^2
    return jnp.square(x[..., 0]) - jnp.square(x[..., 1])


def dudx_polynomial_dirichlet_data_0(x):
    # Assumes x has shape (n, 2).
    # df/dx = 2x
    return 2 * x[..., 0]


def dudy_polynomial_dirichlet_data_0(x):
    # Assumes x has shape (n, 2).
    # df/dy = -2y
    return -2 * x[..., 1]


def part_soln_fn(x: jnp.ndarray) -> jnp.ndarray:
    # v(x,y) = y(x^2 - (pi/2)^2)(y^2 - (pi/2)^2)

    term_1 = x[..., 1] * (jnp.square(x[..., 1]) - (jnp.pi / 2) ** 2)
    term_2 = jnp.square(x[..., 0]) - (jnp.pi / 2) ** 2
    return jnp.expand_dims((term_1 * term_2).flatten(), -1)


def source_fn(x: jnp.ndarray) -> jnp.ndarray:
    # f(x, y) = 6y(x^2 - (pi/2)^2) + 2y(y^2 - (pi/2)^2)
    term_1 = 2 * x[..., 1] * (jnp.square(x[..., 1]) - (jnp.pi / 2) ** 2)
    term_2 = 6 * x[..., 1] * (jnp.square(x[..., 0]) - (jnp.pi / 2) ** 2)
    return jnp.expand_dims(term_1 + term_2, -1)


def d_dx_part_soln_fn(x: jnp.ndarray) -> jnp.ndarray:
    # dv/dx = 2xy(y^2 - (pi/2)^2)
    term_1 = jnp.square(x[..., 1]) - (jnp.pi / 2) ** 2
    return jnp.expand_dims(2 * x[..., 0] * x[..., 1] * term_1, 1)


def d_dy_part_soln_fn(x: jnp.ndarray) -> jnp.ndarray:
    # dv/dy = (3y^2 - (pi/2)^2)(x^2 - (pi/2)^2)
    term_1 = jnp.square(x[..., 0]) - (jnp.pi / 2) ** 2
    term_2 = 3 * jnp.square(x[..., 1]) - (jnp.pi / 2) ** 2
    return jnp.expand_dims(term_1 * term_2, 1)


TEST_CASE_POLY_PART_HOMOG = {
    K_DIRICHLET: polynomial_dirichlet_data_0,
    K_DIRICHLET_DUDX: dudx_polynomial_dirichlet_data_0,
    K_DIRICHLET_DUDY: dudy_polynomial_dirichlet_data_0,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_HOMOG_SOLN: polynomial_dirichlet_data_0,
    K_SOURCE: source_fn,
    K_PART_SOLN: part_soln_fn,
    K_PART_SOLN_DUDX: d_dx_part_soln_fn,
    K_PART_SOLN_DUDY: d_dy_part_soln_fn,
}


TEST_CASE_POLY_ZERO_SOURCE = {
    K_DIRICHLET: polynomial_dirichlet_data_0,
    K_DIRICHLET_DUDX: dudx_polynomial_dirichlet_data_0,
    K_DIRICHLET_DUDY: dudy_polynomial_dirichlet_data_0,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_HOMOG_SOLN: polynomial_dirichlet_data_0,
    K_SOURCE: lambda x: jnp.expand_dims(jnp.zeros_like(x[..., 0]), -1),
    K_PART_SOLN: lambda x: jnp.expand_dims(jnp.zeros_like(x[..., 0]), -1),
    K_PART_SOLN_DUDX: lambda x: jnp.expand_dims(jnp.zeros_like(x[..., 0]), -1),
    K_PART_SOLN_DUDY: lambda x: jnp.expand_dims(jnp.zeros_like(x[..., 0]), -1),
}
