import jax.numpy as jnp
import jax

####################################################################
# Define the keys that will be used in the test cases
K_DIRICHLET = "dirichlet_data_fn"
K_XX_COEFF = "d_xx_coeff_fn"
K_YY_COEFF = "d_yy_coeff_fn"
K_ZZ_COEFF = "d_zz_coeff_fn"
K_I_COEFF = "i_coeff_fn"
K_SOURCE = "source_fn"
K_DIRICHLET_DUDX = "d_dx_dirichlet_data_fn"
K_DIRICHLET_DUDY = "d_dy_dirichlet_data_fn"
K_DIRICHLET_DUDZ = "d_dz_dirichlet_data_fn"
K_PART_SOLN = "part_soln_fn"
K_HOMOG_SOLN = "homog_soln_fn"
K_SOLN = "soln_fn"
K_PART_SOLN_DUDX = "d_dx_part_soln_fn"
K_PART_SOLN_DUDY = "d_dy_part_soln_fn"
K_PART_SOLN_DUDZ = "d_dz_part_soln_fn"
K_DUDX = "d_dx_soln_fn"
K_DUDY = "d_dy_soln_fn"


######################################################################
# Define the domains
XMIN = -jnp.pi / 2
XMAX = jnp.pi / 2
YMIN = -jnp.pi / 2
YMAX = jnp.pi / 2
ETA = 1.0


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


# ######################################################################
# # Test Case 2: Non-polynomial Source Function for ItI Problems
# # \Delta u + q(x) u(x) = f     in \Omega
# # u_n + i u = 0                in \partial \Omega
# #
# #
# # q(x,y) = 1 + e^{- \lambda_1 ||x||^2}
# # f(x,y) = - pi^2 (  e^{i \pi x} + e^{i \pi y} )
# # g(x,y) = defined piecewise
# # Can't separate the solution into homogeneous and particular parts, but here
# # is the solution:
# # u(x,y) = e^{i  \pi x} + e^{i  \pi y}


def q(x):
    # q(x,y) = 1 + e^{- \lambda_1 ||x||^2}
    return jnp.ones_like(x[..., 0]) + jnp.exp(
        -jnp.square(x[..., 0]) - jnp.square(x[..., 1])
    )


def f(x):
    # f(x,y) = - pi^2 u(x) + q(x) u(x)
    return -(jnp.pi**2) * u(x) + q(x) * u(x)


def u(x):
    # u(x,y) = e^{i  \pi x} + e^{i  \pi y}
    return jnp.exp(1j * jnp.pi * x[..., 0]) + jnp.exp(1j * jnp.pi * x[..., 1])


def g(x: jnp.array) -> jnp.array:
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
            -1j * jnp.pi * jnp.exp(1j * jnp.pi * south[..., 1]),
            1j * jnp.pi * jnp.exp(1j * jnp.pi * east[..., 0]),
            1j * jnp.pi * jnp.exp(1j * jnp.pi * north[..., 1]),
            -1j * jnp.pi * jnp.exp(1j * jnp.pi * west[..., 0]),
        ]
    )
    return dudn + 1j * ETA * u(x)


def dudx(x: jax.Array) -> jax.Array:
    return 1j * jnp.pi * jnp.exp(1j * jnp.pi * x[..., 0])


def dudy(x: jax.Array) -> jax.Array:
    return 1j * jnp.pi * jnp.exp(1j * jnp.pi * x[..., 1])


TEST_CASE_HELMHOLTZ_ITI = {
    K_DIRICHLET: g,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_I_COEFF: q,
    K_SOURCE: f,
    K_SOLN: u,
    K_DUDX: dudx,
    K_DUDY: dudy,
}


# ######################################################################
# # Test Case 3: Polynomial Source Function for ItI Problems
# # \Delta u + (k^2 + i \gamma) u(x) = f     in \Omega
# # u_n + i u = g               in \partial \Omega
# #
# #
# # u(x,y) = x^3 + 3 y^2
# # f(x,y) = 6x + 6 + (k^2 + i \gamma) u(x,y)
# # g(x,y) = defined piecewise


# # u(x,y) = x^3 + 3 y^2
# # u_x(x,y) = 3x**2
# # u_xx(x,y) = 6x
# # u_y(x,y) = 6y
# # u_yy(x,y) = 6
# # Note that this solution is entirely the particular solution.


# This is (k^2 + i \gamma)
# k = 1.0; gamma = 0.1
COMPLEX_COEFF = 1j * 0.1 + 1.0


def u_complex_coeffs(x: jax.Array) -> jax.Array:
    # u(x,y) = x^3 + 3 y^2
    return jnp.power(x[..., 0], 3) + 3 * jnp.square(x[..., 1])


def i_complex_coeffs(x: jax.Array) -> jax.Array:
    # q(x,y) = k^2 + i \gamma
    return COMPLEX_COEFF * jnp.ones_like(x[..., 0])


def source_complex_coeffs(x: jax.Array) -> jax.Array:
    # f(x,y) = 6x + 6 + (k^2 + i \gamma) u(x,y)
    return 6 * x[..., 0] + 6 + COMPLEX_COEFF * u_complex_coeffs(x)


def dudx_complex_coeffs(x: jax.Array) -> jax.Array:
    # du/dx = 3x^2
    return 3 * jnp.square(x[..., 0])


def dudy_complex_coeffs(x: jax.Array) -> jax.Array:
    # du/dy = 6y
    return 6 * x[..., 1]


TEST_CASE_HELMHOLTZ_ITI_COMPLEX_COEFFS = {
    K_DIRICHLET: u_complex_coeffs,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_I_COEFF: i_complex_coeffs,
    K_SOURCE: source_complex_coeffs,
    K_SOLN: u_complex_coeffs,
    K_DUDX: dudx_complex_coeffs,
    K_DUDY: dudy_complex_coeffs,
}


# ######################################################################
# # Test Case 4: ItI problem with particular and homogeneous solutions.
# # \Delta u(x) = f     in \Omega
# # u_n + i u   = g     in \partial \Omega
# #
# #
# # u(x,y) = v(x,y) + w(x,y)
# # v(x,y) = e^{\\kappa(x^2 + y^2)}
# # w(x,y) = x^2 - y^2
# # f(x,y) = (4 \\kappa^2 x^2 + 4 \\kappa^2 y^2 + 2 \\kappa) v(x,y)
# # g(x,y) = defined piecewise
# # Note that v is the particular solution and w is the homogeneous solution.


KAPPA = -1j * ETA / jnp.pi


def u_4(x: jax.Array) -> jax.Array:
    # u(x,y) = v(x,y) + w(x,y)
    return v_4(x) + w_4(x)


def v_4(x: jax.Array) -> jax.Array:
    # v(x,y) = e^{\\kappa(x^2 + y^2)}
    return jnp.exp(KAPPA * (jnp.square(x[..., 0]) + jnp.square(x[..., 1])))


def dvdx_4(x: jax.Array) -> jax.Array:
    # dv/dx = 2 \\kappa x e^{\\kappa(x^2 + y^2)}
    return 2 * KAPPA * x[..., 0] * v_4(x)


def dvdy_4(x: jax.Array) -> jax.Array:
    # dv/dy = 2 \\kappa y e^{\\kappa(x^2 + y^2)}
    return 2 * KAPPA * x[..., 1] * v_4(x)


def w_4(x: jax.Array) -> jax.Array:
    # w(x,y) = x^2 - y^2
    return jnp.square(x[..., 0]) - jnp.square(x[..., 1])


def dwdx_4(x: jax.Array) -> jax.Array:
    # dw/dx = 2x
    return 2 * x[..., 0]


def dwdy_4(x: jax.Array) -> jax.Array:
    # dw/dy = -2y
    return -2 * x[..., 1]


def source_4(x: jax.Array) -> jax.Array:
    # f(x,y) = (4 \\kappa^2 x^2 + 4 \\kappa^2 y^2 + 2 \\kappa) v(x,y)
    coeff = (
        4 * KAPPA**2 * (jnp.square(x[..., 0]) + jnp.square(x[..., 1]))
        + 4 * KAPPA
    )
    return coeff * v_4(x)


def dudx_4(x: jax.Array) -> jax.Array:
    return dvdx_4(x) + dwdx_4(x)


def dudy_4(x: jax.Array) -> jax.Array:
    return dvdy_4(x) + dwdy_4(x)


TEST_CASE_ITI_PART_HOMOG = {
    K_DIRICHLET: u_complex_coeffs,
    K_XX_COEFF: default_lap_coeffs,
    K_YY_COEFF: default_lap_coeffs,
    K_I_COEFF: default_zero_source,
    K_SOURCE: source_4,
    K_SOLN: u_4,
    K_DUDX: dudx_4,
    K_DUDY: dudy_4,
    K_PART_SOLN_DUDX: dvdx_4,
    K_PART_SOLN_DUDY: dvdy_4,
    K_PART_SOLN: v_4,
}
