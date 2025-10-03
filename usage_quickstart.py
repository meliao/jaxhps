import jax.numpy as jnp
import jaxhps
import matplotlib.pyplot as plt

root = jaxhps.DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

domain = jaxhps.Domain(
    p=16,  # polynomial degree of leaf Chebyshev points
    q=14,  # polynomial degree of boundary Gauss-Legendre points
    root=root,  # root of the domain tree
    L=3,  # number of levels in the domain tree
)

# It's helpful to use the Domain's quadrature points
source_term = jnp.zeros_like(domain.interior_points[..., 0])
D_xx_coeffs = jnp.ones_like(domain.interior_points[..., 0])
D_yy_coeffs = jnp.ones_like(domain.interior_points[..., 0])

# Create the PDEProblem instance
pde_problem = jaxhps.PDEProblem(
    domain=domain,  # the domain we constructed above
    source=source_term,
    D_xx_coefficients=D_xx_coeffs,
    D_yy_coefficients=D_yy_coeffs,
)

jaxhps.build_solver(pde_problem=pde_problem)

boundary_data = (
    domain.boundary_points[..., 0] ** 2 - domain.boundary_points[..., 1] ** 2
)

# Apply the boundary data to the solver
solution = jaxhps.solve(pde_problem=pde_problem, boundary_data=boundary_data)


# Interpolate the solution onto a regular grid for plotting.
n_pixels = 100
x_pts = jnp.linspace(root.xmin, root.xmax, n_pixels)
y_pts = jnp.linspace(root.ymin, root.ymax, n_pixels)

solution_pixels, pixel_locations = domain.interp_from_interior_points(
    solution, x_pts, y_pts
)

expected_solution = pixel_locations[..., 0] ** 2 - pixel_locations[..., 1] ** 2

# Plot the computed solution and the deviations from the expected solution.
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im_0 = ax[0].imshow(
    solution_pixels,
    extent=(root.xmin, root.xmax, root.ymin, root.ymax),
    origin="lower",
)
plt.colorbar(im_0, ax=ax[0])
ax[0].set_title("Computed Solution")
im_1 = ax[1].imshow(
    jnp.abs(solution_pixels - expected_solution),
    extent=(root.xmin, root.xmax, root.ymin, root.ymax),
    origin="lower",
    cmap="hot",
)
plt.colorbar(im_1, ax=ax[1])
ax[1].set_title("Errors")
plt.tight_layout()
plt.savefig("data/examples/usage_quickstart.svg", bbox_inches="tight")
