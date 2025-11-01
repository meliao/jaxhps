import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from scipy import special
from jaxhps import Domain


@partial(jax.jit, static_argnums=(0,))
def get_ring_points(nrec: int, rad: float) -> jax.Array:
    """
    Generates ``nrec`` points equally-spaced about the ring of radius ``rad`` centered at the origin.

    Args:
        nrec (int): Number of points
        rad (float): Radius

    Returns:
        jax.Array: Has shape (nrec, 2)
    """

    thetas = jnp.linspace(0, 2 * jnp.pi, nrec, endpoint=False)
    out = rad * jnp.array([jnp.cos(thetas), jnp.sin(thetas)]).T
    return out


def get_domain_boundary_scaling(domain: Domain) -> jax.Array:
    """

    Computes quadrature weights to correctly scale integration over the boundary of the domain.

    Args:
        domain (Domain): Gives discretization and geometric information about the boundary.

    Returns:
        jax.Array: Has shape (n_bdry,) where n_bdry is the number of boundary points specified by ``domain``.
    """

    _, gauss_weights = np.polynomial.legendre.leggauss(domain.q)

    panel_len = (domain.root.xmax - domain.root.xmin) / (2**domain.L)
    n_panels = 4 * (2**domain.L)  # Total number of panels in the boundary

    # Scale the weights from length 2 to length panel_len
    scaling = panel_len * gauss_weights / 2

    # Repeat the scaling for each panel
    scaling = jnp.tile(scaling, n_panels)

    return scaling


def gen_S_exterior(
    domain: Domain, k: float, nrec: int, rad: float
) -> jax.Array:
    """
    Generates a matrix S approximating an integral operator for the single-layer Helmholtz kernel.

    Matrix maps from the discretization points specified by ``domain.boundary_points`` to a ring of
    nrec receivers; the ring has radius rad.

    The kernel we are using is

    .. math::

       \frac{i}{4}H_0^{(1)}](k ||x - y||_2)

    where :math:`H_0^{(1)}` is the Hankel function of the first kind of order zero.

    Args:
        domain (Domain): Specifies the spatial discretization of the interior problem.
        k (float): Frequency of the problem.
        nrec (int): Number of receiver points
        rad (float): Radius of the receiver points

    Returns:
        jax.Array: Has shape (nrec, n_bdry) where n_bdry is the number of boundary points specified by ``domain``. This number is 4 q (2^L).
    """
    target_points = get_ring_points(nrec, rad)
    source_points = domain.boundary_points
    source_measure = get_domain_boundary_scaling(domain)
    return _S_exterior(
        target_points=target_points,
        source_points=source_points,
        source_measure=source_measure,
        k=k,
    )

    # # Create an array where arr[i,j] = || target_i - source_j||_2
    # diffs = target_points[:, None, :] - source_points[None, ...]
    # # Compute the Euclidean distance
    # distances = jnp.linalg.norm(diffs, axis=-1)

    # # First get the kernel evals and then we will do the scaling
    # # kernel_evals has shape (nrec, n_bdry)
    # kernel_evals = (1j / 4) * special.hankel1(0, k * distances)

    # # Scalings have shape (n_bdry,)
    # scaling = get_domain_boundary_scaling(domain)

    # # Scale the kernel evaluations
    # kernel_evals_scaled = kernel_evals * scaling

    # return kernel_evals_scaled


def _S_exterior(
    target_points: jax.Array,
    source_points: jax.Array,
    source_measure: jax.Array,
    k: float,
) -> jax.Array:
    # Create an array where arr[i,j] = || target_i - source_j||_2
    diffs = target_points[:, None, :] - source_points[None, ...]
    # Compute the Euclidean distance
    distances = jnp.linalg.norm(diffs, axis=-1)

    # First get the kernel evals and then we will do the scaling
    # kernel_evals has shape (nrec, n_bdry)
    kernel_evals = (1j / 4) * special.hankel1(0, k * distances)

    # Scale the kernel evaluations
    kernel_evals_scaled = kernel_evals * source_measure

    return kernel_evals_scaled


def gen_D_exterior(
    domain: Domain, k: float, nrec: int, rad: float
) -> jax.Array:
    """
    Generates a matrix D approximating an integral operator for the double-layer Helmholtz kernel.

    Matrix maps from the discretization points specified by ``domain.boundary_points`` to a ring of
    nrec receivers; the ring has radius rad.

    The kernel we are using is

    .. math::

       d_{k}(x,y) = n(y) \\cdot \nabla_y \frac{i}{4}H_0^{(1)}](k ||x - y||_2)

    where :math:`H_0^{(1)}` is the Hankel function of the first kind of order zero, and :math:`n(y)` is the
    outward normal vector at point :math:`y`. The points :math:`y` are the boundary points specified by the ``domain`` object.

    Args:
        domain (Domain): Specifies the spatial discretization of the interior problem.
        k (float): Frequency of the problem.
        nrec (int): Number of receiver points
        rad (float): Radius of the receiver points

    Returns:
        jax.Array: Has shape (nrec, n_bdry) where n_bdry is the number of boundary points specified by ``domain``. This number is 4 q (2^L).
    """
    # Target points are x
    target_points = get_ring_points(nrec, rad)
    # Source points are y
    source_points = domain.boundary_points
    source_measure = get_domain_boundary_scaling(domain)
    return _D_exterior(
        target_points=target_points,
        source_points=source_points,
        source_measure=source_measure,
        k=k,
    )

    # # Create an array where arr[i,j] = || target_i - source_j||_2
    # diffs = target_points[:, None, :] - source_points[None, ...]
    # # Compute the Euclidean distance
    # distances = jnp.linalg.norm(diffs, axis=-1)

    # # First get the kernel evals and then we will do the scaling
    # # kernel_evals has shape (nrec, n_bdry, 2)
    # # This is \nabla_y (1/4 i H_0^{(1)}(k ||x - y||_2))
    # kernel_evals = (
    #     (1j * k / 4)
    #     * special.hankel1(1, k * distances)[:, :, None]
    #     / distances[:, :, None]
    #     * diffs
    # )

    # # Get the outward normals
    # n_per_side = domain.boundary_points.shape[0] // 4

    # normals_dot_kernel_evals = jnp.concatenate(
    #     [
    #         -1 * kernel_evals[:, :n_per_side, 1],  # S side
    #         kernel_evals[:, n_per_side : 2 * n_per_side, 0],  # E side
    #         kernel_evals[:, 2 * n_per_side : 3 * n_per_side, 1],  # N side
    #         -1 * kernel_evals[:, 3 * n_per_side :, 0],  # W side
    #     ],
    #     axis=1,
    # )

    # d = normals_dot_kernel_evals * get_domain_boundary_scaling(domain)
    # return d


def _D_exterior(
    target_points: jax.Array,
    source_points: jax.Array,
    source_measure: jax.Array,
    k: float,
) -> jax.Array:
    # Create an array where arr[i,j] = || target_i - source_j||_2
    diffs = target_points[:, None, :] - source_points[None, ...]
    # Compute the Euclidean distance
    distances = jnp.linalg.norm(diffs, axis=-1)

    # First get the kernel evals and then we will do the scaling
    # kernel_evals has shape (nrec, n_bdry, 2)
    # This is \nabla_y (1/4 i H_0^{(1)}(k ||x - y||_2))
    kernel_evals = (
        (1j * k / 4)
        * special.hankel1(1, k * distances)[:, :, None]
        / distances[:, :, None]
        * diffs
    )

    # Get the outward normals
    n_per_side = source_points.shape[0] // 4

    normals_dot_kernel_evals = jnp.concatenate(
        [
            -1 * kernel_evals[:, :n_per_side, 1],  # S side
            kernel_evals[:, n_per_side : 2 * n_per_side, 0],  # E side
            kernel_evals[:, 2 * n_per_side : 3 * n_per_side, 1],  # N side
            -1 * kernel_evals[:, 3 * n_per_side :, 0],  # W side
        ],
        axis=1,
    )

    d = normals_dot_kernel_evals * source_measure
    return d


@jax.jit
def get_T_ext(
    S_int: jax.Array,
    D_int: jax.Array,
) -> jax.Array:
    return jnp.linalg.inv(S_int) @ (0.5 * jnp.eye(D_int.shape[0]) - D_int)
