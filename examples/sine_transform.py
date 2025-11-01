import jax
import jax.numpy as jnp

from jaxhps import DiscretizationNode2D

BASIS_ELT_NORM = 1.0


@jax.jit
def nu_sinetransform(
    samples: jax.Array,
    spatial_points: jax.Array,
    quad_weights: jax.Array,
    freqs: jax.Array,
) -> jax.Array:
    """
    Goes from samples on the spatial domain [-1, 1]^2 to coefficients of the
    sine series.

    This function computs this type of transform:

    .. math::

       c_k = \\frac{1}{N} \\sum_{j=1}^N f(x_j) \\sin(k_1 (x_1 + pi / 2)) \\sin(k_2 (x_2 + pi / 2))



    Args:
        samples (jax.Array): Has shape (a,b)
        spatial_points (jax.Array): Has shape (a,b, 2)
        quad_weights (jax.Array): Quadrature weights. Has shape (a,b)
        freqs (jax.Array): Has shape (c, 2)

    Returns:
        jax.Array: Has shape (c,)
    """

    # Ensure samples have shape (d,) and spatial_points have shape (d, 1, 2). d=a*b
    samples = samples.flatten()
    spatial_points = spatial_points.reshape(-1, 1, 2)
    quad_weights = quad_weights.flatten()

    # Ensure the freqs have shape (1, c)
    freqs = freqs.reshape(1, -1, 2)

    # Compute the sine evaluations. Should have shape (d, c)
    sine_terms = (
        jnp.sin(jnp.pi / 2 * freqs[..., 0] * (spatial_points[..., 0] + 1.0))
        * jnp.sin(jnp.pi / 2 * freqs[..., 1] * (spatial_points[..., 1] + 1.0))
        / BASIS_ELT_NORM
    )
    # Normalize by the quadrature weights
    sine_terms = sine_terms * quad_weights[:, None]

    # Now compute the coefficients
    coefficients = jnp.einsum("dc,d -> c", sine_terms, samples)
    return coefficients


@jax.jit
def adjoint_nu_sinetransform(
    coefficients: jax.Array, freqs: jax.Array, spatial_points: jax.Array
) -> jax.Array:
    """
    Goes from coefficients of the sine series to samples on the spatial domain.

    Args:
        coefficients (jax.Array): Has shape (a,)
        freqs (jax.Array): Has shape (a, 2)
        spatial_points (jax.Array): Has shape (b,c,2)

    Returns:
        jax.Array: Has shape (b, c)
    """
    # Ensure coefficients have shape (a,)
    coefficients = coefficients.flatten()

    # Ensure freqs have shape (a, 2)
    freqs = freqs.reshape(-1, 2)

    # Want these to have shape (a,b,c)
    sine_args_x = (
        jnp.pi
        / 2
        * freqs[:, 0, None, None]
        * (spatial_points[None, :, :, 0] + 1.0)
    )
    sine_args_y = (
        jnp.pi
        / 2
        * freqs[:, 1, None, None]
        * (spatial_points[None, :, :, 1] + 1.0)
    )

    # Compute the sine terms. Should have shape (b, c, a)
    sine_terms = jnp.sin(sine_args_x) * jnp.sin(sine_args_y) / BASIS_ELT_NORM

    # Now compute the samples
    samples = jnp.einsum("abc, a -> bc", sine_terms, coefficients)
    return samples


def get_freqs_up_to_2k(
    k: float, root: DiscretizationNode2D = None
) -> jax.Array:
    """
    Generates the frequency vectors for the Sine Transform up to a given limit ``2k``.

    That is, this function returns a vector of frequency pairs :math:`(k_1, k_2)` such that
    :math:`1 \\leq k_1, k_2 \\leq 2k` and :math:`k_1 + k_2 \\leq 2k`.

    Args:
        k (float): Frequency parameter. This should be the incident wave frequency.

    Returns:
        jax.Array: Array of frequency vectors.
    """

    if root is None:
        max_k = int(k * 2)
    else:
        L = 2 * root.xmax
        max_k = int(k * 2 * L / jnp.pi)
    freqs = []
    for k1 in range(1, max_k + 1):
        for k2 in range(1, max_k + 1):
            l2_nrm = jnp.sqrt(k1**2 + k2**2)
            if l2_nrm <= max_k:
                freqs.append([k1, k2])
    f = jnp.array(freqs)

    # Sort in order of norm
    f_nrms = jnp.linalg.norm(f, axis=1)
    ii = jnp.argsort(f_nrms)
    return f[ii]
