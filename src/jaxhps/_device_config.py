import logging
import jax
import numpy as np
import jax.numpy as jnp

# Logging stuff
jax_logger = logging.getLogger("jax")
jax_logger.setLevel(logging.WARNING)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)

# Jax enable double precision
jax.config.update("jax_enable_x64", True)

# Figure out if GPU is available
GPU_AVAILABLE = any("NVIDIA" in device.device_kind for device in jax.devices())

# On a NVIDIA A40 GPU with standard JAX preallocation, this bytes limit is 35781869568
GPU_SMALLER_THAN_80GB = (
    jax.devices()[0].memory_stats()["bytes_limit"] < 40 * 1024**3
)


# Device configuration

#: Array of acceleration devices (GPUs) available for computation. If no GPUs are available, this will have a single entry for the CPU device.
DEVICE_ARR = np.array(jax.devices()).flatten()

DEVICE_MESH = jax.sharding.Mesh(DEVICE_ARR, axis_names=("x",))

#: The CPU device. Not used for computation but is often used for storing data that needs to be moved off the GPU.
HOST_DEVICE = jax.devices("cpu")[0]


def local_solve_chunksize_2D(p: int, dtype: jax.typing.DTypeLike) -> int:
    """
    Estimates the chunksize that can be used for the local solve stage in 2D probelms.

    Rounds to the nearest power of four that will fit on the device.

    Args:
        p (int): Chebyshev polynomial order.

        dtype (jax.Dtype): Datatype of the input data.

    Returns:

        int: The chunksize that can be used for the local solve stage.
    """

    if p == 7:
        out = 4**2

    if dtype == jnp.complex128:
        out = 4**6

    out = 4**7
    if GPU_SMALLER_THAN_80GB:
        out //= 2
    return out


def local_solve_chunksize_3D(p: int, dtype: jax.typing.DTypeLike) -> int:
    """
    Estimates the chunksize that can be used for the local solve stage in 3D problems.
    Is calibrated for an 80GB GPU, and will divide the chunksize by 2 if a smaller GPU is detected.

    Args:
        p (int): Chebyshev polynomial order.

        dtype (jax.Dtype): Datatype of the input data.

    Returns:

        int: The chunksize that can be used for the local solve stage.
    """

    if p <= 8:
        out = 2_000
    elif p <= 10:
        out = 500
    elif p <= 12:
        out = 100  # bummer how small this must be
    else:
        out = 20
    if GPU_SMALLER_THAN_80GB:
        out //= 2
    return out
