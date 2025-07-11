Data placement across accelerator devices
=========================================

By default, JAX will place all new ``jax.Array`` objects on an accelerator device, if available. 
For our use-case, we do not want this behavior, because we would like to initialize large arrays of data, and then move them to the accelerator devices when needed for a computational step. 
If you plan to use ``jaxhps`` to compute solutions of PDEs with large numbers of discretization points, the default JAX behavior may become problematic.
Because of this, we suggest to override the default device used by ``jax`` by running this line of code:

.. code:: python

   import jax
   jax.config.update("jax_default_device", jax.devices("cpu")[0])

This means that all new ``jax.Array`` objects will be created on the CPU by default, and must be moved to the GPU. All of the example scripts described in :doc:`Examples` run this line of code after importing the necessary modules.

In most of the functions described in :doc:`solution_methods`, there are arguments for ``compute_device`` and ``host_device``. 
The ``compute_device`` is where the computation will be performed; the functions will take care of moving the data to the correct device. 
The ``host_device`` is where the data will be stored after the computation is finished; the functions will move the data to that device before returning.


