Contributing
=========================

We welcome contributions to the ``jaxhps`` package! Please make contributions through pull requests `on GitHub <https://github.com/meliao/jaxhps>`_. To make development easier, forking the repository then installing the package in editable mode with the optional dev requirements is recommended: 

.. code:: bash

   cd jaxhps
   pip install -e .[dev]


Potential contributions include:
---------------------------------
- Any bug fixes or improvements raised in the `issues <https://github.com/meliao/jaxhps/issues>`_.
- Adding a class abstracting different types of boundary conditions, such as Dirichlet, Neumann, Robin, or boundary conditions specified by a boundary integral equation.
- Improving parallelization of the code in the merge step for adaptive discretizations. Currently, the merge step is not parallelized using a jax construct like ``vmap``.
- Improved calculation of the chunksize for the local solve stage. Currently, this is hard-coded for an 80GB GPU, and should be made flexible to adapt to the available GPU memory.