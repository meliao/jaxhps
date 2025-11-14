---
title: 'jaxhps: An elliptic PDE solver built with machine learning in mind'
tags:
  - Python
  - JAX
  - numerical analysis
  - partial differential equations
  - scientific computing
authors:
  - name: Owen Melia
    orcid: 0000-0003-0737-3718
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Daniel Fortunato
    orcid: 0000-0003-1302-7184
    affiliation: "1, 2"
  - name: Jeremy Hoskins
    orcid: 0000-0001-5307-2452
    affiliation: 3
  - name: Rebecca Willett
    orcid: 0000-0002-8109-7582
    affiliation: "3, 4, 5"
affiliations:
 - name: Center for Computational Mathematics, Flatiron Institute, USA
   index: 1
 - name: Center for Computational Biology, Flatiron Institute, USA
   index: 2
 - name: Computational and Applied Mathematics, Department of Statistics, University of Chicago, USA
   index: 3
 - name: Department of Computer Science, University of Chicago, USA
   index: 4
 - name: Data Science Institute, University of Chicago, USA
   index: 5
date: 6 November, 2025
bibliography: paper.bib

---

# Summary

Elliptic partial differential equations (PDEs) can model many physical phenomena, such as electrostatics, acoustics, wave propagation, and diffusion.
In scientific machine learning settings, a high-throughput PDE solver may be required to generate a training dataset, run in the inner loop of an iterative algorithm, or interface directly with a deep neural network.
To provide value to machine learning users, such a PDE solver must be compatible with standard automatic differentiation frameworks, scale efficiently when run on graphics processing units (GPUs), and maintain high accuracy for a large range of input parameters.
We have designed the `jaxhps` package with these use-cases in mind by implementing a highly efficient and accurate solver for elliptic problems with native hardware acceleration and automatic differentiation support. 
This is achieved by expressing a highly-efficient solution method for elliptic PDEs in JAX [@jax2018github]. 
This software implements algorithms specifically designed for fast GPU execution of a family of elliptic PDE solvers, which are described in full in @melia_hardware_2025.

Our Python package can numerically compute solutions $u(x)$ to problems of the form:
\begin{align}
    \mathcal{L}u(x) = f(x), & \qquad x \in \Omega, \\
    u(x) = g(x), & \qquad x \in \partial \Omega. 
\end{align}
In our setting, $\mathcal{L}$ is a linear, elliptic, second-order partial differential operator with spatially varying coefficient functions. The spatial domain, $\Omega$, can be a 2D square or 3D cube.

# Statement of need

While there is a vast array of PDE solvers implemented in JAX, we make a distinct contribution by implementing methods from the hierarchical Poincaré--Steklov (HPS) family of algorithms [@martinsson_direct_2013;@gillman_direct_2014;@gillman_spectrally_2015]. 
These methods use modern numerical analysis tools to resolve physical phenomena that are challenging for simpler tools, such as finite difference or finite element methods. 
One example of such a physical phenomenon is the oscillatory behavior of time-harmonic wave propagation simulations, which HPS methods resolve accurately and finite element methods do not [@babuska_pollution_1997;@yesypenko_slablu_2024].

While open-source implementations of HPS methods exist for users of MATLAB [@fortunato_high-order_2024] and C++ [@chipman_ellipticforest_2024], these packages do not offer native hardware acceleration or automatic differentiation capabilities. In addition, these packages do not offer support for three-dimensional problems.
@yesypenko_SlabLU_software_2024 is a Python implementation of the hardware-accelerated HPS-like method described in @yesypenko_slablu_2024, but it is designed for performance on extremely large 2D systems, which requires different design choices than the machine learning-focused optimizations we include in `jaxhps`.

# Software overview

The software is designed with two goals:

 * Allow users to interact with a simple interface that abstracts the complex HPS algorithms
 * Organize the flow of data to allow the user to reuse computations where possible

A typical user of `jaxhps` will wish to compute a solution $u(x)$ to equations (1) and (2).
The user will first specify $\Omega$ by creating `DiscretizationNode` and `Domain` objects. 
These objects automatically compute a high-order composite spectral discretization of $\Omega$. 
The `Domain` class exposes utilities for interpolating between the composite spectral discretization and a regular discretization specified by the user. 
If $f$ or $\mathcal{L}$ have local regions of high curvature, the `Domain` object's discretization can also be computed in an adaptive way that assigns more discretization points to those parts of $\Omega$. Additional utilities for interacting with the composite spectral discretization can be found in the `quadrature` module.

After the `Domain` is initialized, a `PDEProblem` object is created by the user from data specifying $\mathcal{L}$ and (optionally) $f$. The `PDEProblem` object also stores pre-computed interpolation and differentiation operators that can be reused during repeated calls to the solver. 

The user can then execute the HPS algorithm by calling the `build_solver()` method, specifying $f$ and $g$, and finally calling the `solve()` method. During the `build_solver()` and `solve()` methods, pointers to various solution operators are stored in the `PDEProblem` object. 
If the problem size is large, and these solution operators can not all be stored simultaneously on a GPU, care must be taken to organize the computation and data transfer between the GPU and CPU memory. To facilitate this, we provide `solve_subtree()`, `upward_pass_subtree()`, and `downward_pass_subtree()`, newly developed algorithms designed to minimize data transfer costs. A full description of these algorithms can be found in @melia_hardware_2025.
After computing the solution, the user can use automatic differentiation to compute the gradient of the solution with respect to input parameters by calling `jax.jvp()` or `jax.vjp()`. 
Multiple examples showing these capabilities are included in the [source repository](https://github.com/meliao/jaxhps) and the [documentation](https://jaxhps.readthedocs.io/en/latest/).

Finally, some researchers may want to design new algorithms by operating on the outputs of various subroutines underlying these HPS methods. To facilitate this, we expose a large collection of these subroutines in the `local_solve`, `merge`, `up_pass`, and `down_pass` modules.

# Author Contributions and Disclosure

 * **Owen Melia**: Conceptualization; Methodology; Software; Writing - original draft
 * **Daniel Fortunato**: Conceptualization; Methodology; Supervision; Writing - review & editing
 * **Jeremy Hoskins**: Conceptualization; Methodology; Supervision; Writing - review & editing
 * **Rebecca Willett**: Conceptualization; Methodology; Supervision; Funding acquisition; Project administration; Writing - review & editing

# Acknowledgements

The authors would like to thank Manas Rachh, Leslie Greengard, Vasilis Charisopoulos, and Olivia Tsang for many useful discussions. 
OM, JH, and RW gratefully acknowledge the support of the NSF-Simons National Institute for Theory and Mathematics in Biology (NITMB) via grants NSF DMS-2235451 and Simons Foundation MP-TMPS-00005320.
OM and RW gratefully acknowledge the support of NSF DMS-2023109, DOE DE-SC0022232, AFOSR FA9550-18-1-0166, the Physics Frontier Center for Living Systems funded by the National Science Foundation (PHY-2317138), and the support of the Margot and Tom Pritzker Foundation.
The Flatiron Institute is a division of the Simons Foundation.

# References
