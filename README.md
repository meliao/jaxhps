# jaxhps
A JAX package implementing hardware acceleration for HPS methods in two and three dimensions.

See our preprint on arXiv: [Hardware Acceleration for HPS Methods in Two and Three Dimensions](https://arxiv.org/abs/2503.17535)

## Installation

To install, you can use `pip`: 

```
pip install jaxhps
```

The examples require additional packages `matplotlib` and `h5py`. If you want to install them automatically, use:

```
pip install jaxhps[examples]
```

More installation instructions, including installation for a GPU system, are available in [the documentation](https://jaxhps.readthedocs.io/en/latest/#installation).

## Documentation

[https://jaxhps.readthedocs.io/en/latest/](https://jaxhps.readthedocs.io/en/latest/)

## Examples


### hp convergence on 2D problems with known solutions

Shows convergence using uniform quadtrees using both DtN matrices and ItI matrices.
```
python examples/hp_convergence_2D_problems.py --DtN --ItI
```

### High-wavenumber scattering problem

First, run the matlab script `examples/driver_gen_SD_matrices.m`. This will generate and save exterior single and double-layer kernel matrices. These matrices are necessary to define a boundary integral equation for the scattering problem.
Once in place, we can run the script:
```
python examples/wave_scattering_compute_reference_soln.py --scattering_potential gauss_bumps -k 100 --plot_utot
```
This will generate plots which looks like this, showing the scattering potential and real part and modulus of the total field: 

![Showing the scattering potential, a sum of randomly-placed Gaussian bumps.](.github/assets/k_100_gauss_bumps_q.svg)
![Showing the real part of the total wave field of a scattering problem where k=100 and the scattering potential is a sum of randomly-placed Gaussian bumps.](.github/assets/k_100_gauss_bumps_utot_ground_truth_real.svg)
![Showing the absolute value of the total wave field of a scattering problem where k=100 and the scattering potential is a sum of randomly-placed Gaussian bumps.](.github/assets/k_100_gauss_bumps_utot_ground_truth_abs.svg)


### Adaptive discretization on a 3D problem with known solution

We have a script for generating adaptive discretizations on the wavefront problem presented in our paper:

```
python examples/wavefront_adaptive_discretization_3D.py -p 10 --tol 1e-02 1e-05
```

This should produce an image showing the computed solution, generated grid, and error map:

![Showing the computed solution, the adaptive grid, and the errors on a 2D slice of our 3D wavefont probelm.](.github/assets/wavefront_soln_tol_1e-05.svg)


### Inverse wave scattering using automatic differentiation

We have an implementation of a low-dimensional optimization problem using automatic differentiation:

```
python examples/inverse_wave_scattering.py --n_iter 25
```

This is an inverse scattering problem where we try to recover the locations of four Gaussian bumps which make up the scattering potential. Running the code should produce plots showing the optimization variables converging at the centers of the Gaussian bumps in the scattering potential, as well as a plot showing the convergence of the objective function:

![Showing the convergence of the objective function in our inverse scattering example.](.github/assets/inverse_scattering_residuals.svg)
![Showing the convergence of the iterates to the centers of the Gaussian bumps.](.github/assets/inverse_scattering_iterates.svg)

## Linearized Poisson--Boltzmann equation

Our method can be used to solve the a linearized Poisson-Boltzmann equation, which models the electrostatic properties of a molecule in solution. 

```
python examples/poisson_boltzmann_example.py --tol 1e-01 1e-03 -p 10 
```
