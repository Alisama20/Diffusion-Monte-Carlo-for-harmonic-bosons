# Diffusion Monte Carlo for Interacting Bosons in a Harmonic Trap

This project implements the **Diffusion Monte Carlo (DMC)** method to compute the ground state energy of a system of interacting bosons confined in a harmonic potential. Potencial can be changed to anyone, this model is not suitable for potentials going to minus infinity.

Two approaches are implemented:

1. **Pure Diffusion Monte Carlo**
2. **Importance Sampling Diffusion Monte Carlo**

The numerical results are compared with the **analytical solution** for the harmonic boson model.

---

# Physical Model

The system is described by the Hamiltonian

$$
\hat H = -\frac{1}{2} \sum_{k=1}^{N}\nabla^2_{r_k} + \frac{1}{2} \sum_{k=1}^{N} r_k^2 - \frac{\beta^2}{2N} \sum_{l>k} |r_k-r_l|^2
$$


where

- N is the number of particles
- β² controls the interaction strength

The analytical ground state energy is

$$
  E_0 = \frac32 [1 + (N-1)\sqrt{1-\beta^2}]
$$

E = (3/2)[1 + (N-1)√(1-β²)]

---

# Methods

## Diffusion Monte Carlo (DMC)

The imaginary time Schrödinger equation

$$
\frac{\partial \Psi(\mathbf{r},\tau)}{\partial \tau} = -(\hat H - E_T)\Psi(\mathbf{r},\tau)
$$

is interpreted as a stochastic diffusion process.

The wavefunction is represented by a population of **walkers** evolving through:

- Diffusion
- Branching
- Population control

---

## Importance Sampling

To improve efficiency, a trial wavefunction

$$
f(\mathbf{r},\tau) = \Psi_T(\mathbf{r})\Psi(\mathbf{r},\tau)
$$

is introduced.

This leads to:

- Drift-diffusion process
- Local energy estimator
- Reduced variance


---

# Example Results

The DMC results reproduce the analytical ground state energy with high accuracy.

Without importance sampling, errors increase with particle number due to inefficient exploration of configuration space.

With importance sampling, the method shows excellent agreement even for large particle numbers.




---

## Author

A. S. Amari.

Project developed as part of the evaluation of the subject Mathematical and Numerical Complements at Master's Degree in Physics: Radiation, Nanotechnology, Particles and Astrophysics, University of Granada.

