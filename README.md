# XSpace

Provides implementations of various numerical methods to compute solutions to
both the time-dependent and time-*in*dependent Schrödinger equation for spatial
wavefunctions in arbitrary conservative 1D potentials.

# Known bugs
- I'm trying out a pseudo-spectral Runge-Kutta method with adaptive stepsize for
  the time evolution, and it doesn't really work yet. Use the `split_step*`
  functions for time evolution instead.
- The time-independent solver `solve_secant` consistently overestimates harmonic
  oscillator state energies by around 0.5%. This isn't large enough to be a huge
  problem, but be aware.

