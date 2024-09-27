#![allow(dead_code, non_snake_case)]

//! Provides functions and higher-level constructs for automated solution of the
//! one-dimensional, time-independent Schrödinger equation via Numerov's scheme
//! and time-dependent Schrödinger equation for one-dimensional systems via
//! the split-step and fourth-order Runge-Kutta pseudo-spectral methods.
//!
//! Provides implementations for the following numerical routines:
//! - Time-independent:
//!     - Naive Numerov[^1]
//!     - Matrix Numerov[^2]
//!     - Renormalized Numerov[^3] (secant search and naive root detection via
//!       two-sided shooting)
//! - Time-dependent:
//!     - Pseudo-spectral fourth-order Runge-Kutta (optional adaptive stepsize)
//!     - Pseudo-spectral split-step operator
//!
//! See [`docs`] for theoretical background.
//!
//! [^1]: B. Numerov, "Note on the numerical integration of d2x/dt2 = f(x,t)."
//! Astronomische Nachrichten **230** 19 (1927).
//!
//! [^2]: M. Pillai, J. Goglio, and T. Walker, "Matrix Numerov method for
//! solving Schrödinger's equation." American Journal of Physics **80** 11
//! 1017-1019 (2012).
//!
//! [^3]: B. R. Johnson, "New numerical methods applied to solving the
//! one-dimensional eigenvalue problem." J. Chem. Phys. **67**:4086 (1977).

pub mod error;
pub mod interp;
pub mod units;
pub mod solve;
pub mod timedep;
pub mod utils;

pub mod docs;

pub(crate) const DEF_EPSILON: f64 = 1e-6;
pub(crate) const DEF_MAXITERS: usize = 1000;

/// A one-dimensional array with owned or shared internal data.
pub type Arr1<S> = ndarray::ArrayBase<S, ndarray::Ix1>;

/// A two-dimensional array with owned or shared internal data.
pub type Arr2<S> = ndarray::ArrayBase<S, ndarray::Ix2>;
