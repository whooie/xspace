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

pub mod error;
pub mod interp;
pub mod units;
pub mod solve;
pub mod timedep;
pub mod utils;

pub mod docs;

pub(crate) const DEF_EPSILON: f64 = 1e-6;
pub(crate) const DEF_MAXITERS: usize = 1000;

pub type Arr1<S> = ndarray::ArrayBase<S, ndarray::Ix1>;
pub type Arr2<S> = ndarray::ArrayBase<S, ndarray::Ix2>;
