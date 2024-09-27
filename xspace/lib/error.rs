//! Collection of all error types.
//!
//! All errors derive [`thiserror::Error`], making them composable when allowed
//! and compatible with application code using [`anyhow`][anyhow].
//!
//! [anyhow]: https://crates.io/crates/anyhow

use ndarray as nd;
use ndarray_linalg::error::LinalgError;
use thiserror::Error;

/// Returned when an operation requiring equal-length arrays encounters arrays
/// with unequal length.
#[derive(Debug, Error)]
#[error("encountered arrays with incompatible lengths; got {0} and {1}")]
pub struct LengthError(pub usize, pub usize);

impl LengthError {
    pub(crate) fn check<S, A, T, B>(
        a: &nd::ArrayBase<S, nd::Ix1>,
        b: &nd::ArrayBase<T, nd::Ix1>,
    ) -> Result<(), Self>
    where
        S: nd::Data<Elem = A>,
        T: nd::Data<Elem = B>,
    {
        let na = a.len();
        let nb = b.len();
        (na == nb).then_some(()).ok_or(Self(na, nb))
    }
}

/// Returned from a call to [`find_zeros`][crate::interp::find_zeros] when data
/// arrays are less than 5 elements long.
#[derive(Debug, Error)]
#[error("coordinate arrays in interpolation must be longer than 4 elements; got {0}")]
pub struct LagrangeError(pub usize);

impl LagrangeError {
    pub(crate) fn check<S, A>(a: &nd::ArrayBase<S, nd::Ix1>)
        -> Result<(), Self>
    where S: nd::Data<Elem = A>
    {
        let n = a.len();
        (n > 4).then_some(()).ok_or(Self(n))
    }
}

/// Returned from functions in [`interp`][crate::interp].
#[derive(Debug, Error)]
pub enum InterpError {
    /// [`LengthError`]
    #[error("length error: {0}")]
    Length(#[from] LengthError),

    /// [`LagrangeError`]
    #[error("lagrange error: {0}")]
    Lagrange(#[from] LagrangeError),
}

/// Returned from spatial wavefunction solver functions.
#[derive(Debug, Error)]
pub enum XError {
    /// Returned when a non-positive `epsilon` value is encountered.
    #[error("epsilon values must be greater than 0; got {0}")]
    BadEpsilon(f64),

    /// Returned when a non-positive `maxiters` value is encountered.
    #[error("maxiters must be greater than 0; got {0}")]
    BadMaxiters(usize),

    /// Returned when the initial vibrational level search fails to reach the
    /// desired node count in [`solve_secant`][crate::solve::solve_secant].
    #[error("solve::solve_secant: FATAL: vibrational level search failed to find the correct interval")]
    SecantVib,

    /// [`LengthError`]
    #[error("array length error: {0}")]
    Length(#[from] LengthError),

    /// [`InterpError`].
    #[error("interpolation error: {0}")]
    Interp(#[from] InterpError),

    /// [`LinalgError`].
    #[error("linalg error: {0}")]
    Linalg(#[from] LinalgError),
}

impl XError {
    pub(crate) fn check_epsilon(epsilon: f64) -> Result<(), Self> {
        (epsilon > 0.0).then_some(()).ok_or(Self::BadEpsilon(epsilon))
    }

    pub(crate) fn check_maxiters(maxiters: usize) -> Result<(), Self> {
        (maxiters != 0).then_some(()).ok_or(Self::BadMaxiters(maxiters))
    }
}

/// Returned from time-dependent wavefunction solver functions.
#[derive(Debug, Error)]
pub enum TError {
    /// Returned when a non-positive `epsilon` value is encountered.
    #[error("epsilon values must be greater than 0; got {0}")]
    BadEpsilon(f64),

    #[error("rka error bound could not be satisfied")]
    RKAErrorBound,

    /// [`LengthError`]
    #[error("array length error: {0}")]
    Length(#[from] LengthError),
}

impl TError {
    pub(crate) fn check_epsilon(epsilon: f64) -> Result<(), Self> {
        (epsilon > 0.0).then_some(()).ok_or(Self::BadEpsilon(epsilon))
    }
}

