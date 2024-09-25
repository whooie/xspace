//! Functions to find zeros in array-sampled (continuous) functions via Lagrange
//! interpolation.
//!
//! ```
//! use ndarray as nd;
//! use xspace::interp::{ Zero, find_zeros };
//!
//! let x: nd::Array1<f64> = nd::Array::linspace(-5.0, 5.0, 1000);
//! let y = x.mapv(|xk| (xk + 3.0) * (xk - 0.5) * (xk - 2.0));
//! let zeros = find_zeros(&x, &y, Zero::All).unwrap();
//! assert!(
//!     [-3.0, 0.5, 2.0].into_iter()
//!         .zip(zeros)
//!         .all(|(expected, computed)| (computed - expected).abs() < 1e-6)
//! )
//! ```

use std::cmp;
use ndarray as nd;
use num_traits::Num;
use crate::error::*;

pub type InterpResult<T> = Result<T, InterpError>;

/// Specifies a set of zeros to look for in [`find_zeros`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Zero {
    /// Points at which a function changes from positive to negative.
    Falling,
    /// Points at which a function changes from negative to positive.
    Rising,
    /// Either/both of the above.
    All,
}

impl Zero {
    fn matches<A>(&self, a: &A, b: &A) -> bool
    where A: PartialEq + PartialOrd
    {
        match self {
            Self::Falling if a > b => true,
            Self::Rising if a < b => true,
            Self::All if a != b => true,
            _ => false,
        }
    }
}

/// Compute the value of a sampled function via a Lagrange polynomial.
pub fn lagrange<S, T, A>(
    data_x: &nd::ArrayBase<S, nd::Ix1>,
    data_y: &nd::ArrayBase<T, nd::Ix1>,
    x: A,
) -> InterpResult<A>
where
    S: nd::Data<Elem = A>,
    T: nd::Data<Elem = A>,
    A: Num + Copy
{
    LengthError::check(data_x, data_y)?;
    let res: A
        = data_x.iter().zip(data_y).enumerate()
        .map(|(j, (xj, yj))| {
            let xj = *xj;
            let inner
                = data_x.iter().enumerate()
                .filter(|(m, _)| *m != j)
                .map(|(_, xm)| (x - *xm) / (xj - *xm))
                .fold(A::one(), A::mul);
            *yj * inner
        })
        .fold(A::zero(), A::add);
    Ok(res)
}

/// Compute the value of the derivative of a sampled function via a Lagrange
/// polynomial.
pub fn dlagrange<A>(data_x: &nd::Array1<A>, data_y: &nd::Array1<A>, x: A)
    -> InterpResult<A>
where A: Num + Copy
{
    LengthError::check(data_x, data_y)?;
    let res: A
        = data_x.iter().zip(data_y).enumerate()
        .map(|(j, (xj, yj))| {
            let xj = *xj;
            let inner
                = data_x.iter().enumerate()
                .filter(|(i, _)| *i != j)
                .map(|(i, xi)| {
                    data_x.iter().enumerate()
                        .filter(|(m, _)| *m != j && *m != i)
                        .map(|(_, xm)| (x - *xm) / (xj - *xm))
                        .fold(A::one(), A::mul)
                        / (xj - *xi)
                })
                .fold(A::zero(), A::add);
            *yj * inner
        })
        .fold(A::zero(), A::add);
    Ok(res)
}

/// Return a list of all zeros of a given kind in a sampled function.
///
/// The function must be locally invertible on the scale of a few grid points.
pub fn find_zeros<S, T, A>(
    data_x: &nd::ArrayBase<S, nd::Ix1>,
    data_y: &nd::ArrayBase<T, nd::Ix1>,
    kind: Zero,
) -> InterpResult<Vec<A>>
where
    S: nd::Data<Elem = A>,
    T: nd::Data<Elem = A>,
    A: Num + PartialOrd + Copy,
{
    LengthError::check(data_x, data_y)?;
    LagrangeError::check(data_x)?;
    let n = data_x.len();
    let mut il: usize = 0;
    let mut ir: usize = 0;
    let z = A::zero();
    let zeros: Vec<A>
        = data_x.iter().zip(data_y).skip(1)
        .zip(data_y)
        .enumerate()
        .filter_map(|(i, ((xi, yi), yim1))| {
            if *yi == z {
                Some(Ok(*xi))
            } else if *yi * *yim1 <= z && kind.matches(yim1, yi) {
                il = i.saturating_sub(2);
                ir = cmp::min(n, i + 2);
                if ir - il < 4 {
                    println!(
                        "interp::find_zeros: WARNING: attempting to \
                        interpolate near an edge of the given data; some \
                        accuracy may be lost"
                    );
                }
                let interp
                    = lagrange(
                        &data_y.slice(nd::s![il..ir]),
                        &data_x.slice(nd::s![il..ir]),
                        z,
                    );
                Some(interp)
            } else {
                None
            }
        })
        .collect::<InterpResult<_>>()?;
    Ok(zeros)
}


