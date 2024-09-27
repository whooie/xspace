//! Miscellaneous tools.

use std::ops::Add;
use ndarray::{ self as nd, Ix1, concatenate };
use ndarray_linalg::Scalar;
use num_traits::{ One, Zero };
use rustfft as fft;
// use ndrustfft as ndfft;
use num_complex::Complex64 as C64;

/// Integrate using the trapezoidal rule.
///
/// *Panics if `y` has length less than 2*.
pub fn trapz<S, A>(y: &nd::ArrayBase<S, Ix1>, dx: A) -> A
where
    S: nd::Data<Elem = A>,
    A: Scalar,
{
    let n: usize = y.len();
    let two = A::one() + A::one();
    (dx / two) * (y[0] + two * y.slice(nd::s![1..n - 1]).sum() + y[n - 1])
}

/// Calculate the norm of a wavefunction.
///
/// *Panics if `q` has length less than 2*.
pub fn wf_norm<S, A>(q: &nd::ArrayBase<S, Ix1>, dx: A::Real) -> A::Real
where
    S: nd::Data<Elem = A>,
    A: Scalar,
{
    let n: usize = q.len();
    let two = <A as Scalar>::Real::one() + <A as Scalar>::Real::one();
    (dx / two) * (
        q[0].square()
        + two * q.iter().skip(1).take(n - 2).map(|qk| qk.square())
            .fold(<A as Scalar>::Real::zero(), <A as Scalar>::Real::add)
        + q[n - 1].square()
    )
}

/// Calculate the inner product of two wavefunctions.
///
/// *Panics if either array has length less than 2*.
pub fn wf_dot<S, T, A>(
    q: &nd::ArrayBase<S, Ix1>,
    p: &nd::ArrayBase<T, Ix1>,
    dx: A::Real,
) -> A
where
    S: nd::Data<Elem = A>,
    T: nd::Data<Elem = A>,
    A: Scalar,
{
    let n: usize = q.len().min(p.len());
    let two = A::one() + A::one();
    (A::from_real(dx) / two) * (
        q[0].conj() * p[0]
        + two * q.iter().zip(p).skip(1).take(n - 2)
            .fold(A::zero(), |acc, (qk, pk)| acc + qk.conj() * *pk)
        + q[n - 1].conj() * p[n - 1]
    )
}

/// Renormalize a wavefunction in place.
///
/// *Panics if `q` has length less than 2*.
pub fn wf_renormalize<S, A>(q: &mut nd::ArrayBase<S, Ix1>, dx: A::Real)
where
    S: nd::DataMut<Elem = A>,
    A: Scalar,
{
    let norm = A::from_real(wf_norm(q, dx).sqrt());
    q.iter_mut().for_each(|qk| { *qk /= norm; });
}

/// Return a normalized copy of a wavefunction.
///
/// *Panics if `q` has length less than 2*.
pub fn wf_normalized<S, A>(q: &nd::ArrayBase<S, Ix1>, dx: A::Real)
    -> nd::Array1<A>
where
    S: nd::Data<Elem = A>,
    A: Scalar,
{
    let norm = A::from_real(wf_norm(q, dx).sqrt());
    q.mapv(|qk| qk / norm)
}

/// Generate an array of frequency-space coordinates to accompany a FFT of `n`
/// points for sampling time `dt`.
pub fn fft_freq(n: usize, dt: f64) -> nd::Array1<f64> {
    if n % 2 == 0 {
        let fp: nd::Array1<f64>
            = (0..n / 2)
            .map(|k| k as f64 / (n as f64 * dt))
            .collect();
        let fm: nd::Array1<f64>
            = (1..n / 2 + 1).rev()
            .map(|k| -(k as f64) / (n as f64 * dt))
            .collect();
        concatenate!(nd::Axis(0), fp, fm)
    } else {
        let fp: nd::Array1<f64>
            = (0..(n + 1) / 2)
            .map(|k| k as f64 / (n as f64 * dt))
            .collect();
        let fm: nd::Array1<f64>
            = (1..(n + 1) / 2).rev()
            .map(|k| -(k as f64) / (n as f64 * dt))
            .collect();
        concatenate!(nd::Axis(0), fp, fm)
    }
}

/// Perform the one-dimensional, complex-valued FFT.
pub fn fft<S>(x: &nd::ArrayBase<S, Ix1>) -> nd::Array1<C64>
where S: nd::Data<Elem = C64>
{
    let n: usize = x.len();
    let mut f = x.to_owned();
    let mut plan = fft::FftPlanner::new();
    let fft_plan = plan.plan_fft_forward(n);
    fft_plan.process(f.as_slice_mut().unwrap());
    f
}

/// Perform the one-dimensional, complex-valued FFT in place.
pub fn fft_inplace<S>(f: &mut nd::ArrayBase<S, Ix1>)
where S: nd::DataMut<Elem = C64>
{
    let n: usize = f.len();
    let mut plan = fft::FftPlanner::new();
    let fft_plan = plan.plan_fft_forward(n);
    fft_plan.process(f.as_slice_mut().unwrap());
}

/// Perform the one-dimensional, complex-valued inverse FFT.
pub fn ifft<S>(f: &nd::ArrayBase<S, Ix1>) -> nd::Array1<C64>
where S: nd::Data<Elem = C64>
{
    let n: usize = f.len();
    let mut x = f.to_owned();
    let mut plan = fft::FftPlanner::new();
    let ifft_plan = plan.plan_fft_inverse(n);
    ifft_plan.process(x.as_slice_mut().unwrap());
    let n = n as f64;
    x.map_inplace(|xk| { *xk /= n; });
    x
}

/// Perform the one-dimensional, complex-valued inverse FFT in place.
pub fn ifft_inplace<S>(x: &mut nd::ArrayBase<S, Ix1>)
where S: nd::DataMut<Elem = C64>
{
    let n: usize = x.len();
    let mut plan = fft::FftPlanner::new();
    let ifft_plan = plan.plan_fft_inverse(n);
    ifft_plan.process(x.as_slice_mut().unwrap());
    let n = n as f64;
    x.map_inplace(|xk| { *xk /= n; });
}

/// Perform the one-dimensional, complex-valued FFT and return the result along
/// with the accompanying array of [frequency-space coordinates][fft_freq].
pub fn do_fft<S>(x: &nd::ArrayBase<S, Ix1>, dt: f64)
    -> (nd::Array1<C64>, nd::Array1<f64>)
where S: nd::Data<Elem = C64>
{
    let n: usize = x.len();
    (fft(x), fft_freq(n, dt))
}

/// Return a copy of `x` with indices shifted to map super-Nyquist frequency
/// components to negative frequencies.
pub fn fft_shift<S, A>(x: &nd::ArrayBase<S, Ix1>) -> nd::Array1<A>
where
    S: nd::Data<Elem = A>,
    A: Clone,
{
    let n = x.len();
    let (p, m)
        = if n % 2 == 0 {
            x.view().split_at(nd::Axis(0), n / 2)
        } else {
            x.view().split_at(nd::Axis(0), n / 2 + 1)
        };
    concatenate!(nd::Axis(0), m.into_owned(), p.into_owned())
}

// pub fn fft_nd<D>(X: &nd::Array<C64, D>, axis: usize) -> nd::Array<C64, D>
// where D: nd::Dimension
// {
//     let N: usize = X.shape()[axis];
//     let mut buf: nd::Array::<C64, D> = nd::Array::zeros(X.raw_dim());
//     let mut handler: ndfft::FftHandler<f64> = ndfft::FftHandler::new(N);
//     ndfft::ndfft(X, &mut buf, &mut handler, axis);
//     return buf;
// }
//
// pub fn do_fft_nd<D>(X: &nd::Array<C64, D>, dt: f64, axis: usize)
//     -> (nd::Array<C64, D>, nd::Array1<f64>)
// where D: nd::Dimension
// {
//     let N: usize = X.shape()[axis];
//     return (fft_nd(X, axis), fft_freq(N, dt));
// }
//
// pub fn fft_shifted_axis<T, D>(X: &nd::Array<T, D>, axis: usize)
//     -> nd::Array<T, D>
// where
//     T: Clone,
//     D: nd::Dimension + nd::RemoveAxis,
// {
//     let N: usize = X.shape()[axis];
//     let (P, M) = if N % 2 == 0 {
//         (
//             X.slice_axis(nd::Axis(axis), nd::Slice::from(0..N / 2)),
//             X.slice_axis(nd::Axis(axis), nd::Slice::from(N / 2..N)),
//         )
//     } else {
//         (
//             X.slice_axis(nd::Axis(axis), nd::Slice::from(0..N / 2 + 1)),
//             X.slice_axis(nd::Axis(axis), nd::Slice::from(N / 2 + 1..N)),
//         )
//     };
//     return concatenate!(nd::Axis(axis), M.into_owned(), P.into_owned());
// }
//
// pub fn fft_shift_axis<T, D>(X: nd::ArrayViewMut<T, D>, axis: usize)
// where
//     T: Clone,
//     D: nd::Dimension + nd::RemoveAxis,
// {
//     let N: usize = X.shape()[axis];
//     let (P, M) = if N % 2 == 0 {
//         (
//             X.slice_axis(nd::Axis(axis), nd::Slice::from(0..N / 2)),
//             X.slice_axis(nd::Axis(axis), nd::Slice::from(N / 2..N)),
//         )
//     } else {
//         (
//             X.slice_axis(nd::Axis(axis), nd::Slice::from(0..N / 2 + 1)),
//             X.slice_axis(nd::Axis(axis), nd::Slice::from(N / 2 + 1..N)),
//         )
//     };
//     let new = concatenate!(nd::Axis(axis), M.into_owned(), P.into_owned());
//     new.move_into(X);
// }

