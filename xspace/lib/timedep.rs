//! Provides functions to compute solutions to the 1+1-dimensional
//! (time-dependent) Schrödinger equation (TDSE) for motion in a time-dependent,
//! conservative potential.
//!
//! In all 2D arrays, the first (or zero-th) axis indexes time.

use std::f64::consts::TAU;
use ndarray as nd;
use num_complex::Complex64 as C64;
use crate::{
    Arr1,
    Arr2,
    error::TError,
    utils::{ fft, fft_inplace, ifft_inplace, wf_renormalize },
};

pub type TResult<T> = Result<T, TError>;

// estimate the ratio between truncation errors at different step sizes for a
// fourth-order Runge-Kutta scheme
fn error_ratio(z: C64, w: C64, err: f64) -> f64 {
    let scale: f64 = err * (z.norm() + w.norm()) / 2.0;
    let diff: f64 = (z - w).norm();
    diff / (scale + f64::EPSILON)
}

// estimate the ratio between truncation errors at different step sizes for a
// fourth-order Runge-Kutta scheme with array values
fn error_ratio_arr<S, T>(z: &Arr1<S>, w: &Arr1<T>, err: f64) -> f64
where
    S: nd::Data<Elem = C64>,
    T: nd::Data<Elem = C64>,
{
    z.iter().zip(w)
        .map(|(zk, wk)| error_ratio(*zk, *wk, err))
        .max_by(|l, r| {
            match l.partial_cmp(r) {
                Some(ord) => ord,
                None => std::cmp::Ordering::Less,
            }
        })
        .unwrap()
}

// perform the operation `a + v * b` succinctly
fn array_step<S, T>(a: &Arr1<S>, v: f64, b: &Arr1<T>) -> nd::Array1<C64>
where
    S: nd::Data<Elem = C64>,
    T: nd::Data<Elem = C64>,
{
    nd::Zip::from(a).and(b)
        .map_collect(|ak, bk| ak + v * bk)
}

// return an array of differences between adjacent elements of a source array
fn array_diff<S, A>(a: &Arr1<S>) -> nd::Array1<A>
where
    S: nd::Data<Elem = A>,
    A: std::ops::Sub<A, Output = A> + Copy,
{
    a.iter().zip(a.iter().skip(1))
        .map(|(ak, akp1)| *akp1 - *ak)
        .collect()
}

// calculate the time derivative of the TDSE, i.e. evaluate the action of the
// Hamiltonian on the state `q` with an added factor of `-i`
//
// the Hamiltonian is applied in two stages:
//
// in the first, the state is taken to k-space so that the momentum (squared)
// operator can be applied by simple multiplication; this is the contribution to
// dq/dt from only the kinetic half
//
// in the second, the kinetic contribution is taken back to x-space and the
// potential energy contribution is added via `V * q`
//
// finally, the overall factor of `-i` is included
fn rhs<S, T>(dx: f64, V: &Arr1<S>, q: &Arr1<T>) -> nd::Array1<C64>
where
    S: nd::Data<Elem = f64>,
    T: nd::Data<Elem = C64>,
{
    let n = q.len();
    let m = if n % 2 == 0 { n / 2 } else { (n + 1) / 2 };
    let dk = TAU * (n as f64 * dx).recip();
    let mut dq = fft(q);
    dq.iter_mut().enumerate()
        .for_each(|(i, dqi)| {
            if (0..m).contains(&i) {
                *dqi *= ((i as f64) * dk).powi(2)
            } else {
                *dqi *= (((n - i) as f64) * dk).powi(2)
            }
        });
    ifft_inplace(&mut dq);
    nd::Zip::from(&mut dq).and(V).and(q)
        .for_each(|dqi, Vi, qi| {
            *dqi += Vi * qi;
            *dqi *= -C64::i();
        });
    dq
}

// take a single RK4 step *in place*
//
// for this we need the potential at three points in time:
// - V: the potential at the current time
// - Vh: the potential at the current time + dth
// - Vp: the potential at the current time + dth + dtp
fn rk4_step<S, T>(
    dx: f64,
    V: &Arr1<S>,
    Vh: &Arr1<S>,
    Vp: &Arr1<S>,
    q: &mut Arr1<T>,
    dth: f64,
    dtp: f64,
)
where
    S: nd::Data<Elem = f64>,
    T: nd::DataMut<Elem = C64>,
{
    let k1 = rhs(dx, V, q);
    let k2 = rhs(dx, Vh, &array_step(q, dth, &k1));
    let k3 = rhs(dx, Vh, &array_step(q, dth, &k2));
    let k4 = rhs(dx, Vp, &array_step(q, dth + dtp, &k3));
    nd::Zip::from(q).and(&k1).and(&k2).and(&k3).and(&k4)
        .for_each(|qk, k1k, k2k, k3k, k4k| {
            *qk += (dth + dtp) / 6.0 * (k1k + 2.0 * (k2k + k3k) + k4k);
        });
}

/// Perform fourth-order Runge-Kutta integration for a time-independent
/// potential sampled over a series of time coordinates.
///
/// See also [`rk4`] and [`split_step_const`].
pub fn rk4_const<S, T, U>(dx: f64, V: &Arr1<S>, q0: &Arr1<T>, t: &Arr1<U>)
    -> nd::Array2<C64>
where
    S: nd::Data<Elem = f64>,
    T: nd::Data<Elem = C64>,
    U: nd::Data<Elem = f64>,
{
    let n = t.len();
    let dt = array_diff(t);
    let mut q: nd::Array2<C64> = nd::Array2::zeros((t.len(), V.len()));
    let mut q_temp: nd::Array1<C64> = q0.to_owned();
    q.slice_mut(nd::s![0, ..]).assign(q0);
    let iter
        = dt.iter().zip(dt.iter().skip(1))
        .zip(q.axis_iter_mut(nd::Axis(0)).skip(2))
        .step_by(2);
    for ((&dtk, &dtkp1), qkp2) in iter {
        rk4_step(dx, V, V, V, &mut q_temp, dtk, dtkp1);
        wf_renormalize(&mut q_temp, dx);
        q_temp.clone().move_into(qkp2);
    }
    q_temp = q0.to_owned();
    let mut l: f64;
    let mut r: f64;
    let mut qoffs = q.slice_mut(nd::s![1.., ..]);
    let iter
        = dt.iter().zip(dt.iter().skip(1)).step_by(2)
        .zip(qoffs.axis_chunks_iter_mut(nd::Axis(0), 2));
    for ((&dtk, &dtkp1), mut qkp12) in iter {
        l = dtkp1 / (dtk + dtkp1);
        r = dtk / (dtk + dtkp1);
        nd::Zip::from(&q_temp).and(qkp12.axis_iter_mut(nd::Axis(1)))
            .for_each(|qkx, mut qkp12x| {
                qkp12x[0] = l * qkx + r * qkp12x[1];
            });
        q_temp = qkp12.slice(nd::s![1, ..]).to_owned();
    }
    if n % 2 == 0 {
        q_temp = q.slice(nd::s![n - 2, ..]).to_owned();
        q_temp.move_into(q.slice_mut(nd::s![n - 1, ..]));
    }
    q
}

/// Perform fourth-order Runge-Kutta integration for a time-dependent potential
/// sampled over a series of time coordinates.
///
/// See also [`rk4_func`] and [`rka`].
pub fn rk4<S, T, U>(dx: f64, V: &Arr2<S>, q0: &Arr1<T>, t: &Arr1<U>)
    -> nd::Array2<C64>
where
    S: nd::Data<Elem = f64>,
    T: nd::Data<Elem = C64>,
    U: nd::Data<Elem = f64>,
{
    let n = t.len();
    let dt = array_diff(t);
    let mut q: nd::Array2<C64> = nd::Array2::zeros(V.raw_dim());
    let mut q_temp: nd::Array1<C64> = q0.to_owned();
    q.slice_mut(nd::s![0, ..]).assign(q0);
    let iter
        = dt.iter().zip(dt.iter().skip(1))
        .zip(
            V.axis_iter(nd::Axis(0))
            .zip(V.axis_iter(nd::Axis(0)).skip(1))
            .zip(V.axis_iter(nd::Axis(0)).skip(2))
        )
        .zip(q.axis_iter_mut(nd::Axis(0)).skip(2))
        .step_by(2);
    for (((&dtk, &dtkp1), ((Vk, Vkp1), Vkp2)), qkp2) in iter {
        rk4_step(dx, &Vk, &Vkp1, &Vkp2, &mut q_temp, dtk, dtkp1);
        wf_renormalize(&mut q_temp, dx);
        q_temp.clone().move_into(qkp2);
    }
    q_temp = q0.to_owned();
    let mut l: f64;
    let mut r: f64;
    let iter
        = dt.iter().zip(dt.iter().skip(1)).step_by(2)
        .zip(q.axis_chunks_iter_mut(nd::Axis(0), 2));
    for ((&dtk, &dtkp1), mut qkp12) in iter {
        l = dtkp1 / (dtk + dtkp1);
        r = dtk / (dtk + dtkp1);
        nd::Zip::from(&q_temp).and(qkp12.axis_iter_mut(nd::Axis(1)))
            .for_each(|qkx, mut qkp12x| {
                qkp12x[0] = l * qkx + r * qkp12x[1];
            });
        q_temp = qkp12.slice(nd::s![1, ..]).to_owned();
    }
    if n % 2 == 0 {
        q_temp = q.slice(nd::s![n - 2, ..]).to_owned();
        q_temp.move_into(q.slice_mut(nd::s![n - 1, ..]));
    }
    q
}

/// Perform fourth-order Runge-Kutta integration for a time-dependent potential
/// described by a function over a series of time coordinates.
///
/// See also [`rk4`] and [`rka`].
pub fn rk4_func<F, S, T>(dx: f64, mut V: F, q0: &Arr1<S>, t: &Arr1<T>)
    -> nd::Array2<C64>
where
    F: FnMut(f64) -> nd::Array1<f64>,
    S: nd::Data<Elem = C64>,
    T: nd::Data<Elem = f64>,
{
    let dt = array_diff(t);
    let mut q: nd::Array2<C64> = nd::Array2::zeros((t.len(), q0.len()));
    let mut q_temp: nd::Array1<C64> = q0.to_owned();
    q.slice_mut(nd::s![0, ..]).assign(q0);
    let mut Vk: nd::Array1<f64> = V(t[0]);
    let mut Vkp1h: nd::Array1<f64>;
    let mut Vkp1: nd::Array1<f64>;
    let iter = dt.iter().zip(t).zip(q.axis_iter_mut(nd::Axis(0)).skip(1));
    for ((&dtk, &tk), qkp1) in iter {
        Vkp1h = V(tk + dtk / 2.0);
        Vkp1 = V(tk + dtk);
        rk4_step(dx, &Vk, &Vkp1h, &Vkp1, &mut q_temp, dtk / 2.0, dtk / 2.0);
        wf_renormalize(&mut q_temp, dx);
        q_temp.clone().move_into(qkp1);
        Vk = Vkp1;
    }
    q
}

// take a single RK4 step *in place*, returning the size for the next step
fn rka_step<F, S>(
    dx: f64,
    V: &mut F,
    q: &mut Arr1<S>,
    t: f64,
    dt: f64,
    err: f64,
) -> TResult<f64>
where
    F: FnMut(f64) -> nd::Array1<f64>,
    S: nd::DataMut<Elem = C64>,
{
    // safety numbers -- particular to rk4
    const SAFE1: f64 = 0.9;
    const SAFE2: f64 = 4.0;

    let mut dt_old;
    let mut dt_new = dt;
    let (mut dt_cond1, mut dt_cond2): (f64, f64);
    let mut q_temp_half: nd::Array1<C64>;
    let mut q_temp_full: nd::Array1<C64>;
    let V0: nd::Array1<f64> = V(t);
    let mut Vh: nd::Array1<f64>;
    let mut Vp: nd::Array1<f64>;
    let mut Vp2: nd::Array1<f64>;
    let mut er: f64;
    for _ in 0_usize..100 {
        q_temp_half = q.to_owned();
        q_temp_full = q.to_owned();

        // take two half-sized steps
        Vh = V(t + dt / 4.0);
        Vp = V(t + dt / 2.0);
        rk4_step(dx, &V0, &Vh, &Vp, &mut q_temp_half, dt / 4.0, dt / 4.0);
        Vh = V(t + 3.0 * dt / 4.0);
        Vp2 = V(t + dt);
        rk4_step(dx, &Vp, &Vh, &Vp2, &mut q_temp_half, dt / 4.0, dt / 4.0);

        // take one full-sized step
        rk4_step(dx, &V0, &Vp, &Vp2, &mut q_temp_full, dt / 2.0, dt / 2.0);

        // compute the estimated local truncation error
        er = error_ratio_arr(&q_temp_half, &q_temp_full, err);

        // estimate new step size (with safety factors)
        dt_old = dt_new;
        if er == 0.0 {
            dt_new = dt_old / SAFE2;
            continue;
        }
        dt_new = dt_old * er.powf(-0.2) * SAFE1;
        dt_cond1 = dt_old / SAFE2;
        dt_cond2 = dt_old * SAFE2;
        dt_new = if dt_cond1 > dt_new { dt_cond1 } else { dt_new };
        dt_new = if dt_cond2 < dt_new { dt_cond2 } else { dt_new };

        if er < 1.0 {
            q_temp_half.move_into(q);
            return Ok(dt_new);
        }
    }
    Err(TError::RKAErrorBound)
}

/// Perform fourth-order Runge-Kutta integration with adaptive stepsize for a
/// time-dependent potential described by a function over a series of time
/// coordinates.
///
/// The given time bounds are guaranteed to both be elements of the returned
/// time-coordinate array.
///
/// See also [`rk4`] and [`rk4_func`].
pub fn rka<F, S>(
    dx: f64,
    mut V: F,
    q0: &Arr1<S>,
    t_bounds: (f64, f64),
    dt0: f64,
    epsilon: f64,
) -> TResult<(nd::Array1<f64>, nd::Array2<C64>)>
where
    F: FnMut(f64) -> nd::Array1<f64>,
    S: nd::Data<Elem = C64>,
{
    TError::check_epsilon(epsilon)?;

    let mut t: Vec<f64> = Vec::new();
    t.push(t_bounds.0);
    let mut t_temp = t_bounds.0;
    let mut dt = dt0;
    let mut q: Vec<nd::Array1<C64>> = Vec::new();
    q.push(q0.to_owned());
    let mut q_temp: nd::Array1<C64> = q0.to_owned();
    while t_temp < t_bounds.1 {
        dt = if dt < t_bounds.1 - t_temp { dt } else { t_bounds.1 - t_temp };
        dt = rka_step(dx, &mut V, &mut q_temp, t_temp, dt, epsilon)?;
        t_temp += dt;
        t.push(t_temp);
        q.push(q_temp.clone());
    }
    let t: nd::Array1<f64> = nd::Array1::from_vec(t);
    let q: nd::Array2<C64>
        = nd::stack(
            nd::Axis(0),
            &q.iter().map(|qk| qk.view()).collect::<Vec<_>>(),
        )
        .unwrap();
    Ok((t, q))
}

fn apply_split_kinetic<S>(dx: f64, q: &mut Arr1<S>, dt: f64)
where S: nd::DataMut<Elem = C64>
{
    let n = q.len();
    let m = if n % 2 == 0 { n / 2 } else { (n + 1) / 2 };
    let dk = TAU * (n as f64 * dx).recip();
    fft_inplace(q);
    q.iter_mut().enumerate()
        .for_each(|(i, qi)| {
            if (0..m).contains(&i) {
                *qi *= C64::cis(-((i as f64) * dk).powi(2) * dt);
            } else {
                *qi *= C64::cis(-(((n - i) as f64) * dk).powi(2) * dt);
            }
        });
    ifft_inplace(q);
}

fn apply_split_potential<S, T>(V: &Arr1<S>, q: &mut Arr1<T>, dt: f64)
where
    S: nd::Data<Elem = f64>,
    T: nd::DataMut<Elem = C64>,
{
    q.iter_mut().zip(V)
        .for_each(|(qi, Vi)| { *qi *= C64::cis(-Vi * dt); });
}

/// Perform split-step integration for a time-independent potential over a
/// series of time coordinates.
///
/// See also [`split_step`] and [`split_step_func`].
pub fn split_step_const<S, T, U>(
    dx: f64,
    V: &Arr1<S>,
    q0: &Arr1<T>,
    t: &Arr1<U>,
) -> nd::Array2<C64>
where
    S: nd::Data<Elem = f64>,
    T: nd::Data<Elem = C64>,
    U: nd::Data<Elem = f64>,
{
    let dt = array_diff(t);
    let mut q: nd::Array2<C64> = nd::Array2::zeros((t.len(), V.len()));
    let mut q_temp: nd::Array1<C64> = q0.to_owned();
    q.slice_mut(nd::s![0, ..]).assign(q0);
    let iter = dt.iter().zip(q.axis_iter_mut(nd::Axis(0)).skip(1));
    for (&dtk, qkp1) in iter {
        apply_split_potential(V, &mut q_temp, dtk / 2.0);
        apply_split_kinetic(dx, &mut q_temp, dtk);
        apply_split_potential(V, &mut q_temp, dtk / 2.0);
        // wf_renormalize(&mut q_temp, dx);
        q_temp.clone().move_into(qkp1);
    }
    q
}

/// Perform split-step integration for a time-dependent potantial over a series
/// of time coordinates.
///
/// See also [`split_step_func`].
pub fn split_step<S, T, U>(dx: f64, V: &Arr2<S>, q0: &Arr1<T>, t: &Arr1<U>)
    -> nd::Array2<C64>
where
    S: nd::Data<Elem = f64>,
    T: nd::Data<Elem = C64>,
    U: nd::Data<Elem = f64>,
{
    let dt = array_diff(t);
    let mut q: nd::Array2<C64> = nd::Array2::zeros(V.raw_dim());
    let mut q_temp: nd::Array1<C64> = q0.to_owned();
    q.slice_mut(nd::s![0, ..]).assign(q0);
    let iter =
        dt.iter()
        .zip(V.axis_iter(nd::Axis(0)).zip(V.axis_iter(nd::Axis(0)).skip(1)))
        .zip(q.axis_iter_mut(nd::Axis(0)).skip(1));
    for ((&dtk, (Vk, Vkp1)), qkp1) in iter {
        apply_split_potential(&Vk, &mut q_temp, dtk / 2.0);
        apply_split_kinetic(dx, &mut q_temp, dtk);
        apply_split_potential(&Vkp1, &mut q_temp, dtk / 2.0);
        wf_renormalize(&mut q_temp, dx);
        q_temp.clone().move_into(qkp1);
    }
    q
}

/// Perform split-step integration for a time-dependent potantial described by a
/// function over a series of time coordinates.
///
/// See also [`split_step`].
pub fn split_step_func<F, S, T>(dx: f64, mut V: F, q0: &Arr1<S>, t: &Arr1<T>)
    -> nd::Array2<C64>
where
    F: FnMut(f64) -> nd::Array1<f64>,
    S: nd::Data<Elem = C64>,
    T: nd::Data<Elem = f64>,
{
    let dt = array_diff(t);
    let mut q: nd::Array2<C64> = nd::Array2::zeros((t.len(), q0.len()));
    let mut q_temp: nd::Array1<C64> = q0.to_owned();
    q.slice_mut(nd::s![0, ..]).assign(q0);
    let mut V_temp: nd::Array1<f64> = V(t[0]);
    let iter = dt.iter().zip(t).zip(q.axis_iter_mut(nd::Axis(0)).skip(1));
    for ((&dtk, &tk), qkp1) in iter {
        apply_split_potential(&V_temp, &mut q_temp, dtk / 2.0);
        apply_split_kinetic(dx, &mut q_temp, dtk);
        V_temp = V(tk + dtk);
        apply_split_potential(&V_temp, &mut q_temp, dtk / 2.0);
        wf_renormalize(&mut q_temp, dx);
        q_temp.clone().move_into(qkp1);
    }
    q
}

fn apply_split_kinetic_imag<S>(dx: f64, q: &mut Arr1<S>, dt: f64)
where S: nd::DataMut<Elem = C64>
{
    let n = q.len();
    let m = if n % 2 == 0 { n / 2 } else { (n + 1) / 2 };
    let dk = TAU * (n as f64 * dx).recip();
    fft_inplace(q);
    q.iter_mut().enumerate()
        .for_each(|(i, qi)| {
            if (0..m).contains(&i) {
                *qi *= (-((i as f64) * dk).powi(2) * dt).exp();
            } else {
                *qi *= (-(((n - i) as f64) * dk).powi(2) * dt).exp();
            }
        });
    ifft_inplace(q);
}

fn apply_split_potential_imag<S, T>(V: &Arr1<S>, q: &mut Arr1<T>, dt: f64)
where
    S: nd::Data<Elem = f64>,
    T: nd::DataMut<Elem = C64>,
{
    q.iter_mut().zip(V)
        .for_each(|(qi, Vi)| { *qi *= (-Vi * dt).exp(); });
}

/// Perform imaginary-time split-step integration for a static potential over a
/// series of time coordinates.
pub fn split_step_imag<S, T, U>(
    dx: f64,
    V: &Arr1<S>,
    q0: &Arr1<T>,
    t: &Arr1<U>,
) -> nd::Array2<C64>
where
    S: nd::Data<Elem = f64>,
    T: nd::Data<Elem = C64>,
    U: nd::Data<Elem = f64>,
{
    let dt = array_diff(t);
    let mut q: nd::Array2<C64> = nd::Array2::zeros((t.len(), V.len()));
    let mut q_temp: nd::Array1<C64> = q0.to_owned();
    q.slice_mut(nd::s![0, ..]).assign(q0);
    let iter = dt.iter().zip(q.axis_iter_mut(nd::Axis(0)).skip(1));
    for (&dtk, qkp1) in iter {
        apply_split_potential_imag(V, &mut q_temp, dtk / 2.0);
        apply_split_kinetic_imag(dx, &mut q_temp, dtk);
        apply_split_potential_imag(V, &mut q_temp, dtk / 2.0);
        wf_renormalize(&mut q_temp, dx);
        q_temp.clone().move_into(qkp1);
    }
    q
}

