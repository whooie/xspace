//! Functions to compute solutions to the one-dimensional, time-independent
//! Schrödinger equation (TISE) for conservative potentials.

use std::cmp;
use ndarray as nd;
use ndarray_linalg::{ self as la, EighInto, EigValshInto, InverseInto };
use crate::{
    Arr1,
    error::{ LengthError, XError },
    interp,
    utils::{ wf_renormalize, wf_normalized },
    DEF_EPSILON,
    DEF_MAXITERS,
};

pub type XResult<T> = Result<T, XError>;

/// A single solution to the TISE.
///
/// This struct is usually only returned by a solver function; you probably
/// won't ever instantiate it yourself. The wavefunction is allowed to be
/// missing in the case that `compute_wf = false` is passed to a solver
/// function.
#[derive(Clone, Debug)]
pub struct Solution {
    /// Energy
    pub e: f64,
    /// Wavefunction
    pub wf: Option<nd::Array1<f64>>,
}

impl Solution {
    /// Compare two `Solution`s by their energy.
    pub fn cmp_energy(&self, other: &Self) -> Option<cmp::Ordering> {
        self.e.partial_cmp(&other.e)
    }

    /// Apply a scaling factor to the implicit length dimension associated with
    /// the wavefunction and energy.
    ///
    /// This performs the following:
    /// ```text
    /// wf → √a * wf
    /// e  → e / a²
    /// ```
    pub fn rescale(&mut self, a: f64) {
        self.e /= a.powi(2);
        if let Some(wf) = self.wf.as_mut() { *wf *= a.sqrt(); }
        // if let Some(x) = self.x.as_mut() { *x *= a; }
    }

    /// Like [`Self::rescale`], but create a copy of `self` with the specified
    /// scaling factor.
    pub fn rescaled(&self, a: f64) -> Self {
        let mut new = self.clone();
        new.rescale(a);
        new
    }
}

// fn arraydiff<S, A>(a: &Arr1<S>) -> nd::Array1<A>
// where
//     S: nd::Data<Elem = A>,
//     A: Num + Copy,
// {
//     a.iter().skip(1).zip(a)
//         .map(|(akp1, ak)| *akp1 - *ak)
//         .collect()
// }

#[derive(Copy, Clone, Debug)]
struct Window(f64, f64);

impl Window {
    fn push(&mut self, val: f64) { self.0 = self.1; self.1 = val; }
}

/// Perform a naive Numerov integration, starting on the left, for fixed energy.
///
/// Assumes `V` is sampled over even intervals.
///
/// *Panics if `V` has length less than 2*.
pub fn numerov<S>(dx: f64, V: &Arr1<S>, E: f64) -> nd::Array1<f64>
where S: nd::Data<Elem = f64>
{
    let a = dx.powi(2) / 12.0;
    let mut q: nd::Array1<f64> = nd::Array1::zeros(V.len());
    q[1] = dx;
    let mut qprev = Window(0.0, dx);
    let mut qnext: f64;
    let mut Tprev = Window(a * (E - V[0]), a * (E - V[1]));
    let mut Tnext: f64;
    for (qk, &Vk) in q.iter_mut().skip(2).zip(V.iter().skip(2)) {
        Tnext = a * Vk;
        qnext = (
            2.0 * (1.0 - 5.0 * Tprev.1) * qprev.1
            - (1.0 + Tprev.0) * qprev.0
        ) / (1.0 + a * Tnext);
        *qk = qnext;
        qprev.push(qnext);
        Tprev.push(Tnext);
    }
    wf_renormalize(&mut q, dx);
    q
}

// a = -dx²/12
fn renorm_pot_T(dx: f64, V: f64, E: f64) -> f64 {
    -dx.powi(2) / 12.0 * (E - V)
}

fn renorm_pot_U(dx: f64, V: f64, E: f64) -> f64 {
    let T = renorm_pot_T(dx, V, E);
    (2.0 + 10.0 * T) / (1.0 - T)
}

// left-to-right integration
// integration is performed for the whole `V` array
fn renorm_numerov_Rl<S>(dx: f64, V: &Arr1<S>, E: f64) -> nd::Array1<f64>
where S: nd::Data<Elem = f64>
{
    let mut Rl: nd::Array1<f64> = nd::Array1::zeros(V.len());
    Rl[0] = f64::INFINITY;
    let mut Rlprev: f64 = f64::INFINITY;
    let mut Rlnext: f64;
    let mut U: f64;
    for (Rlk, &Vk) in Rl.iter_mut().skip(1).zip(V) {
        U = renorm_pot_U(dx, Vk, E);
        Rlnext = U - Rlprev.recip();
        *Rlk = Rlnext;
        Rlprev = Rlnext;
    }
    Rl
}

// right-to-left integration
// integration is automatically halted at (and including) the first point for
// which the renormalized wavefunction is ≤1
fn renorm_numerov_Rr<S>(dx: f64, V: &Arr1<S>, E: f64) -> nd::Array1<f64>
where S: nd::Data<Elem = f64>
{
    let n = V.len();
    let mut Rr: nd::Array1<f64> = nd::Array1::zeros(V.len());
    Rr[n - 1] = f64::INFINITY;
    let mut Rrprev: f64 = f64::INFINITY;
    let mut Rrnext: f64;
    let mut U: f64;
    let mut m: usize = 0;
    for (k, (Rrk, &Vk)) in Rr.iter_mut().take(n - 1).zip(V).rev().enumerate() {
        U = renorm_pot_U(dx, Vk, E);
        Rrnext = U - Rrprev.recip();
        *Rrk = Rrnext;
        Rrprev = Rrnext;
        if Rrprev <= 1.0 {
            m = n - 2 - k;
            break;
        }
    }
    Rr.slice_collapse(nd::s![m..n]);
    Rr
}

fn renorm_numerov_ql<S, T>(dx: f64, V: &Arr1<S>, E: f64, Rl: &Arr1<T>)
    -> nd::Array1<f64>
where
    S: nd::Data<Elem = f64>,
    T: nd::Data<Elem = f64>,
{
    let n = Rl.len();
    let mut ql: nd::Array1<f64> = nd::Array1::zeros(n);
    let mut Tprev: f64 = renorm_pot_T(dx, V[n - 1], E);
    let mut Tnext: f64;
    let mut qlprev: f64 = (1.0 - Tprev).recip();
    let mut qlnext: f64;
    ql[n - 1] = qlprev;
    for ((qlk, &Vk), &Rlk) in ql.iter_mut().zip(V).zip(Rl).take(n - 1).rev() {
        Tnext = renorm_pot_T(dx, Vk, E);
        qlnext = (1.0 - Tprev) / (1.0 - Tnext) * qlprev / Rlk;
        *qlk = qlnext;
        Tprev = Tnext;
        qlprev = qlnext;
    }
    ql.iter_mut().for_each(|qlk| { if *qlk == f64::INFINITY { *qlk = 0.0; } });
    ql
}

fn renorm_numerov_qr<S, T>(dx: f64, V: &Arr1<S>, E: f64, Rr: &Arr1<T>)
    -> nd::Array1<f64>
where
    S: nd::Data<Elem = f64>,
    T: nd::Data<Elem = f64>,
{
    let n = Rr.len();
    let mut qr: nd::Array1<f64> = nd::Array1::zeros(n);
    let mut Tprev: f64 = renorm_pot_T(dx, V[0], E);
    let mut Tnext: f64;
    let mut qrprev: f64 = (1.0 - Tprev).recip();
    let mut qrnext: f64;
    qr[0] = qrprev;
    for ((qrk, &Vk), &Rrk) in qr.iter_mut().zip(V).zip(Rr).skip(1) {
        Tnext = renorm_pot_T(dx, Vk, E);
        qrnext = (1.0 - Tprev) / (1.0 - Tnext) * qrprev / Rrk;
        *qrk = qrnext;
        Tprev = Tnext;
        qrprev = qrnext;
    }
    qr.iter_mut().for_each(|qrk| { if *qrk == f64::INFINITY { *qrk = 0.0; } });
    qr
}

/// Integrate using the renormalized Numerov scheme for fixed energy.
///
/// Assumes `V` is sampled over even intervals.
///
/// *Panics if `V` has length less than 2*.
pub fn renorm_numerov<S>(dx: f64, V: &Arr1<S>, E: f64) -> nd::Array1<f64>
where S: nd::Data<Elem = f64>
{
    let n = V.len();
    let Rr = renorm_numerov_Rr(dx, V, E);
    let m = n - Rr.len();
    let Rl = renorm_numerov_Rl(dx, &V.slice(nd::s![0..m + 1]), E);
    let qr = renorm_numerov_qr(dx, &V.slice(nd::s![m..n]), E, &Rr);
    let ql = renorm_numerov_ql(dx, &V.slice(nd::s![0..m + 1]), E, &Rl);
    let mut q = nd::concatenate!(nd::Axis(0), ql, qr.slice(nd::s![1..]));
    wf_renormalize(&mut q, dx);
    q
}

/// Integrate using the renormalized Numerov method from the left-hand side of
/// the coordinate grid only.
///
/// This is intended for calculating scattering states, and the returned
/// wavefunction is not properly normalized.
pub fn renorm_numerov_scatter<S>(dx: f64, V: &Arr1<S>, E: f64)
    -> nd::Array1<f64>
where S: nd::Data<Elem = f64>
{
    let Rl = renorm_numerov_Rl(dx, V, E);
    renorm_numerov_ql(dx, V, E, &Rl)
}

/// Return the number of nodes in the wavefunction within the classically
/// allowed region.
pub fn node_count<S>(dx: f64, V: &Arr1<S>, E: f64) -> usize
where S: nd::Data<Elem = f64>
{
    // let Rl = renorm_numerov_Rl(r, V, E);
    // Rl.iter()
    //     .zip(V.iter().zip(V.iter().skip(1)))
    //     .filter(|(Rlk, (Vk, Vkp1))| {
    //         let Vkp1h = (*Vk + *Vkp1) / 2.0;
    //         Vkp1h < E && *Rlk < 0.0
    //     })
    //     .count()

    let q = renorm_numerov(dx, V, E);
    q.iter().zip(q.iter().skip(1))
        .zip(V.iter().zip(V.iter().skip(1)))
        .filter(|((qkm1, qk), (Vkm1, Vk))| {
            let Vkm1h = (*Vkm1 + *Vk) / 2.0;
            Vkm1h < E && *qkm1 * *qk <= 0.0
        })
        .count()
}

// calculate the matching criterion in the two-sided shooting method for a
// single energy
fn shoot_two_sided_single<S>(dx: f64, V: &Arr1<S>, E: f64) -> f64
where S: nd::Data<Elem = f64>
{
    let n = V.len();
    let Rr = renorm_numerov_Rr(dx, V, E);
    let m = cmp::max(n - Rr.len(), 2);
    let Rl = renorm_numerov_Rl(dx, &V.slice(nd::s![0..m + 1]), E);
    let Tmm1 = renorm_pot_T(dx, V[m - 1], E);
    let Tm = renorm_pot_T(dx, V[m], E);
    let Tmp1 = renorm_pot_T(dx, V[m + 1], E);
    (
        (0.5 - Tmp1) / (1.0 - Tmp1) * (Rr[1].recip() - Rl[m])
        - (0.5 - Tmm1) / (1.0 - Tmm1) * (Rr[0] - Rl[m - 1].recip())
    ) * (1.0 - Tm)
}

/// Calculate the matching criterion in the two-sided shooting method for a set
/// of energies.
pub fn shoot_two_sided<S, T>(dx: f64, V: &Arr1<S>, E: &Arr1<T>)
    -> nd::Array1<f64>
where
    S: nd::Data<Elem = f64>,
    T: nd::Data<Elem = f64>,
{
    E.mapv(|e| shoot_two_sided_single(dx, V, e))
}

/// Compute the (grid resolution-limited) spectrum for bound states in a
/// potential using the matrix Numerov method.
///
/// Assumes that `V` is sampled over even intervals.
///
/// Pass `compute_wf = false` to only calculate energies.
pub fn solve_matrix_numerov<S>(dx: f64, V: &Arr1<S>, compute_wf: bool)
    -> Vec<Solution>
where S: nd::Data<Elem = f64>
{
    let n = V.len();
    let ondx2 = dx.powi(2).recip();
    let mut A: nd::Array2<f64> = nd::Array2::from_diag_elem(n, -2.0 * ondx2);
    A.slice_mut(nd::s![1..n, 0..n - 1]).diag_mut().fill(ondx2);
    A.slice_mut(nd::s![0..n - 1, 1..n]).diag_mut().fill(ondx2);
    let mut B: nd::Array2<f64> = nd::Array2::from_diag_elem(n, -5.0 / 6.0);
    B.slice_mut(nd::s![1..n, 0..n - 1]).diag_mut().fill(12.0_f64.recip());
    B.slice_mut(nd::s![0..n - 1, 1..n]).diag_mut().fill(12.0_f64.recip());
    let Binv = B.inv_into().unwrap();
    let mut H: nd::Array2<f64> = -Binv.dot(&A);
    let mut H_diag = H.diag_mut();
    H_diag += V;
    if compute_wf {
        let (evals, evecs): (nd::Array1<f64>, nd::Array2<f64>)
            = H.eigh_into(la::UPLO::Lower).unwrap();
        evals.into_iter().zip(evecs.columns())
            .map(|(e, v)| Solution { e, wf: Some(wf_normalized(&v, dx)) })
            .collect()
    } else {
        let evals: nd::Array1<f64> = H.eigvalsh_into(la::UPLO::Lower).unwrap();
        evals.into_iter()
            .map(|e| Solution { e, wf: None })
            .collect()
    }
}

/// Find zero or more bound states in a potential using the two-sided shooting
/// method via root-finding performed on the matching criterion for a given set
/// of energies.
///
/// Assumes that `V` is sampled over even intervals.
///
/// Pass `compute_wf = false` to only calculate energies.
pub fn solve_shooting<S, T>(dx: f64, V: &Arr1<S>, E: &Arr1<T>, compute_wf: bool)
    -> XResult<Vec<Solution>>
where
    S: nd::Data<Elem = f64>,
    T: nd::Data<Elem = f64>,
{
    let dlog_diff: nd::Array1<f64> = shoot_two_sided(dx, V, E);
    let e: Vec<f64> = interp::find_zeros(E, &dlog_diff, interp::Zero::Rising)?;
    let sols: Vec<Solution>
        = e.into_iter()
        .map(|ek| {
            Solution {
                e: ek,
                wf: compute_wf.then(|| renorm_numerov(dx, V, ek)),
            }
        })
        .collect();
    Ok(sols)
}

#[derive(Copy, Clone, Debug)]
struct Bounds<T>(T, T);

impl Bounds<f64> {
    fn midpoint(self) -> f64 { (self.0 + self.1) / 2.0 }

    fn diff(self) -> f64 { self.1 - self.0 }
}

impl<T> Bounds<T> {
    fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> Bounds<U> {
        (f(self.0), f(self.1)).into()
    }
}

impl<T: PartialOrd> Bounds<T> {
    fn from_ord(xx: (T, T)) -> Self {
        if xx.0 > xx.1 { Self(xx.1, xx.0) } else { Self(xx.0, xx.1) }
    }
}

impl<T> From<(T, T)> for Bounds<T> {
    fn from(xx: (T, T)) -> Self { Self(xx.0, xx.1) }
}

/// Find a particular bound state in a potential, identified by its node count
/// `nu`, using a secant search applied to the matching criterion in the
/// two-sided shooting method.
///
/// This search is divided into two parts:
///
/// - A coarse, vibrational level search where initial energy bounds are
///   adjusted to be simply within the single energy range associated with a
///   particular node count.
/// - A finer, secant-method search where the matching criterion for the
///   two-sided shooting method is used to converge on a zero in the criterion
///   function.
///
/// The second part will iterate until a precision bound `epsilon > 0` is met;
/// both parts are limited to a maximum number of iterations `maxiters`. This
/// function will return [`XError::SecantVib`] if the first part is unable to
/// find the right vibrational level (this is usually because the correct energy
/// range is not included in the given initial bounds).
pub fn solve_secant<S>(
    dx: f64,
    V: &Arr1<S>,
    bounds: (f64, f64),
    nu: usize,
    epsilon: f64,
    maxiters: usize,
    compute_wf: bool,
) -> XResult<Solution>
where S: nd::Data<Elem = f64>
{
    XError::check_epsilon(epsilon)?;
    XError::check_maxiters(maxiters)?;

    let (mut E, mut Elast, mut D): (f64, f64, f64);
    let (mut Eb, mut Db, mut nb): (Bounds<f64>, Bounds<f64>, Bounds<usize>);
    let mut n: usize;
    let mut k: usize = 0;

    Eb = Bounds::from_ord(bounds);
    E = Eb.midpoint();
    nb = Eb.map(|e| node_count(dx, V, e));
    n = node_count(dx, V, E);
    for i in 0..maxiters {
        k = i;
        if nb.0 == nu && nb.1 == nu { break; }
        match n.cmp(&nu) {
            cmp::Ordering::Less => {
                Eb.0 = E;
                E = Eb.midpoint();
                n = node_count(dx, V, E);
            },
            cmp::Ordering::Greater => {
                Eb.1 = E;
                E = Eb.midpoint();
                n = node_count(dx, V, E);
            },
            cmp::Ordering::Equal => {
                if nb.0 < nu { Eb.0 = (Eb.0 + E) / 2.0; }
                if nb.1 < nu { Eb.1 = (E + Eb.1) / 2.0; }
            },
        }
        nb = Eb.map(|e| node_count(dx, V, e));
    }
    if nb.0 != nu || nb.1 != nu { return Err(XError::SecantVib); }
    if k >= maxiters - 1 {
        println!(
            "solve::solve_secant: WARNING: vibrational level search reached \
            maxiters"
        );
    }

    for i in 0..maxiters {
        k = i;
        Db = Eb.map(|e| shoot_two_sided_single(dx, V, e));
        while Db.0 == Db.1 {
            Eb.0 -= epsilon * Eb.0.abs();
            Eb.1 += epsilon * Eb.1.abs();
            Db = Eb.map(|e| shoot_two_sided_single(dx, V, e));
        }
        Elast = E;
        E = -Db.1 * (Eb.1 - Eb.0) / (Db.1 - Db.0) + Eb.1;
        n = node_count(dx, V, E);
        while n > nu {
            E = Eb.1 + (E - Eb.1).abs() / 2.0;
            n = node_count(dx, V, E);
        }
        while n < nu {
            E = Eb.0 - (E - Eb.0).abs() / 2.0;
            n = node_count(dx, V, E);
        }
        D = shoot_two_sided_single(dx, V, E);
        if Db.0 * Db.1 > 0.0 {
            Eb.0 = if Db.0 > 0.0 { E } else { Eb.1 };
            Eb.1 = if Db.0 > 0.0 { Eb.0 } else { E };
        } else if Db.0 * Db.1 < 0.0 {
            if D > 0.0 {
                Eb.1 = E;
            } else if D < 0.0 {
                Eb.0 = E;
            }
        } else {
            E = if Db.1 == 0.0 { Eb.1 } else { Eb.0 };
            break;
        }
        if ((Eb.diff() / Eb.midpoint()).abs() < epsilon && Db.0 * Db.1 < 0.0)
            || ((E - Elast) / Eb.midpoint()).abs() < epsilon
        { break; }
    }
    if k == maxiters - 1 {
        println!(
            "solve::solve_secant: WARNING: energy convergence reached maxiters"
        );
    }

    if compute_wf {
        Ok(Solution { e: E, wf: Some(renorm_numerov(dx, V, E)) })
    } else {
        Ok(Solution { e: E, wf: None })
    }
}

/// Solving method selector and parameters.
#[derive(Clone, Debug)]
pub enum Method {
    /// Use the [matrix Numerov method][solve_matrix_numerov].
    MatrixNumerov,
    /// Use the [two-sided shooting method][solve_shooting].
    Shooting {
        /// Energy range over which to compute the [matching
        /// criterion][shoot_two_sided].
        E: nd::Array1<f64>,
    },
    /// Use the [secant search-based shooting method][solve_secant].
    Secant {
        /// Initial guess bounds on energy.
        bounds: (f64, f64),
        /// Desired vibrational level (node count).
        nu: usize,
        /// Desired accuracy bound (default: `1e-6`).
        epsilon: Option<f64>,
        /// Maximum number of iterations (default: `1000`).
        maxiters: Option<usize>,
    },
}

impl Method {
    /// Return `true` if `self` is `MatrixNumerov`.
    pub fn is_matrix_numerov(&self) -> bool {
        matches!(self, Self::MatrixNumerov)
    }

    /// Return `true` if `self` is `Shooting`.
    pub fn is_shooting(&self) -> bool {
        matches!(self, Self::Shooting { .. })
    }

    /// Return `true` if `self` is `Secant`.
    pub fn is_secant(&self) -> bool {
        matches!(self, Self::Secant { .. })
    }
}

/// Master solving function for all [methods][Method].
pub fn solve<S>(dx: f64, V: &Arr1<S>, method: Method, compute_wf: bool)
    -> XResult<Vec<Solution>>
where S: nd::Data<Elem = f64>
{
    match method {
        Method::MatrixNumerov => {
            let sols = solve_matrix_numerov(dx, V, compute_wf);
            Ok(sols)
        },
        Method::Shooting { E } => {
            solve_shooting(dx, V, &E, compute_wf)
        },
        Method::Secant { bounds, nu, epsilon, maxiters } => {
            let sol = solve_secant(
                dx,
                V,
                bounds,
                nu,
                epsilon.unwrap_or(DEF_EPSILON),
                maxiters.unwrap_or(DEF_MAXITERS),
                compute_wf,
            );
            sol.map(|s| vec![s])
        },
    }
}

/// Simple record to keep track of coordinate and potential arrays.
///
/// Arrays borrowed from this type are guaranteed to have the same length and to
/// sampled (or generated) for a coordinate grid with uniform spacing.
#[derive(Clone, Debug)]
pub struct System {
    // coordinate array
    x: nd::Array1<f64>,
    // coordinate array grid spacing
    dx: f64,
    // potential array
    V: nd::Array1<f64>,
    // array sizes
    n: usize,
}

impl System {
    /// Create a new `System`, generating the coordinate array from
    /// "linspace-style" arguments (start, inclusive end, and an array length).
    ///
    /// *Panics if the number of points is less than 2*.
    pub fn new_linspace<F>(xargs: (f64, f64, usize), V: F) -> Self
    where F: FnMut(f64) -> f64
    {
        let x: nd::Array1<f64>
            = nd::Array1::linspace(xargs.0, xargs.1, xargs.2);
        let dx = x[1] - x[0];
        let V: nd::Array1<f64> = x.mapv(V);
        let n = xargs.2;
        Self { x, dx, V, n }
    }

    /// Create a new `System`, generating the coordinate array from
    /// "range-style" arguments (start, exclusive end, and a step size).
    pub fn new_range<F>(xargs: (f64, f64, f64), V: F) -> Self
    where F: FnMut(f64) -> f64
    {
        let x: nd::Array1<f64>
            = nd::Array1::range(xargs.0, xargs.1, xargs.2);
        let dx = xargs.2;
        let V: nd::Array1<f64> = x.mapv(V);
        let n = x.len();
        Self { x, dx, V, n }
    }

    /// Create a new `System` from bare coordinate and potential arrays.
    ///
    /// *Panics if the number of points is less than 2*.
    pub fn new_arrays(x: nd::Array1<f64>, V: nd::Array1<f64>) -> XResult<Self> {
        LengthError::check(&x, &V)?;
        let dx = x[1] - x[0];
        let n = x.len();
        Ok(Self { x, dx, V, n })
    }

    /// Get a reference to the coordinate array.
    pub fn get_x(&self) -> &nd::Array1<f64> { &self.x }

    /// Get a reference to the potential array.
    pub fn get_V(&self) -> &nd::Array1<f64> { &self.V }

    /// Get the coordinate array grid spacing.
    pub fn get_dx(&self) -> f64 { self.dx }

    /// Get the length of the coordinate and potential arrays.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize { self.n }

    /// Thin interface to [`solve`].
    pub fn solve(&self, method: Method, compute_wf: bool)
        -> XResult<Vec<Solution>>
    {
        solve(self.dx, &self.V, method, compute_wf)
    }

    /// Apply a scaling factor the coordinate array, propagating the factor to
    /// the implicit length dimension associated with the potential as well.
    ///
    /// This performs the following:
    /// ```text
    /// x → a * x
    /// V → V / a²
    /// ```
    pub fn rescale(&mut self, a: f64) {
        self.x *= a;
        self.V /= a.powi(2);
    }

    /// Like [`Self::rescale`], but create a copy of `self` with the specified
    /// scaling factor.
    pub fn rescaled(&self, a: f64) -> Self {
        let mut new = self.clone();
        new.rescale(a);
        new
    }
}

