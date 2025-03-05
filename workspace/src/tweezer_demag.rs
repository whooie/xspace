#![allow(dead_code)]

use std::{ f64::consts::PI, path::PathBuf };
use ndarray as nd;
use num_complex::Complex64 as C64;
#[allow(unused_imports)]
use rand::{ thread_rng, Rng };
use whooie::{ mkdir, read_npz, write_npz };
use xspace::{ solve, timedep, units };

const MASS: f64 = 2.8384644058191703e-25; // ¹⁷¹Yb; kg
const TRAP_FREQ: f64 = 2.0 * PI * 30e3; // s⁻¹
const WAIST0: f64 = 1.0e-6; // μm
const DEPTH: f64 = WAIST0 * WAIST0 * MASS * TRAP_FREQ * TRAP_FREQ / 16.0;
const R: f64 = 100.0; // ratio between "k-lattice" tweezer waists
const Z: f64 = 1.25; // ratio between asymmetric potential trap frequencies
const TEMPERATURE: f64 = 1e-6; // K
const THERM_CUTOFF: f64 = 1e-6; // ratio P(nmax) / P(0)
const MC: usize = 20; // # monte carlo draws for thermal state evolutions

fn tweezer(depth: f64, waist: f64, x: f64) -> f64 {
    depth * (1.0 - (-2.0 * (x / (waist / 2.0)).powi(2)).exp())
}

fn get_units() -> units::Units {
    let gs_length = (units::hbar / MASS / TRAP_FREQ).sqrt();
    units::Units::from_mks(MASS, gs_length)
}

fn gen_basis() {
    let uu = get_units();
    // maximum vibrational mode
    let nmax = (
        -units::kB * TEMPERATURE / units::hbar / TRAP_FREQ * THERM_CUTOFF.ln()
    ).ceil() as usize * 2;
    // minimum local wavelength
    let lmin =
        2.0 * PI
        / (2.0 * MASS * TRAP_FREQ * (nmax as f64 + 0.5) / units::hbar).sqrt();

    let hw = units::hbar * TRAP_FREQ / uu.e;
    let e = move |n: usize| hw * (0.5 + n as f64);

    let dx = lmin / 1000.0 / uu.a;
    let xmin = -1.0e-6 / uu.a;
    let xmax =  1.0e-6 / uu.a;
    let npoints = ((xmax - xmin) / dx).ceil() as usize + 1;
    let x: nd::Array1<f64> = nd::Array1::linspace(xmin, xmax, npoints);

    let v: nd::Array1<f64> =
        x.mapv(|xk| tweezer(DEPTH / uu.e, WAIST0 / uu.a, xk));

    let sols: Vec<solve::Solution> =
        (0..=nmax).map(|n| {
            eprint!("\r  {} / {} ", n, nmax);
            // let bounds = (e(n) - hw / 4.0, e(n) + hw / 4.0);
            let bounds = (0.0, e(nmax + 1));
            solve::solve_secant(dx, &v, bounds, n, 1e-6, 10000, true).unwrap()
        })
        .collect();
    eprintln!();

    let energies: nd::Array1<f64> = sols.iter().map(|sol| sol.e).collect();
    let wfs: nd::Array2<f64> =
        nd::stack(
            nd::Axis(0),
            &sols.iter()
                .map(|sol| sol.wf.as_ref().unwrap().view())
                .collect::<Vec<_>>(),
        )
        .unwrap();

    let outdir = PathBuf::from("output");
    mkdir!(outdir);
    write_npz!(
        outdir.join("tweezer.npz"),
        arrays: {
            "x" => &x,
            "v" => &v,
            "e" => &energies,
            "wf" => &wfs,
        }
    );
}

#[derive(Clone, Debug, PartialEq)]
struct Basis {
    x: nd::Array1<f64>,
    e: nd::Array1<f64>,
    wfs: nd::Array2<C64>,
}

fn get_basis() -> Basis {
    let (x, e, wfs): (nd::Array1<f64>, nd::Array1<f64>, nd::Array2<f64>) =
        read_npz!(
            PathBuf::from("output").join("tweezer.npz"),
            arrays: { "x", "e", "wf" }
        );
    let wfs = wfs.mapv(C64::from);
    Basis { x, e, wfs }
}

#[derive(Clone, Debug, PartialEq)]
struct Wf {
    x: nd::Array1<f64>,
    q: nd::Array1<C64>,
}

fn gen_phase<R>(rng: &mut R) -> C64
where R: Rng + ?Sized
{
    C64::cis(2.0 * PI * rng.gen::<f64>())
}

fn gen_thermal_state<R>(basis: &Basis, rng: &mut R) -> Wf
where R: Rng + ?Sized
{
    let uu = get_units();
    let beta = uu.e / units::kB / TEMPERATURE;

    let mut amps: nd::Array1<C64> =
        basis.e.mapv(|ek| (-beta * ek).exp().into());
    let z = amps.sum();
    amps.mapv_inplace(|pk| gen_phase(rng) * (pk / z).sqrt());
    let q: nd::Array1<C64> =
        nd::Zip::from(amps.view()).and(basis.wfs.outer_iter())
            .fold(
                nd::Array1::<C64>::zeros(basis.x.len()),
                |mut acc, amp, wf| { acc += &(&wf * *amp); acc },
            );
    Wf { x: basis.x.to_owned(), q }
}

fn interp_shift(wf: &Wf, x_new: nd::Array1<f64>, sh: f64) -> Wf {
    let mut q_new: nd::Array1<C64> = nd::Array1::zeros(x_new.len());
    nd::Zip::from(x_new.view()).and(&mut q_new)
        .for_each(|xk, qk| {
            let xk_sh = *xk - sh;
            let mb_bounds =
                wf.x.iter().zip(wf.q.iter())
                .zip(wf.x.iter().zip(wf.q.iter()).skip(1))
                .find(|((xl, _), (xr, _))| **xl <= xk_sh && xk_sh <= **xr)
                .map(|((xl, ql), (xr, qr))| ((*xl, *ql), (*xr, *qr)));
            if let Some(((xl, ql), (xr, qr))) = mb_bounds {
                *qk = ql + (qr - ql) / (xr - xl) * (xk_sh - xl);
            }
        });
    Wf { x: x_new, q: q_new }
}

#[derive(Clone, Debug, PartialEq)]
struct TParams {
    x: nd::Array1<f64>, // length n
    dx: f64,
    v: nd::Array1<f64>, // length n
    t: nd::Array1<f64>, // length m
}

fn gen_timedep_params() -> TParams {
    let uu = get_units();

    /* demag potentials */
    let mut x: nd::Array1<f64> =
        nd::Array1::linspace(-(R / Z) * WAIST0, R * WAIST0, 10000);
    x.mapv_inplace(|xk| xk / uu.a);
    let v: nd::Array1<f64> =
        x.mapv(|xk| {
            if xk < 0.0 {
                0.5 * MASS * (TRAP_FREQ * (Z / R) * uu.a * xk).powi(2) / uu.e
                // tweezer(DEPTH / uu.e, (Z / R) * WAIST0 / uu.a, xk)
            } else {
                0.5 * MASS * (TRAP_FREQ / R * uu.a * xk).powi(2) / uu.e
                // tweezer(DEPTH / uu.e, R * WAIST0 / uu.a, xk)
            }
        });

    /* symmetric potentials */
    // let mut x: nd::Array1<f64> =
    //     nd::Array1::linspace(-R * WAIST0, R * WAIST0, 10000);
    // x.mapv_inplace(|xk| xk / uu.a);
    // let v: nd::Array1<f64> =
    //     x.mapv(|xk| {
    //         // 0.5 * MASS * (TRAP_FREQ / R * uu.a * xk).powi(2) / uu.e
    //         tweezer(DEPTH / uu.e, R * WAIST0 / uu.a, xk)
    //     });
    
    let dx = x[1] - x[0];
    let tmax = 1.5 * (2.0 * PI / TRAP_FREQ * R) / uu.t;
    let t: nd::Array1<f64> = nd::Array1::linspace(0.0, tmax, 1000);
    TParams { x, dx, v, t }
}

fn evolve_thermal_state<R>(
    basis: &Basis,
    params: TParams,
    sh: f64,
    rng: &mut R,
) -> nd::Array3<C64>
where R: Rng + ?Sized
{
    let mut runs: nd::Array3<C64> =
        nd::Array3::zeros((MC, params.t.len(), params.x.len()));
    let mut wf: Wf;
    eprint!(" {:2} ", 0);
    for (k, mut run) in runs.outer_iter_mut().enumerate() {
        wf = gen_thermal_state(basis, rng);
        wf = interp_shift(&wf, params.x.clone(), sh);
        timedep::split_step_const(params.dx, &params.v, &wf.q, &params.t)
            .move_into(&mut run);
        eprint!("\x1b[3D{:2} ", k + 1);
    }
    runs
}

fn evolve_sites() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let uu = get_units();
    let mut rng = thread_rng();
    let basis = get_basis();
    let mut sites_x: nd::Array1<f64> =
        nd::array![
            1.0,
            5.0,
            10.0,
            15.0,
            20.0,
            // 30.0,
            // 40.0,
            // 50.0,
        ]; // μm
    sites_x.mapv_inplace(|xk| xk * 1e-6 / uu.a);

    let tparams = gen_timedep_params();
    eprint!("  0 ");
    for (k, sh) in sites_x.iter().enumerate() {
        let outfile =
            outdir.join(format!("sites_ev_sh={:.1}.npz", *sh * uu.a * 1e6));
        let q = evolve_thermal_state(&basis, tparams.clone(), *sh, &mut rng);
        write_npz!(
            outfile,
            arrays: {
                "x" => &tparams.x,
                "v" => &tparams.v,
                "t" => &tparams.t,
                "sh" => &nd::array![*sh],
                "q" => &q,
            }
        );
        eprint!("\r  {} ", k + 1);
    }
    eprintln!();
}

fn main() {
    // gen_basis();
    evolve_sites();
}

