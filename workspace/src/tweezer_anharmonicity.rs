#![allow(unused_imports)]
#![allow(dead_code, unused_variables, unused_mut)]

use std::path::PathBuf;
use ndarray as nd;
use whooie::{ mkdir, write_npz };
use xspace::{ interp, solve, units };


const MASS: f64 = 5.00823449476748e-27; // ³He; kg
const LSCALE: f64 = 1e-6; // length scale; m
const DEPTH: f64 = 1.0; // MHz
const WAIST: f64 = 1.0; // μm

fn solve_wfs() {
    let uu = units::Units::from_mks(MASS, LSCALE);
    let u0 = DEPTH * 1e6 * units::h / uu.e;
    let w0 = WAIST / 2.0 * 1e-6 / uu.a;

    let x: nd::Array1<f64> = nd::Array1::linspace(-5.0 * w0, 5.0 * w0, 5501);
    let dx = x[1] - x[0];
    let v: nd::Array1<f64> =
        x.mapv(|xk| u0 * (1.0 - (-2.0 * (xk / w0).powi(2)).exp()));

    let escan: nd::Array1<f64> = nd::Array1::linspace(0.0, u0, 10000);
    let sols: Vec<solve::Solution> =
        solve::solve_shooting(dx, &v, &escan, true).unwrap();
    println!("{} solutions found", sols.len());
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
        outdir.join("he-tweezer.npz"),
        arrays: {
            "x" => &x,
            "v" => &v,
            "e" => &energies,
            "enat" => &nd::array![uu.e],
            "a" => &nd::array![uu.a],
            "wf" => &wfs,
        }
    );
}

#[derive(Copy, Clone, Debug)]
struct Qutrit {
    e0: f64,
    e1: f64,
    e2: f64,
    uu: units::Units,
}

// expect depth in MHz
fn solve_depth(depth: f64) -> Option<Qutrit> {
    let uu = units::Units::from_mks(MASS, LSCALE);
    let u0 = depth * 1e6 * units::h / uu.e;
    let w0 = WAIST / 2.0 * 1e-6 / uu.a;

    let x: nd::Array1<f64> = nd::Array1::linspace(-3.0 * w0, 3.0 * w0, 5001);
    let dx = x[1] - x[0];
    let v: nd::Array1<f64> =
        x.mapv(|xk| u0 * (1.0 - (-2.0 * (xk / w0).powi(2)).exp()));

    let escan: nd::Array1<f64> = nd::Array1::linspace(0.0, u0, 10000);
    let sols: Vec<solve::Solution> =
        solve::solve_shooting(dx, &v, &escan, false).unwrap();
    if sols.len() >= 3 {
        let result = Qutrit { e0: sols[0].e, e1: sols[1].e, e2: sols[2].e, uu };
        Some(result)
    } else {
        None
    }
}

fn anharmonicity_depth_scan() {
    let u0: nd::Array1<f64> = nd::Array1::linspace(0.1, 1.0, 100); // MHz
    eprint!("  0 / {}", u0.len());
    let anharm: nd::Array1<f64> =
        u0.iter().enumerate()
        .map(|(k, u0_k)| {
            let mb_sol = solve_depth(*u0_k);
            eprint!("\r  {} / {}", k + 1, u0.len());
            mb_sol
                .map(|sol| (sol.e1 - sol.e0) / (sol.e2 - sol.e1) - 1.0)
                .unwrap_or(f64::INFINITY)
        })
        .collect();
    eprintln!();
    let outdir = PathBuf::from("output");
    mkdir!(outdir);
    write_npz!(
        outdir.join("he-tweezer-anharm.npz"),
        arrays: {
            "depth" => &u0,
            "anharm" => &anharm,
            "waist" => &nd::array![WAIST],
            "mass" => &nd::array![MASS],
            "lscale" => &nd::array![LSCALE],
        }
    );
}

fn main() {
    solve_wfs();
    // anharmonicity_depth_scan();
}

