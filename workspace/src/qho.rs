use std::{ f64::consts::PI, path::PathBuf };
use ndarray as nd;
use num_complex::Complex64 as C64;
use whooie::{ mkdir, write_npz };
use xspace::{ solve, timedep, units };

const MASS: f64 = 2.8384644058191703e-25; // ¹⁷¹Yb; kg
const TRAP_FREQ: f64 = 2.0 * PI * 30e3; // s⁻¹

fn main() {
    let gs_length = (units::hbar / MASS / TRAP_FREQ).sqrt();
    let uu = units::Units::from_mks(MASS, gs_length);

    let hw = units::hbar * TRAP_FREQ / uu.e;
    let e = move |n: usize| hw * (0.5 + n as f64);
    let x: nd::Array1<f64> = nd::Array1::linspace(-10.0, 10.0, 3001);
    let dx = x[1] - x[0];
    let v: nd::Array1<f64>
        = x.mapv(|xk| 0.5 * MASS * (TRAP_FREQ * gs_length * xk).powi(2) / uu.e);

    let sols: Vec<solve::Solution>
        = (0..=10).map(|n| {
            let bounds = (e(n) - hw / 4.0, e(n) + hw / 4.0);
            solve::solve_secant(dx, &v, bounds, n, 1e-6, 1000, true).unwrap()
        })
        .collect();
    let energies: nd::Array1<f64> = sols.iter().map(|sol| sol.e).collect();
    let wfs: nd::Array2<f64>
        = nd::stack(
            nd::Axis(0),
            &sols.iter()
                .map(|sol| sol.wf.as_ref().unwrap().view())
                .collect::<Vec<_>>(),
        ).unwrap();

    let v_sh: nd::Array1<f64>
        = x.mapv(|xk| {
            0.5 * MASS * (TRAP_FREQ * gs_length * (xk + 2.5)).powi(2) / uu.e
        });
    let gs_sh: solve::Solution = solve::solve_secant(
        dx, &v_sh, (0.25 * hw, 0.75 * hw), 0, 1e-6, 1000, true).unwrap();

    let period = 2.0 * PI / TRAP_FREQ / (uu.t / 2.0 / PI);
    let t: nd::Array1<f64>
        = nd::Array1::linspace(0.0, 2.0 * period, 3000);
    let q0: nd::Array1<C64> = gs_sh.wf.as_ref().unwrap().mapv(C64::from);
    let q: nd::Array2<C64> = timedep::split_step_const(dx, &v, &q0, &t);

    let outdir = PathBuf::from("output");
    mkdir!(outdir);
    write_npz!(
        outdir.join("qho.npz"),
        arrays: {
            "x" => &x,
            "v" => &v,
            "e" => &energies,
            "wf" => &wfs,
            "v_sh" => &v_sh,
            "gs_sh" => gs_sh.wf.as_ref().unwrap(),
            "t" => &t,
            "q" => &q,
        }
    );
}

