use std::{ f64::consts::PI, path::PathBuf };
use ndarray as nd;
use num_complex::Complex64 as C64;
use whooie::write_npz;
use xspace::{ solve, timedep, units };

// solve for eigenstates of the quantum harmonic oscillator

fn main() {
    const MASS: f64 = 2.8384644058191703e-25; // ¹⁷¹Yb; kg
    const TRAP_FREQ: f64 = 2.0 * PI * 30e3; // s⁻¹
    const TARGET_N: usize = 5; // target number of nodes

    // characteristic length scale of the ground state
    let gs_length = (units::hbar / MASS / TRAP_FREQ).sqrt();
    // create a unit system based on this length, normalizing:
    // * state energies to (2n + 1)
    // * turning points to ±√(2n + 1)
    let uu = units::Units::from_mks(MASS, gs_length);

    let hw = units::hbar * TRAP_FREQ / uu.e;
    let e = move |n: usize| units::hbar * TRAP_FREQ * (0.5 + n as f64) / uu.e;

    // coordinate and potential arrays
    let x: nd::Array1<f64> = nd::Array1::linspace(-10.0, 10.0, 3000);
    let dx = x[1] - x[0];
    let v: nd::Array1<f64>
        = x.mapv(|xk| {
            0.5 * MASS * TRAP_FREQ.powi(2) * (gs_length * xk).powi(2) / uu.e
        });

    // solve for the target solution
    let bounds = (e(TARGET_N) - hw / 4.0, e(TARGET_N) + hw / 4.0);
    let sol = solve::solve_secant(
        dx, &v, bounds, TARGET_N, 1e-6, 1000, true).unwrap();
    println!("expected: {:.3e}", e(TARGET_N));
    println!("computed: {:.3e}", sol.e);

    // do the same for a horizontally shifted potential
    let v_shifted: nd::Array1<f64>
        = x.mapv(|xk| {
            0.5 * MASS * TRAP_FREQ.powi(2) * (gs_length * (xk + 2.5)).powi(2)
                / uu.e
        });
    let sol_shifted = solve::solve_secant(
        dx, &v_shifted, bounds, TARGET_N, 1e-6, 1000, true).unwrap();

    // use the shifted solution as the initial state for time evolution in the
    // original potential
    let period = 2.0 * PI / TRAP_FREQ / uu.t;
    let t: nd::Array1<f64>
        = nd::Array1::linspace(0.0, 2.0 * period, 3000);
    let q0: nd::Array1<C64> = sol_shifted.wf.as_ref().unwrap().mapv(C64::from);
    let q: nd::Array2<C64> = timedep::split_step_const(dx, &v, &q0, &t);

    write_npz!(
        PathBuf::from("qho.npz"),
        arrays: {
            "x" => &x,
            "v" => &v,
            "e" => &nd::array![sol.e],
            "wf" => sol.wf.as_ref().unwrap(),
            "v_shifted" => &v_shifted,
            "wf_shifted" => &sol_shifted.wf.as_ref().unwrap(),
            "t" => &t,
            "q" => &q,
        }
    );
}

