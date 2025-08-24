use std::{ f64::consts::TAU, path::PathBuf };
use ndarray as nd;
use num_complex::Complex64 as C64;
use whooie::{ mkdir, write_npz };
use xspace::{ solve, timedep, units, utils };

const MASS: f64 = 5.00823449476748e-27; // ³He; kg
const LSCALE: f64 = 1e-6; // length scale; m
const DEPTH: f64 = 0.3; // MHz
const WAIST: f64 = 1.0; // μm

fn main() {
    let uu = units::Units::from_mks(MASS, LSCALE);
    let u0 = DEPTH * 1e6 * units::h / uu.e;
    let w0 = WAIST / 2.0 * 1e-6 / uu.a;

    let x: nd::Array1<f64> = nd::Array1::linspace(-3.0 * w0, 3.0 * w0, 3001);
    let dx = x[1] - x[0];
    let v_static: nd::Array1<f64> =
        x.mapv(|xk| u0 * (1.0 - (-2.0 * (xk / w0).powi(2)).exp()));
    let escan: nd::Array1<f64> = nd::Array1::linspace(0.0, u0, 10000);
    let sols: Vec<solve::Solution> =
        solve::solve_shooting(dx, &v_static, &escan, true).unwrap();
    if sols.len() < 3 { panic!("couldn't find enough solutions"); }

    eprintln!("E0 = {:.6}", sols[0].e);
    eprintln!("E1 = {:.6}", sols[1].e);
    eprintln!("E2 = {:.6}", sols[2].e);
    eprintln!("(E1 - E0)/(E2 - E1) = {:.6}",
        (sols[1].e - sols[0].e) / (sols[2].e - sols[1].e));

    let q0: nd::Array1<C64> = sols[0].wf.as_ref().unwrap().mapv(C64::from);
    let q1: nd::Array1<C64> = sols[1].wf.as_ref().unwrap().mapv(C64::from);
    let q2: nd::Array1<C64> = sols[2].wf.as_ref().unwrap().mapv(C64::from);
    let fdiff = (sols[2].e - sols[0].e) * uu.e / units::h * uu.t;

    // let ampl: f64 = 0.7045;
    let ampl: f64 = 0.0010;
    let freq: f64 = 1.00049 * fdiff;

    let t: nd::Array1<f64> =
        nd::Array1::linspace(0.0, 3000.0 / fdiff, 75000);
    eprintln!("ampl = {:.6}", ampl);
    eprintln!("f/f0 = {:.6}", freq / fdiff);

    let vt: nd::Array2<f64> =
        t.iter().copied()
        .flat_map(|ti| &v_static * (1.0 + ampl * (TAU * freq * ti).sin()))
        .collect::<nd::Array1<f64>>()
        .into_shape((t.len(), x.len())).unwrap();
    let q: nd::Array2<C64> = timedep::split_step(dx, &vt, &q0, &t);
    let a0: nd::Array1<C64> =
        q.outer_iter()
        .map(|qi| utils::wf_dot(&q0, &qi, dx))
        .collect();
    let a1: nd::Array1<C64> =
        q.outer_iter()
        .map(|qi| utils::wf_dot(&q1, &qi, dx))
        .collect();
    let a2: nd::Array1<C64> =
        q.outer_iter()
        .map(|qi| utils::wf_dot(&q2, &qi, dx))
        .collect();

    let outdir = PathBuf::from("output");
    mkdir!(outdir);
    write_npz!(
        outdir.join("tweezer_amod.npz"),
        arrays: {
            "mass" => &nd::array![MASS],
            "anat" => &nd::array![uu.a],
            "enat" => &nd::array![uu.e],
            "tnat" => &nd::array![uu.t],
            "x" => &x,
            "v" => &v_static,
            "q0" => &q0,
            "e0" => &nd::array![sols[0].e],
            "q1" => &q1,
            "e1" => &nd::array![sols[1].e],
            "q2" => &q2,
            "e2" => &nd::array![sols[2].e],
            "fdiff" => &nd::array![fdiff],
            "ampl" => &nd::array![ampl],
            "freq" => &nd::array![freq],
            "t" => &t,
            "vt" => &vt,
            "q" => &q,
            "a0" => &a0,
            "a1" => &a1,
            "a2" => &a2,
        }
    );
}

