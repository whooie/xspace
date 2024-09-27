use std::{ f64::consts::TAU, path::PathBuf };
use ndarray as nd;
use num_complex::Complex64 as C64;
use whooie::{ mkdir, write_npz };
use xspace::{ solve, timedep, units, utils };

const MASS: f64 = 5.00823449476748e-27; // ³He; kg
const LSCALE: f64 = 1e-6; // length scale; m

fn main() {
    let outdir = PathBuf::from("output");
    mkdir!(outdir);

    let uu = units::Units::from_mks(MASS, LSCALE);
    let a: f64 = 10500.0; // base potential parameter
    let x: nd::Array1<f64> = nd::Array1::linspace(-1.5, 1.5, 3001);
    let dx = x[1] - x[0];
    let v_static: nd::Array1<f64> = x.mapv(|xk| a * xk.powi(4));

    let escan: nd::Array1<f64> = nd::Array1::linspace(0.0, 200.0, 1000);
    let sols: Vec<solve::Solution>
        = solve::solve_shooting(dx, &v_static, &escan, true).unwrap();
    if sols.len() < 3 { panic!("couldn't find enough solutions"); }

    let q0: nd::Array1<C64> = sols[0].wf.as_ref().unwrap().mapv(C64::from);
    let q1: nd::Array1<C64> = sols[1].wf.as_ref().unwrap().mapv(C64::from);
    let q2: nd::Array1<C64> = sols[2].wf.as_ref().unwrap().mapv(C64::from);
    let fdiff = (sols[1].e - sols[0].e) * uu.e / units::h * uu.t;

    // 0.999869
    let ampl: f64 = 0.001;
    let freq: f64 = 0.999024 * fdiff;
    let t: nd::Array1<f64>
        = nd::Array1::linspace(0.0, 150.0 / fdiff, 10000);
    //
    // 0.999305
    // let ampl: f64 = 0.0025;
    // let freq: f64 = 0.998313 * fdiff;
    // let t: nd::Array1<f64>
    //     = nd::Array1::linspace(0.0, 75.0 / fdiff, 7000);
    //
    // 0.996880
    // let ampl: f64 = 0.005;
    // let freq: f64 = 0.997243 * fdiff;
    // let t: nd::Array1<f64>
    //     = nd::Array1::linspace(0.0, 40.0 / fdiff, 6500);
    //
    // 0.993217
    // let ampl: f64 = 0.0075;
    // let freq: f64 = 0.996000 * fdiff;
    // let t: nd::Array1<f64>
    //     = nd::Array1::linspace(0.0, 35.0 / fdiff, 6000);
    //
    // 0.989240
    // let ampl: f64 = 0.01;
    // let freq: f64 = 0.994613 * fdiff;
    // let t: nd::Array1<f64>
    //     = nd::Array1::linspace(0.0, 30.0 / fdiff, 5000);
    //

    eprintln!("{:.6}", freq / fdiff);

    let vt: nd::Array2<f64>
        = t.iter().copied()
        .flat_map(|ti| {
            x.iter().copied()
                .map(move |xj| {
                    a * (xj - ampl * (TAU * freq * ti).sin()).powi(4)
                })
        })
        .collect::<nd::Array1<f64>>()
        .into_shape((t.len(), x.len())).unwrap();
    let q: nd::Array2<C64> = timedep::split_step(dx, &vt, &q0, &t);
    let a0: nd::Array1<C64>
        = q.outer_iter()
        .map(|qi| utils::wf_dot(&q0, &qi, dx))
        .collect();
    let a1: nd::Array1<C64>
        = q.outer_iter()
        .map(|qi| utils::wf_dot(&q1, &qi, dx))
        .collect();
    let a2: nd::Array1<C64>
        = q.outer_iter()
        .map(|qi| utils::wf_dot(&q2, &qi, dx))
        .collect();

    write_npz!(
        outdir.join("quartic_xmod.npz"),
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

