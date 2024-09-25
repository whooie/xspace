use std::path::PathBuf;
use ndarray as nd;
use whooie::{ mkdir, write_npz, utils::FExtremum };
use xspace::solve;

fn main() {
    let a: f64 = 1.0; // potential parameter
    let x: nd::Array1<f64> = nd::Array1::linspace(-10.0, 10.0, 3001);
    let dx = x[1] - x[0];
    let v: nd::Array1<f64> = x.mapv(|xk| a * xk.powi(4));
    let e: nd::Array1<f64>
        = nd::Array1::linspace(0.0, v.fmax().unwrap(), 50000);
    let d: nd::Array1<f64> = solve::shoot_two_sided(dx, &v, &e);

    let outdir = PathBuf::from("output");
    mkdir!(outdir);
    write_npz!(
        outdir.join("quartic_shooting.npz"),
        arrays: {
            "e" => &e,
            "d" => &d,
        }
    );
}

