use std::path::PathBuf;
use ndarray as nd;
use whooie::{ mkdir, write_npz };
use xspace::{ interp, solve, units };

const MASS: f64 = 5.00823449476748e-27; // ³He; kg
const LSCALE: f64 = 1e-6; // length scale; m

fn main() {
    let uu = units::Units::from_mks(MASS, LSCALE);
    let a: f64 = 1.6e-3; // potential parameter
    let x: nd::Array1<f64> = nd::Array1::linspace(-15.0, 15.0, 3001);
    let dx = x[1] - x[0];
    let v: nd::Array1<f64> = x.mapv(|xk| a * xk.powi(4));

    let escan: nd::Array1<f64> = nd::Array1::linspace(0.0, 20.0, 10000);
    let sols: Vec<solve::Solution>
        = solve::solve_shooting(dx, &v, &escan, true).unwrap();
    println!("{} solutions found", sols.len());
    let energies: nd::Array1<f64> = sols.iter().map(|sol| sol.e).collect();
    let wfs: nd::Array2<f64>
        = nd::stack(
            nd::Axis(0),
            &sols.iter()
                .map(|sol| sol.wf.as_ref().unwrap().view())
                .collect::<Vec<_>>(),
        ).unwrap();
    let turning_points: nd::Array1<f64>
        = sols.iter()
        .map(|sol| {
            interp::find_zeros(&x, &(&v - sol.e), interp::Zero::Rising)
                .unwrap()[0]
        })
        .collect();

    let outdir = PathBuf::from("output");
    mkdir!(outdir);
    write_npz!(
        outdir.join("quartic.npz"),
        arrays: {
            "x" => &x,
            "v" => &v,
            "e" => &energies,
            "enat" => &nd::array![uu.e],
            "wf" => &wfs,
            "tps" => &turning_points,
        }
    );
}

