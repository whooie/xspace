#![allow(non_upper_case_globals)]

//! Convenience functions and constructs to handle minutiae associated with
//! conversion to and from naturalized units.
//!
//! Concrete physical constants are taken from NIST.

use std::f64::consts::PI;

/// Planck constant (kg m^2 s^-1)
pub const h: f64 = 6.62607015e-34;
//             +/- 0 (exact)

/// reduced Planck constant (kg m^2 s^-1)
pub const hbar: f64 = h / 2.0 / PI;
//                +/- 0 (exact)

/// speed of light in vacuum (m s^-1)
pub const c: f64 = 2.99792458e8;
//             +/- 0 (exact)

/// Avogadro's number
pub const NA: f64 = 6.02214076e23;
//              +/- 0 (exact)

/// Boltzmann's constant (J K^-1)
pub const kB: f64 = 1.380649e-23;
//              +/- 0 (exact)

/// electric permittivity in vacuum (F m^-1)
pub const e0: f64 = 8.8541878128e-12;
//              +/- 0.0000000013e-12

/// magnetic permeability in vacuum (N A^-2)
pub const u0: f64 = 1.25663706212e-6;
//              +/- 0.00000000019e-6

/// Newtonian gravitational constant (m^3 kg^-1 s^-2)
pub const G: f64 = 6.67430e-11;
//             +/- 0.00015e-11

/// gravitational acceleration near Earth's surface (m s^-2)
pub const g: f64 = 9.80665;
//             +/- 0 (exact)

/// elementary charge (C)
pub const e: f64 = 1.602176634e-19;
//             +/- 0 (exact)

/// electron mass (kg)
pub const me: f64 = 9.1093837015e-31;
//              +/- 0.0000000028e-31

/// proton mass (kg)
pub const mp: f64 = 1.67262192369e-27;
//              +/- 0.00000000051e-27

/// unified atomic mass unit (kg)
pub const mu: f64 = 1.66053906660e-27;
//              +/- 0.00000000050e-27

/// Rydberg constant for an infinite-mass nucleus (m^-1)
pub const Rinf: f64 = 10973731.568160;
//                       +/- 0.000021

/// fine structure constant
pub const alpha: f64 = 7.2973525693e-3;
//                 +/- 0.0000000011e-3

/// molar gas constant
pub const R: f64 = 8.314462618;
//             +/- 0 (exact)

/// Stefan-Boltzmann constant (W m^-2 K^-4)
pub const SB: f64 = ( PI * PI * kB * kB * kB * kB )
                    / ( 60.0 * hbar * hbar * hbar * c * c );
//              +/- 0 (exact)

/// Bohr radius (m)
pub const a0: f64 = 5.29177210903e-11;
//              +/- 0.00000000080e-11

/// Bohr magneton (J T^-1)
pub const uB: f64 = 9.2740100783e-24;
//              +/- 0.0000000028e-24

/// Hartree energy (J) = 2\*Rinf\*h\*c
pub const Eh: f64 = 4.3597447222071e-18;
//              +/- 0.0000000000085e-18

/// A collection of natural unit scaling factors relative to some base unit
/// system.
///
/// Constructor methods produce scaling constants whose numerical values are
/// represented in the base unit system.
///
/// See [`docs/units`][crate::docs#units] for more information.
#[derive(Copy, Clone, Debug)]
pub struct Units {
    /// Particle mass.
    pub m: f64,
    /// Base length scale.
    pub a: f64,
    /// Associated energy scale.
    pub e: f64,
    /// Associated (angular) time scale.
    pub t: f64,
}

impl Units {
    /// Construct from a mass and length scale given in meters/kilograms/seconds
    /// (MKS) units.
    pub fn from_mks(mass: f64, a: f64) -> Self {
        let e_unit = hbar.powi(2) / 2.0 / mass / a.powi(2);
        let t_unit = hbar / e_unit;
        Self { m: mass, a, e: e_unit, t: t_unit }
    }

    /// Construct from a mass and length scale given in
    /// centimeters/grams/seconds (CGS) units.
    pub fn from_cgs(mass: f64, a: f64) -> Self {
        const hbar_cgs: f64 = hbar * 1e7;
        let e_unit = hbar_cgs.powi(2) / 2.0 / mass / a.powi(2);
        let t_unit = hbar_cgs / e_unit;
        Self { m: mass, a, e: e_unit, t: t_unit }
    }

    /// Construct from a mass and length scale in atomic (Bohr radii/electron
    /// masses/angular Hartree periods) units (AU).
    pub fn from_au(mass: f64, a: f64) -> Self {
        let m_si = me * mass;
        let a_si = a0 * a;
        let e_unit = hbar.powi(2) / 2.0 / m_si / a_si.powi(2) / Eh;
        let t_unit = 2.0 * m_si * a_si.powi(2) / Eh;
        Self { m: mass, a, e: e_unit, t: t_unit }
    }

    /// Convert a quantity with dimensions of length in the base unit system to
    /// natural units.
    pub fn to_nat_length<T, U>(&self, x: T) -> U
    where T: std::ops::Mul<f64, Output = U>
    {
        x * self.a.recip()
    }

    /// Convert a dimensionless quantity to one with length units in the base
    /// unit system.
    pub fn from_nat_length<T, U>(&self, x: T) -> U
    where T: std::ops::Mul<f64, Output = U>
    {
        x * self.a
    }

    /// Convert a quantity with dimensions of energy in the base unit system to
    /// natural units.
    pub fn to_nat_energy<T, U>(&self, x: T) -> U
    where T: std::ops::Mul<f64, Output = U>
    {
        x * self.e.recip()
    }

    /// Convert a dimensionless quantity to one with energy units in the base
    /// unit system.
    pub fn from_nat_energy<T, U>(&self, x: T) -> U
    where T: std::ops::Mul<f64, Output = U>
    {
        x * self.e
    }

    /// Convert a quantity with dimensions of time in the base unit system to
    /// natural units.
    pub fn to_nat_time<T, U>(&self, x: T) -> U
    where T: std::ops::Mul<f64, Output = U>
    {
        x * self.t.recip()
    }

    /// Convert a dimensionless quantity to one with time units in the base
    /// unit system.
    pub fn from_nat_time<T, U>(&self, x: T) -> U
    where T: std::ops::Mul<f64, Output = U>
    {
        x * self.t
    }
}


