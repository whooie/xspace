//! Theoretical background.
//!
//! # Contents
//! - [Background](#background)
//! - [Units](#units)
//! - [Radial wavefunctions](#radial-wavefunctions)
//! - [Time dependence](#time-dependence)
//!
//! # Background
//! Solution of the one-dimensional time-independent Schrödinger equation (TISE)
//! amounts to solving equations of the form
//! ```text
//! ∂²f
//! --- = -Q(x) f(x)
//! ∂x²
//! ```
//! where, in the particular case of the TISE, one typically takes
//! ```text
//!        2 m
//! Q(x) = --- (E - V(x))
//!         ħ²
//! ```
//! with *V*(*x*) being a (conservative) potential for some non-trivial range of
//! energies *E* and *m* being a relevant mass. Solutions to this equation are
//! eigenpairs of the Hamiltonian operator (*ħ*²/2 *m*) (*∂*²/*∂*x²) + *V*(*x*)
//! and hence take the form of a wavefunction defined over *x*, along with a
//! (real-valued) energy that is strictly greater than the minimum value of
//! *V*(*x*). Note that, since the eigenvalues of the Hamiltonian operator are
//! real (by Hermiticity), their associated wavefunctions are guaranteed to be
//! real-valued as well.
//!
//! This module allows for the solution of this module using, in particular,
//! Numerov's method for second-order ordinary differential equations[^1].
//! Assuming a discretization
//! ```text
//! x[i] = x₀ + i δx, i ∊ {0, ..., N - 1}
//! f[i] = f(x[i])
//! Q[i] = Q(x[i])
//! ```
//! Numerov's method is a three-point numerical integration scheme,
//! ```text
//!      δx²                             5 δx²                   δx²
//! (1 + --- Q[i + 1]) f[i + 1] = 2 (1 - ----- Q[i]) f[i] - (1 + --- Q[i - 1]) u[i - 1]
//!      12                               12                     12
//! ```
//! which has an error term of only *O*(*δx*⁶) (c.f. the more generally used
//! fourth-order Runge-Kutta scheme, which has a *O*(*δx*⁴) error term).
//!
//! This relation can be integrated forward and backward, but also allows for
//! a matrix representation[^2] specialized to the TISE that allows for large
//! swaths of a system's energy spectrum (with associated wavefunctions) to be
//! found quickly and accurately for decently sized grids. The matrix
//! representation translates Numerov's scheme into a direct matrix eigenvalue
//! problem
//! ```text
//!     ħ²
//! (- --- inv(B) A + V) f = E f
//!    2 m
//!
//! A = (I{-1} - 2 I{0} + I{+1}) / δx²
//! B = (I{-1} - 10 I{0} + I{+1}) / 12
//! V[i, j] = δ[i, j] V(x[i])
//! ```
//! where *I*{*k*} is the *N*×*N* matrix with elements equal to 1 on the *k*-th
//! diagonal and 0 elsewhere.
//!
//! The matrix approach may not be desirable when using large grids or when only
//! few solutions are required. In this case, one may use a kind of variational
//! approach known as a shooting method: The wavefunction is integrated for an
//! initial guess energy starting deep in the classically forbidden region
//! (where *E* < *V* and the value of the wavefunction is exponentially close to
//! 0), heading in toward the classically allowed region. This is done for
//! starting points on both sides of the allowed region, and the two solutions
//! are compared at some middle point. If they match by some criterion, then the
//! solution is valid; if not then another guess energy is used. Typically one
//! chooses the matching criterion to be that the difference in the derivatives
//! of the logarithms of the two trial solutions is zero at the matching point,
//! i.e.
//! ```text
//! ∂             |        ∂              |
//! -- ln(f_left) |      - -- ln(f_right) |     == 0
//! ∂x            |x=x₀    ∂x             |x=x₀
//! ```
//! (Using the derivative of the logarithm allows overall signs and
//! normalization factors in *f*<sub>left</sub> and *f*<sub>right</sub> to be
//! discarded.)
//!
//! However, naive use of the Numerov scheme is liable to cause problems because
//! the integration will more easily couple to exponential growth solutions than
//! exponential decay solutions in classically forbidden regions, leading to
//! numerical overflow. To overcome this limitation, we perform a transformation
//! on the bare Numerov scheme to describe the wavefunction by the ratios
//! between wavefunction values at adjacent grid points rather than by the
//! values of the wavefunction themselves, termed the "renormalized" Numerov
//! method[^3]. This admits two descriptions *R* and *L*, depending on whether
//! right or left neighbors are used.
//! ```text
//!        (1 + (δx²/12) Q[i + 1]) f[i + 1]
//! R[i] = --------------------------------
//!            (1 + (δx²/12) Q[i]) f[i]
//!
//!            (1 + (δx²/12) Q[i]) f[i]
//! L[i] = --------------------------------
//!        (1 + (δx²/12) Q[i - 1]) f[i - 1]
//! ```
//! These representations of the wavefunction can then be integrated using
//! ```text
//!        2 - 10 (δx²/12) Q[i]      1
//! R[i] = -------------------- - --------
//!          1 + (δx²/12) Q[i]    R[i - 1]
//!
//!        2 - 10 (δx²/12) Q[i]      1
//! L[i] = -------------------- - --------
//!          1 + (δx²/12) Q[i]    L[i + 1]
//! ```
//! Finally, the matching point *x*\[*M*\] is chosen as the grid point just to
//! the left of the rightmost peak in the wavefunction, i.e. the coordinate of
//! the first point for which the integration of *L*\[*i*\] encounters
//! *L*\[*i*\] ≤ 1. At this point, the renormalization of the wavefunction can
//! be inverted and the mathing criterion checked as described above.
//!
//! Two approaches can then be employed to find solutions. In the first, the
//! shooting procedure can be performed for a series of trial energies with the
//! matching criterion computed for each one, and solutions can be computed by
//! interpolating to find the locations (energies) where the criterion crosses
//! zero. In the second, one can use a simple [secant method][secant] to
//! iteratively converge to a single solution until a given accuracy bound is
//! satisfied. In either case, it can be noted that the number of nodes *ν* ∊
//! {0, 1, ...} in the wavefunction (which can be computed from only *R*)
//! uniquely identifies a particular solution to the TISE, which can be used to
//! provide additional constraints on energy bounds when searching for
//! solutions. Specifically, given node count identifies a range of energies for
//! which the chosen matching criterion monotonically increases over the range
//! (-∞, +∞), guaranteeing the convergence of the secant method to a unique
//! solution for fixed node count.
//!
//! # Units
//! All solver functions in this crate are designed to work with the TISE in
//! natural (dimensionless) units. Starting from the usual expression of the
//! TISE,
//! ```text
//!    ħ² ∂²
//! - --- --- ψ(x) + V(x) ψ(x) = E ψ(x)
//!   2 m ∂x²
//! ```
//! we first choose a particular characteristic length scale *a* and change
//! variables using *x'* ≡ *x* / *a*, giving
//! ```text
//! dx = a dx' ⇒ (∂²/∂x²) = (1/a²) (∂²/∂(x')²)
//! ψ(x) → ψ'(x') = ψ(a x') / √a
//! ```
//! This change of variables introduces a factor of 1/*a*² into the coefficient
//! of the kinetic energy term, *ħ*²/2 *m* → *ħ*²/2 *m* *a*² ≡ *ε*. This
//! quantity has units of energy and is a convenient choice for a natural energy
//! scale, corresponding roughly to the ground-state energy of a particle of
//! mass *m* confined to a box of size *a*. Dividing through by this factor
//! gives the potential and particle energy in natural units as well,
//! ```text
//! V(x) → V'(x') = V(a x') / ε
//! E → E' = E / ε
//! ```
//! This can be extended to the time-*dependent* Schrödinger equation as well.
//! where we start with
//! ```text
//!    ħ² ∂²                               ∂
//! - --- --- ψ(x, t) + V(x) ψ(x, t) = i ħ -- ψ(x, t)
//!   2 m ∂x²                              ∂t
//! ```
//! After introducing *a* and dividing through by *ε*, the coefficient on the
//! right-hand side becomes *ħ* → 2 *m* *a* / *ħ* ≡ *τ*, which has units of
//! time and acts as a natural time scale. Performing a second change of
//! variables *t'* ≡ *t* / *τ* gives
//! ```text
//! dt = τ dt' ⇒ (∂/∂t) = (1/τ) (∂/∂t')
//! ψ'(x', t) → ψ''(x', t') = ψ'(x', τ t')
//! ```
//! which removes the remaining coefficient and produces the dimensionless
//! time-(in)dependent Schrödinger equations
//! ```text
//!     ∂²
//! - ------ ψ'(x') + V'(x') ψ'(x') = E' ψ'(x')
//!   ∂(x')²
//!
//!     ∂²                                          ∂
//! - ------ ψ''(x', t') + V'(x') ψ''(x', t') = i ----- ψ''(x', t')
//!   ∂(x')²                                      ∂(t')
//! ```
//!
//! Items in [`units`][crate::units] are provided to handle the minutiae
//! associated with conversion to and from naturalized units.
//!
//! # Radial wavefunctions
//! One potential use for this crate is to calculate bound states in atomic (or
//! otherwise radially symmetric) potentials. In this case, the total
//! wavefunction is separable into radial and angular components *ψ*(*r*, *θ*,
//! *φ*) = *R*(*r*) *Y*(*θ*, *φ*) with *Y* being a spherical harmonic. One is
//! usually only interested in *R*, for which the Hamiltonian operator is
//! expressed in radial coordinates as
//! ```text
//!    ħ² 1       ∂    ∂²                      l (l + 1) ħ²
//! - --- -- (2 r -- + ---) R(r) + V(r) R(r) + ------------ R(r) = E R(r)
//!   2 m r²      ∂r   ∂r²                        2 m r²
//! ```
//! which does not fit the form of equation that is solvable by the Numerov
//! method. But by switching focus to the quantity *u*(*r*) ≡ *r* *R*(*r*)
//! instead, the radial TISE can be rewritten as
//! ```text
//!    ħ² ∂²u               l (l + 1) ħ²
//! - --- --- + V(r) u(r) + ------------ u(r) = E u(r)
//!   2 m ∂r²                  2 m r²
//! ```
//! This form is identical to the ordinary expression of the Schrödinger
//! equation in Cartesian coordinates (which *is* integrable by the Numerov
//! method) when the angular momentum term is absorbed into the potential, and
//! the above can be applied.
//!
//! # Time dependence
//! Evolution of a spatial wavefunction in time is performed via the split-step
//! method. This is a particularly nice method to use for the Schrödinger
//! equation because the Hamiltonian opposite the time derivative is
//! conveniently expressed as the sum of two operators, kinetic and potential.
//! In our natural units:
//! ```text
//!   ∂ψ
//! i -- = (H_k + H_v) ψ
//!   ∂t
//!
//! H_k = k²
//! H_v = V
//! ```
//! The general form of the time-dependent solution then looks like
//! ```text
//!               -i H_v dt  -i H_k dt  -i [H_v, H_k] dt²/2  O(dt³)
//! ψ(t + dt) = [e          e          e                    e      ] ψ(t)
//! ```
//! using the Baker-Campbell-Hausdorff formula. Of course, the terms involving
//! the \[*H*<sub>*v*</sub>, *H*<sub>*v*</sub>\] commutator are messy; they can
//! simply be discarded, but doing so would generate a *O*(*dt*²) error term,
//! which is relatively large. However, it turns out that by sandwiching the
//! *H*<sub>*k*</sub>-step term between two half-sized *H*<sub>*v*</sub>-step
//! terms reduces the error do *O*(*dt*³), which is more acceptable.
//! ```text
//!               -i H_v dt/2  -i H_k dt  -i H_v dt/2
//! ψ(t + dt) = [e            e          e           ] ψ(t) + O(dt³)
//! ```
//! Now the true benefit to using this method is that the kinetic and potential
//! terms can be dealt with independently. The potential term is
//! straightforward: Since the *x* operator is diagonal in position space space,
//! analytic functions of *x* are likewise diagonal, and hence the action of the
//! potential term reduces to applying a simple phase factor pointwise to ψ:
//! ```text
//!  -i H_v dt/2         -i V(x) dt/2
//! e            ψ(x) = e             ψ(x)
//! ```
//! The kinetic term, however, involves the *k* operator, which is usually
//! translated to a spatial derivative implemented as a finite difference with a
//! *O*(*dx*²) error term. In an exponential, this would be quite messy, but one
//! can note that *k* is diagonal in momentum space, which can be accessed using
//! the (fast) Fourier transform! Thus the action of the kinetic term can
//! likewise be reduced to applying a pointwise phase factor in momentum space:
//! ```text
//!  -i H_k dt                -i k² dt
//! e          F[ψ](k) = F⁻¹[e         F[ψ](k)]
//! ```
//! where *F* and *F*⁻¹ denote the Fourier transform and its inverse.
//!
//! Using this scheme, taking a step *dt* in time looks like this:
//! ```text
//!        ψ(t, x)
//!           |
//!           V
//!     -i V(t, x) dt/2
//!    e
//!           |
//!           '--> FFT ---.
//!                       |
//!                       V
//!                    -i k² dt
//!                   e
//!                       |
//!           .-- iFFT <--'
//!           |
//!           V
//!  -i V(t + dt, x) dt/2
//! e
//!           |
//!           V
//!     ψ(t + dt, x)
//! ```
//! Further, this scheme is particularly amenable to extensions to 2D and 3D as
//! well as nonlinear terms in either position or momentum space, and even to
//! evolution in imaginary time (i.e. taking *t* → -*i* *t*), which can be used
//! to find the ground state of the potential.
//!
//! [^1]: B. Numerov, "Note on the numerical integration of d2x/dt2 = f(x,t)."
//! Astronomische Nachrichten **230** 19 (1927).
//!
//! [^2]: M. Pillai, J. Goglio, and T. Walker, "Matrix Numerov method for
//! solving Schrödinger's equation." American Journal of Physics **80** 11
//! 1017-1019 (2012).
//!
//! [^3]: B. R. Johnson, "New numerical methods applied to solving the
//! one-dimensional eigenvalue problem." J. Chem. Phys. **67**:4086 (1977).
//!
//! [secant]: https://en.wikipedia.org/wiki/Secant_method

