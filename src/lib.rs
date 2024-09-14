//! [CR-FM-NES](https://arxiv.org/pdf/2201.11422) is a derivative free optimiser developed by [Masahiro Nomura](https://github.com/nomuramasahir0/crfmnes) and Isao Ono specifically for high dimensional black-box problems.
//! This implementation is a translation of the fast-cma-es library implementation by [Dietmar Wolz](https://github.com/dietmarwo/fast-cma-es) from cpp/eigen to nalgebra.
//!
//! Similar to CMA-ES and NES optimisers at its core is sampling of a multivariate normal distribution.
//! To allow use on high dimensional problems the covariance matrix is approximated by a simplified form to reduce the time and space complexity:
//!
//! `C = sigma*sigma*D(I + v*v_T)*D`
//!
//! This is similar to the VD-CMA optimiser where `D` is a diagonal scaling matrix, `v` is a principal component vector, and `sigma` is the size of the sampling distribution.
//! These along with the mean position vector `m` are gradually adjusted based on feedback from evaluations of samples by the user supplied objective function.
//! This optimiser includes features for better behaviour on constrained problems. The user can be indicate that a sample falls outside the feasible region by returning a function evaluation of `f64::INFINITY` and learning rates will be adapted for that trial accordingly.
//!
//! An Ask-Tell interface is exposed allowing arbitrary stopping criteria to be implemented, and allowing the optimiser to be wrapped in a struct which provides stopping criteria, evaluation looping, or BIPOP functionality.

// Copyright (c) Dietmar Wolz. (Source Cpp Implementation)
// Copyright (c) James Millard. (Rust Translation)
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

use core::f64;
use std::slice::from_ref;

use nalgebra::{
    ComplexField, DMatrix, DMatrixView, DVector, DVectorView, Dyn, Matrix, RowDVector, Scalar,
    ViewStorage,
};

use rand::Rng;
use rand_distr::StandardNormal;

pub mod test_functions;

/// broadcast single element to a matrix view
fn bc_element<T: Scalar>(
    elem: &T,
    nrows: usize,
    ncols: usize,
) -> Matrix<T, Dyn, Dyn, ViewStorage<'_, T, Dyn, Dyn, Dyn, Dyn>> {
    DMatrixView::from_slice_with_strides(from_ref(elem), nrows, ncols, 0, 0)
}

/// broadcast single column to a matrix view
fn bc_column<T: Scalar>(
    vec: &DVector<T>,
    ncols: usize,
) -> Matrix<T, Dyn, Dyn, ViewStorage<'_, T, Dyn, Dyn, Dyn, Dyn>> {
    DMatrixView::from_slice_with_strides(vec.as_slice(), vec.len(), ncols, 1, 0)
}

/// broadcast single row to a matrix view
fn bc_row<T: Scalar>(
    vec: &RowDVector<T>,
    nrows: usize,
) -> Matrix<T, Dyn, Dyn, ViewStorage<'_, T, Dyn, Dyn, Dyn, Dyn>> {
    DMatrixView::from_slice_with_strides(vec.as_slice(), nrows, vec.len(), 0, 1)
}

/// Recommended lambda for a given dim size for typical problems.
///
/// Noisy or highly multi-modal objective functions should use higher values, e.g. 4*dim.
pub fn rec_lamb(dim: usize) -> usize {
    let x = ((dim as f64).ln() * 3.0).floor() as usize;
    if x % 2 == 0 {
        x + 4
    } else {
        x + 5
    }
}

fn cexp(a: f64) -> f64 {
    a.min(100.0).exp() // avoid overflow
}

fn f(a: f64, dim: usize) -> f64 {
    ((1. + a * a) * cexp(a * a / 2.) / 0.24) - 10. - dim as f64
}

fn f_prime(a: f64) -> f64 {
    (1. / 0.24) * a * cexp(a * a / 2.) * (3. + a * a)
}

fn get_h_inv(dim: usize) -> f64 {
    let mut h_inv = 1.0;
    while (f(h_inv, dim)).abs() > 1e-10 {
        h_inv = h_inv - 0.5 * (f(h_inv, dim) / f_prime(h_inv));
    }
    h_inv
}

fn num_feasible(evals: &[f64]) -> usize {
    evals.iter().filter(|e| e.is_finite()).count()
}

fn sort_indices_by(evals: &[f64], z: DMatrixView<f64>) -> Vec<usize> {
    let lam = evals.len();

    let distances: Vec<f64> = (0..lam).map(|i| z.column(i).norm_squared()).collect();
    sort_index(evals, &distances)
}

/// sort index by primary, and if primary is not finite treat as greater than and sort by secondary.
/// Panics if `primary.len() != secondary.len()`. Panics if secondary contains non-finite values.
fn sort_index(primary: &[f64], secondary: &[f64]) -> Vec<usize> {
    assert_eq!(primary.len(), secondary.len());
    let mut indices: Vec<usize> = (0..primary.len()).collect();

    indices.sort_unstable_by(
        |a, b| match (primary[*a].is_finite(), primary[*b].is_finite()) {
            (true, true) => primary[*a].total_cmp(&primary[*b]),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => secondary[*a].total_cmp(&secondary[*b]),
        },
    );
    indices
}

#[derive(Clone, Debug)]
#[allow(non_snake_case)]
struct State {
    sigma: f64,

    /// dim x 1
    m: DVector<f64>,

    /// dim x 1
    D: DVector<f64>,

    /// dim x 1
    v: DVector<f64>,

    /// dim x 1
    pc: DVector<f64>,

    /// dim x 1
    ps: DVector<f64>,

    /// number of trials/generations
    g: usize,
}

/// High-dimension black box optimiser with an ask-tell interface.
///
/// # Example
/// In the example below the 40D Rosenbrock test function is optimised.
/// ```rust
/// use rand::{thread_rng, Rng, SeedableRng};
/// use rand_xoshiro::Xoroshiro128PlusPlus;
/// use nalgebra::DVector;
/// use crfmnes::{rec_lamb, CrfmnesOptimizer, test_functions::rosenbrock};
///
/// let mut rng = Xoroshiro128PlusPlus::seed_from_u64(thread_rng().gen());
/// let dim = 40;
/// let start_m = DVector::zeros(dim);
/// let start_sigma = 10.0;
/// let mut opt = CrfmnesOptimizer::new(start_m.clone(), start_sigma, rec_lamb(dim), &mut rng);
///
/// let mut best = f64::INFINITY;
/// let mut best_x = start_m;
///
/// for i in 0..10000 {
///     let mut trial = opt.ask(&mut rng);
///
///     let mut evs = Vec::new();
///     for (i, x) in trial.x().column_iter().enumerate() {
///         let eval = rosenbrock(x.as_slice(), 1.0, 100.0);
///         evs.push(eval);
///         if eval < best {
///             best = eval;
///             best_x = x.into_owned();
///         }
///     }
///
///     trial.tell(evs).unwrap();
///
///     if best < 0.001 {
///         break;
///     }
/// }
/// println!("best: {} best_x: {}", best, best_x);
/// ```
#[derive(Clone, Debug)]
pub struct CrfmnesOptimizer {
    /// number of dimensions in the problem
    dim: usize,

    /// number of samples per trial
    lamb: usize,

    w_rank_hat: DVector<f64>,
    w_rank: DVector<f64>,

    mueff: f64,
    cs: f64,
    cc: f64,
    c1_cma: f64,

    // expected value for dim size
    chi_n: f64,

    // distance weight parameter
    h_inv: f64,

    // learning rate
    eta_m: f64,
    eta_move_sigma: f64,

    state: State,
}

impl CrfmnesOptimizer {
    /// Create a new optimiser with the provided parameters and state.
    ///
    /// Default initialisation is used for `D` and `v`. `D` is set to the identity matrix and `v` is set to a small random vector.
    ///
    /// See `with_v_D` for more details.
    #[allow(non_snake_case)]
    pub fn new<R: Rng>(m: DVector<f64>, sigma: f64, lamb: usize, rand: &mut R) -> Self {
        let dim = m.len();

        let v = DVector::from_fn(dim, |_, _| {
            rand.sample::<f64, _>(StandardNormal) / (dim as f64).sqrt()
        });
        let D = DVector::from_element(dim, 1.0);

        Self::with_v_D(m, sigma, v, D, lamb)
    }

    /// Create a new optimiser with the provided parameters and state.
    ///
    /// The parameters in order of application when generating samples from a standard normal distribution:
    /// * `lamb` determines the number of sample vectors generated for each trial. If an odd number is provided, the next even number is used. For smooth, uni-modal problems use the value provided by `rec_lamb`.
    /// * `v` is a principal vector which stretches the sampling distribution in an arbitrary direction.
    /// * `D` is a diagonal matrix (stored as a vector) which scales the distribution along each axis of the problem.
    /// * `sigma` is the initial size, standard deviation, of the sampling distribution.
    /// * `m` is the initial mean position vector of the sampling distribution.
    ///
    /// # Panics
    /// * If `m.is_empty()`
    /// * If `lamb < 4`
    /// * If `sigma <= 0.0`
    /// * If `m.len() != v.len()`
    /// * If `m.len() != D.len()`
    #[allow(non_snake_case)]
    pub fn with_v_D(
        m: DVector<f64>,
        sigma: f64,
        v: DVector<f64>,
        D: DVector<f64>,
        lamb: usize,
    ) -> Self {
        let lamb = lamb + lamb % 2;
        assert!(!m.is_empty());
        assert!(lamb >= 4);
        assert!(sigma > 0.0);
        assert_eq!(m.len(), v.len());
        assert_eq!(m.len(), D.len());

        let dim = m.len();
        let mu = lamb / 2;
        let w_rank_hat = DVector::from_fn(lamb, |row, _| {
            ((mu as f64 + 1.0).ln() - ((row + 1) as f64).ln()).max(0.0)
        });

        let w_rank: DVector<f64> = (w_rank_hat.clone() / w_rank_hat.sum())
            - DVectorView::from_slice_with_strides(&[1. / lamb as f64], lamb, 0, 0);

        let mueff = 1.
            / w_rank.fold(0.0, |acc, e| {
                let q = e + (1. / lamb as f64);
                acc + q * q
            });

        let cs = (mueff + 2.) / (dim as f64 + mueff + 5.);
        let cc = (4. + mueff / dim as f64) / (dim as f64 + 4. + 2. * mueff / dim as f64);
        let c1_cma = 2. / ((dim as f64 + 1.3).powi(2) + mueff);
        // initialisation
        let chi_n = (dim as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * dim as f64) + 1.0 / (21.0 * dim as f64 * dim as f64));
        let pc = DVector::zeros(dim);
        let ps = DVector::zeros(dim);
        // distance weight parameter
        let h_inv = get_h_inv(dim);
        // learning rate
        let eta_m = 1.0;
        let eta_move_sigma = 1.0;

        Self {
            dim,

            lamb,

            w_rank_hat,
            w_rank,
            mueff,
            cs,
            cc,
            c1_cma,
            chi_n,

            h_inv,
            eta_m,
            eta_move_sigma,

            state: State {
                sigma,

                m,
                D,
                v,

                pc,
                ps,

                g: 0,
            },
        }
    }

    /// Ask the optimiser to supply a new set of sample points to be evaluated.
    pub fn ask<'a, R: Rng>(&'a mut self, rand: &mut R) -> Trial<'a> {
        let State {
            sigma,
            ref m,
            ref D,
            ref v,
            ..
        } = &self.state;

        let mut z = DMatrix::<f64>::zeros(self.dim, self.lamb);

        for i in 0..self.lamb / 2 {
            for j in 0..self.dim {
                let val: f64 = rand.sample(StandardNormal);
                *z.get_mut((j, i)).unwrap() = val;
                *z.get_mut((j, i + self.lamb / 2)).unwrap() = -val;
            }
        }

        let normv2 = v.norm_squared();
        let normv = normv2.sqrt();
        let vbar = v / normv;
        let y = &z + (((1.0 + normv2).sqrt() - 1.0) * (&vbar * (vbar.transpose() * &z)));

        let x = *sigma * y.component_mul(&bc_column(D, self.lamb)) + bc_column(m, self.lamb);

        Trial { opt: self, z, y, x }
    }

    /// Returns the number of sucessful updates performed via the `Trial::tell` method.
    pub fn update_count(&self) -> usize {
        self.state.g
    }

    /// The number of samples per `Trial`.
    pub fn lamb(&self) -> usize {
        self.lamb
    }

    /// The dimensions of the problem space.
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn c1(&self, lamb_feas: usize) -> f64 {
        self.c1_cma * (self.dim.saturating_sub(6) + 1) as f64 / 6.0
            * (lamb_feas as f64 / self.lamb as f64)
    }

    #[allow(non_snake_case)]
    fn eta_B(&self, lamb_feas: usize) -> f64 {
        (((0.02 * lamb_feas as f64).min(3.0 * (self.dim as f64).ln()) + 5.0)
            / (0.23 * self.dim as f64 + 25.0))
            .tanh()
    }

    fn alpha_dist(&self, lamb_feas: usize) -> f64 {
        self.h_inv
            * ((self.lamb as f64) / self.dim as f64).sqrt().min(1.0)
            * (lamb_feas as f64 / self.lamb as f64).sqrt()
    }

    fn w_dist_hat(&self, z: DVectorView<f64>, lamb_feas: usize) -> f64 {
        cexp(self.alpha_dist(lamb_feas) * z.norm())
    }

    fn eta_stag_sigma(&self, lamb_feas: usize) -> f64 {
        ((0.024 * lamb_feas as f64 + 0.7 * self.dim as f64 + 20.) / (self.dim as f64 + 12.)).tanh()
    }

    fn eta_conv_sigma(&self, lamb_feas: usize) -> f64 {
        2. * ((0.025 * lamb_feas as f64 + 0.75 * self.dim as f64 + 10.) / (self.dim as f64 + 4.))
            .tanh()
    }
}

/// Error types potentially returned by `Trial::tell`.
///
/// In all cases, the trial is discarded and can be retried.
#[derive(Debug, Clone)]
pub enum TrialError {
    /// All supplied evs were `f64::INFINITY` (Non-feasible).
    NoFeasibleSolutions,
    /// Update of `D` and `v` failed due to a singularity when calculating the approximate Fischer Information matrix.
    DivByZero,
    /// An element of the diagonal became negative during the update.
    DiagonalInverted,
}

/// The next set of trial values to be evaluated by the user.
#[derive(Debug)]
pub struct Trial<'a> {
    opt: &'a mut CrfmnesOptimizer,

    /// Isotropic sample vectors. Accessing these is typically not required.
    /// Drawn from a standard normal distribution, with the second half mirrored.
    ///
    /// Shape: dims x lamb
    z: DMatrix<f64>,

    /// Skewed sample vectors. Accessing these is typically not required.
    /// z sample vectors updated to account for the distortion of the learned v vector
    ///
    /// Shape: dims x lamb
    y: DMatrix<f64>,

    /// y sample vectors updated to account for current mean m, the diagonal scaling, and the sample std-dev, `sigma`.
    ///
    /// Shape: dims x lamb
    x: DMatrix<f64>,
}

impl<'a> Trial<'a> {
    /// A matrix of shape (dim, lamb) containing sample vectors to be evaluated as columns.
    pub fn x(&self) -> DMatrixView<f64> {
        self.x.as_view()
    }

    /// Tell the optimiser the results of evaluating the sample points, updating its internal state based on the results of this trial.
    ///
    /// The user provided evaluations in `evs` should be in the same order as the columns of `x`.
    ///
    /// If an error is returned, the state of the parent optimiser is not updated, and a new trial can be attempted.
    ///
    /// # Panics
    /// If `evs.len() != x.len()` this method will panic.
    #[allow(non_snake_case)]
    pub fn tell(&mut self, evs: Vec<f64>) -> Result<(), TrialError> {
        // Read this method in conjunction with the paper, as the same variable names are used.

        assert_eq!(evs.len(), self.x.ncols());

        // This operation assumes that if the solution is infeasible, infinity comes in as input.
        let lamb_feas = num_feasible(&evs);

        if lamb_feas == 0 {
            return Err(TrialError::NoFeasibleSolutions);
        }

        let mut new_state = self.opt.state.clone();

        let normv2 = new_state.v.norm_squared();
        let normv = normv2.sqrt();
        let normv4 = normv2 * normv2;

        let vbar = &new_state.v / normv;

        let lamb = self.opt.lamb;
        let dim = self.opt.dim;

        let sorted_indices = sort_indices_by(&evs, self.z.as_view());

        let x = DMatrix::from_fn(dim, lamb, |row, col| {
            *self.x.get((row, sorted_indices[col])).unwrap()
        });
        let y = DMatrix::from_fn(dim, lamb, |row, col| {
            *self.y.get((row, sorted_indices[col])).unwrap()
        });
        let z = DMatrix::from_fn(dim, lamb, |row, col| {
            *self.z.get((row, sorted_indices[col])).unwrap()
        });

        new_state.g += 1;

        // evolution path p_sigma
        new_state.ps = (1.0 - self.opt.cs) * new_state.ps
            + (&z * &self.opt.w_rank) * (self.opt.cs * (2. - self.opt.cs) * self.opt.mueff).sqrt();
        let ps_norm = new_state.ps.norm();

        // distance weight
        let weights_dist: DVector<f64> = {
            let mut w_tmp: Vec<f64> = (0..lamb)
                .map(|k| self.opt.w_rank_hat[k] * self.opt.w_dist_hat(z.column(k), lamb_feas))
                .collect();
            let sum: f64 = w_tmp.iter().sum();
            for e in &mut w_tmp {
                *e = (*e / sum) - 1. / lamb as f64;
            }
            DVector::from_vec(w_tmp)
        };

        // switching weights and learning rate
        let weights: DVectorView<f64> = if ps_norm >= self.opt.chi_n {
            weights_dist.as_view()
        } else {
            self.opt.w_rank.as_view()
        };
        let eta_sigma = if ps_norm >= self.opt.chi_n {
            self.opt.eta_move_sigma
        } else if ps_norm >= 0.1 * self.opt.chi_n {
            self.opt.eta_stag_sigma(lamb_feas)
        } else {
            self.opt.eta_conv_sigma(lamb_feas)
        };

        // update pc, m
        let wxm: DVector<f64> = (x - bc_column(&new_state.m, lamb)) * weights;
        new_state.pc = (1. - self.opt.cc) * &new_state.pc
            + (self.opt.cc * (2. - self.opt.cc) * self.opt.mueff).sqrt() / new_state.sigma * &wxm;
        new_state.m += self.opt.eta_m * wxm;

        // calculate s, t
        // step1
        let exY = DMatrix::from_fn(self.opt.dim, lamb + 1, |r, c| {
            if c < lamb {
                *y.get((r, c)).unwrap()
            } else {
                *new_state.pc.get(r).unwrap() / new_state.D.get(r).unwrap()
            }
        }); // dim x lamb+1

        let yy: DMatrix<f64> = exY.map(|e| e * e); // dim x lamb+1

        let ip_yvbar: RowDVector<f64> = vbar.transpose() * &exY; // 1 x lamb+1

        let vbar_bc = bc_column(&vbar, lamb + 1); // broadcasting vbar

        let yvbar: DMatrix<f64> = exY.component_mul(&vbar_bc); // dim x lamb+1. exYのそれぞれの列にvbarがかかる
        let gammav: f64 = 1. + normv2;
        let vbarbar: DVector<f64> = vbar.map(|e| e * e);
        let alphavd: f64 = 1.0f64
            .min((normv4 + (2.0 * gammav - gammav.sqrt()) / vbarbar.max()).sqrt() / (2. + normv2)); // scalar

        let ibg: RowDVector<f64> = ip_yvbar.map(|e| e * e + gammav); // 1 x lamb+1
        let mut t: DMatrix<f64> = (exY.component_mul(&bc_row(&ip_yvbar, dim)))
            - (vbar_bc.component_mul(&bc_row(&ibg, dim))) / 2.; // dim x lamb+1

        let b: f64 = -(1.0 - alphavd * alphavd) * normv4 / gammav + 2.0 * alphavd * alphavd;
        let H: DVector<f64> =
            DVector::from_element(self.opt.dim, 2.0) - (b + 2.0 * alphavd * alphavd) * &vbarbar; // dim x 1
        let invH: DVector<f64> = H.map(|e| 1.0 / e); // dim x 1
        let s_step1: DMatrix<f64> = yy
            - normv2 / gammav * (yvbar.component_mul(&bc_row(&ip_yvbar, dim)))
            - bc_element(&1.0, dim, lamb + 1); // dim x lamb+1

        let ip_vbart: RowDVector<f64> = vbar.transpose() * &t; // 1 x lamb+1
        let s_step2: DMatrix<f64> = s_step1
            - (alphavd / gammav
                * ((2.0 + normv2) * (t.component_mul(&vbar_bc))
                    - (normv2 * (&vbarbar * ip_vbart)))); // dim x lamb+1

        let invHvbarbar: DVector<f64> = invH.component_mul(&vbarbar);
        let ip_s_step2invHvbarbar: RowDVector<f64> = invHvbarbar.transpose() * &s_step2; // 1 x lamb+1

        let div: f64 = 1.0 + b * (vbarbar.transpose() * &invHvbarbar).as_scalar();
        if div.abs() < 1e-10 {
            return Err(TrialError::DivByZero);
        }

        let s: DMatrix<f64> = (s_step2.component_mul(&bc_column(&invH, lamb + 1)))
            - ((b / div) * (invHvbarbar * ip_s_step2invHvbarbar)); // dim x lamb+1

        let ip_svbarbar: RowDVector<f64> = vbarbar.transpose() * &s; // 1 x lamb+1
        t -= alphavd * ((2.0 + normv2) * (s.component_mul(&vbar_bc)) - (&vbar * ip_svbarbar)); // dim x lamb+1

        // update v, D
        let mut exw = DVector::zeros(lamb + 1);
        let eta_B = self.opt.eta_B(lamb_feas);
        for k in 0..lamb {
            exw[k] = eta_B * weights[k];
        }
        exw[lamb] = self.opt.c1(lamb_feas);

        new_state.v += (t * &exw) / normv;
        new_state.D += (s * &exw).component_mul(&new_state.D);

        // calculate detA
        if new_state.D.min() < 0.0 {
            return Err(TrialError::DiagonalInverted);
        }
        let nthrootdetA = cexp(
            new_state.D.map(|e| e.ln()).sum() / dim as f64
                + (1.0 + (new_state.v.transpose() * &new_state.v).as_scalar()).ln()
                    / (2.0 * dim as f64),
        );

        new_state.D = new_state.D.map(|e| e / nthrootdetA);

        // update sigma
        let G_s = ((z.map(|e| e * e) - bc_element(&1.0, dim, lamb)) * weights).sum() / dim as f64;
        new_state.sigma *= cexp(eta_sigma / 2.0 * G_s);

        // update state only if no errors arise.
        self.opt.state = new_state;

        Ok(())
    }
}
