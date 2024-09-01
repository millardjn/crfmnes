<p align="center">
    <a href="https://crates.io/crates/crfmnes"><img src="https://img.shields.io/crates/v/crfmnes.svg" alt="crates.io"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>

# CR-FM-NES Optimiser

A rust implementation of the [CR-FM-NES](https://arxiv.org/pdf/2201.11422) derivative free optimiser developed by [Masahiro Nomura](https://github.com/nomuramasahir0/crfmnes) and Isao Ono specifically for high dimensional black-box problems. This implementation is a translation of the fast-cma-es library implementation by [Dietmar Wolz](https://github.com/dietmarwo/fast-cma-es) from cpp/eigen to nalgebra.

Similar to CMA-ES and NES optimisers at the core of this optimiser is sampling of a multivariate normal distribution.
To allow use on high dimensional problems the covariance matrix is approximated by a simplified form to reduce the time and space complexity:

`C = sigma*sigma*D(I + v*v_T)*D`

This is similar to the VD-CMA optimiser where `D` is a diagonal scaling matrix, `v` is a principal component vector, and `sigma` is the size of the sampling distribution.
These along with the mean position vector `m` are gradually adjusted based on feedback from evaluations of samples by the user's objective function.
This optimiser includes features for better behaviour on constrained problems. The user can be indicate that a sample falls outside the feasible region by returning a function evaluation of `f64::INFINITY` and learning rates will be adapted for that trial accordingly.

An Ask-Tell interface is exposed allowing arbitrary stopping criteria to be implemented, and allowing the optimiser to be wrapped in a struct which provides stopping criteria, evaluation looping, or BIPOP functionality.

# Example
```rust
use rand::{thread_rng, Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;
use nalgebra::DVector;
use crfmnes::{rec_lamb, CrfmnesOptimizer, test_functions::rosenbrock};

let mut rng = Xoroshiro128PlusPlus::seed_from_u64(thread_rng().gen());
let dim = 40;
let start_m = DVector::zeros(dim);
let start_sigma = 10.0;
let mut opt = CrfmnesOptimizer::new(start_m.clone(), start_sigma, rec_lamb(dim), &mut rng);

let mut best = f64::INFINITY;
let mut best_x = start_m;

for i in 0..10000 {
    let mut trial = opt.ask(&mut rng);

    let mut evs = Vec::new();
    for (i, x) in trial.x().column_iter().enumerate() {
        let eval = rosenbrock(x.as_slice(), 1.0, 100.0);
        evs.push(eval);
        if eval < best {
            best = eval;
            best_x = x.into_owned();
        }
    }

    trial.tell(evs).unwrap();

    if best < 0.001 {
        break;
    }
}
println!("best: {} best_x: {}", best, best_x);
```


# Performance
<img width="1215" alt="188303830-aa7b11d0-c6ff-4d1a-9bd8-2ccbf4d7e2dd" src="https://user-images.githubusercontent.com/10880858/211967554-65d632bd-3e77-4725-998c-20f69bb8f5ce.png">