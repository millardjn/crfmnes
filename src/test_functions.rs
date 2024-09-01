//! These optimisation test functions have been copied from the now defunct [argmin-testfunctions](https://github.com/argmin-rs/argmin-testfunctions) crate by Stefan Kroboth.

// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:

// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

use std::f64::consts::PI;

/// Rastrigin test function
pub fn rastrigin(param: &[f64]) -> f64 {
    rastrigin_a(param, 10.0)
}

/// Rastrigin test function
///
/// The same as `rastrigin`; however, it allows to set the parameter a.
pub fn rastrigin_a(param: &[f64], a: f64) -> f64 {
    a * param.len() as f64
        + param
            .iter()
            .map(|&x| x.powi(2) - a * (2.0 * PI * x).cos())
            .sum::<f64>()
}

/// Schaffer test function No. 2
///
/// Defined as
///
/// `f(x_1, x_2) = 0.5 + (sin^2(x_1^2 - x_2^2) - 0.5) / (1 + 0.001*(x_1^2 + x_2^2))^2`
///
/// where `x_i \in [-100, 100]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.
pub fn schaffer_n2(param: &[f64]) -> f64 {
    let plen = param.len();
    assert!(plen == 2);
    let (x1, x2) = (param[0], param[1]);
    let n05 = 0.5;
    let n1 = 1.0;
    let n0001 = 0.0001;
    n05 + ((x1.powi(2) - x2.powi(2)).sin().powi(2) - n05)
        / (n1 + n0001 * (x1.powi(2) + x2.powi(2))).powi(2)
}

/// Schaffer test function No. 4
///
/// Defined as
///
/// `f(x_1, x_2) = 0.5 + (cos(sin(abs(x_1^2 - x_2^2)))^2 - 0.5) / (1 + 0.001*(x_1^2 + x_2^2))^2`
///
/// where `x_i \in [-100, 100]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, 1.25313) = 0.291992`.
pub fn schaffer_n4(param: &[f64]) -> f64 {
    let plen = param.len();
    assert!(plen == 2);
    let (x1, x2) = (param[0], param[1]);
    let n05 = 0.5;
    let n1 = 1.0;
    let n0001 = 0.0001;
    n05 + ((x1.powi(2) - x2.powi(2)).abs().sin().cos().powi(2) - n05)
        / (n1 + n0001 * (x1.powi(2) + x2.powi(2))).powi(2)
}

/// Multidimensional Rosenbrock test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 \right]`
///
/// where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.
pub fn rosenbrock(param: &[f64], a: f64, b: f64) -> f64 {
    param
        .iter()
        .zip(param.iter().skip(1))
        .map(|(&xi, &xi1)| (a - xi).powi(2) + b * (xi1 - xi.powi(2)).powi(2))
        .sum()
}

/// 2D Rosenbrock test function
///
/// Defined as
///
/// `f(x_1, x_2) = (a - x_1)^2 + b * (x_2 - x_1^2)^2`
///
/// where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
///
/// For 2D problems, this function is much faster than `rosenbrock`.
///
/// The global minimum is at `f(x_1, x_2) = f(1, 1) = 0`.
pub fn rosenbrock_2d(param: &[f64], a: f64, b: f64) -> f64 {
    if let [x, y] = *param {
        (a - x).powi(2) + b * (y - x.powi(2)).powi(2)
    } else {
        panic!("rosenbrock_2d only works for a parameter vector with two values.");
    }
}

/// Derivative of 2D Rosenbrock function
pub fn rosenbrock_2d_derivative(param: &[f64], a: f64, b: f64) -> Vec<f64> {
    let num2 = 2.0;
    let num4 = 4.0;
    if let [x, y] = *param {
        vec![
            -num2 * a + num4 * b * x.powi(3) - num4 * b * x * y + num2 * x,
            num2 * b * (y - x.powi(2)),
        ]
    } else {
        panic!("rosenbrock function only accepts 2 parameters.");
    }
}

/// Hessian of 2D Rosenbrock function
pub fn rosenbrock_2d_hessian(param: &[f64], _a: f64, b: f64) -> Vec<f64> {
    let num2 = 2.0;
    let num4 = 4.0;
    let num12 = 12.0;
    if let [x, y] = *param {
        vec![
            // d/dxdx
            num12 * b * x.powi(2) - num4 * b * y + num2,
            // d/dxdy
            -num4 * b * x,
            // d/dydx
            -num4 * b * x,
            // d/dydy
            num2 * b,
        ]
    } else {
        panic!("rosenbrock_hessian only accepts 2 parameters.");
    }
}

/// Sphere test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = \sum_{i=1}^n x_i^2
///
/// where `x_i \in (-\infty, \infty)` and `n > 0`.
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.
pub fn sphere(param: &[f64]) -> f64 {
    param.iter().map(|x| x.powi(2)).sum()
}

/// Derivative of sphere test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = (2 * x_1, 2 * x_2, ... 2 * x_n)`
///
/// where `x_i \in (-\infty, \infty)` and `n > 0`.
pub fn sphere_derivative(param: &[f64]) -> Vec<f64> {
    let num2 = 2.0;
    param.iter().map(|x| num2 * *x).collect()
}

/// Ackley test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = - a * exp( -b \sqrt{\frac{1}{d}\sum_{i=1}^n x_i^2 ) -
/// exp( \frac{1}{d} cos(c * x_i) ) + a + exp(1)`
///
/// where `x_i \in [-32.768, 32.768]` and usually `a = 10`, `b = 0.2` and `c = 2*pi`
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.
pub fn ackley(param: &[f64]) -> f64 {
    ackley_param(param, 20.0, 0.2, 2.0 * PI)
}

/// Ackley test function
///
/// The same as `ackley`; however, it allows to set the parameters a, b and c.
pub fn ackley_param(param: &[f64], a: f64, b: f64, c: f64) -> f64 {
    let num1 = 1.0;
    let n = param.len() as f64;
    -a * (-b * ((num1 / n) * param.iter().map(|x| x.powi(2)).sum::<f64>()).sqrt()).exp()
        - ((num1 / n) * param.iter().map(|x| (c * *x).cos()).sum::<f64>()).exp()
        + a
        + num1.exp()
}

/// Beale test function
///
/// Defined as
///
/// `f(x_1, x_2) = (1.5 - x_1 + x_1 * x_2)^2 + (2.25 - x_1 + x_1 * x_2^2)^2 +
///                (2.625 - x_1 + x1 * x_2^3)^2`
///
/// where `x_i \in [-4.5, 4.5]`.
///
/// The global minimum is at `f(x_1, x_2) = f(3, 0.5) = 0`.
pub fn beale(param: &[f64]) -> f64 {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    (1.5 - x1 + x1 * x2).powi(2)
        + (2.25 - x1 + x1 * (x2.powi(2))).powi(2)
        + (2.625 - x1 + x1 * (x2.powi(3))).powi(2)
}

/// Booth test function
///
/// Defined as
///
/// `f(x_1, x_2) = (x_1 + 2*x_2 - 7)^2 + (2*x_1 + x_2 - 5)^2
///
/// where `x_i \in [-10, 10]`.
///
/// The global minimum is at `f(x_1, x_2) = f(1, 3) = 0`.
pub fn booth(param: &[f64]) -> f64 {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    (x1 + 2.0 * x2 - 7.0).powi(2) + (2.0 * x1 + x2 - 5.0).powi(2)
}
