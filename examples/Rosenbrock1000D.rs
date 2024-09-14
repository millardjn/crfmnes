


use rand::{thread_rng, Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;
use nalgebra::DVector;
use crfmnes::{rec_lamb, CrfmnesOptimizer, test_functions::rosenbrock};

fn main() {
    let mut rng = Xoroshiro128PlusPlus::seed_from_u64(thread_rng().gen());
    let dim = 1000;
    let start_m = DVector::zeros(dim);
    let start_sigma = 1.0;
    let mut opt = CrfmnesOptimizer::new(start_m.clone(), start_sigma, rec_lamb(dim), &mut rng);
    
    let mut best = f64::INFINITY;
    let mut best_x = start_m;
    
    for i in 0..1000000 {
        let mut trial = opt.ask(&mut rng);
    
        let mut evs = Vec::new();
        for x in trial.x().column_iter() {
            let eval = rosenbrock(x.as_slice(), 1.0, 100.0);
            evs.push(eval);
            if eval < best {
                best = eval;
                best_x = x.into_owned();
            }
        }
        trial.tell(&evs).unwrap();

        println!("{} {} {:.3e} {:.3e}", i, evs[0], opt.state().sigma, opt.state().v.norm());
    
        if opt.state().sigma < 1e-8 {
            break;
        }
    }
    println!("best: {} best_x: {}", best, best_x);
}
