use std::fmt::Display;
use std::ops::Sub;

use nalgebra::ClosedAddAssign;
use nalgebra::DMatrix;
use nalgebra::RealField;
use nalgebra::Scalar;
use num_traits::{One, Zero};
use rand::Rng;
// use rand::distributions::uniform::SampleUniform;
// use rand::distributions::Uniform;

// Fun story: The inverse matrix of an integer matrix is not forcibly an integer
// matrix.
pub struct SecurityMatrix<T> {
    pub m : DMatrix<T>,
    pub size : usize
}

// Generate a Diagonally Dominant Matrix. It is invertible by definition.
pub trait InvertibleMatrixBuilder<T :
     Scalar +
     RealField +
     PartialEq  +
     Display +
     ?Sized +
     Zero +
     One +
     Into<<T as Sub>::Output> +
     From<<T as Sub>::Output> +
     ClosedAddAssign +
     Sub +
     Copy > {
    fn _rand () -> T;

    fn populate (m : &mut DMatrix<T>) -> DMatrix<T> {
        let (l,c) = m.shape();
        for i in 0..l {
            for j in 0..c {
                m[(i,j)] = Self::_rand();
            }
        }
        m.clone()
    }

    fn make_diagonally_dominant (m : &mut DMatrix<T>) -> DMatrix<T> {
        let (l,_) = m.shape();
        for i in 0..l {
            m[(i,i)] = <<T as Sub>::Output as Into<T>>::into(m.row(i).sum() - m[(i,i)])
        }
        m.clone()
    }

    fn build(dim : usize) -> SecurityMatrix<T> {
        let mut sm =  SecurityMatrix {
            m : DMatrix::<T>::identity(dim + 1, dim + 1),
            size : dim};
        sm.m = Self::populate(&mut sm.m);
        sm.m = Self::make_diagonally_dominant(&mut sm.m);
        debug_assert!(sm.m.is_invertible());
        sm
    }
}

impl InvertibleMatrixBuilder<f64> for SecurityMatrix<f64> {
    fn _rand () -> f64 {
        let mut rng = rand::thread_rng();
        rng.gen::<f64>().abs()
    }
}

impl InvertibleMatrixBuilder<f32> for SecurityMatrix<f32> {
    fn _rand () -> f32 {
        let mut rng = rand::thread_rng();
        rng.gen::<f32>().abs()
    }
}

// impl InvertibleMatrixBuilder<i32> for SecurityMatrix<i32> {
//     fn _rand () -> i32 {
//         let mut rng = rand::thread_rng();
//         return rng.gen::<i32>().abs()
//     }
// }
