use std::{fmt::Display, ops::Sub};

use nalgebra::{ClosedAddAssign, RealField, Scalar};
use num_traits::{One, Zero};
use rand::Rng;

use crate::matrix::{InvertibleMatrixBuilder, SecurityMatrix};

pub struct Configuration<T> {
    pub m1 : SecurityMatrix<T>,
    pub m2 : SecurityMatrix<T>,
    pub s : Vec<i8>
}

pub trait ConfigurationInitializer<T :
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
    Copy> {

    fn generate_random_vec(dim : usize) -> Vec<i8> {
        let mut v = Vec::<i8>::new();
        let mut rng = rand::thread_rng();
        for _ in 0..dim{
            if rng.gen::<i8>() % 2 == 0 {
                v.push(0)
            } else {
                v.push(1)
            }
        }
        v
    }

    fn init (dim : usize) -> Configuration<T>;
}

impl ConfigurationInitializer<f64> for Configuration<f64> {
    fn init (dim : usize) -> Configuration<f64> {
        Configuration {
            m1 : SecurityMatrix::<f64>::build(dim + 1),
            m2 : SecurityMatrix::<f64>::build(dim + 1),
            s : Self::generate_random_vec(dim + 1)
        }
    }
}

impl ConfigurationInitializer<f32> for Configuration<f32> {

    fn init (dim : usize) -> Configuration<f32> {
        Configuration {
            m1 : SecurityMatrix::<f32>::build(dim + 1),
            m2 : SecurityMatrix::<f32>::build(dim + 1),
            s : Self::generate_random_vec(dim + 1)
        }
    }
}