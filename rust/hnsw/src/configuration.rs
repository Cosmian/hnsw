use std::{fmt::Display, ops::Sub};

use nalgebra::{ClosedAddAssign, RealField, Scalar};
use num_traits::{One, Zero};

use crate::matrix::{InvertibleMatrixBuilder, SecurityMatrix};

pub struct Configuration<T> {
    pub m1 : SecurityMatrix<T>,
    pub m2 : SecurityMatrix<T>,
    pub s : Vec<T>
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
    fn init (dim : usize) -> Configuration<T>;
}

impl ConfigurationInitializer<f64> for Configuration<f64> {
    fn init (dim : usize) -> Configuration<f64> {
        return Configuration {
            m1 : SecurityMatrix::<f64>::build(dim),
            m2 : SecurityMatrix::<f64>::build(dim),
            s : Vec::<f64>::new()
        }
    }
}

impl ConfigurationInitializer<f32> for Configuration<f32> {

    fn init (dim : usize) -> Configuration<f32> {
        return Configuration {
            m1 : SecurityMatrix::<f32>::build(dim),
            m2 : SecurityMatrix::<f32>::build(dim),
            s : Vec::<f32>::new()
        }
    }
}