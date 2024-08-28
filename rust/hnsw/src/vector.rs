use std::ops::{Mul, Sub};
use std::iter::Sum;

#[derive(Clone, Debug)]
pub struct VectorItem<T> {
    pub id: usize,
    pub vector: Vec<T>,
}


pub trait DistanceCalculator<T> {
    fn calculate(&self, item1: &VectorItem<T>, item2: &VectorItem<T>) -> T;
}

pub enum DistanceType {
    Euclidean,
}

pub struct Distance {
    pub _type : DistanceType,
}

pub trait EuclideanDistanceCalculator<
    T : Sub<T> +
    Mul<T> +
    Copy +
    Into<<T as Sub>::Output> +
    From<<T as Sub>::Output> +
    std::convert::From<<T as std::ops::Mul>::Output> +
    Sum<T>> : DistanceCalculator<T> {
    fn sqrt(&self, item : &T) -> T;
    fn calculate(&self, item1: &VectorItem<T>, item2: &VectorItem<T>) -> T {
        <Self as EuclideanDistanceCalculator<T>>::sqrt(self,
        &item1
                .vector
                .iter()
                .zip(item2.vector.iter())
                .map(|(x, y)| (<<T as Sub>::Output as Into<T>>::into(*x - *y) *
                                       <<T as Sub>::Output as Into<T>>::into(*x - *y))
                                       .into()) // to implement as power
            .sum::<T>())
    }
}

impl EuclideanDistanceCalculator<f64> for Distance {
    fn sqrt(&self, item : &f64) -> f64{
        item.sqrt()
    }
}

impl DistanceCalculator<f64> for Distance {
    fn calculate(&self, item1: &VectorItem<f64>, item2: &VectorItem<f64>) -> f64 {
        match self._type {
            DistanceType::Euclidean => EuclideanDistanceCalculator::calculate(self, item1, item2)
        }
    }
}
