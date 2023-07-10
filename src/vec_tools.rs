use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

pub trait ValidNumber<T>:
    Copy
    + Add<T, Output = T>
    + Mul<T, Output = T>
    + Sub<T, Output = T>
    + Div<T, Output = T>
    + From<f64>
    + Into<f64>
    + Debug
    + PartialOrd
{
}

impl<T> ValidNumber<T> for T where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Sub<T, Output = T>
        + Div<T, Output = T>
        + From<f64>
        + Into<f64>
        + Debug
        + PartialOrd
{
}

pub trait AddVec<T> {
    fn add_vec(&self, other: &Vec<T>) -> Vec<T>;
}

impl<T: ValidNumber<T>> AddVec<T> for Vec<T> {
    fn add_vec(&self, other: &Vec<T>) -> Vec<T> {
        self.iter()
            .zip(other.iter())
            .map(|(x, y)| *x + *y)
            .collect()
    }
}

pub trait ElementwiseMatrixMultiply<T> {
    fn elementwise_multiply(&self, other: &Vec<Vec<T>>) -> Vec<Vec<T>>;
}

impl<T: ValidNumber<T>> ElementwiseMatrixMultiply<T> for Vec<Vec<T>> {
    fn elementwise_multiply(&self, other: &Vec<Vec<T>>) -> Vec<Vec<T>> {
        let mut out = vec![vec![T::from(0.0); self[0].len()]; self.len()];

        for row in 0..out.len() {
            for col in 0..out[row].len() {
                out[row][col] = self[row][col] * other[row][col];
            }
        }

        out
    }
}

pub trait DotProduct<T> {
    fn dot_product(&self, right: &Vec<T>) -> T;
}

impl<T: ValidNumber<T>> DotProduct<T> for Vec<T> {
    fn dot_product(&self, right: &Vec<T>) -> T {
        self.iter()
            .zip(right.iter())
            .fold(T::from(0.0), |acc, (x, y)| acc + (*x) * (*y))
    }
}

pub trait Transpose<T> {
    fn transposed(&self) -> Vec<Vec<T>>;
}

impl<T: ValidNumber<T>> Transpose<T> for Vec<Vec<T>> {
    fn transposed(&self) -> Vec<Vec<T>> {
        let mut out = vec![vec![self[0][0]; self.len()]; self[0].len()];

        for row in 0..self.len() {
            for col in 0..self[row].len() {
                out[col][row] = self[row][col];
            }
        }

        out
    }
}

pub trait MatrixMultiply<T> {
    fn matrix_multiply(&self, other: &Vec<Vec<T>>) -> Vec<Vec<T>>;
}

impl<T: ValidNumber<T>> MatrixMultiply<T> for Vec<Vec<T>> {
    fn matrix_multiply(&self, other: &Vec<Vec<T>>) -> Vec<Vec<T>> {
        let mut out: Vec<Vec<T>> = vec![vec![other[0][0]; other[0].len()]; self.len()];
        let other = other.transposed();

        for row in 0..out.len() {
            for col in 0..out[0].len() {
                out[row][col] = self[row].dot_product(&other[col]);
            }
        }

        out
    }
}

mod test {
    use super::*;

    #[test]
    fn mat_mul_test_1() {
        let mat1 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mat2 = vec![vec![1.0], vec![2.0]];
        let res = mat1.matrix_multiply(&mat2);

        assert_eq!(res, mat2);
    }

    #[test]
    fn mat_mul_test_2() {
        let mat1 = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mat2 = vec![vec![1.0], vec![2.0], vec![3.0]];
        let res = mat1.matrix_multiply(&mat2);
        let truth = vec![vec![14.0], vec![32.0], vec![50.0]];
    }

    #[test]
    fn row_transpose_test() {
        let input = vec![1.0, 2.0, 3.0];
        let res = vec![input.clone()].transposed().transposed();

        assert_eq!(vec![input], res)
    }

    #[test]
    fn elem_mult_test() {
        let mat1 = vec![vec![1.0, 2.0, 3.0]];
        let mat2 = vec![vec![3.0, 2.0, 1.0]];

        let res = mat1.transposed().elementwise_multiply(&mat2.transposed());

        assert_eq!(res, vec![vec![3.0, 4.0, 3.0]].transposed());
    }
}
