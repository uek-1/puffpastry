use crate::vec_tools::ValidNumber;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<const RANK: usize, T: ValidNumber<T>> {
    pub shape: [usize; RANK],
    pub data: Vec<T>,
}

impl<const RANK: usize, T: ValidNumber<T>> Tensor<RANK, T> {
    pub fn new(shape: [usize; RANK]) -> Tensor<RANK, T> {
        let size = shape.iter().fold(1, |size, x| size * x);
        Tensor {
            shape: shape,
            data: vec![T::from(0.0); size],
        }
    }

    fn calculate_data_index(&self, loc: [usize; RANK]) -> usize {
        loc.into_iter()
            .rev()
            .enumerate()
            .fold(0, |data_idx, (dim_idx, idx)| {
                data_idx
                    + idx
                        * match dim_idx {
                            0 => 1,
                            _ => self.shape[dim_idx],
                        }
            })
    }

    pub fn get(&self, loc: [usize; RANK]) -> Option<&T> {
        let idx = self.calculate_data_index(loc);
        self.data.get(idx)
    }

    pub fn get_mut(&mut self, loc: [usize; RANK]) -> Option<&mut T> {
        let idx = self.calculate_data_index(loc);
        self.data.get_mut(idx)
    }

    pub fn elementwise_product(&self, other: &Tensor<RANK, T>) -> Result<Tensor<RANK, T>, ()> {
        if self.shape != other.shape {
            return Err(());
        }

        let mut out = Tensor::new(self.shape.clone());
        println!("{:?}, {:?}", self.shape, out.shape);
        println!("{:?}", self.data);
        println!("{}", out.data.len());
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] * other.data[i];
        }

        Ok(out)
    }
}

impl<const RANK: usize, T: ValidNumber<T>> Add<Tensor<RANK, T>> for Tensor<RANK, T> {
    type Output = Tensor<RANK, T>;

    fn add(self, rhs: Tensor<RANK, T>) -> Self::Output {
        if self.shape != rhs.shape {
            panic!("Cannot add tensors of different shape!")
        }

        Tensor {
            shape: self.shape,
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(x, y)| *x + *y)
                .collect(),
        }
    }
}

impl<const RANK: usize, T: ValidNumber<T>> Sub<Tensor<RANK, T>> for Tensor<RANK, T> {
    type Output = Tensor<RANK, T>;

    fn sub(self, rhs: Tensor<RANK, T>) -> Self::Output {
        if self.shape != rhs.shape {
            panic!("Cannot add tensors of different shape")
        }

        Tensor {
            shape: self.shape,
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(x, y)| *x - *y)
                .collect(),
        }
    }
}

impl<const RANK: usize, T: ValidNumber<T>> Mul<T> for Tensor<RANK, T> {
    type Output = Tensor<RANK, T>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor {
            shape: self.shape,
            data: self.data.iter().map(|x| *x * rhs).collect(),
        }
    }
}

impl<const RANK: usize, T: ValidNumber<T>> Div<T> for Tensor<RANK, T> {
    type Output = Tensor<RANK, T>;

    fn div(self, rhs: T) -> Self::Output {
        if rhs == T::from(0.0) {
            panic!("Dividing by 0!")
        }

        Tensor {
            shape: self.shape,
            data: self.data.iter().map(|x| *x * rhs).collect(),
        }
    }
}

impl<T: ValidNumber<T>> Tensor<1, T> {
    pub fn from(v: Vec<T>) -> Tensor<1, T> {
        Tensor {
            shape: [v.len()],
            data: v,
        }
    }

    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data.iter()
    }
    pub fn dot_product(&self, other: &Tensor<1, T>) -> T {
        self.iter()
            .zip(other.iter())
            .fold(T::from(0.0), |res, (s, o)| res + *s * *o)
    }
}

impl<T: ValidNumber<T>> IntoIterator for Tensor<1, T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<T: ValidNumber<T>> Tensor<2, T> {
    pub fn from(v: Vec<Vec<T>>) -> Tensor<2, T> {
        let shape = [v.len(), v[0].len()];
        let data: Vec<T> = v.into_iter().fold(vec![], |mut data, mut x| {
            data.append(&mut x);
            data
        });

        Tensor { shape, data }
    }

    pub fn row_count(&self) -> usize {
        self.shape[0]
    }

    pub fn col_count(&self) -> usize {
        self.shape[1]
    }

    pub fn get_dims(&self) -> (usize, usize) {
        (self.shape[0], self.shape[1])
    }

    pub fn as_rows(&self) -> Vec<Tensor<1, T>> {
        let (_, cols) = self.get_dims();

        self.data
            .chunks_exact(cols)
            .map(|x| Tensor::<1, T>::from(x.to_vec()))
            .collect()
    }

    pub fn as_columns(&self) -> Vec<Tensor<1, T>> {
        let (rows, cols) = self.get_dims();

        (0..cols)
            .map(|x| {
                self.data
                    .iter()
                    .skip(x)
                    .step_by(cols)
                    .take(rows)
                    .cloned()
                    .collect()
            })
            .map(|x| Tensor::<1, T>::from(x))
            .collect()
    }

    pub fn matrix_multiply(&self, other: &Tensor<2, T>) -> Result<Tensor<2, T>, ()> {
        if self.shape[1] != other.shape[0] {
            return Err(());
        }

        let new_data = self
            .as_rows()
            .iter()
            .map(|row| {
                other
                    .as_columns()
                    .iter()
                    .map(|col| row.dot_product(col))
                    .collect()
            })
            .collect();

        Ok(Tensor::<2, T>::from(new_data))
    }
}

impl<T: ValidNumber<T>> Display for Tensor<2, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (rows, cols) = self.get_dims();

        for row in 0..rows {
            for col in 0..cols {
                write!(f, "{:?} ", self.get([row, col]).unwrap())?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn get_generic_tensor2d() -> Tensor<2, f64> {
        Tensor::<2, f64>::from(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
    }

    #[test]
    fn dot_product_test() {
        let x = Tensor::<1, f64>::from(vec![1.0, 2.0, 3.0]);
        let y = Tensor::<1, f64>::from(vec![1.0, 2.0, 3.0]);

        let z = x.dot_product(&y);

        assert_eq!(z, 14.0)
    }

    #[test]
    fn elementwise2d_test() {
        let x = Tensor::<2, f64>::from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        let y = Tensor::<2, f64>::from(vec![vec![1.0, 1.0, 1.0], vec![0.0, 1.0, 1.0]]);

        let res = x
            .elementwise_product(&y)
            .expect("Incorrect Dimension Error");

        assert_eq!(
            Tensor::<2, f64>::from(vec![vec![1.0, 2.0, 3.0], vec![0.0, 5.0, 6.0]]),
            res
        )
    }

    #[test]
    fn incorrect_elementwise2d_test() {
        let x = Tensor::<2, f64>::from(vec![vec![2.0, 3.0]]);
        let y = Tensor::<2, f64>::from(vec![vec![1.0, 2.0, 3.0]]);

        let res = x.elementwise_product(&y);

        assert!(res.is_err())
    }

    #[test]
    fn as_rows_test() {
        let x = Tensor::<2, f64>::from(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        assert_eq!(x.as_rows()[0], Tensor::<1, f64>::from(vec![1.0, 2.0, 3.0]));
        assert_eq!(x.as_rows()[1], Tensor::<1, f64>::from(vec![4.0, 5.0, 6.0]));
        assert_eq!(x.as_rows()[2], Tensor::<1, f64>::from(vec![7.0, 8.0, 9.0]))
    }

    #[test]
    fn tensor2d_matmul_1() {
        let x = get_generic_tensor2d();
        let y = get_generic_tensor2d();
        let res = x.matrix_multiply(&y).expect("e");

        println!("{res}");

        assert_eq!(
            Tensor::<2, f64>::from(vec![
                vec![30.0, 36.0, 42.0],
                vec![66.0, 81.0, 96.0],
                vec![102.0, 126.0, 150.0]
            ]),
            res
        )
    }

    #[test]
    fn main() {
        let x = get_generic_tensor2d();
        let y = get_generic_tensor2d();
        let res = x - y;
        println!("{res}");
        assert!(false)
    }
}
