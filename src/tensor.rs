use crate::vec_tools::ValidNumber;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<T: ValidNumber<T>> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

/// Functions for all ranks
impl<T: ValidNumber<T>> Tensor<T> {
    pub fn new(shape: Vec<usize>) -> Tensor<T> {
        let size = shape.iter().fold(1, |size, x| size * x);
        Tensor {
            shape: shape,
            data: vec![T::from(0.0); size],
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn calculate_data_index(&self, loc: &[usize]) -> usize {
        loc.into_iter()
            .rev()
            .enumerate()
            .fold(0, |data_idx, (loc_idx, loc_val)| {
                data_idx
                    + loc_val
                        * match loc_idx {
                            0 => 1,
                            _ => self
                                .shape
                                .clone()
                                .iter()
                                .rev()
                                .take(loc_idx)
                                .fold(1, |prod, x| prod * x),
                        }
            })
    }

    pub fn get(&self, loc: &[usize]) -> Option<&T> {
        if loc.len() != self.rank() {
            return None;
        }

        if (0..loc.len())
            .into_iter()
            .any(|idx| loc[idx] >= self.shape[idx])
        {
            return None;
        }

        let idx = self.calculate_data_index(loc);
        self.data.get(idx)
    }

    pub fn get_mut(&mut self, loc: &[usize]) -> Option<&mut T> {
        if loc.len() != self.rank() {
            return None;
        }

        let idx = self.calculate_data_index(loc);
        self.data.get_mut(idx)
    }

    pub fn elementwise_product(&self, other: &Tensor<T>) -> Result<Tensor<T>, ()> {
        if self.shape != other.shape {
            return Err(());
        }

        let mut out = Tensor::new(self.shape.clone());
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] * other.data[i];
        }

        Ok(out)
    }
}

impl<T: ValidNumber<T>> Add<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Tensor<T>) -> Self::Output {
        if self.shape != rhs.shape {
            panic!("Cannot add tensors of different shape: lhs {self:?},  rhs: {rhs:?}")
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

impl<T: ValidNumber<T>> Sub<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Tensor<T>) -> Self::Output {
        if self.shape != rhs.shape {
            panic!("Cannot subtract tensors of different shape: lhs {self:?},  rhs: {rhs:?}")
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

impl<T: ValidNumber<T>> Mul<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor {
            shape: self.shape,
            data: self.data.iter().map(|x| *x * rhs).collect(),
        }
    }
}

impl<T: ValidNumber<T>> Div<T> for Tensor<T> {
    type Output = Tensor<T>;

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

impl<T: ValidNumber<T>> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        Tensor {
            shape: vec![value.len()],
            data: value,
        }
    }
}

impl<T: ValidNumber<T>> From<Vec<Vec<T>>> for Tensor<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        let shape = vec![value.len(), value[0].len()];
        let data: Vec<T> = value.into_iter().flatten().collect();

        Tensor { shape, data }
    }
}

impl<T: ValidNumber<T>> From<Vec<Vec<Vec<T>>>> for Tensor<T> {
    fn from(value: Vec<Vec<Vec<T>>>) -> Self {
        let shape = vec![value.len(), value[0].len(), value[0][0].len()];
        let data: Vec<T> = value.into_iter().flatten().flatten().collect();

        Tensor { shape, data }
    }
}

// Functions for rank 1

impl<T: ValidNumber<T>> Tensor<T> {
    pub fn dot_product(&self, other: &Tensor<T>) -> Result<T, ()> {
        if self.rank() != 1 || (self.shape() != other.shape()) {
            return Err(());
        }

        Ok(self
            .iter()
            .zip(other.iter())
            .fold(T::from(0.0), |res, (s, o)| res + *s * *o))
    }
}

impl<T: ValidNumber<T>> Tensor<T> {
    pub fn row_count(&self) -> usize {
        self.shape[0]
    }

    pub fn col_count(&self) -> usize {
        self.shape[1]
    }

    pub fn get_2dims(&self) -> (usize, usize) {
        if self.rank() != 2 {
            panic!("only defined for rank 2 tensors!")
        }
        (self.shape[0], self.shape[1])
    }

    pub fn as_rows(&self) -> Vec<Tensor<T>> {
        let (_, cols) = self.get_2dims();

        self.data
            .chunks_exact(cols)
            .map(|x| Tensor::from(x.to_vec()))
            .collect()
    }

    pub fn as_columns(&self) -> Vec<Tensor<T>> {
        let (rows, cols) = self.get_2dims();

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
            .map(|x: Vec<T>| Tensor::from(x))
            .collect()
    }

    pub fn matrix_multiply(&self, other: &Tensor<T>) -> Result<Tensor<T>, ()> {
        if self.shape[1] != other.shape[0] {
            return Err(());
        }

        let new_data: Vec<Vec<T>> = self
            .as_rows()
            .iter()
            .map(|row| {
                other
                    .as_columns()
                    .iter()
                    .map(|col| row.dot_product(col))
                    .collect::<Result<Vec<T>, ()>>()
            })
            .collect::<Result<Vec<Vec<T>>, ()>>()?;

        Ok(Tensor::from(new_data))
    }

    pub fn transposed(&self) -> Tensor<T> {
        Tensor {
            shape: vec![self.shape[1], self.shape[0]],
            data: self.data.clone(),
        }
    }

    pub fn column(data: Vec<T>) -> Tensor<T> {
        let data: Vec<Vec<T>> = data.into_iter().map(|x| vec![x]).collect();

        Tensor::from(data)
    }
}

impl<T: ValidNumber<T>> Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (rows, cols) = self.get_2dims();

        for row in 0..rows {
            for col in 0..cols {
                write!(f, "{:?} ", self.get(&[row, col]).unwrap())?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn get_generic_tensor2d() -> Tensor<f64> {
        Tensor::<f64>::from(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
    }

    #[test]
    fn dot_product_test() {
        let x = Tensor::<f64>::from(vec![1.0, 2.0, 3.0]);
        let y = Tensor::<f64>::from(vec![1.0, 2.0, 3.0]);

        let z = x.dot_product(&y);

        assert_eq!(z.unwrap(), 14.0)
    }

    #[test]
    fn elementwise2d_test() {
        let x = Tensor::<f64>::from(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        let y = Tensor::<f64>::from(vec![vec![1.0, 1.0, 1.0], vec![0.0, 1.0, 1.0]]);

        let res = x
            .elementwise_product(&y)
            .expect("Incorrect Dimension Error");

        assert_eq!(
            Tensor::<f64>::from(vec![vec![1.0, 2.0, 3.0], vec![0.0, 5.0, 6.0]]),
            res
        )
    }

    #[test]
    fn incorrect_elementwise2d_test() {
        let x = Tensor::<f64>::from(vec![vec![2.0, 3.0]]);
        let y = Tensor::<f64>::from(vec![vec![1.0, 2.0, 3.0]]);

        let res = x.elementwise_product(&y);

        assert!(res.is_err())
    }

    #[test]
    fn as_rows_test() {
        let x = Tensor::<f64>::from(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        assert_eq!(x.as_rows()[0], Tensor::<f64>::from(vec![1.0, 2.0, 3.0]));
        assert_eq!(x.as_rows()[1], Tensor::<f64>::from(vec![4.0, 5.0, 6.0]));
        assert_eq!(x.as_rows()[2], Tensor::<f64>::from(vec![7.0, 8.0, 9.0]))
    }

    #[test]
    fn tensor2d_matmul_1() {
        let x = get_generic_tensor2d();
        let y = get_generic_tensor2d();
        let res = x.matrix_multiply(&y).expect("e");

        println!("{res}");

        assert_eq!(
            Tensor::<f64>::from(vec![
                vec![30.0, 36.0, 42.0],
                vec![66.0, 81.0, 96.0],
                vec![102.0, 126.0, 150.0]
            ]),
            res
        )
    }

    #[test]
    fn index_calc_test() {
        let data: Vec<f64> = (0..=8).into_iter().map(|x| x as f64).collect();

        let tensor1d = Tensor {
            shape: vec![8],
            data: data.clone(),
        };

        let tensor2d = Tensor {
            shape: vec![2, 4],
            data: data.clone(),
        };

        let tensor3d = Tensor {
            shape: vec![2, 2, 2],
            data: data.clone(),
        };

        assert_eq!(tensor1d.calculate_data_index(&[2]), 2);
        assert_eq!(tensor2d.calculate_data_index(&[1, 1]), 5);
        assert_eq!(tensor3d.calculate_data_index(&[1, 1, 0]), 6);
    }

    #[test]
    fn main() {
        let x = get_generic_tensor2d();
        let y = get_generic_tensor2d();
        let res = x - y;
        println!("{res}");
        assert!(true)
    }
}
