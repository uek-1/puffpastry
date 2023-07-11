use crate::vec_tools::ValidNumber;

#[derive(Debug)]
pub struct Tensor<const DIM: usize, T: ValidNumber<T>> {
    shape: [usize; DIM],
    data: Vec<T>,
}

impl<const DIM: usize, T: ValidNumber<T>> Tensor<DIM, T> {
    pub fn new(shape: [usize; DIM]) -> Tensor<DIM, T> {
        let size = shape.iter().fold(0, |size, x| size * x);
        Tensor {
            shape: shape,
            data: vec![T::from(0.0); size],
        }
    }

    fn calculate_data_index(&self, loc: [usize; DIM]) -> usize {
        loc.into_iter()
            .enumerate()
            .fold(0, |data_idx, (dim_idx, idx)| {
                data_idx + idx * self.shape[dim_idx]
            })
    }

    pub fn get(&self, loc: [usize; DIM]) -> Option<&T> {
        let idx = self.calculate_data_index(loc);
        self.data.get(idx)
    }

    pub fn get_mut(&mut self, loc: [usize; DIM]) -> Option<&mut T> {
        let idx = self.calculate_data_index(loc);
        self.data.get_mut(idx)
    }

    pub fn elementwise_product(&self, other: Tensor<DIM, T>) -> Result<Tensor<DIM, T>, ()> {
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

    // pub fn as_rows(&self) -> Vec<Tensor<1, T>> {}

    // pub fn as_columns(&self) -> Vec<Tensor<1, T>> {}
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dot_product_test() {
        let x = Tensor::<1, f64>::from(vec![1.0, 2.0, 3.0]);
        let y = Tensor::<1, f64>::from(vec![1.0, 2.0, 3.0]);

        let z = x.dot_product(&y);

        assert_eq!(z, 14.0)
    }
}
