use rand::Rng;
use std::fmt::Debug;

use crate::activation::Activation;
use crate::rand;
use crate::tensor::Tensor;
use crate::vec_tools::ValidNumber;

pub trait Layer<T: ValidNumber<T>>: Debug {
    fn evaluate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()>;

    fn preactivate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()>;

    fn activate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()>;

    fn get_weights(&self) -> Tensor<T>;

    fn set_weights(&mut self, new_weights: Tensor<T>);

    fn get_activation(&self) -> Activation;
}

#[derive(Clone, Debug)]
pub struct Dense<T: ValidNumber<T>> {
    pub weights: Tensor<T>,
    pub biases: Tensor<T>,
    pub activation: Activation,
}

impl<T: ValidNumber<T>> Dense<T> {
    pub fn from_size(input_size: usize, output_size: usize, activation: Activation) -> Dense<T> {
        let mut rng = rand::thread_rng();
        let mut weights = Tensor::new(vec![output_size, input_size]);

        for elem in &mut weights.data {
            *elem = T::from(rng.gen_range(0.0..1.0));
        }

        let biases = Tensor::new(vec![output_size, 1]);

        Dense {
            weights,
            biases,
            activation,
        }
    }

    pub fn from_weights_biases(
        weights: Vec<Vec<T>>,
        biases: Vec<Vec<T>>,
        activation: Activation,
    ) -> Dense<T> {
        Dense {
            weights: Tensor::<T>::from(weights),
            biases: Tensor::<T>::from(biases),
            activation,
        }
    }
}

impl<T: ValidNumber<T>> Layer<T> for Dense<T> {
    fn evaluate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        let preactivation = self.preactivate(input);
        self.activate(&preactivation?)
    }

    fn preactivate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        Ok(self.weights.matrix_multiply(input)? + self.biases.clone())
    }

    fn activate(&self, preactivation: &Tensor<T>) -> Result<Tensor<T>, ()> {
        self.activation.activate_tensor2d(preactivation)
    }

    fn get_weights(&self) -> Tensor<T> {
        self.weights.clone()
    }

    fn set_weights(&mut self, new_weights: Tensor<T>) {
        if self.weights.shape != new_weights.shape {
            panic!(
                "Incorrect shape to update weights! Current: {:?} Input {:?}",
                self.weights.shape, new_weights.shape
            );
        }
        self.weights = new_weights;
    }

    fn get_activation(&self) -> Activation {
        self.activation.clone()
    }
}

#[derive(Clone, Debug)]
pub struct Conv2d<T: ValidNumber<T>> {
    weights: Tensor<T>,
    activation: Activation,
}

// Note: assumes 0 if out of boundary on input
impl<T: ValidNumber<T>> Conv2d<T> {
    pub fn from_size(
        input_depth: usize,
        filter_size: usize,
        filter_count: usize,
        stride: (usize, usize),
        activation: Activation,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = Tensor::new(vec![filter_count, input_depth, filter_size, filter_size]);

        weights
            .data
            .iter_mut()
            .for_each(|x| *x = T::from(rng.gen_range(0.0..1.0)));

        Conv2d {
            weights,
            activation,
        }
    }

    // export .depth, .cols, .rows into sep function
    fn convolve2d(&self, input: &Tensor<T>, loc: Vec<usize>, filter_num: usize) -> Result<T, ()> {
        if input.rank() != 3 || self.get_weights().shape()[1] != input.shape()[0] || loc.len() != 2
        {
            return Err(());
        }

        let depth = self.get_weights().shape()[1];
        let height = self.get_weights().shape()[2];
        let width = self.get_weights().shape()[3];

        let mut out = T::from(0.0);
        for i in 0..depth {
            for j in 0..height {
                for k in 0..width {
                    let input_loc = vec![i, loc[0] + j, loc[1] + k];
                    let filter_loc: Vec<usize> = vec![filter_num, i, j, k];

                    println!("multiplying filter {filter_loc:?} with input {input_loc:?}");
                    let filter_item = *self.weights.get(&filter_loc).ok_or(())?;
                    let input_item = *input.get(&input_loc).unwrap_or(&T::from(0.0));
                    println!("filter {filter_item:?} * input {input_item:?}");
                    out = out + (filter_item * input_item);
                }
            }
        }

        Ok(out)
    }

    fn convolve2d_all(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        let mut data: Vec<T> = vec![];

        let [filter_count, _filter_depth, _filter_height, _filter_width] = self.weights.shape()[..] else {
            return Err(())
        };
        let [input_height, input_width] = input.shape()[1..] else {
            return Err(())
        };

        for fnum in 0..filter_count {
            for height in 0..(input_height) {
                for width in 0..(input_width) {
                    let loc = vec![fnum, height, width];
                    println!("current conv loc {loc:?}");
                    let item = self.convolve2d(input, vec![height, width], fnum)?;
                    data.push(item);
                }
            }
        }

        Ok(Tensor {
            shape: vec![filter_count, input_height, input_width],
            data: data,
        })
    }
}

impl<T: ValidNumber<T>> Layer<T> for Conv2d<T> {
    fn evaluate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        let preactivation = self.preactivate(input)?;
        self.activate(&preactivation)
    }

    fn preactivate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        self.convolve2d_all(input)
    }

    fn activate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        self.activation.activate_tensor(input)
    }

    fn get_weights(&self) -> Tensor<T> {
        self.weights.clone()
    }

    fn set_weights(&mut self, new_weights: Tensor<T>) {
        self.weights = new_weights;
    }

    fn get_activation(&self) -> Activation {
        self.activation.clone()
    }
}

#[derive(Debug)]
pub struct MaxPool2d {
    window: (usize, usize),
}

impl MaxPool2d {
    pub fn new(window_height: usize, window_width: usize) -> Self {
        MaxPool2d {
            window: (window_height, window_width),
        }
    }

    fn max_pool_tensor<T: ValidNumber<T>>(
        &self,
        loc: Vec<usize>,
        input: &Tensor<T>,
    ) -> Result<T, ()> {
        let (window_height, window_width) = self.window;

        let mut elems: Vec<T> = vec![];

        for i in 0..window_height {
            for j in 0..window_width {
                let mut current_loc = loc.clone();
                current_loc[1] += i;
                current_loc[2] += j;

                elems.push(*input.get(&current_loc).ok_or(())?);
            }
        }

        elems
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or(())
    }
}

impl<T: ValidNumber<T>> Layer<T> for MaxPool2d {
    fn evaluate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        let preactivation = self.preactivate(input)?;
        self.activate(&preactivation)
    }

    fn preactivate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        let [input_depth, input_height, input_width] = input.shape()[0..3] else {
            return Err(())
        };

        let (window_height, window_width) = self.window;

        let mut data = vec![];
        let shape = vec![
            input_depth,
            input_height / window_height,
            input_width / window_width,
        ];

        for depth in 0..input_depth {
            for height in (0..input_height).step_by(window_height) {
                for width in (0..input_width).step_by(window_width) {
                    let loc = vec![depth, height, width];
                    data.push(self.max_pool_tensor(loc, input)?);
                }
            }
        }

        Ok(Tensor { shape, data })
    }
    fn activate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        Ok(input.clone())
    }
    fn get_weights(&self) -> Tensor<T> {
        panic!("No weights on a MaxPool layer!")
    }
    fn set_weights(&mut self, new_weights: Tensor<T>) {
        panic!("No weights on a MaxPool layer!")
    }
    fn get_activation(&self) -> Activation {
        panic!("No activation on a MaxPool layer!")
    }
}

#[derive(Debug)]
pub struct Flatten {}

impl<T: ValidNumber<T>> Layer<T> for Flatten {
    fn evaluate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        self.preactivate(input)
    }

    fn preactivate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        Ok(Tensor::column(input.data.clone()))
    }

    fn activate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        Ok(input.clone())
    }

    fn get_weights(&self) -> Tensor<T> {
        panic!("No weights on a flatten layer!")
    }

    fn set_weights(&mut self, new_weights: Tensor<T>) {
        panic!("No weights on a flatten layer!")
    }

    fn get_activation(&self) -> Activation {
        panic!("No weights on a flatten layer!")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn convolve_single_test() {
        let weights = Tensor {
            shape: vec![1, 2, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        };

        let conv2d: Conv2d<f64> = Conv2d {
            weights,
            activation: Activation::None,
        };

        let input = Tensor::from(vec![
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        ]);

        let res = conv2d.convolve2d(&input, vec![0, 0], 0);

        assert_eq!(res, Ok(10.0));
    }

    #[test]
    fn convolve_multiple_test() {
        let weights = Tensor {
            shape: vec![1, 2, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        };

        let conv2d: Conv2d<f64> = Conv2d {
            weights,
            activation: Activation::None,
        };

        let input = Tensor::from(vec![
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        ]);

        let res = conv2d.convolve2d_all(&input);

        let expec = Tensor::from(vec![vec![vec![10.0, 4.0], vec![3.0, 1.0]]]);

        println!("{res:?}");
        assert_eq!(res, Ok(expec));
    }

    #[test]
    fn convolve_multiple_test2() {
        let mut data = vec![];

        let filter1 = vec![vec![vec![1.0, 0.0], vec![1.0, 0.0]]]
            .into_iter()
            .flatten()
            .flatten()
            .for_each(|x| data.push(x));

        let filter2 = vec![vec![vec![1.0, 1.0], vec![0.0, 0.0]]]
            .into_iter()
            .flatten()
            .flatten()
            .for_each(|x| data.push(x));

        println!("{data:?}");

        let weights = Tensor {
            shape: vec![2, 1, 2, 2],
            data: data,
        };

        let conv2d: Conv2d<f64> = Conv2d {
            weights,
            activation: Activation::None,
        };

        let input = Tensor::from(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);

        let res = conv2d.convolve2d_all(&input).unwrap();

        println!("{res:?}");

        assert_eq!(res.shape, vec![2, 2, 2])
    }
}
