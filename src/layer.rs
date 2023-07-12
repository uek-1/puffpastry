use rand::Rng;

use crate::activation::Activation;
use crate::rand;
use crate::tensor::Tensor;
use crate::vec_tools::ValidNumber;

#[derive(Clone, Debug)]
pub struct Dense<T: ValidNumber<T>> {
    pub weights: Tensor<2, T>,
    pub biases: Tensor<2, T>,
    pub activation: Activation,
}

impl<T: ValidNumber<T>> Dense<T> {
    pub fn from_size(input_size: usize, output_size: usize, activation: Activation) -> Dense<T> {
        let mut rng = rand::thread_rng();
        let mut weights = Tensor::new([output_size, input_size]);

        for elem in &mut weights.data {
            *elem = T::from(rng.gen_range(0.0..1.0));
        }

        let biases = Tensor::new([input_size, 1]);

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
            weights: Tensor::<2, T>::from(weights),
            biases: Tensor::<2, T>::from(biases),
            activation,
        }
    }
}

impl<T: ValidNumber<T>> Layer<T> for Dense<T> {
    fn evaluate<const I: usize, const O: usize>(&self, input: &Tensor<I, T>) -> Tensor<O, T> {
        let out: Vec<Vec<T>> = self
            .weights
            .matrix_multiply(input)
            .transposed()
            .into_iter()
            .map(|x| x.add_vec(&self.biases))
            .map(|x| self.activation.activate_vec(x))
            .collect();

        out.transposed()
    }

    fn preactivate(&self, input: &Tensor<1, T>) -> Tensor<1, T> {
        self.weights
            .matrix_multiply(input)
            .transposed()
            .iter()
            .map(|x| x.add_vec(&self.biases))
            .collect::<Vec<Vec<T>>>()
            .transposed()
    }

    fn activate(&self, preactivation: Vec<Vec<T>>) -> Vec<Vec<T>> {
        preactivation
            .transposed()
            .into_iter()
            .map(|x| self.activation.activate_vec(x))
            .collect::<Vec<Vec<T>>>()
            .transposed()
    }
}

pub trait Layer<T: ValidNumber<T>> {
    fn evaluate<const I: usize, const O: usize>(&self, input: &Tensor<I, T>) -> Tensor<O, T>
    where
        Self: Sized;

    fn preactivate<const I: usize, const O: usize>(&self, input: &Tensor<I, T>) -> Tensor<O, T>
    where
        Self: Sized;

    fn activate<const I: usize, const O: usize>(&self, input: &Tensor<I, T>) -> Tensor<O, T>
    where
        Self: Sized;
}
