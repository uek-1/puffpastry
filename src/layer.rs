use rand::Rng;

use crate::activation::Activation;
use crate::rand;
use crate::tensor::Tensor;
use crate::vec_tools::ValidNumber;

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

        let biases = Tensor::new(vec![input_size, 1]);

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
        let preactivation = self.weights.matrix_multiply(input)? + self.biases.clone();
        self.activate(&preactivation)
    }

    fn preactivate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        Ok(self.weights.matrix_multiply(input)? + self.biases.clone())
    }

    fn activate(&self, preactivation: &Tensor<T>) -> Result<Tensor<T>, ()> {
        self.activation.activate_tensor2d(preactivation)
    }
}

pub trait Layer<T: ValidNumber<T>> {
    fn evaluate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()>;

    fn preactivate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()>;

    fn activate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()>;
}
