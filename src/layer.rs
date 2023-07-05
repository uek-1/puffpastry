use rand::Rng;

use crate::activation::Activation;
use crate::vec_tools::{self, MatrixMultiply, Transpose, AddVec};
use crate::rand;


#[derive(Clone, Debug)]
pub struct Layer<T>{
    pub weights : Vec<Vec<T>>,
    pub biases : Vec<T>,
    pub activation : Activation,
}

impl<T : vec_tools::ValidNumber<T>> Layer<T> {
    pub fn from_size(input_size : usize, output_size : usize, activation : Activation) -> Layer<T> {
        let mut rng = rand::thread_rng();
        let mut weights = vec![vec![T::from(1.0); input_size]; output_size]; 

        for row in &mut weights  {
            for elem in row {
                *elem = T::from(rng.gen_range(0.0..1.0));
            }
        }

        let biases = vec![T::from(0.0); output_size];

        Layer {
            weights,
            biases,
            activation
        }

    }

    pub fn from_weights_biases(weights : Vec<Vec<T>>, biases : Vec<T>, activation : Activation) -> Layer<T> {
        Layer {
            weights,
            biases,
            activation
        }
    }

    pub fn evaluate(&self, input: &Vec<Vec<T>>) -> Vec<Vec<T>> {
        let out : Vec<Vec<T>> = self.weights
            .matrix_multiply(input)
            .transposed()
            .into_iter()
            .map(|x| x.add_vec(&self.biases))
            .map(|x| self.activation.activate_vec(x))
            .collect();

        out.transposed()
    }

    pub fn preactivation(&self, input: &Vec<Vec<T>>) -> Vec<Vec<T>> {
        self.weights
            .matrix_multiply(input)
            .transposed()
            .iter()
            .map(|x| x.add_vec(&self.biases))
            .collect::<Vec<Vec<T>>>()
            .transposed()
    }

    pub fn activate(&self, preactivation : Vec<Vec<T>>) -> Vec<Vec<T>> {
        preactivation
            .transposed()
            .into_iter()
            .map(|x| self.activation.activate_vec(x))
            .collect::<Vec<Vec<T>>>()
            .transposed()
    }
}
