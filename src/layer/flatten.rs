use super::{Activation, Layer, Tensor, ValidNumber};

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

    fn get_weights(&self) -> Option<Tensor<T>> {
        None
    }

    fn set_weights(&mut self, new_weights: Tensor<T>) -> Result<(), ()> {
        Err(())
    }

    fn get_activation(&self) -> Option<Activation> {
        None
    }

    // UNTESTED
    fn input_derivative(&self, input: &Tensor<T>, step_grad: &Tensor<T>) -> Result<Tensor<T>, ()> {
        let reshaped = Tensor {
            shape: input.shape().clone(),
            data: step_grad.data.clone(),
        };
        Ok(reshaped)
    }

    fn weights_derivative(
        &self,
        step_grad: &Tensor<T>,
        input: &Tensor<T>,
    ) -> Result<Option<Tensor<T>>, ()> {
        Ok(None)
    }
}
