use super::{Activation, Debug, Tensor, ValidNumber};

pub trait Layer<T: ValidNumber<T>>: Debug {
    fn evaluate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()>;

    fn preactivate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()>;

    fn activate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()>;

    fn get_weights(&self) -> Option<Tensor<T>>;

    fn set_weights(&mut self, new_weights: Tensor<T>) -> Result<(), ()>;

    fn get_activation(&self) -> Option<Activation>;

    fn input_derivative(&self, input: &Tensor<T>, step_grad: &Tensor<T>) -> Result<Tensor<T>, ()>;

    fn weights_derivative(
        &self,
        input: &Tensor<T>,
        step_grad: &Tensor<T>,
    ) -> Result<Option<Tensor<T>>, ()>;
}
