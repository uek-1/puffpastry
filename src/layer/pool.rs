use super::{Activation, Layer, Tensor, ValidNumber};

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
    fn get_weights(&self) -> Option<Tensor<T>> {
        None
    }
    fn set_weights(&mut self, new_weights: Tensor<T>) -> Result<(), ()> {
        Err(())
    }

    fn get_activation(&self) -> Option<Activation> {
        None
    }

    fn input_derivative(&self, step_grad: &Tensor<T>) -> Result<Tensor<T>, ()> {
        todo!()
    }

    fn weights_derivative(
        &self,
        input: &Tensor<T>,
        step_grad: &Tensor<T>,
    ) -> Result<Option<Tensor<T>>, ()> {
        Ok(None)
    }
}
