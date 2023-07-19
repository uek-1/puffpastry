use super::{Activation, Layer, Rng, Tensor, ValidNumber};

#[derive(Clone, Debug)]
pub struct Conv2d<T: ValidNumber<T>> {
    weights: Tensor<T>,
    activation: Activation,
    stride: (usize, usize),
    input_shape: Vec<usize>,
}

// Note: assumes 0 if out of boundary on input
impl<T: ValidNumber<T>> Conv2d<T> {
    pub fn from_size(
        input_shape: Vec<usize>,
        filter_size: usize,
        filter_count: usize,
        stride: (usize, usize),
        activation: Activation,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let input_depth = input_shape[0];
        let mut weights = Tensor::new(vec![filter_count, input_depth, filter_size, filter_size]);

        weights
            .data
            .iter_mut()
            .for_each(|x| *x = T::from(rng.gen_range(0.0..1.0)));

        Conv2d {
            weights,
            activation,
            stride,
            input_shape,
        }
    }

    // export .depth, .cols, .rows into sep function
    fn convolve2d(&self, input: &Tensor<T>, loc: Vec<usize>, filter_num: usize) -> Result<T, ()> {
        let layer_weights = self.get_weights().ok_or(())?;

        if input.rank() != 3 || layer_weights.shape()[1] != input.shape()[0] || loc.len() != 2 {
            return Err(());
        }

        let [depth, height, width] = layer_weights.shape()[1..=3] else {
            return Err(())
        };

        let mut out = T::from(0.0);
        for i in 0..depth {
            for j in 0..height {
                for k in 0..width {
                    let input_loc = vec![i, loc[0] + j, loc[1] + k];
                    let filter_loc: Vec<usize> = vec![filter_num, i, j, k];

                    let filter_item = *self.weights.get(&filter_loc).ok_or(())?;
                    let input_item = *input.get(&input_loc).unwrap_or(&T::from(0.0));
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

    fn input_derivative_helper(&self, loc: &[usize], step_grad: &Tensor<T>, out: &mut Tensor<T>) {
        // iterate through the volume that created this output value:
        // rangewidth [width, width + filter_width)
        // rangeheight [height, height + filter_height)
        // rangedepth [0, input_depth)
        // add fitler[rangedepth][rangeheight][rangewidth] * stepgrad[d,h,w] to input[rd, rh, rw].
        //conv_input_derivative_adder (priv functino)
        // - creates volume
        // - iterates through volume
        // - calculates dj/dx the position (filter w * dj/dz)
        // - adds to &mut out.
        let &[_, filter_depth, filter_height, filter_width] = &self.weights.shape()[..] else {
            panic!()
        };

        let &[output_depth, output_height, output_width] = loc else {
            panic!()
        };

        let range_depth = 0..filter_depth; //here filter_depth == input_depth
        let range_height = output_height..(output_height + filter_height);
        let range_width = output_width..(output_width + filter_width);

        for depth in range_depth {
            for height in range_height.clone() {
                for width in range_width.clone() {
                    let filter_weight = *self
                        .weights
                        .get(&[
                            output_depth,
                            depth,
                            height - output_height,
                            width - output_width,
                        ])
                        .expect("There should always be a weight here!");

                    //SAFETY, loc is created by iterating through step_grad.
                    let step_grad_derivative = *step_grad.get(loc).unwrap();

                    match out.get_mut(&[depth, height, width]) {
                        //broek!
                        Some(x) => *x = *x + (filter_weight * step_grad_derivative),
                        None => (),
                    }
                }
            }
        }
    }

    fn weights_derivative_helper(
        &self,
        loc: &[usize],
        input: &Tensor<T>,
        step_grad: &Tensor<T>,
        out: &mut Tensor<T>,
    ) {
        let &[_, filter_depth, filter_height, filter_width] = &self.weights.shape()[..] else {
            panic!()
        };

        let &[output_depth, output_height, output_width] = loc else {
            panic!()
        };

        let range_depth = 0..filter_depth;
        let range_height = output_height..(output_height + filter_height);
        let range_width = output_width..(output_width + filter_width);

        let step_grad_derivative = *step_grad.get(loc).unwrap();

        for depth in range_depth {
            for height in range_height.clone() {
                for width in range_width.clone() {
                    let input_at_loc = *input.get(&[depth, height, width]).unwrap_or(&T::from(0.0));

                    // subtract origin (output) to get relative (filter) positions
                    match out.get_mut(&[
                        output_depth, // thisis teh filter number
                        depth,
                        height - output_height,
                        width - output_width,
                    ]) {
                        Some(x) => *x = *x + (input_at_loc * step_grad_derivative),
                        None => panic!("There should always be a weight in this space"),
                    }
                }
            }
        }
    }
}

impl<T: ValidNumber<T>> Layer<T> for Conv2d<T> {
    fn evaluate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        let preactivation = self.preactivate(input)?;
        self.activate(&preactivation)
    }

    fn preactivate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        if input.shape() != &self.input_shape {
            return Err(());
        }

        self.convolve2d_all(input)
    }

    fn activate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        self.activation.activate_tensor(input)
    }

    fn get_weights(&self) -> Option<Tensor<T>> {
        Some(self.weights.clone())
    }

    fn set_weights(&mut self, new_weights: Tensor<T>) -> Result<(), ()> {
        self.weights = new_weights;
        Ok(())
    }

    fn get_activation(&self) -> Option<Activation> {
        Some(self.activation.clone())
    }

    fn input_derivative(&self, input: &Tensor<T>, step_grad: &Tensor<T>) -> Result<Tensor<T>, ()> {
        // step_grad is dj/dz at this point, so it's the size of the output of this layer. This function calculates the derivative for the input

        let &[input_depth, input_height, input_width] = &self.input_shape[..] else {
            panic!("Should be unreachable! - input shape should be rank 3 for conv")
        };

        let [output_depth, output_height, output_width] = step_grad.shape()[..] else {
            panic!("Should be unreachable - output shape should be rank 3 for conv")
        };

        let mut out = Tensor::new(vec![input_depth, input_height, input_width]);

        for depth in 0..output_depth {
            for height in 0..output_height {
                for width in 0..output_width {
                    self.input_derivative_helper(&[depth, width, height], step_grad, &mut out);
                }
            }
        }

        Ok(out)
    }

    fn weights_derivative(
        &self,
        step_grad: &Tensor<T>,
        input: &Tensor<T>,
    ) -> Result<Option<Tensor<T>>, ()> {
        let &[input_depth, input_height, input_width] = &self.input_shape[..] else {
            panic!("Should be unreachable! - input shape should be rank 3 for conv")
        };

        let [output_depth, output_height, output_width] = step_grad.shape()[..] else {
            panic!("Should be unreachable - output shape should be rank 3 for conv")
        };

        let mut out = Tensor::new(self.weights.shape().clone());

        for depth in 0..output_depth {
            for height in 0..output_height {
                for width in 0..output_width {
                    self.weights_derivative_helper(
                        &[depth, width, height],
                        input,
                        step_grad,
                        &mut out,
                    );
                }
            }
        }

        Ok(Some(out))
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
            stride: (1, 1),
            input_shape: vec![1, 2, 2],
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
            stride: (1, 1),
            input_shape: vec![1, 2, 2],
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
            stride: (1, 1),
            input_shape: vec![1, 2, 2],
        };

        let input = Tensor::from(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);

        let res = conv2d.convolve2d_all(&input).unwrap();

        println!("{res:?}");

        assert_eq!(res.shape, vec![2, 2, 2])
    }
}
