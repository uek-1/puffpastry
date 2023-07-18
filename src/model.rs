use crate::activation::Activation;
use crate::layer::{Conv2d, Dense, Flatten, Layer, MaxPool2d};
use crate::loss::Loss;
use crate::tensor::Tensor;
use crate::vec_tools::{
    self, AddVec, ElementwiseMatrixMultiply, MatrixMultiply, Transpose, ValidNumber,
};

#[derive(Debug)]
pub struct Model<T: ValidNumber<T>> {
    pub layers: Vec<Box<dyn Layer<T>>>,
    pub loss: Loss,
}

impl<T: ValidNumber<T>> Model<T> {
    pub fn new(loss: Loss) -> Model<T> {
        Model {
            layers: vec![],
            loss: loss,
        }
    }

    pub fn from_layers(layers: Vec<impl Layer<T> + 'static>, loss: Loss) -> Model<T> {
        let mut new: Vec<Box<dyn Layer<T>>> = vec![];

        for layer in layers {
            let item: Box<dyn Layer<T>> = Box::new(layer);
            new.push(item);
        }

        Model { layers: new, loss }
    }

    pub fn push_layer(&mut self, layer: impl Layer<T> + 'static) {
        let item: Box<dyn Layer<T>> = Box::new(layer);
        self.layers.push(item);
    }

    fn forward_pass(&self, input: &Tensor<T>) -> Result<(Vec<Tensor<T>>, Vec<Tensor<T>>), ()> {
        // Input should be a column vector
        if input.rank() != 2 || input.shape()[1] != 1 {
            panic!("ERROR")
        }

        let mut temp: Tensor<T> = input.clone();
        let mut z_steps: Vec<Tensor<T>> = vec![];
        let mut a_steps: Vec<Tensor<T>> = vec![temp.clone()];

        for layer in &self.layers {
            let z = layer.preactivate(&temp)?;
            let a = layer.activate(&z)?;
            temp = a.clone();

            z_steps.push(z);
            a_steps.push(a);
        }

        Ok((z_steps, a_steps))
    }

    fn backward_pass(
        &self,
        mut a_steps: Vec<Tensor<T>>,
        mut z_steps: Vec<Tensor<T>>,
        loss_gradient: Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, ()> {
        let mut weight_updates: Vec<Tensor<T>> = vec![];
        let mut prev_layer: Option<&Box<dyn Layer<T>>> = None;
        let mut step_gradient = loss_gradient;

        // println!("{a_steps:?}");

        for (num, layer) in self.layers.iter().rev().enumerate() {
            // println!("{num}");
            let current_preactivation = z_steps
                .pop()
                .expect("Backprop couldn't find required preactivations!");
            let previous_activation = a_steps
                .pop()
                .expect("Backprop couldn't find required activations!");
            // println!(
            //     "backprop using z: {:?}, a: {:?}",
            //     current_preactivation, previous_activation
            // );

            // Column vector
            // println!("CALCULATING dz/da");
            // println!("prev {prev_layer:?}, sg : {step_gradient:?}");
            let partial_prevpreactiv_activation = match prev_layer {
                Some(prev) => prev
                    .get_weights()
                    .transposed()
                    .matrix_multiply(&step_gradient)?,
                None => step_gradient,
            };

            // println!("{:?}", partial_prevpreactiv_activation);

            // Column vector
            // println!("CALCULATING da/dz");
            // println!("derivative of {current_preactivation:?}");
            let partial_activation_preactiv: Tensor<T> =
                layer.get_activation().derivative(&current_preactivation);
            // println!("{:?}", partial_activation_preactiv);

            // Elementwise multiply - Hadamard product, unless it's the last layer (first
            // iteration). Then pick between matmul / elementwise depending on sizes.
            // println!("CALCULATING dj/dz");
            match num {
                0 => {
                    // check matrix dimensions
                    if partial_activation_preactiv.shape()
                        == partial_prevpreactiv_activation.shape()
                    {
                        step_gradient = partial_activation_preactiv
                            .elementwise_product(&partial_prevpreactiv_activation)?
                    } else {
                        step_gradient = partial_activation_preactiv
                            .matrix_multiply(&partial_prevpreactiv_activation)?
                    }
                }
                _ => {
                    step_gradient = partial_activation_preactiv
                        .elementwise_product(&partial_prevpreactiv_activation)?
                }
            };
            // println!("\n dj/dz {:?}\n\n", step_gradient);

            // Multiply by respective previous layers' activation
            // println!("CALCULATING dj/dw");
            let partial_loss_weight: Tensor<T> =
                step_gradient.matrix_multiply(&previous_activation.transposed())?;
            // println!("{:?} \n", partial_loss_weight);
            weight_updates.push(partial_loss_weight);
            // println!("{weight_updates:?}");
            prev_layer = Some(layer);
        }

        // Note weight updates stores the LAST layers' weight FIRST!
        //println!("weight updates : {weight_updates:?}");
        Ok(weight_updates)
    }

    pub fn one_pass(
        &self,
        input: &Tensor<T>,
        output: &Tensor<T>,
    ) -> Result<(Vec<Tensor<T>>, T), ()> {
        let result = self.evaluate(input)?;
        // println!("\nRES {result:?} \n REAL {output:?}\n");
        let loss: T = self.loss.calculate_loss(result, output.clone())?;

        let (z_steps, mut a_steps) = self.forward_pass(input)?;

        // TODO use popped a_step to calculate loss gradient.
        //a_steps.pop();
        let last_activation = a_steps
            .pop()
            .expect("No activations created during forward pass!");
        let loss_gradient = self.loss.get_gradient(last_activation, output.clone())?;

        // println!("loss grad {loss_gradient:?}");

        Ok((self.backward_pass(a_steps, z_steps, loss_gradient)?, loss))
    }

    pub fn update_weights(
        &mut self,
        weight_updates: Vec<Tensor<T>>,
        learning_rate: T,
    ) -> Result<(), ()> {
        self.layers
            .iter_mut()
            .zip(weight_updates.iter().rev())
            .map(|(layer, update)| {
                let current_weights = layer.get_weights();
                if current_weights.shape() != update.shape() {
                    Err(())
                } else {
                    let new_weights = current_weights - (update.clone() * learning_rate);
                    layer.set_weights(new_weights);
                    Ok(())
                }
            })
            .collect::<Result<(), ()>>()
    }

    // TODO: Very hacky implementation, needs fixing!
    pub fn gradient_check(
        &mut self,
        mut weight_updates: Vec<Tensor<T>>,
        input: &Tensor<T>,
        output: &Tensor<T>,
        epsilon: f64,
    ) -> Result<(), ()> {
        // VERY INTENSIVE! WILL SLOW DOWN NETWORK SIGNFICANTLY!
        for layer in 0..self.layers.len() {
            let update = weight_updates
                .pop()
                .expect("Couldn't find weight updates for layer {layer}");

            let layer_weights = self.layers[layer].get_weights();
            let param_count = layer_weights.data.len();

            for param in 0..param_count {
                // Check only 5 trainable values per layer.
                if param > 5 {
                    break;
                }

                let original = update.data[param];

                let mut inc_weights = layer_weights.clone();
                let mut dec_weights = layer_weights.clone();
                inc_weights.data[param] = inc_weights.data[param] + T::from(epsilon);
                dec_weights.data[param] = dec_weights.data[param] - T::from(epsilon);

                self.layers[layer].set_weights(inc_weights);
                let (_, inc) = self
                    .one_pass(input, output)
                    .expect("Failed to train during gradient checking!");

                self.layers[layer].set_weights(dec_weights);
                let (_, dec) = self
                    .one_pass(input, output)
                    .expect("Failed to train during gradient checking!");

                self.layers[layer].set_weights(layer_weights.clone());

                let res = (inc - dec) / T::from(2.0 * epsilon);
                match (original - res).into().abs() < epsilon {
                    true => (),
                    false => {
                        println!("Gradient checking failed : layer: #{:?},  finite_difference: {:?} output{:?}", layer, res, original);
                        return Err(());
                    }
                }
            }
        }

        Ok(())
    }

    pub fn fit(
        &mut self,
        train: Vec<Tensor<T>>,
        validate: Vec<Tensor<T>>,
        epochs: usize,
        learning_rate: T,
    ) -> Result<(), ()> {
        let data_iter = train.iter().zip(validate.iter());

        for epoch in 0..epochs {
            println!("EPOCH #{epoch}");
            let (mut average_loss, mut inputs) = (0.0, 0.0);
            for (input, output) in data_iter.clone() {
                // println!("train input - {:?} output - {:?}", input, output);
                let (weight_updates, loss) = self.one_pass(input, output)?;
                // println!("INPUT {inputs:?} LOSS {loss:?}");
                //self.gradient_check(weight_updates.clone(), input, output, 0.01)?;
                self.update_weights(weight_updates, learning_rate)?;

                // println!("l1w = {}", self.layers[0].get_weights());

                average_loss += loss.into();
                inputs += 1.0;
            }

            average_loss = average_loss / inputs;
            println!("{average_loss}");
        }

        Ok(())
    }

    pub fn evaluate(&self, input: &Tensor<T>) -> Result<Tensor<T>, ()> {
        self.layers
            .iter()
            .fold(Ok(input.clone()), |temp, layer| layer.evaluate(&temp?))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn conv_feedforward_test() {
        let input = Tensor::from(vec![vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ]]);

        let mut model: Model<f64> = Model::new(Loss::MeanSquaredError);
        model.push_layer(Conv2d::from_size(1, 2, 4, (1, 1), Activation::None));
        model.push_layer(MaxPool2d::new(2, 2));
        model.push_layer(Conv2d::from_size(4, 2, 8, (1, 1), Activation::Sigmoid));
        model.push_layer(Flatten {});
        model.push_layer(Dense::from_size(32, 10, Activation::Softmax));

        let res = model.evaluate(&input).unwrap();

        println!("{res:?}");
        assert!(false)
    }
}
