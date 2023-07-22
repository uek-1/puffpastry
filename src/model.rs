use super::{layer::*, Loss, Tensor, ValidNumber};
use std::io::Write;

#[derive(Debug)]
pub struct Model<T: ValidNumber<T>> {
    pub layers: Vec<Box<dyn Layer<T>>>,
    pub loss: Loss,
}

impl<T: ValidNumber<T>> Model<T> {
    pub fn new(loss: Loss) -> Model<T> {
        Model {
            layers: vec![],
            loss,
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
        // println!("input {input:?}");
        let mut temp: Tensor<T> = input.clone();
        let mut z_steps: Vec<Tensor<T>> = vec![];
        let mut a_steps: Vec<Tensor<T>> = vec![temp.clone()];

        for layer in &self.layers {
            let z = layer.preactivate(&temp)?;
            let a = layer.activate(&z)?;
            temp = a.clone();

            // println!(" pre-act {:?}", z.data[..10].to_vec());
            // println!(" act {:?}\n\n", a.data[..10].to_vec());
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
    ) -> Result<Vec<Option<Tensor<T>>>, ()> {
        let mut weight_updates: Vec<Option<Tensor<T>>> = vec![];
        let mut prev_layer: Option<&Box<dyn Layer<T>>> = None;
        let mut step_gradient = loss_gradient;
        let mut current_activation: Option<Tensor<T>> = None;

        for (num, layer) in self.layers.iter().rev().enumerate() {
            // println!("\n {num:?} {layer:.5?} \n");
            let current_preactivation = z_steps
                .pop()
                .expect("Backprop couldn't find required preactivations!");
            let previous_activation = a_steps
                .pop()
                .expect("Backprop couldn't find required activations!");

            // How the next (in model) layer's preactivation depends on this layers activation
            let partial_prevpreactiv_activation = match prev_layer {
                Some(prev) => {
                    let current_activation = current_activation.as_ref().ok_or(())?;
                    prev.input_derivative(&current_activation, &step_gradient)?
                }
                None => step_gradient.clone(),
            };

            // How this layer's activation (if it exists) depends on this layer's preactivation
            let partial_activation_preactiv: Option<Tensor<T>> = layer
                .get_activation()
                .map(|activation| activation.derivative(&current_preactivation));

            // Elementwise multiply - Hadamard product, unless it's the last layer (first
            // iteration).

            // How the next (in model) layer's preactivation depends on this layer's activation
            // Since layer preactivation.shape() == layer.activation.shape() == (next in model layer).input.shape(), this is an elementwise product

            // Caveat: the last (in model) layer's next (in model) layer is the loss function, which is why there has to be a seperate case handling it.

            match num {
                0 if partial_activation_preactiv.is_some() => {
                    // Safe because is_some() check above
                    let partial_activation_preactiv = partial_activation_preactiv.unwrap();
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
                    step_gradient = match partial_activation_preactiv {
                        Some(tens) => tens.elementwise_product(&partial_prevpreactiv_activation)?,
                        None => partial_prevpreactiv_activation,
                    }
                }
            };

            let partial_loss_weight: Option<Tensor<T>> =
                layer.weights_derivative(&previous_activation, &step_gradient)?;

            weight_updates.push(partial_loss_weight);
            prev_layer = Some(layer);
            current_activation = Some(previous_activation.clone());
        }

        // Note weight updates stores the LAST layers' weight FIRST!
        Ok(weight_updates)
    }

    pub fn one_pass(
        &self,
        input: &Tensor<T>,
        output: &Tensor<T>,
    ) -> Result<(Vec<Option<Tensor<T>>>, T), ()> {
        let result = self.evaluate(input)?;
        let loss: T = self.loss.calculate_loss(result, output.clone())?;

        let (z_steps, mut a_steps) = self.forward_pass(input)?;

        let last_activation = a_steps
            .pop()
            .expect("No activations created during forward pass!");
        let loss_gradient = self.loss.get_gradient(last_activation, output.clone())?;

        Ok((self.backward_pass(a_steps, z_steps, loss_gradient)?, loss))
    }

    pub fn update_weights(
        &mut self,
        weight_updates: Vec<Option<Tensor<T>>>,
        learning_rate: T,
    ) -> Result<(), ()> {
        self.layers
            .iter_mut()
            .zip(weight_updates.iter().rev())
            .try_for_each(|(layer, update)| {
                let Some(update) = update else {
                    return Ok(())
                };

                let current_weights = match layer.get_weights() {
                    Some(x) => x,
                    None => return Ok(()),
                };

                if current_weights.shape() != update.shape() {
                    Err(())
                } else {
                    let new_weights = current_weights - (update.clone() * learning_rate);
                    layer.set_weights(new_weights)
                }
            })
    }

    // TODO: Very hacky implementation, needs fixing!
    pub fn gradient_check(
        &mut self,
        mut weight_updates: Vec<Option<Tensor<T>>>,
        input: &Tensor<T>,
        output: &Tensor<T>,
        epsilon: f64,
    ) -> Result<(), ()> {
        // VERY INTENSIVE! WILL SLOW DOWN NETWORK SIGNFICANTLY!
        for layer in 0..self.layers.len() {
            println!("{layer}");
            let Some(update) = weight_updates
                .pop()
                .expect("Couldn't find weight updates for layer {layer}") 
            else {
                continue
            };

            let layer_weights = match self.layers[layer].get_weights() {
                Some(x) => x,
                None => continue,
            };

            let param_count = layer_weights.data.len();

            for param in (0..param_count).step_by(100) {
                // Check only 5 trainable values per layer.
                if param > 10 * 100 {
                    break;
                }

                let original = update.data[param];

                let mut inc_weights = layer_weights.clone();
                let mut dec_weights = layer_weights.clone();
                inc_weights.data[param] = inc_weights.data[param] + T::from(epsilon);
                dec_weights.data[param] = dec_weights.data[param] - T::from(epsilon);

                self.layers[layer].set_weights(inc_weights)?;
                let (_, inc) = self
                    .one_pass(input, output)
                    .expect("Failed to train during gradient checking!");

                self.layers[layer].set_weights(dec_weights)?;
                let (_, dec) = self
                    .one_pass(input, output)
                    .expect("Failed to train during gradient checking!");

                self.layers[layer].set_weights(layer_weights.clone())?;

                let res = (inc - dec) / T::from(2.0 * epsilon);
                // println!("{res:?}, {original:?}");
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
        let mut stdout = std::io::stdout();

        for epoch in 0..epochs {
            println!("EPOCH #{epoch}");
            let (mut total_loss, mut inputs) = (0.0, 0.0);
            for (input, output) in data_iter.clone() {
                let (weight_updates, loss) = self.one_pass(input, output)?;
                print!("\rINPUT {inputs:?} AVERAGE LOSS {}", total_loss / inputs);
                stdout.flush();
                // self.gradient_check(weight_updates.clone(), input, output, 0.0001)?;
                self.update_weights(weight_updates, learning_rate)?;

                total_loss += loss.into();
                inputs += 1.0;
            }

            let average_loss = total_loss / inputs;
            println!("\n Epoch loss: {average_loss}");
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
    use crate::Activation;

    #[test]
    fn conv_feedforward_test() {
        let input = Tensor::from(vec![vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ]]);

        let output = Tensor::column(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let mut model: Model<f64> = Model::new(Loss::MeanSquaredError);
        model.push_layer(Conv2d::from_size(
            vec![1, 4, 4],
            2,
            4,
            (1, 1),
            Activation::None,
        ));
        model.push_layer(MaxPool2d::new(2, 2));
        model.push_layer(Conv2d::from_size(
            vec![4, 2, 2],
            2,
            8,
            (1, 1),
            Activation::Sigmoid,
        ));
        model.push_layer(Flatten {});
        model.push_layer(Dense::from_size(32, 10, Activation::Softmax));

        let (weight_updates, loss) = model.one_pass(&input, &output).unwrap();

        println!("\nWEIGHT UPDATES: \n");
        println!("{weight_updates:#?}");
        println!("\nLoss: \n");
        println!("{loss:#?}");
        assert!(false)
    }
}
