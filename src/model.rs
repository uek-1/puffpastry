use crate::activation::Activation;
use crate::layer::Layer;
use crate::loss::Loss;
use crate::tensor::Tensor;
use crate::vec_tools::{
    self, AddVec, ElementwiseMatrixMultiply, MatrixMultiply, Transpose, ValidNumber,
};

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

    pub fn from_layers(layers: Vec<impl Layer<T>>, loss: Loss) -> Model<T> {
        let mut new: Vec<Box<dyn Layer<T>>> = vec![];

        for layer in layers {
            let item: Box<dyn Layer<T>> = Box::new(layer);
        }

        Model { layers: new, loss }
    }

    //     fn forward_pass(&self, input: &Tensor<T>) -> (Vec<Vec<Vec<T>>>, Vec<Vec<Vec<T>>>) {
    //         if input.get_rank() != 2 || input.shape()[1] != 1 {
    //             panic!("ERROR")
    //         }
    //         // Input should be a column vector
    //         let mut temp: Vec<Vec<T>> = vec![input.to_vec()].transposed();
    //         let mut z_steps: Vec<Vec<Vec<T>>> = vec![];
    //         // Could have error here!
    //         let mut a_steps: Vec<Vec<Vec<T>>> = vec![temp.clone()];

    //         for layer in &self.layers {
    //             let z = layer.preactivation(&temp);
    //             let a = layer.activate(z.clone());
    //             temp = a.clone();

    //             if z[0][0].into().is_nan() {
    //                 //panic!("NAN");
    //             }

    //             z_steps.push(z);
    //             a_steps.push(a);
    //         }

    //         (z_steps, a_steps)
    //     }

    //     fn backward_pass(
    //         &self,
    //         mut a_steps: Vec<Vec<Vec<T>>>,
    //         mut z_steps: Vec<Vec<Vec<T>>>,
    //         loss_gradient: Vec<Vec<T>>,
    //     ) -> Vec<Vec<Vec<T>>> {
    //         let mut weight_updates: Vec<Vec<Vec<T>>> = vec![];
    //         let mut prev_layer: Option<Layer<T>> = None;
    //         let mut step_gradient = loss_gradient;

    //         for (num, layer) in self.layers.iter().rev().enumerate() {
    //             // println!("{num}");
    //             let current_preactivation = z_steps
    //                 .pop()
    //                 .expect("Backprop couldn't find required preactivations!");
    //             let previous_activation = a_steps
    //                 .pop()
    //                 .expect("Backprop couldn't find required activations!");
    //             //println!("backprop using z: {:?}, a: {:?}", current_z, previous_a);

    //             // Column vector
    //             //println!("CALCULATING dz/da");
    //             let partial_prevpreactiv_activation = match prev_layer.to_owned() {
    //                 Some(prev) => prev.weights.transposed().matrix_multiply(&step_gradient),
    //                 None => step_gradient,
    //             };

    //             //println!("{:?}", dz_da);

    //             // Column vector
    //             //println!("CALCULATING da/dz");
    //             let partial_activation_preactiv: Vec<Vec<T>> =
    //                 layer.activation.derivative(current_preactivation);
    //             //println!("{:?}", da_dz);

    //             // Elementwise multiply - Hadamard product, unless it's the last layer (first
    //             // iteration). Then pick between matmul / elementwise depending on sizes.
    //             //println!("CALCULATING dj/dz");
    //             match num {
    //                 0 => {
    //                     // check matrix dimensions
    //                     if Self::matrix_equal_dims(
    //                         &partial_activation_preactiv,
    //                         &partial_prevpreactiv_activation,
    //                     ) {
    //                         step_gradient = partial_activation_preactiv
    //                             .elementwise_multiply(&partial_prevpreactiv_activation)
    //                     } else {
    //                         step_gradient = partial_activation_preactiv
    //                             .matrix_multiply(&partial_prevpreactiv_activation)
    //                     }
    //                 }
    //                 _ => {
    //                     step_gradient = partial_activation_preactiv
    //                         .elementwise_multiply(&partial_prevpreactiv_activation)
    //                 }
    //             };
    //             // println!("\n dj/dz {:?}\n\n", step_gradient);

    //             // Multiply by respective previous layers' activation
    //             //println!("CALCULATING dj/dw");
    //             let partial_loss_weight: Vec<Vec<T>> =
    //                 step_gradient.matrix_multiply(&previous_activation.transposed());
    //             //println!("{:?} \n", dj_dw);
    //             weight_updates.push(partial_loss_weight);
    //             // println!("{weight_updates:?}");
    //             prev_layer = Some(layer.clone());
    //         }

    //         // Note weight updates stores the LAST layers' weight FIRST!
    //         weight_updates
    //     }

    //     fn matrix_equal_dims(left: &Vec<Vec<T>>, right: &Vec<Vec<T>>) -> bool {
    //         // println!("\nda/dz \n{left:?} \ndj/da \n{right:?}");

    //         if left.len() != right.len() {
    //             return false;
    //         }

    //         if left.len() == 0 || right.len() == 0 {
    //             return false;
    //         }

    //         if left[0].len() != right[0].len() {
    //             return false;
    //         }

    //         // println!("EQUAL");

    //         true
    //     }

    //     pub fn one_pass(&self, input: &Tensor<T>, output: &Tensor<T>) -> (Vec<Vec<Vec<T>>>, T) {
    //         let result = self.evaluate(input);
    //         //println!("\nRES {result:?} \n REAL {output:?}\n");
    //         let loss: T = self.loss.calculate_loss(result, output.to_vec());

    //         let (z_steps, mut a_steps) = self.forward_pass(input);

    //         // TODO use popped a_step to calculate loss gradient.
    //         //a_steps.pop();
    //         let last_activation = a_steps
    //             .pop()
    //             .expect("No activations created during forward pass!");
    //         let loss_gradient = self
    //             .loss
    //             .get_gradient(last_activation, vec![output.clone()].transposed());

    //         (self.backward_pass(a_steps, z_steps, loss_gradient), loss)
    //     }

    //     pub fn update_weights(&mut self, weight_updates: Vec<Vec<Vec<T>>>, learning_rate: T) {
    //         let mut weight_updates = weight_updates;

    //         for layer in 0..self.layers.len() {
    //             let current_weight_update = weight_updates
    //                 .pop()
    //                 .expect("Not enough weight updates for the layers in the model!");
    //             for row in 0..self.layers[layer].weights.len() {
    //                 for col in 0..self.layers[layer].weights[row].len() {
    //                     self.layers[layer].weights[row][col] = self.layers[layer].weights[row][col]
    //                         - (learning_rate * current_weight_update[row][col]);
    //                 }
    //             }
    //         }
    //     }

    //     pub fn gradient_check(
    //         &self,
    //         mut weight_updates: Vec<Vec<Vec<T>>>,
    //         input: &Vec<T>,
    //         output: &Vec<T>,
    //         epsilon: f64,
    //     ) {
    //         // VERY INTENSIVE! WILL SLOW DOWN NETWORK SIGNFICANTLY!
    //         for layer in 0..self.layers.len() {
    //             let update = weight_updates
    //                 .pop()
    //                 .expect("Couldn't find weight updates for layer {layer}");
    //             for neuron in 0..self.layers[layer].weights.len() {
    //                 for weight in 0..self.layers[layer].weights[neuron].len() {
    //                     match (neuron, weight) {
    //                         (x, y) if x > 5 || y > 5 => continue,
    //                         _ => (),
    //                     }

    //                     let mut alt = self.clone();
    //                     alt.layers[layer].weights[neuron][weight] =
    //                         alt.layers[layer].weights[neuron][weight] + T::from(epsilon);
    //                     let (_, inc) = alt.one_pass(input, output);

    //                     let mut alt = self.clone();
    //                     alt.layers[layer].weights[neuron][weight] =
    //                         alt.layers[layer].weights[neuron][weight] - T::from(epsilon);
    //                     let (_, dec) = alt.one_pass(input, output);

    //                     let res = (inc - dec) / T::from(2.0 * epsilon);
    //                     match (update[neuron][weight] - res).into().abs() < epsilon {
    //                         true => (),
    //                         false => {
    //                             panic!("Gradient checking failed : layer: {:?} {:?},  neuron: {} weight: {:?} finite_difference: {:?} output{:?}", layer, self.layers[layer].activation, neuron, update[neuron][weight], res, output);
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     pub fn fit(
    //         &mut self,
    //         train: Vec<Vec<T>>,
    //         validate: Vec<Vec<T>>,
    //         epochs: usize,
    //         learning_rate: T,
    //     ) {
    //         let data_iter = train.iter().zip(validate.iter());

    //         for epoch in 0..epochs {
    //             println!("EPOCH #{epoch}");
    //             let (mut average_loss, mut inputs) = (0.0, 0.0);
    //             for (input, output) in data_iter.clone() {
    //                 //println!("train input - {:?} output - {:?}", input, output);
    //                 let (weight_updates, loss) = self.one_pass(input, output);
    //                 // println!("INPUT {inputs:?} LOSS {loss:?}");
    //                 // self.gradient_check(weight_updates.clone(), input, output, 0.01);
    //                 self.update_weights(weight_updates, learning_rate);

    //                 average_loss += loss.into();
    //                 inputs += 1.0;
    //             }

    //             average_loss = average_loss / inputs;
    //             println!("{average_loss}");
    //         }
    //     }

    pub fn evaluate(&self, input: &Tensor<T>) -> Tensor<T> {
        self.layers
            .iter()
            .fold(input.clone(), |temp, layer| layer.evaluate(&temp).unwrap())
    }
}
