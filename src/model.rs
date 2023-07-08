use crate::vec_tools::{self, Transpose, MatrixMultiply, ElementwiseMatrixMultiply, AddVec};
use crate::activation::Activation;
use crate::layer::Layer;
use crate::loss::Loss;


#[derive(Clone, Debug)]
pub struct Model<T : vec_tools::ValidNumber<T>> {
    pub layers : Vec<Layer<T>>,
    pub loss : Loss,
}

impl<T : vec_tools::ValidNumber<T>> Model<T> {
    pub fn new(loss: Loss) -> Model<T> {
        Model {
            layers : vec![],
            loss : loss,
        } 
    }

    pub fn from_layers(layers : Vec<Layer<T>>, loss: Loss) -> Model<T> {
        Model {
            layers,
            loss
        }
    }

    fn forward_pass(&self, input: &Vec<T>) -> (Vec<Vec<Vec<T>>>, Vec<Vec<Vec<T>>>) {
        // Input should be a column vector
        let mut temp : Vec<Vec<T>> = vec![input.to_vec()].transposed();
        let mut z_steps : Vec<Vec<Vec<T>>> = vec![];
        // Could have error here!
        let mut a_steps : Vec<Vec<Vec<T>>> = vec![temp.clone()];

        for layer in &self.layers {
            let z = layer.preactivation(&temp);
            let a = layer.activate(z.clone());
            temp = a.clone();

            if z[0][0].into().is_nan() {
                //panic!("NAN");
            }

            z_steps.push(z);
            a_steps.push(a);
        }

        (z_steps, a_steps)
    }

    fn backward_pass(&self, a_steps: Vec<Vec<Vec<T>>>, z_steps : Vec<Vec<Vec<T>>>, loss_gradient: Vec<Vec<T>>) -> Vec<Vec<Vec<T>>> {
        let mut weight_updates : Vec<Vec<Vec<T>>> = vec![]; 
        let mut z_steps = z_steps;
        let mut a_steps = a_steps;
        let mut prev_layer : Option<Layer<T>> = None;
        let mut step_gradient = loss_gradient;
        
        for layer in self.layers.iter().rev() {
            let current_preactivation = z_steps.pop().expect("Backprop couldn't find required preactivations!");
            let previous_activation = a_steps.pop().expect("Backprop couldn't find required activations!");
            //println!("backprop using z: {:?}, a: {:?}", current_z, previous_a); 
            
            // Column vector
            //println!("CALCULATING dz/da");
            let partial_prevpreactiv_activation = match prev_layer.to_owned() {
                Some(prev) => prev.weights.transposed().matrix_multiply(&step_gradient),
                None => step_gradient,
            };

            //println!("{:?}", dz_da);

            // Column vector
            //println!("CALCULATING da/dz");
            let partial_activation_preactiv : Vec<Vec<T>> = layer.activation.derivative(current_preactivation);
            //println!("{:?}", da_dz);

            // Elementwise multiply (Hadamard product)
            //println!("CALCULATING dj/dz");
            step_gradient = partial_prevpreactiv_activation.elementwise_multiply(&partial_activation_preactiv);
            //println!("{:?}", step_gradient);
            
            // Multiply by respective previous layers' activation
            //println!("CALCULATING dj/dw");
            let partial_loss_weight : Vec<Vec<T>> = step_gradient.matrix_multiply(&previous_activation.transposed());
            //println!("{:?} \n", dj_dw);
            weight_updates.push(partial_loss_weight); 
            prev_layer = Some(layer.clone());
        }
            
        // Note weight updates stores the LAST layers' weight FIRST!
        weight_updates
    }

    pub fn one_pass(&self, input: &Vec<T>, output: &Vec<T>) -> (Vec<Vec<Vec<T>>>, T){
        let result = self.evaluate(input);
        //println!("\nRES {result:?} \n REAL {output:?}\n");
        let loss : T =  self.loss.calculate_loss(result, output.to_vec()); 

        let (z_steps, mut a_steps) = self.forward_pass(input);

        // TODO use popped a_step to calculate loss gradient.
        //a_steps.pop();
        let last_activation = a_steps.pop().expect("No activations created during forward pass!");
        let loss_gradient = self.loss.get_gradient(last_activation, vec![output.clone()].transposed());

        (self.backward_pass(a_steps, z_steps, loss_gradient), loss)
    }

    pub fn update_weights(&mut self, weight_updates: Vec<Vec<Vec<T>>>, learning_rate : T) {
        //println!("\nUPDATING WEIGHTS {:?} with {:?} \n", self.layers.last(), weight_updates.first());
        let mut weight_updates = weight_updates;

        for layer in 0..self.layers.len() {
            let current_weight_update = weight_updates.pop().expect("Not enough weight updates for the layers in the model!");
            for row in 0..self.layers[layer].weights.len() {
                for col in 0..self.layers[layer].weights[row].len() {
                    self.layers[layer].weights[row][col] = self.layers[layer].weights[row][col] - (learning_rate * current_weight_update[row][col]);
                }
            } 
        }
    }

    pub fn gradient_check(&self, mut weight_updates: Vec<Vec<Vec<T>>>, input: &Vec<T>, output: &Vec<T>, epsilon : f64) {
        // VERY INTENSIVE! WILL SLOW DOWN NETWORK SIGNFICANTLY!
        for layer in 0..self.layers.len() {
            let update = weight_updates.pop().expect("Couldn't find weight updates for layer {layer}");
            for neuron in 0..self.layers[layer].weights.len() {
                for weight in 0..self.layers[layer].weights[neuron].len() {
                    match (neuron, weight) {
                        (x,y) if x > 5 || y > 5 => continue,
                        _ => (),
                    }

                    let mut alt = self.clone();
                    alt.layers[layer].weights[neuron][weight] = alt.layers[layer].weights[neuron][weight] + T::from(epsilon);
                    let (_, inc) = alt.one_pass(input, output);

                    let mut alt = self.clone();
                    alt.layers[layer].weights[neuron][weight] = alt.layers[layer].weights[neuron][weight] - T::from(epsilon * 2.0);
                    let (_, dec) = alt.one_pass(input, output);

                    let res = (inc - dec) / T::from(2.0 * epsilon);
                    match (update[neuron][weight] - res).into().abs() < epsilon {
                        true => (),
                        false => {
                            println!("Gradient checking failed : layer: {} neuron: {} weight: {:?} res: {:?} output{:?}", layer, neuron, update[neuron][weight], res, output);
                            assert!(false)
                        }
                    }
                }
            }


        }

    }

    pub fn fit(&mut self, train : Vec<Vec<T>>, validate : Vec<Vec<T>>, epochs : usize, learning_rate: T) {
        let data_iter = train
            .iter()
            .zip(validate.iter());

        for epoch in 0..epochs {
            println!("EPOCH #{epoch}");
            let (mut average_loss, mut inputs) = (0.0, 0.0);
            for (input, output) in data_iter.clone() {
                //println!("train input - {:?} output - {:?}", input, output);
                let (weight_updates, loss) = self.one_pass(input, output);
                //println!("INPUT {inputs:?} LOSS {loss:?}");
                self.gradient_check(weight_updates.clone(), input, output, 0.00001);
                self.update_weights(weight_updates, learning_rate);
                
                average_loss += loss.into();
                inputs += 1.0;
            }

            average_loss = average_loss / inputs;
            println!("{average_loss}");
        }
    }

    pub fn evaluate(&self, input: &Vec<T>) -> Vec<T> {
        self.layers
            .iter()
            .fold(vec![input.clone()].transposed(), |temp, layer| layer.evaluate(&temp))
            .transposed()
            .get(0)
            .unwrap()
            .to_vec()
    }

}


