# puffpastry

[![crates.io](https://img.shields.io/crates/v/puffpastry.svg)](https://crates.io/crates/puffpastry)
[![docs.rs](https://docs.rs/puffpastry/badge.svg)](https://docs.rs/puffpastry/)

```puffpastry``` is a very basic feedforward neuron network library with a focus on parity with mathematical representations. It can be used to create and train simple models. 
## Usage
```puffpastry``` is used very similarly to keras - stack layers and fit to training data.  
### Learning XOR
```rust
let mut model : Model<f64> = Model::new(Loss::MeanSquaredError);

// Dense::from_size(input_neurons, output_neurons, activation) -> Dense
model.push_layer(Dense::from_size(2, 2, Activation::Sigmoid));
model.push_layer(Dense::from_size(2, 1, Activation::None));


let train_inputs = vec![
    tensor!([[0.0], [0.0]]),
    tensor!([[1.0], [0.0]]),
    tensor!([[0.0], [1.0]]),
    tensor!([[1.0], [1.0]]),
];

let train_outputs = vec![
    tensor!([[0.0]]),
    tensor!([[1.0]]),
    tensor!([[1.0]]),
    tensor!([[0.0]]),
];

// fit(&mut self, inputs, outputs, epochs, learning_rate) -> Result
model.fit(train_inputs, train_outputs, 100, 1.2).unwrap();  

// evaluate(&self, input: Tensor) -> Result<Tensor>
model.evaluate(&tensor!([[1.0], [0.0]]).unwrap();
// stdout: Tensor {shape: [1], data: [0.9179620463347642]}
```

## Features
Activation functions: ```[softmax, relu, sigmoid, linear]``` <br />
Loss functions: ```[categorical cross entropy, mean squared error]``` <br />
Layers: ```[dense, conv2d, maxpool2d, flatten]``` <br />

## Roadmap
1. Improve convolution layer performance
2. Make variations more explicit i.e. CategoricalCrossentropy vs ClippedCategoricalCrossEntropy
3. Documentation
4. Add pptimizers, weight initializers
5. Add more losses, layers, metrics
