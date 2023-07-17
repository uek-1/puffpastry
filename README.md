# puffpastry

[![crates.io](https://img.shields.io/crates/v/puffpastry.svg)](https://crates.io/crates/puffpastry)
[![docs.rs](https://docs.rs/puffpastry/badge.svg)](https://docs.rs/puffpastry/)

```puffpastry``` is a very basic feedforward neuron network library with a focus on parity with mathematical representations. It can be used to create and train simple models. 
## Usage
```puffpastry``` is used very similarly to keras - stack layers and fit to training data.  
### Learning XOR
```rust
// from_layers(layers: Vec<impl Layer, loss: Loss) -> Model
let mut model : Model<f64> = Model::from_layers(vec![
        Dense::from_size(2, 2, Activation::Sigmoid),
        Dense::from_size(2, 1, Activation::None)
    ],
    Loss::MeanSquaredError
);

let train_inputs = vec![
    Tensor::column(vec![0.0, 0.0]),
    Tensor::column(vec![1.0, 0.0]),
    Tensor::column(vec![0.0, 1.0]),
    Tensor::column(vec![1.0, 1.0]),
];

let train_outputs = vec![
    Tensor::column(vec![0.0]),
    Tensor::column(vec![1.0]),
    Tensor::column(vec![1.0]),
    Tensor::column(vec![0.0]),
];

// fit(&mut self, inputs, outputs, epochs, learning_rate) -> Result
model.fit(train_inputs, train_outputs, 100, 1.2).unwrap();  

// evaluate(&self, input: Tensor) -> Result<Tensor>
model.evaluate(&Tensor::column(vec![1.0, 0.0])).unwrap()
// stdout: Tensor {shape: [1], data: [0.9179620463347642]}
```

## Features
Activation functions: ```[softmax, relu, sigmoid, linear]``` <br />
Loss functions: ```[categorical cross entropy, mean squared error]``` <br />
Layers: ```[dense]``` <br />
## Roadmap
1. Convulational Layers (Layer rework in general) [75%]
2. Documentation
3. Tools to build GANs
