# puffpastry

```puffpastry``` is a very basic feedforward neuron network library with a focus on parity with mathematical represenations. It can be used to create and train very basic models without much overhead. 
## Usage
```puffpastry``` is used very similarly to keras - stack layers and fit to training data.  
### Learning XOR
```rust
let mut model : Model<f64> = Model {
    layers: vec![
        Layer::from_size(2, 2, Activation::Sigmoid),
        Layer::from_size(2, 1, Activation::None)
    ],
    loss: Loss::MeanSquaredError
};

let train_inputs = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0]
];

let train_outputs = vec![
    vec![0.0],
    vec![1.0],
    vec![1.0],
    vec![0.0]
];

// fit(&mut self, inputs, outputs, epochs, learning_rate)
model.fit(train_inputs, train_outputs, 100, 1.2);  

model.evaluate(vec![1.0, 0.0])
// stdout: [0.9179620463347642]
```

## Features
Activation functions: ```[softmax, relu, sigmoid, linear]``` <br />
Loss functions: ```[categorical cross entropy, mean squared error]``` <br />
Layers: ```[dense]``` <br />
## Roadmap
1. Convulational Layers (Layer rework in general)
2. Documentation
3. Tools to build GANs
