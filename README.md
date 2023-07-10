# puffpastry

```puffpastry``` is a very basic feedforward neuron network library that allows for very simple models to be created. ```puffpastry``` focuses on explicit code and explicit usage. The library will not ensure the effectiveness or numerical stability of a neural network. ```puffpastry``` was not designed for efficiency. It was designed to mimic as close as possible the mathematical formulae that feedforward neural networks rely on. This results in the unnecessary presence of certain operations to preserve that quality.

## Usage
```puffpastry``` is used very similarly to keras. Create a model from layers and a loss function using the constructor. Create the layers using either predecided weight matrices or generated weights with an input/output size and an activation. Train using model.fit(), evaluate using model.evaluate()

## Roadmap
1. Convulational Layers (Layer rework in general)
2. Documentation
3. Tools to build GANs
