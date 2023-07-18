use rand::Rng;
use std::fmt::Debug;

use crate::activation::Activation;
use crate::rand;
use crate::tensor::Tensor;
use crate::vec_tools::ValidNumber;
pub mod convolutional;
pub mod dense;
pub mod flatten;
pub mod layer_trait;
pub mod pool;
pub use convolutional::Conv2d;
pub use dense::Dense;
pub use flatten::Flatten;
pub use layer_trait::Layer;
pub use pool::MaxPool2d;
