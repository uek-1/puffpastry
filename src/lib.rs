extern crate puffpastry_tensor_macro;
extern crate rand;
pub use puffpastry_tensor_macro::tensor;
pub mod activation;
pub mod layer;
pub mod loss;
pub mod model;
pub mod tensor;
pub mod vec_tools;

pub use activation::Activation;
pub use layer::{Conv2d, Dense, Flatten, MaxPool2d};
pub use loss::Loss;
pub use model::Model;
pub use tensor::Tensor;
pub use vec_tools::ValidNumber;
