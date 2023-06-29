extern crate rand;
pub mod model;
pub mod loss;
pub mod layer;
pub mod activation;
pub mod vec_tools;

pub use model::Model;
pub use layer::Layer;
pub use loss::Loss;
pub use activation::Activation;

