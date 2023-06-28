use crate::{model::Model, layer::Layer};
use rand::{self, Rng};
use csv::{Reader, StringRecord};
use std::fmt;

mod model;
mod loss;
mod layer;
mod activation;
mod vec_tools;

use activation::{Activation};
use loss::Loss;

fn main () {
    let mut model : Model<f64> = Model {
        layers: vec![
            Layer::from_size(784, 28, Activation::Sigmoid),
            Layer::from_size(28, 9, Activation::Sigmoid),
            Layer::from_size(9, 1, Activation::None)
        ],
        loss : Loss::MeanSquaredError,
    };
        
    let mut train : Vec<Vec<f64>> = vec![];
    
    let mut validate : Vec<Vec<f64>>= vec![];
    
    let mut mnist_reader = Reader::from_path("data/mnist_train.csv").unwrap();
    

    for (num, record) in mnist_reader.records().enumerate() { 
        if num > 3000 {
            break;
        }

        if let Ok(x) = record {
            validate.push(
                x.into_iter()
                    .take(1)
                    .map(|x| x.parse().unwrap())
                    .collect()
            ); 

            train.push(
                x.into_iter()
                    .skip(1)
                    .map(|x| x.parse().unwrap())
                    .collect()
            );
        }
    }

    println!("{:?} \n {}", validate[0].to_vec(), Pretty(train[0].to_vec()));
 
    model.fit(train.clone(), validate.clone(), 3, 0.001);

    println!("trained model : \n");
    //println!("{:?}", model);
    
    println!("testing on input {:?} : \n {}", validate[5].clone(), Pretty(train[5].clone()));
    let res = model.evaluate(&validate[5]);
    println!("{:?}", res);
    
}

pub struct Pretty(Vec<f64>);

impl fmt::Display for Pretty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..28 {
            for col in 0..28 {
                let elem = match self.0[28 * row + col] < 2.0 {
                    true => "  ",
                    false => "1.0"
                };

                write!(f, "{} ", elem);
            }
            writeln!(f);
        }

        Ok(())
    }
}
