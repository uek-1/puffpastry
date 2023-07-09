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
            Layer::from_size(784, 128, Activation::Sigmoid),
            Layer::from_size(128, 10, Activation::Softmax),
        ],
        loss : Loss::CategoricalCrossEntropy,
    };
        
    let mut train : Vec<Vec<f64>> = vec![];
    
    let mut validate : Vec<Vec<f64>>= vec![];
    
    let mut mnist_reader = Reader::from_path("data/mnist_train.csv").unwrap();
    
    let labels = 10;

    for (num, record) in mnist_reader.records().enumerate() { 
        if num > 5 {
            break;
        }

        if let Ok(x) = record {
            let label : usize = x
                .into_iter()
                .take(1)
                .map(|x| x.parse().unwrap())
                .next()
                .unwrap();

            let mut val_vec = vec![0.0; labels];
            val_vec[label] = 1.0;

            validate.push(val_vec); 

            train.push(
                x.into_iter()
                    .skip(1)
                    .map(|x| x.parse::<f64>().unwrap() / 255.0)
                    .collect()
            );
        }
    }

    println!("{:?} \n {}", validate[0].to_vec(), Pretty(train[0].to_vec()));
 
    model.fit(train.clone(), validate.clone(), 5, 0.2);

    println!("trained model : \n");
    //println!("{:?}", model);
    
    println!("testing on input {:?} : \n {}", validate[5].clone(), Pretty(train[5].clone()));
    let res = model.evaluate(&train[5]);
    println!("{:?}", res);
    println!("Calculated loss for this input {:?}", model.loss.calculate_loss(res, validate[5].clone()));
    
}

pub struct Pretty(Vec<f64>);

impl fmt::Display for Pretty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..28 {
            for col in 0..28 {
                let elem = match self.0[28 * row + col] < 0.01 {
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn xor_test() {
        let mut model : Model<f64> = Model {
            layers: vec![
                Layer::from_size(2, 2, Activation::None),
                Layer::from_size(2,1, Activation::Sigmoid)
            ],
            loss: Loss::MeanSquaredError
        };

        let train = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0]
        ];

        let validate = vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0]
        ];

        model.fit(train, validate, 10000, 1.2);
        
        assert!(model.evaluate(&vec![0.0, 1.0])[0] > 0.8);
        assert!(model.evaluate(&vec![1.0, 1.0])[0] < 0.3);
    }
}
