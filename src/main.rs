use crate::{layer::Dense, model::Model};
use csv::{Reader, StringRecord};
use rand::{self, Rng};
use std::fmt;

mod activation;
mod layer;
mod loss;
mod model;
mod tensor;
mod vec_tools;

use activation::Activation;
use loss::Loss;
use tensor::Tensor;

fn main() {
    /*
    let mut model: Model<f64> = Model {
        layers: vec![Layer::from_size(784, 10, Activation::Softmax)],
        loss: Loss::CategoricalCrossEntropy,
    };

    let mut train: Vec<Vec<f64>> = vec![];

    let mut validate: Vec<Vec<f64>> = vec![];

    let mut mnist_reader = Reader::from_path("data/mnist_train.csv").unwrap();

    let labels = 10;

    for (num, record) in mnist_reader.records().enumerate() {
        if num > 60000 {
            break;
        }

        if let Ok(x) = record {
            let label: usize = x
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
                    .collect(),
            );
        }
    }

    println!(
        "{:?} \n {}",
        validate[0].to_vec(),
        Pretty(train[0].to_vec())
    );

    model.fit(train.clone(), validate.clone(), 5, 0.002);

    println!("trained model : \n");
    //println!("{:?}", model);

    for i in 0..10 {
        println!(
            "testing on input {:?} : \n {}",
            validate[i].clone(),
            Pretty(train[i].clone())
        );

        let res = model.evaluate(&train[i]);
        println!("{:?}", res);
        println!(
            "Calculated loss for this input {:?}",
            model.loss.calculate_loss(res, validate[i].clone())
        );
    }
    */
}

pub struct Pretty(Vec<f64>);

impl fmt::Display for Pretty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..28 {
            for col in 0..28 {
                let elem = match self.0[28 * row + col] < 0.01 {
                    true => "  ",
                    false => "1.0",
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
        // std::env::set_var("RUST_BACKTRACE", "1");
        let mut model: Model<f64> = Model::from_layers(
            vec![
                Dense::from_size(2, 2, Activation::Sigmoid),
                Dense::from_size(2, 1, Activation::None),
            ],
            Loss::MeanSquaredError,
        );
        println!("Model: {model:?}");

        let train = vec![
            Tensor::from(vec![vec![0.0], vec![0.0]]),
            Tensor::from(vec![vec![1.0], vec![0.0]]),
            Tensor::from(vec![vec![0.0], vec![1.0]]),
            Tensor::from(vec![vec![1.0], vec![1.0]]),
        ];

        let validate = vec![
            Tensor::from(vec![vec![0.0]]),
            Tensor::from(vec![vec![1.0]]),
            Tensor::from(vec![vec![1.0]]),
            Tensor::from(vec![vec![0.0]]),
        ];

        model.fit(train, validate, 100, 1.2).unwrap();

        let res = model
            .evaluate(&Tensor::from(vec![vec![0.0], vec![1.0]]))
            .unwrap();

        println!("0 XOR 1 =  {:?}", res);
        // assert!(res > 0.8);

        let res = model
            .evaluate(&Tensor::from(vec![vec![1.0], vec![1.0]]))
            .unwrap();

        println!("1 XOR 1 = {res}");
        // assert!(res < 0.3);
        assert!(false);
    }
}
