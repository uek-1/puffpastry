use csv::{Reader, StringRecord};
use puffpastry::*;
use rand::{self, Rng};
use std::fmt;

fn main() {
    let mut model: Model<f64> = Model::new(Loss::CategoricalCrossEntropy);

    model.push_layer(Conv2d::from_size(
        vec![1, 28, 28],
        3,
        32,
        (1, 1),
        Activation::None,
    ));

    model.push_layer(MaxPool2d::new(2, 2));
    model.push_layer(Flatten {});
    model.push_layer(Dense::from_size(32 * 14 * 14, 100, Activation::None));
    // Output values of ^ are too large, causing softmax to output 0.0s into the CCE and introducing NANs into the weights.
    model.push_layer(Dense::from_size(100, 10, Activation::Softmax));
    // model.push_layer(Dense::from_size(28 * 28, 10, Activation::Softmax));

    let mut train: Vec<Tensor<f64>> = vec![];

    let mut validate: Vec<Tensor<f64>> = vec![];

    let mut mnist_reader = Reader::from_path("data/mnist_train.csv").unwrap();

    let labels = 10;

    let mut zero_count = 0;
    let train_count = 10;

    for (num, record) in mnist_reader.records().enumerate() {
        if num > train_count {
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

            validate.push(Tensor::column(val_vec));

            let image_data: Vec<f64> = x
                .into_iter()
                .skip(1)
                .map(|x| x.parse::<f64>().unwrap() / 255.0)
                .map(|x| {
                    if x == 0.0 {
                        zero_count += 1
                    };
                    x
                })
                .collect();

            train.push(Tensor {
                shape: vec![1, 28, 28],
                data: image_data,
            });
        }
    }

    println!(
        "zero count {}, train_count: {}, train_pixels: {}, proportion: {}",
        zero_count,
        train_count,
        train_count * 784,
        zero_count as f64 / (train_count as f64 * 784.0)
    );

    // return ();

    // println!("{} \n {}", validate[0], Pretty(train[0].data.clone()));

    // println!("{:?}", model.layers.last());
    let data1 = model.layers.last().unwrap().get_weights().unwrap().data;

    model
        .fit(train.clone(), validate.clone(), 1, 0.02)
        .expect("failed to train");

    println!("trained model : \n");
    // println!("{:?}", model.layers.last());

    let data2 = model.layers.last().unwrap().get_weights().unwrap().data;

    assert_ne!(data1, data2);

    for i in 0..2 {
        println!(
            "testing on input {} : \n {}",
            validate[i].clone(),
            Pretty(train[i].data.clone())
        );

        let res = model.evaluate(&train[i]).unwrap();
        println!("{:?}", res);
        println!(
            "Calculated loss for this input {:?}",
            model.loss.calculate_loss(res, validate[i].clone())
        );
    }
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
            Tensor::column(vec![0.0, 0.0]),
            Tensor::column(vec![1.0, 0.0]),
            Tensor::column(vec![0.0, 1.0]),
            Tensor::column(vec![1.0, 1.0]),
        ];

        let validate = vec![
            Tensor::column(vec![0.0]),
            Tensor::column(vec![1.0]),
            Tensor::column(vec![1.0]),
            Tensor::column(vec![0.0]),
        ];

        model.fit(train, validate, 100, 1.2).unwrap();

        let res = model.evaluate(&Tensor::column(vec![0.0, 1.0])).unwrap();

        println!("0 XOR 1 =  {:?}", res);
        assert!(res.data[0] > 0.8);

        let res = model.evaluate(&Tensor::column(vec![1.0, 1.0])).unwrap();

        println!("1 XOR 1 = {res}");
        assert!(res.data[0] < 0.3);
    }

    #[test]
    fn softmax_output_test() {
        let model: Model<f64> = Model::from_layers(
            vec![Dense::from_size(5, 5, Activation::Softmax)],
            Loss::MeanSquaredError,
        );

        let res = model
            .evaluate(&Tensor::column(vec![1.0, 2.0, 3.0, 4.0, 5.0]))
            .unwrap();

        let sum = res.iter().fold(0.0, |sum, x| sum + x);

        assert!(sum <= 1.0)
    }
}
