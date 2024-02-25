use csv::Reader;
use puffpastry::*;
use std::fmt;

fn main() {
    let mut model: Model<f64> = Model::new(Loss::CategoricalCrossEntropy);
    model.push_layer(Conv2d::from_size(
        vec![1, 28, 28],
        3,
        32,
        (1, 1),
        Activation::Relu,
    ));
    model.push_layer(MaxPool2d::new(2, 2));
    model.push_layer(Flatten::new());
    model.push_layer(Dense::from_size(32 * 14 * 14, 10, Activation::Softmax));

    let mut train_inputs: Vec<Tensor<f64>> = vec![];
    let mut train_outputs: Vec<Tensor<f64>> = vec![];
    load_mnist_with(&mut train_inputs, &mut train_outputs, 5000);

    model
        .fit(train_inputs.clone(), train_outputs.clone(), 3, 0.02)
        .expect("failed to train");

    for i in 0..5 {
        println!(
            "testing on input {} : \n {}",
            train_outputs[i].clone(),
            Pretty(train_inputs[i].data.clone())
        );

        let res = model.evaluate(&train_inputs[i]).unwrap();
        println!("{:?}", res);
        println!(
            "Calculated loss for this input {:?}",
            model.loss.calculate_loss(res, train_outputs[i].clone())
        );
    }
}

fn load_mnist_with<T: ValidNumber<T>>(
    train: &mut Vec<Tensor<T>>,
    validate: &mut Vec<Tensor<T>>,
    samples: usize,
) {
    let mut mnist_reader = Reader::from_path("data/mnist_train.csv").unwrap();
    let labels = 10;
    let mut zero_count = 0;

    for (num, record) in mnist_reader.records().enumerate() {
        if num > samples {
            break;
        }

        if let Ok(x) = record {
            let label: usize = x
                .into_iter()
                .take(1)
                .map(|x| x.parse().unwrap())
                .next()
                .unwrap();

            let mut val_vec = vec![T::from(0.0); labels];
            val_vec[label] = T::from(1.0);

            validate.push(Tensor::column(val_vec));

            let image_data: Vec<T> = x
                .into_iter()
                .skip(1)
                .map(|x| x.parse::<f64>().unwrap() / 255.0)
                .map(|x| {
                    if x == 0.0 {
                        zero_count += 1
                    };
                    x
                })
                .map(|x| T::from(x))
                .collect();

            train.push(Tensor {
                shape: vec![1, 28, 28],
                data: image_data,
            });
        }
    }
}

pub struct Pretty<T: ValidNumber<T>>(Vec<T>);

impl<T: ValidNumber<T>> fmt::Display for Pretty<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..28 {
            for col in 0..28 {
                let elem = match self.0[28 * row + col] < T::from(0.01) {
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

    #[test]
    fn proc_macro_test() {
        let x = tensor!([[1, 2, 3], [1, 2, 3], [1, 2, 3]]);
        println!("{x}");
        assert!(x.shape == vec![3, 3]);
        assert!(x.get(&[1, 2]) == Some(&3.0));
    }
}
