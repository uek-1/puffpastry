use crate::vec_tools::{self, Transpose};

#[derive(Clone, Debug, PartialEq)]
pub enum Activation {
    None,
    Sigmoid,
    Relu,
    Softmax,
}

impl Activation {
    pub fn activate_vec<T: vec_tools::ValidNumber<T>>(&self, preactivations: Vec<T>) -> Vec<T> {
        preactivations
            .iter()
            .map(|num| match self {
                Activation::None => *num,
                Activation::Relu => Self::relu(*num),
                Activation::Sigmoid => Self::sigmoid(*num),
                Activation::Softmax => Self::softmax(*num, preactivations.clone()),
            })
            .collect()
    }

    pub fn derivative<T: vec_tools::ValidNumber<T>>(
        &self,
        preactivations: Vec<Vec<T>>,
    ) -> Vec<Vec<T>> {
        preactivations
            .transposed()
            .iter()
            .map(|x| {
                x.iter()
                    .enumerate()
                    .map(|(num, y)| match self {
                        Activation::Sigmoid => Self::sigmoid_derivative(*y),
                        Activation::Relu => Self::relu_derivative(*y),
                        Activation::None => vec![T::from(1.0)],
                        Activation::Softmax => Self::softmax_derivative(num, x.clone()),
                    })
                    .collect::<Vec<Vec<T>>>()
                    .clone()
            })
            .collect::<Vec<Vec<Vec<T>>>>()
            .get(0)
            .cloned()
            .unwrap()
    }

    pub fn activate_num<T: vec_tools::ValidNumber<T>>(&self, num: T) -> T {
        match self {
            Activation::Sigmoid => Self::sigmoid(num),
            Activation::Relu => Self::relu(num),
            Activation::Softmax => todo!("SOFTMAX ACTIVATION HASN'T BEEN IMPLMENETED"),
            Activation::None => num,
        }
    }

    fn sigmoid<T: vec_tools::ValidNumber<T>>(num: T) -> T {
        T::from(1.0 / (1.0 + (-1.0 * num.into()).exp()))
    }

    fn sigmoid_derivative<T: vec_tools::ValidNumber<T>>(num: T) -> Vec<T> {
        let delta = Self::sigmoid(num);
        vec![delta * (T::from(1.0) - delta)]
    }

    fn relu<T: vec_tools::ValidNumber<T>>(num: T) -> T {
        if num < T::from(0.0) {
            return T::from(0.0);
        }
        num
    }

    fn relu_derivative<T: vec_tools::ValidNumber<T>>(num: T) -> Vec<T> {
        if num < T::from(0.0) {
            return vec![T::from(0.0)];
        }
        vec![T::from(1.0)]
    }

    pub fn softmax<T: vec_tools::ValidNumber<T>>(num: T, classes: Vec<T>) -> T {
        // To saturate to zero instead of infinity, subtract max(classes) from top and bottom.
        /*let max_classes : f64 = classes
            .iter()
            .map(|x| (*x).into())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        */
        let top: f64 = (num.into() - 0.0).exp();
        let bottom: f64 = classes
            .iter()
            .fold(0.0, |sum: f64, x: &T| sum + ((*x).into() - 0.0).exp());
        let out = T::from(top / bottom);

        //println!("softmax ({:?}, {:?}) = {:?}", top, bottom, out);
        //println!("max_classes - {max_classes}");

        out
    }

    fn softmax_derivative<T: vec_tools::ValidNumber<T>>(neuron: usize, classes: Vec<T>) -> Vec<T> {
        //ERROR: Softmax has {classes.len()} outputs and input nerons which are all codependent. Here we
        //only calculate the effect an input neuron has on its respective output activation,
        //ignoring the fact that the input neuron affects all {classes.len()} outputs.
        classes
            .iter()
            .enumerate()
            .map(|(idx, output)| {
                let smax_output = Self::softmax(*output, classes.clone());
                let smax_input = Self::softmax(classes[neuron], classes.clone());
                let delta = match idx == neuron {
                    true => T::from(1.0),
                    false => T::from(0.0),
                };

                // println!("si {smax_input:?}, so {smax_output:?}, delta {delta:?}, idx {idx:?} neuron {neuron}");

                smax_output * (delta - smax_input)
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use crate::vec_tools::MatrixMultiply;

    use super::*;

    #[test]
    fn softmax_test() {
        let items = vec![1.0, 2.0, 3.0];
        let res = Activation::softmax(items[0], items);
        assert!((res - 0.09003057).abs() < 0.001)
        // close eneough.
    }

    #[test]
    fn softmax_derivative_test() {
        let classes = vec![3.0, 4.0, 5.0];
        let res = Activation::softmax_derivative(0, classes);
        assert!((res[0] - 0.08192506).abs() < 0.001);
    }

    #[test]
    fn softmax_derivative_vector_test() {
        let classes = vec![1.0, 2.0, 3.0];
        let res = Activation::softmax_derivative(0, classes);
        println!("{res:?}");

        let error = res
            .into_iter()
            .zip([0.08192, -0.02203, -0.059892])
            .fold(0.0, |err, (res, expec)| err + (expec - res).abs());

        assert!(error < 0.001)
    }

    #[test]
    fn total_softmax_derivative_test() {
        let preactivations = vec![1.0, 2.0, 3.0];
        let res = Activation::Softmax.derivative(vec![preactivations].transposed());
        println!("{:?}", res);

        let truth = vec![
            vec![0.0819, -0.0220, -0.0598],
            vec![-0.0220, 0.1848, -0.1628],
            vec![-0.0598, -0.1628, 0.2226],
        ];

        for i in 0..truth.len() {
            for j in 0..truth[i].len() {
                assert!((res[i][j] - truth[i][j]).abs() < 0.01);
            }
        }
    }
}
