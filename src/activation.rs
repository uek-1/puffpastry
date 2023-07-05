use crate::vec_tools;

#[derive(Clone, Debug, PartialEq)]
pub enum Activation {
    None,
    Sigmoid,
    Relu,
    Softmax,
}

impl Activation {
    pub fn activate_vec<T : vec_tools::ValidNumber<T>>(&self, row : Vec<T>) -> Vec<T> {
        row
            .iter()
            .map(|num| self.activate_num(*num))
            .collect()
    } 

    pub fn derivative<T: vec_tools::ValidNumber<T>>(&self, row : Vec<Vec<T>>) -> Vec<Vec<T>> {
        match self {
            Activation::Sigmoid => {
                row
                    .iter()
                    .map(
                        |x| x.iter().map(
                            |y| Self::sigmoid_derivative(*y)
                        )
                        .collect()
                    )
                    .collect()
            }
            Activation::Relu => {
                row
                    .iter()
                    .map(
                        |x| x.iter().map(
                            |y| Self::relu_derivative(*y)
                        )
                        .collect()
                    )
                    .collect()
            }
            Activation::Softmax => todo!("SOFTMAX DERIVATIVE HASN'T BEEN IMPLEMENTED YET"),
            Activation::None => vec![vec![T::from(1.0); row[0].len()]; row.len()]
        }
    }

    pub fn activate_num<T : vec_tools::ValidNumber<T>>(&self, num : T) -> T {
        match self {
            Activation::Sigmoid => Self::sigmoid(num),
            Activation::Relu => Self::relu(num),
            Activation::Softmax => todo!("SOFTMAX ACTIVATION HASN'T BEEN IMPLMENETED"),
            Activation::None => num
        }
    }

    fn sigmoid<T : vec_tools::ValidNumber<T>>(num: T) -> T {
        T::from(1.0 / (1.0 + (-1.0 * num.into()).exp()))
    }

    fn sigmoid_derivative<T : vec_tools::ValidNumber<T>>(num: T) -> T {
        let delta = Self::sigmoid(num);
        delta * (T::from(1.0) - delta)
    }

    fn relu<T : vec_tools::ValidNumber<T>>(num : T) -> T {
        if num < T::from(0.0) {
            return T::from(0.0);
        }
        num
    }

    fn relu_derivative<T : vec_tools::ValidNumber<T>>(num: T) ->T {
        if num < T::from(0.0) {
            return T::from(0.0)
        }
        T::from(1.0)
    }

    fn softmax<T : vec_tools::ValidNumber<T>>(num : T, classes : Vec<T>) -> T {
        let top : f64 = num.into().exp(); 
        let bottom : f64 = classes.iter().fold(0.0, |sum : f64, x : &T| sum + (*x).into().exp());
        
        T::from(top / bottom)
    }

    fn softmax_derivative<T: vec_tools::ValidNumber<T>>(num : T, classes : Vec<T>) -> T {
        todo!()
    }
}

