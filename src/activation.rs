use crate::vec_tools;

#[derive(Clone, Debug, PartialEq)]
pub enum Activation {
    None,
    Sigmoid,
    Relu,
    Softmax,
}

impl Activation {
    pub fn activate_vec<T : vec_tools::ValidNumber<T>>(&self, preactivations : Vec<T>) -> Vec<T> {
        preactivations
            .iter()
            .map(|num| {
                match self {
                    Activation::None => *num,
                    Activation::Relu => Self::relu(*num),
                    Activation::Sigmoid => Self::sigmoid(*num),
                    Activation::Softmax => Self::softmax(*num, preactivations.clone())
                }
            })
            .collect()
    } 

    pub fn derivative<T: vec_tools::ValidNumber<T>>(&self, preactivations : Vec<Vec<T>>) -> Vec<Vec<T>> {
        preactivations
            .iter()
            .map(
                |x| x.iter().map(
                    |y|{ 
                        match self {
                            Activation::Sigmoid => Self::sigmoid_derivative(*y),
                            Activation::Relu => Self::relu_derivative(*y),
                            Activation::None => T::from(1.0),
                            Activation::Softmax => Self::softmax_derivative(*y, x.clone()),
                            _ => todo!("Invalid")
                        }
                    }
                    )
                    .collect()
            )
            .collect()
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

    pub fn softmax<T : vec_tools::ValidNumber<T>>(num : T, classes : Vec<T>) -> T {
        let top : f64 = num.into().exp(); 
        let bottom : f64 = classes.iter().fold(0.0, |sum : f64, x : &T| sum + (*x).into().exp());
        
        T::from(top / bottom)
    }

    fn softmax_derivative<T: vec_tools::ValidNumber<T>>(num : T, classes : Vec<T>) -> T {
        todo!()
    }
}

