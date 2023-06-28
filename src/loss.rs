use crate::vec_tools::{self, ValidNumber};

#[derive(Debug, Clone, PartialEq)]
pub enum Loss {
    MeanSquaredError,
    AbsoluteMeanSquaredError,
}

impl Loss {
    pub fn calculate_loss<T : ValidNumber<T>>(&self, result : Vec<T>, expected : Vec<T>) -> T { 
        match self {
            Loss::MeanSquaredError => Loss::mean_squared_error(result[0], expected[0]),
            Loss::AbsoluteMeanSquaredError => Loss::absolute_mean_squared_error(result[0], expected[0]),
        }
    }

    pub fn get_gradient<T : ValidNumber<T>>(&self, result : Vec<Vec<T>>, expected : Vec<Vec<T>>) -> Vec<Vec<T>> {
        match self {
            _ => Loss::mean_squared_error_grad(result, expected)
        }
    }

    fn mean_squared_error<T : ValidNumber<T>>(result : T, expected : T) -> T {
        (expected - result) * (expected - result)
    }

    fn mean_squared_error_grad<T: ValidNumber<T>>(result: Vec<Vec<T>>, expected: Vec<Vec<T>>) -> Vec<Vec<T>> {
        // Result and expected should be column vectors!
        
        result
            .iter()
            .zip(
                expected.iter()
            )
            .map(|(res, expec)| vec![T::from(-2.0) * (expec[0] - res[0])])
            .collect()
    }

    fn absolute_mean_squared_error<T : ValidNumber<T>>(result : T, expected : T) -> T {
        ((expected - result) * (expected - result)).into().abs().into()
    }
}
