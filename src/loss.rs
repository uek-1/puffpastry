use crate::{vec_tools::{self, ValidNumber, Transpose}, Activation};

#[derive(Debug, Clone, PartialEq)]
pub enum Loss {
    MeanSquaredError,
    CategoricalCrossEntropy,
}

impl Loss {
    pub fn calculate_loss<T : ValidNumber<T>>(&self, result : Vec<T>, expected : Vec<T>) -> T { 
        match self {
            Loss::MeanSquaredError => Loss::mean_squared_error(result, expected),
            Loss::CategoricalCrossEntropy => Loss::categorical_cross_entropy(result, expected)
        }
    }

    pub fn get_gradient<T : ValidNumber<T>>(&self, result : Vec<Vec<T>>, expected : Vec<Vec<T>>) -> Vec<Vec<T>> {
        match self {
            Loss::MeanSquaredError => Loss::mean_squared_error_grad(result, expected),
            Loss::CategoricalCrossEntropy => Loss::categorical_cross_entropy_grad(result, expected)
        }
    }

    fn mean_squared_error<T : ValidNumber<T>>(result : Vec<T>, expected : Vec<T>) -> T {
        let total : f64 = result
            .iter()
            .zip(expected.iter())
            .fold(T::from(0.0), |error, (res, expec)| error + (*expec - *res) * (*expec - *res))
            .into();

        T::from(total / result.len() as f64)

    }

    fn categorical_cross_entropy<T: ValidNumber<T>>(result: Vec<T>, expected: Vec<T>) -> T {
        // Requires the output to be one-hot encoded.
        T::from(-1.0) * result
            .into_iter()
            .zip(expected.into_iter())
            .map(|(res, expec)| (res.into(), expec.into()))
            .find(|(_, expec) : & (f64, f64)| *expec == 1.0)
            .and_then(|(res, _)| Some(T::from(res.ln())))
            .unwrap()    
    }

    fn mean_squared_error_grad<T: ValidNumber<T>>(result: Vec<Vec<T>>, expected: Vec<Vec<T>>) -> Vec<Vec<T>> {
        // Result and expected should be column vectors!
        
        result
            .iter()
            .zip(expected.iter())
            .map(|(res, expec)| vec![T::from(-2.0) * (expec[0] - res[0])])
            .collect()
    }
    
    fn categorical_cross_entropy_grad<T: ValidNumber<T>>(result: Vec<Vec<T>>, expected: Vec<Vec<T>>) -> Vec<Vec<T>> {
        // Result and expected should be column vectors! 

        result
            .iter()
            .zip(expected.iter())
            .map(|(res, expec)| (res[0].into(), expec[0].into()))
            .map(|(res, expec) : (f64,f64)| vec![T::from(-1.0 * expec * (1.0/res))])
            .collect()
    }

    /* Actually, it's unlikely that I need this. The last softmax layer should be able to calculate
    * its own loss.

    fn softmax_cce<T : ValidNumber<T>>(result: Vec<T>, expected: Vec<T>) -> T {
        let result_softmax = result
            .clone()
            .into_iter()
            .map(|x| Activation::softmax(x, result.clone()));

        Self::categorical_cross_entropy(result, expected)
    }

    fn softmax_cce_grad<T : ValidNumber<T>>(result: Vec<Vec<T>>, expected : Vec<Vec<T>>) -> Vec<Vec<T>> {
        //Result and expected should be column vectors!
        todo!()
    }

    */
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn mean_sq_test() {
        let v1 = vec![1.0 ,2.0 ,3.0];
        let v2 = vec![2.0, 3.0, 4.0];
        let res = Loss::mean_squared_error(v1, v2);

        assert_eq!(res, 1.0)
    }

    #[test]
    fn cce_test() {
        let v1 = vec![0.3 ,0.0 ,0.0];
        let v2 = vec![1.0, 0.0, 0.8];
        let res = Loss::categorical_cross_entropy(v1, v2);

        assert_eq!(res, -1.0 *  0.3f64.ln());
    }
}
