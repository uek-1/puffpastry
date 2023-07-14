use crate::{
    vec_tools::{self, Transpose, ValidNumber},
    Activation, Tensor,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Loss {
    MeanSquaredError,
    CategoricalCrossEntropy,
}

impl Loss {
    pub fn calculate_loss<T: ValidNumber<T>>(
        &self,
        result: Tensor<T>,
        expected: Tensor<T>,
    ) -> Result<T, ()> {
        match self {
            Loss::MeanSquaredError => Loss::mean_squared_error(result, expected).ok_or_else(|| ()),
            Loss::CategoricalCrossEntropy => {
                Loss::categorical_cross_entropy(result, expected).ok_or_else(|| ())
            }
        }
    }

    pub fn get_gradient<T: ValidNumber<T>>(
        &self,
        result: Tensor<T>,
        expected: Tensor<T>,
    ) -> Result<Tensor<T>, ()> {
        match self {
            Loss::MeanSquaredError => Loss::mean_squared_error_grad(result, expected),
            Loss::CategoricalCrossEntropy => Loss::categorical_cross_entropy_grad(result, expected),
        }
    }

    fn mean_squared_error<T: ValidNumber<T>>(result: Tensor<T>, expected: Tensor<T>) -> Option<T> {
        if result.shape() != expected.shape() {
            return None;
        }
        let total: f64 = result
            .data
            .iter()
            .zip(expected.data.iter())
            .fold(T::from(0.0), |error, (res, expec)| {
                error + (*expec - *res) * (*expec - *res)
            })
            .into();

        Some(T::from(total / result.data.len() as f64))
    }

    fn categorical_cross_entropy<T: ValidNumber<T>>(
        result: Tensor<T>,
        expected: Tensor<T>,
    ) -> Option<T> {
        if result.shape() != expected.shape() || result.shape()[1] != 1 && expected.shape()[1] != 1
        {
            return None;
        }
        // Requires the output to be one-hot encoded.
        Some(
            T::from(-1.0)
                * result
                    .data
                    .into_iter()
                    .zip(expected.data.into_iter())
                    .map(|(res, expec)| (res.into(), expec.into()))
                    .find(|(_, expec): &(f64, f64)| *expec == 1.0)
                    .and_then(|(res, _)| {
                        // clipped to 1000.0
                        let out = res.ln();
                        if 1000.0 < -1.0 * out || out.is_nan() {
                            return Some(T::from(-1000.0));
                        }
                        Some(T::from(out))
                    })
                    .unwrap(),
        )
    }

    fn mean_squared_error_grad<T: ValidNumber<T>>(
        result: Tensor<T>,
        expected: Tensor<T>,
    ) -> Result<Tensor<T>, ()> {
        // Result and expected should be column vectors!
        if result.shape() != expected.shape() || result.shape()[1] != 1 {
            return Err(());
        }

        let data: Vec<Vec<T>> = result
            .as_rows()
            .iter()
            .zip(expected.as_rows().iter())
            .map(|(res, expec)| vec![T::from(-2.0) * (expec.data[0] - res.data[0])])
            .collect();

        Ok(Tensor::from(data))
    }

    fn categorical_cross_entropy_grad<T: ValidNumber<T>>(
        result: Tensor<T>,
        expected: Tensor<T>,
    ) -> Result<Tensor<T>, ()> {
        // Result and expected should be column vectors!
        if result.shape() != expected.shape() || result.shape()[1] != 1 {
            return Err(());
        }
        // clipped to 1000.0 as max
        let data: Vec<Vec<T>> = result
            .as_rows()
            .iter()
            .zip(expected.as_rows().iter())
            .map(|(res, expec)| (res.data[0].into(), expec.data[0].into()))
            .map(|(res, expec): (f64, f64)| {
                vec![T::from(
                    if 1000.0 < expec * (1.0 / res) || (1.0 / res).is_nan() {
                        -1000.0
                    } else {
                        -1.0 * expec * (1.0 / res)
                    },
                )]
            })
            .collect();

        Ok(Tensor::from(data))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn mean_sq_test() {
        let v1 = Tensor::from(vec![1.0, 2.0, 3.0]);
        let v2 = Tensor::from(vec![2.0, 3.0, 4.0]);
        let res = Loss::mean_squared_error(v1, v2);

        assert_eq!(res, Some(1.0))
    }

    #[test]
    fn cce_test() {
        let v1 = Tensor::from(vec![vec![0.3], vec![0.0], vec![0.0]]);
        let v2 = Tensor::from(vec![vec![1.0], vec![0.0], vec![0.8]]);
        let res = Loss::categorical_cross_entropy(v1, v2);

        assert_eq!(res, Some(-1.0 * 0.3f64.ln()));
    }
}
