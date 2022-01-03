use linfa::prelude::Fit;
use linfa::{Dataset, DatasetBase, DatasetView, Float};
use linfa_clustering::{generate_blobs, KMeans};
use nalgebra as na;
use ndarray::{s, Array, Array2, Array3, Dim};
use num_traits::Zero;
use rand::{Rng, SeedableRng};
use std::marker::PhantomData;
use std::process::Output;

#[derive(Debug, Clone)]
struct PQ<Input, Code> {
    sub_space: usize,
    code_space: usize,
    datas: Option<usize>,
    codewords: Option<Array3<Input>>,
    _code: PhantomData<Code>,
}

impl<Input: Default, Code: Default> Default for PQ<Input, Code> {
    fn default() -> Self {
        todo!()
    }
}

impl<Input: Default + Clone + Zero + Float, Code: Default + Clone> PQ<Input, Code> {
    pub fn new(sub_space: usize, code_space: usize) -> Self {
        PQ {
            sub_space,
            code_space,
            ..Default::default()
        }
    }

    pub fn fit<R: Rng + Clone + SeedableRng>(
        &mut self,
        input: &Array2<Input>,
        rng: &mut R,
    ) -> anyhow::Result<()> {
        let datas = input.shape()[1] / self.sub_space;

        self.codewords = Some(Array3::zeros((self.sub_space, self.code_space, datas)));

        for i in 0..self.sub_space {
            let input_sub = input.slice(s![.., i * datas..(i + 1) * datas]).to_owned();
            let dataset = DatasetBase::from(input_sub);
            let model = linfa_clustering::KMeans::params(self.code_space)
                .max_n_iterations(200)
                .tolerance(Input::from_f64(1e-5).unwrap())
                .fit(&dataset)?;
            let centroids = model.centroids();

            self.codewords
                .as_mut()
                .unwrap()
                .slice_mut(s![i, .., ..])
                .assign(centroids);
        }

        unimplemented!();
    }

    pub fn encode(&self, input: &Array2<Input>) -> Array2<Code> {
        unimplemented!()
    }

    pub fn decode(&self, code: &Array2<Code>) -> Array2<Input> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Dim;
    use ndarray::{Array, ArrayBase};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand_isaac::Isaac64Rng;
    use std::ops::Sub;

    #[test]
    fn test_encode_decode() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let (N, D, M, Ks) = (100, 12, 4, 10);

        let X = Array::random((N, D), Uniform::new(0.0, 1.0));

        let mut pq = PQ::<f64, u8>::new(M, Ks);
        pq.fit(&X, &mut rng);

        let X_ = pq.encode(&X);
        let X__ = pq.decode(&X_);

        //assert!(X.approx_eq(&X__));
    }
}
