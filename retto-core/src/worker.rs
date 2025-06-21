pub mod ort_worker;

use crate::error::RettoResult;
use ndarray::prelude::*;
use std::fmt::Debug;

pub(crate) trait RettoInnerWorker {
    fn det(&mut self, input: Array4<f32>) -> RettoResult<Array4<f32>>;
    fn cls(&mut self, input: Array4<f32>) -> RettoResult<Array4<f32>>;
    fn rec(&mut self, input: Array4<f32>) -> RettoResult<Array4<f32>>;
}

pub trait RettoWorker: RettoInnerWorker {
    type RettoWorkerConfig: Debug + Default + Clone;
    fn new(cfg: Self::RettoWorkerConfig) -> RettoResult<Self>
    where
        Self: Sized;
    fn init(&self) -> RettoResult<()>;
}

pub mod prelude {
    pub use super::ort_worker::*;
}
