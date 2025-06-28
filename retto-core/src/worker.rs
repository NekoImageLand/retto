pub mod ort_worker;

use crate::error::RettoResult;
use crate::serde::*;
use ndarray::prelude::*;
use std::fmt::Debug;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RettoWorkerModelProvider {
    #[cfg(not(target_arch = "wasm32"))]
    Path(String),
    Blob(Vec<u8>),
}

impl Default for RettoWorkerModelProvider {
    fn default() -> Self {
        RettoWorkerModelProvider::Blob(Vec::new())
    }
}

// TODO: Split each worker into different cases so that GAT can be fully utilised,
// TODO: and take advantage of the metadata functionality of the ONNX model
pub(crate) trait RettoInnerWorker {
    fn det(&mut self, input: Array4<f32>) -> RettoResult<Array4<f32>>;
    fn cls(&mut self, input: Array4<f32>) -> RettoResult<Array2<f32>>;
    fn rec(&mut self, input: Array4<f32>) -> RettoResult<Array3<f32>>;
}

pub trait RettoWorker: RettoInnerWorker {
    type RettoWorkerConfig: Debug + Default + Clone + MaybeSerde;
    fn new(cfg: Self::RettoWorkerConfig) -> RettoResult<Self>
    where
        Self: Sized;
    fn init(&self) -> RettoResult<()>;
}

pub mod prelude {
    pub use super::RettoWorkerModelProvider;
    pub use super::ort_worker::*;
}
