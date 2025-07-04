#[cfg(feature = "backend-ort")]
pub mod ort_worker;

use crate::error::{RettoError, RettoResult};
use crate::serde::*;
use ndarray::prelude::*;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub(crate) enum RettoWorkerModelResolvedSource {
    #[cfg(not(target_family = "wasm"))]
    Path(std::path::PathBuf),
    Blob(Vec<u8>),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RettoWorkerModelSource {
    #[cfg(not(target_family = "wasm"))]
    Path(String),
    Blob(Vec<u8>),
    #[cfg(feature = "hf-hub")]
    HuggingFace {
        repo: String,
        model: String,
    },
}

impl RettoWorkerModelSource {
    pub(crate) fn resolve(self) -> RettoResult<RettoWorkerModelResolvedSource> {
        match self {
            #[cfg(not(target_family = "wasm"))]
            RettoWorkerModelSource::Path(path) => {
                let path = std::path::PathBuf::from(path);
                match path.exists() {
                    true => Ok(RettoWorkerModelResolvedSource::Path(path)),
                    false => Err(RettoError::ModelNotFoundError(
                        path.into_os_string().to_string_lossy().to_string(),
                    )),
                }
            }
            RettoWorkerModelSource::Blob(blob) => match blob.is_empty() {
                true => Err(RettoError::ModelNotFoundError(
                    "Empty model blob!".to_string(),
                )),
                false => Ok(RettoWorkerModelResolvedSource::Blob(blob)),
            },
            #[cfg(feature = "hf-hub")]
            RettoWorkerModelSource::HuggingFace { repo, model } => {
                use crate::hf_hub_helper::HfHubHelper;
                let helper = HfHubHelper::new();
                let path = helper.get_model_file(&repo, &model)?;
                Ok(RettoWorkerModelResolvedSource::Path(path))
            }
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RettoWorkerModelProvider {
    pub det: RettoWorkerModelSource,
    pub rec: RettoWorkerModelSource,
    pub cls: RettoWorkerModelSource,
}

// TODO: Split each worker into different cases so that GAT can be fully utilised,
// TODO: and take advantage of the metadata functionality of the ONNX model
pub(crate) trait RettoInnerWorker {
    fn det(&mut self, input: Array4<f32>) -> RettoResult<Array4<f32>>;
    fn cls(&mut self, input: Array4<f32>) -> RettoResult<Array2<f32>>;
    fn rec(&mut self, input: Array4<f32>) -> RettoResult<Array3<f32>>;
}

pub trait RettoWorkerModelProviderBuilder: Debug + Clone + MaybeSerde {
    #[cfg(all(not(target_family = "wasm"), feature = "hf-hub"))]
    fn from_hf_hub_v4_default() -> Self;
    #[cfg(not(target_family = "wasm"))]
    fn from_local_v4_path_default() -> Self;
    fn from_local_v4_blob_default() -> Self;
    fn default_provider() -> Self {
        #[cfg(all(not(target_family = "wasm"), feature = "hf-hub"))]
        return Self::from_hf_hub_v4_default();
        #[cfg(all(not(target_family = "wasm"), not(feature = "hf-hub")))]
        return Self::from_local_v4_path_default();
        #[cfg(target_family = "wasm")]
        return Self::from_local_v4_blob_default();
    }
}

pub trait RettoWorker: RettoInnerWorker {
    type RettoWorkerModelProvider: RettoWorkerModelProviderBuilder;
    type RettoWorkerConfig: Debug + Default + Clone + MaybeSerde;
    fn new(cfg: Self::RettoWorkerConfig) -> RettoResult<Self>
    where
        Self: Sized;
    fn init(&self) -> RettoResult<()>;
}

pub mod prelude {
    #[cfg(feature = "backend-ort")]
    pub use super::ort_worker::*;
    pub use super::{
        RettoWorkerModelProvider, RettoWorkerModelProviderBuilder, RettoWorkerModelSource,
    };
}
