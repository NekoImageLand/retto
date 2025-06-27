use crate::error::{RettoError, RettoResult};
use crate::worker::{RettoInnerWorker, RettoWorker, RettoWorkerModelProvider};
use ndarray::prelude::*;
#[cfg(feature = "backend-ort-directml")]
use ort::execution_providers::DirectMLExecutionProvider;
#[cfg(feature = "backend-ort-cuda")]
use ort::execution_providers::{
    ArenaExtendStrategy::NextPowerOfTwo, CUDAExecutionProvider,
    cuda::CuDNNConvAlgorithmSearch::Exhaustive,
};
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use ort::value::TensorRef;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RettoOrtWorkerDeviceConfig {
    #[default]
    /// Use CPU Only
    CPU,
    /// Use CUDA with the specified device ID
    #[cfg(feature = "backend-ort-cuda")]
    Cuda(i32),
    /// Use DirectML with the specified device ID
    #[cfg(feature = "backend-ort-directml")]
    DirectML(i32),
}

#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RettoOrtWorkerConfig {
    pub device: RettoOrtWorkerDeviceConfig,
    pub det_model_source: RettoWorkerModelProvider,
    pub rec_model_source: RettoWorkerModelProvider,
    pub cls_model_source: RettoWorkerModelProvider,
}

#[derive(Debug)]
pub struct RettoOrtWorker {
    cfg: RettoOrtWorkerConfig,
    det_session: ort::session::Session,
    rec_session: ort::session::Session,
    cls_session: ort::session::Session,
}

fn build_ort_session(
    model_source: RettoWorkerModelProvider,
    providers: &[ExecutionProviderDispatch],
) -> RettoResult<ort::session::Session> {
    let builder = ort::session::Session::builder()?.with_execution_providers(providers)?;
    match model_source {
        #[cfg(not(target_arch = "wasm32"))]
        RettoWorkerModelProvider::Path(path) => builder
            .commit_from_file(path)
            .map_err(|e| RettoError::from(e)),
        RettoWorkerModelProvider::Blob(blob) => builder
            .commit_from_memory(&blob)
            .map_err(|e| RettoError::from(e)),
    }
}

impl RettoWorker for RettoOrtWorker {
    type RettoWorkerConfig = RettoOrtWorkerConfig;
    fn new(cfg: Self::RettoWorkerConfig) -> RettoResult<Self>
    where
        Self: Sized,
    {
        #[cfg(target_arch = "wasm32")]
        {
            tracing::debug!("Initializing ort in wasi...");
            ort::init()
                .with_global_thread_pool(ort::environment::GlobalThreadPoolOptions::default())
                .commit()
                .expect("Cannot initialize ort.");
        }
        let mut providers = Vec::new();
        match cfg.device {
            #[cfg(feature = "backend-ort-cuda")]
            RettoOrtWorkerDeviceConfig::Cuda(id) => providers.push(
                CUDAExecutionProvider::default()
                    .with_arena_extend_strategy(NextPowerOfTwo)
                    .with_conv_algorithm_search(Exhaustive)
                    .with_device_id(id)
                    .build(),
            ),
            #[cfg(feature = "backend-ort-directml")]
            RettoOrtWorkerDeviceConfig::DirectML(id) => providers.push(
                DirectMLExecutionProvider::default()
                    .with_device_id(id)
                    .build(),
            ),
            _ => {}
        };
        providers.push(CPUExecutionProvider::default().build());
        let det_session = build_ort_session(cfg.det_model_source.clone(), &providers)?;
        let cls_session = build_ort_session(cfg.cls_model_source.clone(), &providers)?;
        let rec_session = build_ort_session(cfg.rec_model_source.clone(), &providers)?;
        let worker = RettoOrtWorker {
            cfg,
            det_session,
            rec_session,
            cls_session,
        };
        Ok(worker)
    }

    fn init(&self) -> RettoResult<()> {
        Ok(())
    }
}

impl RettoInnerWorker for RettoOrtWorker {
    fn det(&mut self, input: Array4<f32>) -> RettoResult<Array4<f32>> {
        let outputs = self.det_session.run(ort::inputs! {
            "x" => TensorRef::from_array_view(&input.as_standard_layout())?
        })?;
        let val = &outputs[0]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix4>()?;
        let output = val.to_owned();
        Ok(output)
    }

    fn cls(&mut self, input: Array4<f32>) -> RettoResult<Array2<f32>> {
        let outputs = self.cls_session.run(ort::inputs! {
            "x" => TensorRef::from_array_view(&input.as_standard_layout())?
        })?;
        let val = &outputs[0]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix2>()?;
        let output = val.to_owned();
        Ok(output)
    }

    fn rec(&mut self, input: Array4<f32>) -> RettoResult<Array3<f32>> {
        let outputs = self.rec_session.run(ort::inputs! {
            "x" => TensorRef::from_array_view(&input.as_standard_layout())?
        })?;
        let val = &outputs[0]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix3>()?;
        let output = val.to_owned();
        Ok(output)
    }
}
