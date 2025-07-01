use crate::error::{RettoError, RettoResult};
use crate::serde::*;
use crate::worker::{
    RettoInnerWorker, RettoWorker, RettoWorkerModelProvider, RettoWorkerModelProviderBuilder,
    RettoWorkerModelResolvedSource, RettoWorkerModelSource,
};
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
use std::ops::Deref;

#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RettoOrtWorkerDevice {
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

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RettoOrtWorkerModelProvider(pub RettoWorkerModelProvider);

impl Default for RettoOrtWorkerModelProvider {
    fn default() -> Self {
        Self::default_provider()
    }
}

impl Deref for RettoOrtWorkerModelProvider {
    type Target = RettoWorkerModelProvider;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RettoOrtWorkerConfig {
    pub device: RettoOrtWorkerDevice,
    pub models: RettoOrtWorkerModelProvider,
}

impl RettoWorkerModelProviderBuilder for RettoOrtWorkerModelProvider {
    #[cfg(all(not(target_family = "wasm"), feature = "hf-hub"))]
    fn from_hf_hub_v4_default() -> Self {
        let hf_repo = "pk5ls20/PaddleModel";
        Self(RettoWorkerModelProvider {
            det: RettoWorkerModelSource::HuggingFace {
                repo: hf_repo.to_string(),
                model: "retto/onnx/ch_PP-OCRv4_det_infer.onnx".to_string(),
            },
            rec: RettoWorkerModelSource::HuggingFace {
                repo: hf_repo.to_string(),
                model: "retto/onnx/ch_PP-OCRv4_rec_infer.onnx".to_string(),
            },
            cls: RettoWorkerModelSource::HuggingFace {
                repo: hf_repo.to_string(),
                model: "retto/onnx/ch_ppocr_mobile_v2.0_cls_infer.onnx".to_string(),
            },
        })
    }

    #[cfg(not(target_family = "wasm"))]
    fn from_local_v4_path_default() -> Self {
        Self(RettoWorkerModelProvider {
            det: RettoWorkerModelSource::Path("ch_PP-OCRv4_det_infer.onnx".into()),
            rec: RettoWorkerModelSource::Path("ch_PP-OCRv4_rec_infer.onnx".into()),
            cls: RettoWorkerModelSource::Path("ch_ppocr_mobile_v2.0_cls_infer.onnx".into()),
        })
    }

    #[cfg(target_family = "wasm")]
    fn from_local_v4_blob_default() -> Self {
        Self(RettoWorkerModelProvider {
            det: RettoWorkerModelSource::Blob(
                include_bytes!("../../../ch_PP-OCRv4_det_infer.onnx").to_vec(),
            ),
            rec: RettoWorkerModelSource::Blob(
                include_bytes!("../../../ch_PP-OCRv4_rec_infer.onnx").to_vec(),
            ),
            cls: RettoWorkerModelSource::Blob(
                include_bytes!("../../../ch_ppocr_mobile_v2.0_cls_infer.onnx").to_vec(),
            ),
        })
    }
}

#[derive(Debug)]
pub struct RettoOrtWorker {
    cfg: RettoOrtWorkerConfig,
    det_session: ort::session::Session,
    rec_session: ort::session::Session,
    cls_session: ort::session::Session,
}

fn build_ort_session(
    model_source: RettoWorkerModelSource,
    providers: &[ExecutionProviderDispatch],
) -> RettoResult<ort::session::Session> {
    let builder = ort::session::Session::builder()?.with_execution_providers(providers)?;
    let model_source = model_source.resolve()?;
    match model_source {
        #[cfg(not(target_family = "wasm"))]
        RettoWorkerModelResolvedSource::Path(path) => builder
            .commit_from_file(path)
            .map_err(|e| RettoError::from(e)),
        RettoWorkerModelResolvedSource::Blob(blob) => builder
            .commit_from_memory(&blob)
            .map_err(|e| RettoError::from(e)),
    }
}

impl RettoWorker for RettoOrtWorker {
    type RettoWorkerModelProvider = RettoOrtWorkerModelProvider;
    type RettoWorkerConfig = RettoOrtWorkerConfig;
    fn new(cfg: Self::RettoWorkerConfig) -> RettoResult<Self>
    where
        Self: Sized,
    {
        #[cfg(target_family = "wasm")]
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
            RettoOrtWorkerDevice::Cuda(id) => providers.push(
                CUDAExecutionProvider::default()
                    .with_arena_extend_strategy(NextPowerOfTwo)
                    .with_conv_algorithm_search(Exhaustive)
                    .with_device_id(id)
                    .build(),
            ),
            #[cfg(feature = "backend-ort-directml")]
            RettoOrtWorkerDevice::DirectML(id) => providers.push(
                DirectMLExecutionProvider::default()
                    .with_device_id(id)
                    .build(),
            ),
            _ => {}
        };
        providers.push(CPUExecutionProvider::default().build());
        let det_session = build_ort_session(cfg.models.det.clone(), &providers)?;
        let cls_session = build_ort_session(cfg.models.cls.clone(), &providers)?;
        let rec_session = build_ort_session(cfg.models.rec.clone(), &providers)?;
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
