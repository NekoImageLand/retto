use crate::error::RettoResult;
use crate::worker::{RettoInnerWorker, RettoWorker};
use ndarray::{Array4, Ix4};
use ort::execution_providers::ArenaExtendStrategy::NextPowerOfTwo;
use ort::execution_providers::cuda::CuDNNConvAlgorithmSearch::Exhaustive;
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider,
};
use ort::value::TensorRef;

#[derive(Debug, Default, Clone)]
pub enum RettoOrtWorkerDeviceConfig {
    #[default]
    /// Use CPU Only
    CPU,
    /// Use CUDA with the specified device ID
    Cuda(i32),
    /// Use DirectML with the specified device ID
    DirectML(i32),
}

#[derive(Debug, Default, Clone)]
pub struct RettoOrtWorkerConfig {
    pub device: RettoOrtWorkerDeviceConfig,
    pub det_model_path: String,
    pub rec_model_path: String,
    pub cls_model_path: String,
}

#[derive(Debug)]
pub struct RettoOrtWorker {
    cfg: RettoOrtWorkerConfig,
    det_session: ort::session::Session,
    rec_session: ort::session::Session,
    cls_session: ort::session::Session,
}

impl RettoWorker for RettoOrtWorker {
    type RettoWorkerConfig = RettoOrtWorkerConfig;
    fn new(cfg: Self::RettoWorkerConfig) -> RettoResult<Self>
    where
        Self: Sized,
    {
        let mut providers = Vec::new();
        match cfg.device {
            RettoOrtWorkerDeviceConfig::Cuda(id) => providers.push(
                CUDAExecutionProvider::default()
                    .with_arena_extend_strategy(NextPowerOfTwo)
                    .with_conv_algorithm_search(Exhaustive)
                    .with_device_id(id)
                    .build(),
            ),
            RettoOrtWorkerDeviceConfig::DirectML(id) => providers.push(
                DirectMLExecutionProvider::default()
                    .with_device_id(id)
                    .build(),
            ),
            _ => {}
        };
        providers.push(CPUExecutionProvider::default().build());
        let det_session = ort::session::Session::builder()?
            .with_execution_providers(providers.as_slice())?
            .commit_from_file(cfg.det_model_path.clone())?;
        let rec_session = ort::session::Session::builder()?
            .with_execution_providers(providers.as_slice())?
            .commit_from_file(cfg.rec_model_path.clone())?;
        let cls_session = ort::session::Session::builder()?
            .with_execution_providers(providers.as_slice())?
            .commit_from_file(cfg.cls_model_path.clone())?;
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

    fn cls(&mut self, input: Array4<f32>) -> RettoResult<Array4<f32>> {
        todo!()
    }

    fn rec(&mut self, input: Array4<f32>) -> RettoResult<Array4<f32>> {
        todo!()
    }
}
