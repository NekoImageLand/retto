use crate::error::RettoResult;
use crate::image_helper::ImageHelper;
use crate::processor::prelude::*;
use crate::worker::RettoWorker;
use ndarray::Array3;

#[derive(Debug)]
pub struct RettoSession<W: RettoWorker> {
    worker: W,
    config: RettoSessionConfig<W>,
}

#[derive(Debug)]
pub struct RettoSessionConfig<W: RettoWorker> {
    pub worker_config: W::RettoWorkerConfig,
    pub max_side_len: usize,
    pub min_side_len: usize,
    pub det_processor_config: DetProcessorConfig,
    pub rec_processor_config: RecProcessorConfig,
    pub cls_processor_config: ClsProcessorConfig,
}

impl<W> Default for RettoSessionConfig<W>
where
    W: RettoWorker,
{
    fn default() -> Self {
        RettoSessionConfig {
            worker_config: <_>::default(),
            max_side_len: 2000,
            min_side_len: 30,
            det_processor_config: DetProcessorConfig::default(),
            rec_processor_config: RecProcessorConfig::default(),
            cls_processor_config: ClsProcessorConfig::default(),
        }
    }
}

#[derive(Debug)]
pub struct RettoWorkerResult;

impl<W> RettoSession<W>
where
    W: RettoWorker,
{
    pub fn new(cfg: RettoSessionConfig<W>) -> RettoResult<Self> {
        let worker = W::new(cfg.worker_config.clone())?; // TODO:
        worker.init()?;
        Ok(RettoSession {
            worker,
            config: cfg,
        })
    }

    pub fn run(&mut self, input: impl AsRef<[u8]>) -> RettoResult<RettoWorkerResult> {
        let mut image = ImageHelper::new_from_raw_img(input)?; // TODO: args
        let image_ori_size = image.ori_size();
        let (ratio_h, ratio_w) =
            image.resize_both(self.config.max_side_len, self.config.min_side_len)?;
        let arr: RettoResult<Array3<u8>> = image.into();
        let arr = arr?;
        let det = DetProcessor::new(
            &self.config.det_processor_config,
            image_ori_size.0,
            image_ori_size.1,
        )?;
        let det = det.process(&arr, |i| self.worker.det(i))?;
        tracing::info!("det result: {:#?}", det);
        Ok(RettoWorkerResult)
    }
}
