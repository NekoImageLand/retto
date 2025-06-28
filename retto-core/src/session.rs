use crate::error::RettoResult;
use crate::image_helper::ImageHelper;
use crate::processor::prelude::*;
use crate::serde::*;
use crate::worker::RettoWorker;

#[derive(Debug)]
pub struct RettoSession<W: RettoWorker> {
    worker: W,
    rec_character: RecCharacter,
    config: RettoSessionConfig<W>,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RettoSessionConfig<W: RettoWorker> {
    pub worker_config: W::RettoWorkerConfig,
    pub max_side_len: usize,
    pub min_side_len: usize,
    pub det_processor_config: DetProcessorConfig,
    pub cls_processor_config: ClsProcessorConfig,
    pub rec_processor_config: RecProcessorConfig,
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
            cls_processor_config: ClsProcessorConfig::default(),
            rec_processor_config: RecProcessorConfig::default(),
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RettoWorkerResult {
    pub det_result: DetProcessorResult,
    pub cls_result: ClsProcessorResult,
    pub rec_result: RecProcessorResult,
}

impl<W> RettoSession<W>
where
    W: RettoWorker,
{
    pub fn new(cfg: RettoSessionConfig<W>) -> RettoResult<Self> {
        // load dict
        let worker = W::new(cfg.worker_config.clone())?; // TODO:
        let rec_character =
            RecCharacter::new(cfg.rec_processor_config.character_source.clone(), vec![0])?;
        worker.init()?;
        Ok(RettoSession {
            worker,
            rec_character,
            config: cfg,
        })
    }

    pub fn run(&mut self, input: impl AsRef<[u8]>) -> RettoResult<RettoWorkerResult> {
        let mut image = ImageHelper::new_from_raw_img_flow(input)?; // TODO: args
        let (ratio_h, ratio_w) =
            image.resize_both(self.config.max_side_len, self.config.min_side_len)?;
        let (after_h, after_w) = image.size();
        let arr = image.array_view()?; // cheap
        let det = DetProcessor::new(&self.config.det_processor_config, after_h, after_w)?;
        let det_res = det.process(arr, |i| self.worker.det(i))?;
        tracing::debug!("det result: {:?}", det_res);
        // As you can see, crop_images is unsanitary, but currently only limited to changing incorrect cls angles
        let mut crop_images = det_res
            .0
            .iter()
            .map(|(pb, _)| ImageHelper::new_from_rgb_image(image.get_crop_img(&pb)))
            .collect::<Vec<_>>();
        let cls = ClsProcessor::new(&self.config.cls_processor_config);
        let cls_res = cls.process(&mut crop_images, |i| self.worker.cls(i))?;
        tracing::debug!("cls result: {:?}", cls_res);
        let rec = RecProcessor::new(&self.config.rec_processor_config, &self.rec_character);
        let rec_res = rec.process(&crop_images, |i| self.worker.rec(i))?;
        tracing::debug!("rec result: {:?}", rec_res);
        // TODO: RettoWorkerResult
        Ok(RettoWorkerResult {
            det_result: det_res,
            cls_result: cls_res,
            rec_result: rec_res,
        })
    }
}
