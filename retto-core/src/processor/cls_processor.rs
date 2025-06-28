use crate::error::{RettoError, RettoResult};
use crate::image_helper::ImageHelper;
use crate::processor::{Processor, ProcessorInner, ProcessorInnerIO, ProcessorInnerRes};
use crate::serde::*;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::fmt::Display;

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClsProcessorConfig {
    pub image_shape: [usize; 3],
    pub batch_num: usize,
    pub thresh: f32,
    pub label: Vec<u16>,
}

impl Default for ClsProcessorConfig {
    fn default() -> Self {
        ClsProcessorConfig {
            image_shape: [3, 48, 192],
            batch_num: 6,
            thresh: 0.9,
            label: vec![0, 180],
        }
    }
}

#[derive(Debug)]
pub(crate) struct ClsProcessor<'p> {
    config: &'p ClsProcessorConfig,
}

#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClsPostProcessLabel {
    pub label: u16,
    pub score: f32,
}

#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClsProcessorSingleResult {
    pub label: ClsPostProcessLabel,
}

impl Display for ClsProcessorSingleResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClsProcessorSingleResult")
            .field("label", &self.label)
            .finish()
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClsProcessorResult(pub Vec<ClsProcessorSingleResult>);

impl Display for ClsProcessorResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let joined = self
            .0
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "[{}]", joined)
    }
}

impl ProcessorInnerRes for ClsProcessor<'_> {
    type FinalResult = ClsProcessorResult;
}

impl ProcessorInnerIO for ClsProcessor<'_> {
    type PreProcessInput<'ppl> = Array4<f32>;
    type PreProcessInputExtra<'ppl> = ();
    type PreProcessOutput<'ppl> = Array4<f32>;
    type PostProcessInput<'ppl> = Array2<f32>;
    type PostProcessInputExtra<'ppl> = ();
    type PostProcessOutput<'ppl> = Vec<ClsPostProcessLabel>;
}

impl<'a> ClsProcessor<'a> {
    pub fn new(config: &'a ClsProcessorConfig) -> Self {
        ClsProcessor { config }
    }
}

impl ProcessorInner for ClsProcessor<'_> {
    fn preprocess<'a>(
        &self,
        input: Self::PreProcessInput<'a>,
        _: Self::PreProcessInputExtra<'a>,
    ) -> RettoResult<Self::PreProcessOutput<'a>> {
        Ok(input)
    }

    fn postprocess<'a>(
        &self,
        input: Self::PostProcessInput<'a>,
        _: Self::PostProcessInputExtra<'a>,
    ) -> RettoResult<Self::PostProcessOutput<'a>> {
        let pred_idxs = input.map_axis(Axis(1), |row| row.argmax().unwrap());
        let mut out = Vec::with_capacity(pred_idxs.len());
        for (i, &class_idx) in pred_idxs.iter().enumerate() {
            let score = input[(i, class_idx)];
            let label = self.config.label[class_idx];
            out.push(ClsPostProcessLabel { label, score });
        }
        Ok(out)
    }
}

impl<'p> Processor for ClsProcessor<'p> {
    type Config = ClsProcessorConfig;
    type ProcessInput<'pl> = &'pl mut Vec<ImageHelper>;
    fn process<'a, F>(
        &self,
        crop_images: &'a mut Vec<ImageHelper>,
        mut worker_fun: F,
    ) -> RettoResult<Self::FinalResult>
    where
        F: FnMut(Self::PreProcessOutput<'a>) -> RettoResult<Self::PostProcessInput<'a>>,
    {
        let mut final_res: Vec<ClsProcessorSingleResult> = Vec::with_capacity(crop_images.len());
        final_res.resize_with(crop_images.len(), || ClsProcessorSingleResult::default());
        let mut image_index_asc_size: Vec<usize> = (0..crop_images.len()).collect();
        image_index_asc_size.sort_by_key(|&i| Reverse(OrderedFloat(crop_images[i].ori_ratio())));
        let batched = image_index_asc_size
            .chunks(self.config.batch_num)
            .map(|batch| {
                let mats = batch
                    .iter()
                    .map(|&i| {
                        crop_images[i]
                            .resize_norm_image(self.config.image_shape, None)
                            .insert_axis(Axis(0))
                    })
                    .collect::<Vec<_>>();
                let norm_img_batch =
                    concatenate(Axis(0), &mats.iter().map(|a| a.view()).collect::<Vec<_>>())?;
                Ok((batch, norm_img_batch))
            })
            .collect::<RettoResult<Vec<(&[usize], Array4<f32>)>>>()?;
        batched
            .into_iter()
            .try_for_each(|(batch_idxs, norm_img_batch)| {
                let worker_res = worker_fun(norm_img_batch)?;
                let post_processed = self.postprocess(worker_res, ())?;
                batch_idxs
                    .iter()
                    .zip(post_processed)
                    .try_for_each(|(&idx, label)| {
                        if label.label == 180 && label.score >= self.config.thresh {
                            crop_images[idx].rotate_180_in_place()?;
                        }
                        final_res[idx].label = label;
                        Ok::<(), RettoError>(())
                    })
            })?;
        Ok(ClsProcessorResult(final_res))
    }
}
