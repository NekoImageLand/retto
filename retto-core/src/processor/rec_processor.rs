use crate::error::{RettoError, RettoResult};
use crate::image_helper::ImageHelper;
use crate::processor::{Processor, ProcessorInner, ProcessorInnerIO, ProcessorInnerRes};
use ndarray::prelude::*;
use ndarray::{Zip, concatenate};
use ndarray_stats::QuantileExt;
use ordered_float::OrderedFloat;
use std::cmp::{Reverse, max};

#[derive(Debug)]
pub(crate) struct RecCharacter {
    inner: Vec<String>,
    ignored_tokens: Vec<usize>,
}

impl RecCharacter {
    pub fn new(dict_path: &str, ignored_tokens: Vec<usize>) -> RettoResult<Self> {
        let content = std::fs::read_to_string(dict_path)?;
        let mut dict: Vec<String> = content.lines().map(str::trim).map(str::to_owned).collect();
        // insert_special_char
        dict.push(" ".to_string());
        dict.insert(0, "blank".to_string());
        Ok(Self {
            inner: dict,
            ignored_tokens,
        })
    }

    fn decode(
        &self,
        text_index: &Array2<usize>,
        text_prob: &Array2<f32>, // TODO: let it be optional
        wh_ratio_list: &[OrderedFloat<f32>],
        max_wh_ratio: OrderedFloat<f32>,
        remove_duplicate: bool,
        return_word_box: bool, // TODO: implement this
    ) -> Vec<(String, f32)> {
        text_index
            .axis_iter(Axis(0))
            .zip(text_prob.axis_iter(Axis(0)))
            .map(|(token_indices, prob)| {
                let mut selection = token_indices.mapv(|i| i != 0);
                if remove_duplicate {
                    Zip::from(selection.slice_mut(s![1..]))
                        .and(token_indices.slice(s![1..]))
                        .and(token_indices.slice(s![..-1]))
                        .for_each(|sel, &curr, &prev| {
                            *sel = *sel && curr != prev;
                        });
                }
                self.ignored_tokens.iter().for_each(|ignored| {
                    Zip::from(&mut selection)
                        .and(token_indices)
                        .for_each(|sel, &idx| {
                            *sel = *sel && idx != *ignored;
                        });
                });
                debug_assert_eq!(token_indices.len(), prob.len());
                let text_len = token_indices.len();
                let pre_res = selection
                    .iter()
                    .zip(token_indices.iter())
                    .zip(prob.iter())
                    .filter_map(|((&sel, &idx), &p)| match sel {
                        true => Some((self.inner[idx].clone(), p)),
                        false => None,
                    })
                    .fold(
                        (String::with_capacity(text_len), 0.0, 0u32),
                        |(mut acc_str, acc_sum, sum), (seg, p)| {
                            acc_str.push_str(&seg);
                            (acc_str, acc_sum + p, sum + 1)
                        },
                    );
                (pre_res.0, pre_res.1 / pre_res.2 as f32)
            })
            .collect::<Vec<_>>()
    }
}

#[derive(Debug)]
pub struct RecProcessorConfig {
    pub character_path: String,
    pub image_shape: [usize; 3],
    pub batch_num: usize,
}

impl Default for RecProcessorConfig {
    fn default() -> Self {
        RecProcessorConfig {
            character_path: String::from("ppocr_keys_v1.txt"),
            image_shape: [3, 48, 320],
            batch_num: 6,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RecProcessor<'p> {
    character: &'p RecCharacter,
    config: &'p RecProcessorConfig,
}

impl<'a> RecProcessor<'a> {
    pub fn new(config: &'a RecProcessorConfig, character: &'a RecCharacter) -> Self {
        RecProcessor { character, config }
    }
}

#[derive(Debug)]
pub struct RecProcessorSingleResult {
    pub text: String,
    pub score: f32,
    // TODO: word_results
}

#[derive(Debug)]
pub struct RecProcessorResult(pub Vec<RecProcessorSingleResult>);

impl ProcessorInnerRes for RecProcessor<'_> {
    type FinalResult = RecProcessorResult;
}

#[derive(Debug)]
pub(crate) struct PostProcessExtraInput {
    wh_ratios: Vec<OrderedFloat<f32>>,
    max_wh_ratio: OrderedFloat<f32>,
}

#[derive(Debug)]
struct PostProcessOutput;

impl ProcessorInnerIO for RecProcessor<'_> {
    type PreProcessInput<'ppl> = Array4<f32>;
    type PreProcessInputExtra<'ppl> = ();
    type PreProcessOutput<'ppl> = Array4<f32>;
    type PostProcessInput<'ppl> = Array3<f32>;
    type PostProcessInputExtra<'ppl> = PostProcessExtraInput;
    type PostProcessOutput<'ppl> = Vec<(String, f32)>;
}

impl ProcessorInner for RecProcessor<'_> {
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
        PostProcessExtraInput {
            wh_ratios,
            max_wh_ratio,
        }: Self::PostProcessInputExtra<'a>,
    ) -> RettoResult<Self::PostProcessOutput<'a>> {
        let preds_idx = input.map_axis(Axis(2), |lane| lane.argmax().unwrap());
        let preds_prob = input.map_axis(Axis(2), |lane| *lane.max().unwrap());
        Ok(self.character.decode(
            &preds_idx,
            &preds_prob,
            &wh_ratios,
            max_wh_ratio,
            true,
            false,
        ))
    }
}

impl<'p> Processor for RecProcessor<'p> {
    type Config = RecProcessorConfig;
    type ProcessInput<'pl> = &'pl Vec<ImageHelper>;
    fn process<'a, F>(
        &self,
        images: &'a Vec<ImageHelper>,
        mut worker_fun: F,
    ) -> RettoResult<Self::FinalResult>
    where
        F: FnMut(Self::PreProcessOutput<'a>) -> RettoResult<Self::PostProcessInput<'a>>,
    {
        let mut final_res: Vec<Option<RecProcessorSingleResult>> = Vec::with_capacity(images.len());
        final_res.resize_with(images.len(), || None);
        let mut image_index_asc_size: Vec<usize> = (0..images.len()).collect();
        image_index_asc_size.sort_by_key(|&i| Reverse(OrderedFloat(images[i].ori_ratio())));
        let [_, h, w] = self.config.image_shape;
        let mut max_wh_ratio = OrderedFloat(w as f32 / h as f32);
        image_index_asc_size
            .chunks(self.config.batch_num)
            .try_for_each(|batch_idx| {
                let mut wh_ratios = Vec::with_capacity(batch_idx.len());
                batch_idx.iter().for_each(|&i| {
                    let img = &images[i];
                    let (img_h, img_w) = img.size();
                    let wh_ratio = OrderedFloat(img_w as f32 / img_h as f32);
                    wh_ratios.push(wh_ratio);
                    max_wh_ratio = max(max_wh_ratio, wh_ratio);
                });
                let mats = batch_idx
                    .into_iter()
                    .map(|&i| {
                        images[i]
                            .resize_norm_image(
                                self.config.image_shape,
                                Some(max_wh_ratio.into_inner()),
                            )
                            .insert_axis(Axis(0))
                    })
                    .collect::<Vec<_>>();
                let norm_img_batch =
                    concatenate(Axis(0), &mats.iter().map(|a| a.view()).collect::<Vec<_>>())?;
                let worker_res = worker_fun(norm_img_batch)?;
                let post_processed = self.postprocess(worker_res, {
                    PostProcessExtraInput {
                        wh_ratios,
                        max_wh_ratio,
                    }
                })?;
                batch_idx.iter().zip(post_processed).for_each(|(idx, res)| {
                    final_res[*idx] = Some(RecProcessorSingleResult {
                        text: res.0,
                        score: res.1,
                    });
                });
                Ok::<(), RettoError>(())
            })?;
        Ok(RecProcessorResult(
            final_res.into_iter().map(|x| x.unwrap()).collect(),
        ))
    }
}
