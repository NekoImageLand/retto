use crate::error::RettoResult;
use crate::processor::{Processor, ProcessorInner, ProcessorInnerIO, ProcessorInnerRes};
use ndarray::prelude::*;

#[derive(Debug, Default)]
pub struct RecProcessorConfig;

#[derive(Debug)]
pub(crate) struct RecProcessor<'p> {
    config: &'p RecProcessorConfig,
}

#[derive(Debug, Default)]
pub struct RecProcessorResult;

impl ProcessorInnerRes for RecProcessor<'_> {
    type FinalResult = RecProcessorResult;
}

impl ProcessorInnerIO for RecProcessor<'_> {
    type PreProcessInput<'ppl> = ArrayView3<'ppl, u8>;
    type PreProcessOutput<'ppl> = Array4<f32>;
    type PostProcessInput<'ppl> = Array4<f32>;
    type PostProcessOutput<'ppl> = RecProcessorResult;
}

impl ProcessorInner for RecProcessor<'_> {
    fn preprocess<'a>(
        &self,
        input: Self::PreProcessInput<'a>,
    ) -> RettoResult<Self::PreProcessOutput<'a>> {
        todo!()
    }

    fn postprocess<'a>(
        &self,
        input: Self::PostProcessInput<'a>,
    ) -> RettoResult<Self::PostProcessOutput<'a>> {
        todo!()
    }
}

impl<'p> Processor for RecProcessor<'p> {
    type Config = RecProcessorConfig;
    type ProcessInput<'pl> = Self::PreProcessInput<'pl>;
    fn process<'a, F>(
        &self,
        input: Self::PreProcessInput<'a>,
        mut worker_fun: F,
    ) -> RettoResult<Self::FinalResult>
    where
        F: FnMut(Self::PreProcessOutput<'a>) -> RettoResult<Self::PostProcessInput<'a>>,
    {
        let pre_processed = self.preprocess(input)?;
        let worker_res = worker_fun(pre_processed)?;
        let post_processed = self.postprocess(worker_res)?;
        Ok(post_processed)
    }
}
