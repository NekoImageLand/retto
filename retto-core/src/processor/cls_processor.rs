use crate::error::RettoError;
use crate::processor::{Processor, ProcessorInner, ProcessorInnerRes};
use ndarray::prelude::*;

#[derive(Debug, Default)]
pub struct ClsProcessorConfig;

#[derive(Debug)]
pub struct ClsProcessor<'a> {
    config: &'a ClsProcessorConfig,
}

#[derive(Debug, Default)]
pub struct ClsProcessorResult;

impl ProcessorInnerRes for ClsProcessor<'_> {
    type FinalResult = ClsProcessorResult;
}

impl ProcessorInner for ClsProcessor<'_> {
    fn preprocess(&self, input: &Array3<u8>) -> Result<Array4<f32>, RettoError> {
        todo!()
    }

    fn postprocess(&self, input: &Array4<f32>) -> Result<Self::FinalResult, RettoError> {
        todo!()
    }
}

impl<'a> Processor<'a> for ClsProcessor<'a> {
    type Config = ClsProcessorConfig;
}
