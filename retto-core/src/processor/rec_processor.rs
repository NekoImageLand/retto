use crate::error::RettoError;
use crate::processor::{Processor, ProcessorInner, ProcessorInnerRes};
use ndarray::prelude::*;

#[derive(Debug, Default)]
pub struct RecProcessorConfig;

#[derive(Debug)]
pub struct RecProcessor<'a> {
    config: &'a RecProcessorConfig,
}

#[derive(Debug, Default)]
pub struct RecProcessorResult;

impl ProcessorInnerRes for RecProcessor<'_> {
    type FinalResult = RecProcessorResult;
}

impl ProcessorInner for RecProcessor<'_> {
    fn preprocess(&self, input: &Array3<u8>) -> Result<Array4<f32>, RettoError> {
        todo!()
    }

    fn postprocess(&self, input: &Array4<f32>) -> Result<Self::FinalResult, RettoError> {
        todo!()
    }
}

impl<'a> Processor<'a> for RecProcessor<'a> {
    type Config = RecProcessorConfig;
}
