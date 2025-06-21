pub mod cls_processor;
pub mod det_processor;
pub mod rec_processor;

use crate::error::RettoResult;
use ndarray::prelude::*;

pub trait ProcessorInnerRes {
    type FinalResult;
}

trait ProcessorInner: ProcessorInnerRes {
    fn preprocess(&self, input: &Array3<u8>) -> RettoResult<Array4<f32>>;
    fn postprocess(&self, input: &Array4<f32>) -> RettoResult<Self::FinalResult>;
}

pub trait Processor<'a>: ProcessorInner {
    type Config;
    fn process<F>(&self, input: &Array3<u8>, mut worker_fun: F) -> RettoResult<Self::FinalResult>
    where
        F: FnMut(Array4<f32>) -> RettoResult<Array4<f32>>,
    {
        let pre_processed = self.preprocess(input)?;
        let worker_res = worker_fun(pre_processed)?;
        let post_processed = self.postprocess(&worker_res)?;
        Ok(post_processed)
    }
}

pub mod prelude {
    pub use super::Processor;
    pub use super::cls_processor::*;
    pub use super::det_processor::*;
    pub use super::rec_processor::*;
}
