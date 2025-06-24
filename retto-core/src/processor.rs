pub mod cls_processor;
pub mod det_processor;
pub mod rec_processor;

use crate::error::RettoResult;

pub trait ProcessorInnerRes {
    type FinalResult;
}

pub(crate) trait ProcessorInnerIO: ProcessorInnerRes {
    type PreProcessInput<'ppl>;
    type PreProcessInputExtra<'ppl>;
    type PreProcessOutput<'ppl>;
    type PostProcessInput<'ppl>;
    type PostProcessInputExtra<'ppl>;
    type PostProcessOutput<'ppl>;
}

trait ProcessorInner: ProcessorInnerIO {
    fn preprocess<'a>(
        &self,
        input: Self::PreProcessInput<'a>,
        extra: Self::PreProcessInputExtra<'a>,
    ) -> RettoResult<Self::PreProcessOutput<'a>>;
    fn postprocess<'a>(
        &self,
        input: Self::PostProcessInput<'a>,
        extra: Self::PostProcessInputExtra<'a>,
    ) -> RettoResult<Self::PostProcessOutput<'a>>;
}

pub(crate) trait Processor: ProcessorInner {
    type Config;
    type ProcessInput<'pl>;
    fn process<'a, F>(
        &self,
        input: Self::ProcessInput<'a>,
        worker_fun: F,
    ) -> RettoResult<Self::FinalResult>
    where
        F: FnMut(Self::PreProcessOutput<'a>) -> RettoResult<Self::PostProcessInput<'a>>;
}

pub mod prelude {
    pub(crate) use super::Processor;
    pub use super::cls_processor::*;
    pub use super::det_processor::*;
    pub use super::rec_processor::*;
}
