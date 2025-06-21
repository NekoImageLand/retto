#[derive(Debug, thiserror::Error)]
pub enum RettoError {
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error(transparent)]
    ImageError(#[from] image::ImageError),
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[error(transparent)]
    OrtError(#[from] ort::error::Error),
}

pub type RettoResult<T> = Result<T, RettoError>;
