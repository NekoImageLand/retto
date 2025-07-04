#[derive(Debug, thiserror::Error)]
pub enum RettoError {
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error(transparent)]
    ImageError(#[from] image::ImageError),
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[cfg(feature = "backend-ort")]
    #[error(transparent)]
    OrtError(#[from] ort::error::Error),
    #[error(transparent)]
    Utf8Error(#[from] std::string::FromUtf8Error),
    #[cfg(feature = "hf-hub")]
    #[error(transparent)]
    HfHubError(#[from] hf_hub::api::sync::ApiError),
    #[error("Model not found: {0}")]
    ModelNotFoundError(String),
}

pub type RettoResult<T> = Result<T, RettoError>;
