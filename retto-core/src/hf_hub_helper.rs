use crate::error::{RettoError, RettoResult};
use hf_hub::api::sync::{Api, ApiBuilder};
use std::path::PathBuf;

pub struct HfHubHelper(Api);

impl HfHubHelper {
    pub fn new() -> Self {
        let api = ApiBuilder::new().with_progress(true).build().unwrap();
        Self(api)
    }

    pub fn get_model_file(&self, repo: &str, path: &str) -> RettoResult<PathBuf> {
        self.0
            .model(repo.to_string())
            .get(path)
            .map_err(|e| RettoError::HfHubError(e))
    }
}
