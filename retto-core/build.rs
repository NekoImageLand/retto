use hf_hub::api::sync::ApiBuilder;
use std::env;
use std::path::Path;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

const MODELS: [&str; 4] = [
    "ch_PP-OCRv4_det_infer.onnx",
    "ch_PP-OCRv4_rec_infer.onnx",
    "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "ppocr_keys_v1.txt",
];

fn main() {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("debug"));
    tracing_subscriber::registry().with(stdout).init();
    let download_enabled = env::var("CARGO_FEATURE_DOWNLOAD_MODELS").is_ok();
    if !download_enabled {
        tracing::warn!("Model downloading is disabled");
    }
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let model_dir = Path::new(&manifest_dir).join("models");
    if !model_dir.exists() {
        std::fs::create_dir_all(&model_dir).expect("Failed to create model directory");
    }
    let need_download_models = MODELS
        .iter()
        .cloned()
        .filter(|&model| !model_dir.join(model).exists())
        .collect::<Vec<_>>();
    if need_download_models.len() > 0 {
        tracing::info!("Downloading models...");
    }
    let api = ApiBuilder::new().with_progress(true).build().unwrap();
    let downloaded_path = need_download_models
        .iter()
        .map(|&model| {
            tracing::info!("Downloading model: {}", model);
            api.model("pk5ls20/PaddleModel".to_string())
                .get(&format!("retto/onnx/{}", model))
                .expect("Failed to download model")
        })
        .collect::<Vec<_>>();
    for src in &downloaded_path {
        let file_name = src.file_name().unwrap();
        let link_path = model_dir.join(file_name);
        if link_path.exists() {
            tracing::warn!("Link {} already exists, skip", file_name.to_string_lossy());
            continue;
        }
        tracing::info!(
            "Creating symlink: {} → {}",
            link_path.display(),
            src.display()
        );
        #[cfg(unix)]
        std::os::unix::fs::symlink(src, &link_path)
            .unwrap_or_else(|e| panic!("failed to create unix symlink: {}", e));
        #[cfg(windows)]
        std::os::windows::fs::symlink_file(src, &link_path)
            .unwrap_or_else(|e| panic!("failed to create windows symlink: {}", e));
    }
}
