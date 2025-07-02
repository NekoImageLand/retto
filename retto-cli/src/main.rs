use clap::{Parser, ValueEnum};
use retto_core::prelude::*;
use std::fs;
use std::time::Instant;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;
use walkdir::WalkDir;

#[derive(ValueEnum, Clone, Debug)]
pub enum DeviceKind {
    Cpu,
    #[cfg(feature = "backend-ort-cuda")]
    Cuda,
    #[cfg(feature = "backend-ort-directml")]
    DirectMl,
}

#[derive(Parser, Debug)]
#[command(name = "ratio-cli", version)]
pub struct Cli {
    #[arg(long, default_value = "ch_PP-OCRv4_det_infer.onnx")]
    det_model_path: String,
    #[arg(long, default_value = "ch_ppocr_mobile_v2.0_cls_infer.onnx")]
    cls_model_path: String,
    #[arg(long, default_value = "ch_PP-OCRv4_rec_infer.onnx")]
    rec_model_path: String,
    #[arg(long, default_value = "ppocr_keys_v1.txt")]
    rec_keys_path: String,
    #[arg(short, long)]
    images: String,
    #[arg(long, value_enum, default_value_t = DeviceKind::Cpu)]
    device: DeviceKind,
    #[cfg(any(feature = "backend-ort-cuda", feature = "backend-ort-directml"))]
    #[arg(long, default_value_t = 0)]
    device_id: i32,
    #[cfg(feature = "hf-hub")]
    #[arg(long, default_value = "true")]
    use_hf_hub: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let stdout = tracing_subscriber::fmt::layer()
        .with_filter(EnvFilter::new("ort=debug,retto_core=debug,retto_cli=debug"));
    tracing_subscriber::registry().with(stdout).init();
    let device = match cli.device {
        DeviceKind::Cpu => RettoOrtWorkerDevice::CPU,
        #[cfg(feature = "backend-ort-cuda")]
        DeviceKind::Cuda => RettoOrtWorkerDevice::Cuda(cli.device_id),
        #[cfg(feature = "backend-ort-directml")]
        DeviceKind::DirectMl => RettoOrtWorkerDevice::DirectML(cli.device_id),
    };
    #[allow(unused_mut)] // false positive due to dynamic target cfg
    let mut models = RettoOrtWorkerModelProvider(RettoWorkerModelProvider {
        det: RettoWorkerModelSource::Path(cli.det_model_path),
        rec: RettoWorkerModelSource::Path(cli.rec_model_path),
        cls: RettoWorkerModelSource::Path(cli.cls_model_path),
    });
    #[cfg(feature = "hf-hub")]
    match cli.use_hf_hub {
        true => {
            tracing::info!("Using Hugging Face Hub for models");
            models = RettoOrtWorkerModelProvider::from_hf_hub_v4_default();
        }
        false => {}
    }
    let cfg: RettoSessionConfig<RettoOrtWorker> = RettoSessionConfig {
        worker_config: RettoOrtWorkerConfig { device, models },
        ..Default::default()
    };
    let mut session = RettoSession::new(cfg)?;
    let walkers = WalkDir::new(&cli.images);
    let files = walkers
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .collect::<Vec<_>>();
    tracing::info!("Found {} files, processing...", files.len());
    let start = Instant::now();
    let results = files
        .iter()
        .map(|e| {
            let img = fs::read(e.path()).expect("Failed to read image file");
            session.run(img)
        })
        .collect::<Vec<_>>();
    let duration = start.elapsed();
    // TODO: Now we can handle the results
    tracing::info!(
        "Successfully processed {} images, avg time: {:.2?}",
        results.len(),
        duration / results.len() as u32
    );
    Ok(())
}
