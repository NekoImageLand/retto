use clap::Parser;
use retto_core::prelude::*;
use std::fs;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "ratio-cli", version)]
pub struct Cli {
    #[arg(short, long, default_value = "ch_PP-OCRv4_det_infer.onnx")]
    det_model_path: String,
    #[arg(short, long, default_value = "ch_ppocr_mobile_v2.0_cls_infer.onnx")]
    cls_model_path: String,
    #[arg(short, long, default_value = "ch_PP-OCRv4_rec_infer.onnx")]
    rec_model_path: String,
    #[arg(short, long)]
    image: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("info"));
    tracing_subscriber::registry().with(stdout).init();
    let cfg: RettoSessionConfig<RettoOrtWorker> = RettoSessionConfig {
        worker_config: RettoOrtWorkerConfig {
            device: RettoOrtWorkerDeviceConfig::Cuda(0),
            det_model_path: cli.det_model_path,
            rec_model_path: cli.rec_model_path,
            cls_model_path: cli.cls_model_path,
        },
        ..Default::default()
    };
    let mut session = RettoSession::new(cfg)?;
    let img = fs::read(cli.image)?;
    for _ in 0..1 {
        session.run(&img)?;
    }
    Ok(())
}
