use crate::error::RettoResult;
use crate::image_helper::ImageHelper;
use crate::processor::prelude::*;
use crate::serde::*;
use crate::worker::RettoWorker;
use std::sync::mpsc;

#[derive(Debug)]
pub struct RettoSession<W: RettoWorker> {
    worker: W,
    rec_character: RecCharacter,
    config: RettoSessionConfig<W>,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RettoSessionConfig<W: RettoWorker> {
    pub worker_config: W::RettoWorkerConfig,
    pub max_side_len: usize,
    pub min_side_len: usize,
    pub det_processor_config: DetProcessorConfig,
    pub cls_processor_config: ClsProcessorConfig,
    pub rec_processor_config: RecProcessorConfig,
}

impl<W> Default for RettoSessionConfig<W>
where
    W: RettoWorker,
{
    fn default() -> Self {
        RettoSessionConfig {
            worker_config: <_>::default(),
            max_side_len: 2000,
            min_side_len: 30,
            det_processor_config: DetProcessorConfig::default(),
            cls_processor_config: ClsProcessorConfig::default(),
            rec_processor_config: RecProcessorConfig::default(),
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RettoWorkerResult {
    pub det_result: DetProcessorResult,
    pub cls_result: ClsProcessorResult,
    pub rec_result: RecProcessorResult,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RettoWorkerStageResult {
    Det(DetProcessorResult),
    Cls(ClsProcessorResult),
    Rec(RecProcessorResult),
}

impl<W> RettoSession<W>
where
    W: RettoWorker,
{
    pub fn new(cfg: RettoSessionConfig<W>) -> RettoResult<Self> {
        // load dict
        let worker = W::new(cfg.worker_config.clone())?; // TODO:
        let rec_character =
            RecCharacter::new(cfg.rec_processor_config.character_source.clone(), vec![0])?;
        worker.init()?;
        Ok(RettoSession {
            worker,
            rec_character,
            config: cfg,
        })
    }

    fn process_pipeline<F>(&mut self, input: impl AsRef<[u8]>, mut callback: F) -> RettoResult<()>
    where
        F: FnMut(RettoWorkerStageResult),
    {
        let mut image = ImageHelper::new_from_raw_img_flow(input)?; // TODO: args
        let (ori_h, ori_w) = image.size();
        let (ratio_h, ratio_w) =
            image.resize_both(self.config.max_side_len, self.config.min_side_len)?;
        let (after_h, after_w) = image.size();
        let arr = image.array_view()?; // cheap
        let det = DetProcessor::new(&self.config.det_processor_config, after_h, after_w)?;
        let mut det_res = det.process(arr, |i| self.worker.det(i))?;
        // As you can see, crop_images is mutable, but currently only limited to changing incorrect cls angles
        let mut crop_images = det_res
            .0
            .iter()
            .map(|res| ImageHelper::new_from_rgb_image(image.get_crop_img(&res.boxes)))
            .collect::<Vec<_>>();
        // So we have to resample the point boxes (to ensure consistency of coordinates)...
        for res in &mut det_res.0 {
            res.boxes
                .scale_and_clip(after_w as f64, after_h as f64, ori_w as f64, ori_h as f64);
        }
        callback(RettoWorkerStageResult::Det(det_res));
        let cls = ClsProcessor::new(&self.config.cls_processor_config);
        let cls_res = cls.process(&mut crop_images, |i| self.worker.cls(i))?;
        callback(RettoWorkerStageResult::Cls(cls_res));
        let rec = RecProcessor::new(&self.config.rec_processor_config, &self.rec_character);
        let rec_res = rec.process(&crop_images, |i| self.worker.rec(i))?;
        callback(RettoWorkerStageResult::Rec(rec_res));
        Ok(())
    }

    pub fn run(&mut self, input: impl AsRef<[u8]>) -> RettoResult<RettoWorkerResult> {
        let mut det_opt = None;
        let mut cls_opt = None;
        let mut rec_opt = None;
        self.process_pipeline(input, |stage| match stage {
            RettoWorkerStageResult::Det(r) => {
                tracing::debug!("Det result: {:?}", r);
                det_opt = Some(r)
            }
            RettoWorkerStageResult::Cls(r) => {
                tracing::debug!("Cls result: {:?}", r);
                cls_opt = Some(r)
            }
            RettoWorkerStageResult::Rec(r) => {
                tracing::debug!("Rec result: {:?}", r);
                rec_opt = Some(r)
            }
        })?;
        Ok(RettoWorkerResult {
            det_result: det_opt.unwrap(),
            cls_result: cls_opt.unwrap(),
            rec_result: rec_opt.unwrap(),
        })
    }

    pub fn run_stream(
        &mut self,
        input: impl AsRef<[u8]>,
        sender: mpsc::Sender<RettoWorkerStageResult>,
    ) -> RettoResult<()> {
        self.process_pipeline(input, |stage| {
            if let Err(e) = sender.send(stage) {
                tracing::error!("Error sending request {:?}", e);
            }
        })
    }
}

// allow us auto download models
#[cfg(all(test, feature = "hf-hub", feature = "backend-ort"))]
mod tests {
    use crate::points::Point;
    use crate::prelude::*;
    use ab_glyph::{FontVec, PxScale};
    use anyhow::Result as AnyResult;
    use image::{ImageFormat, Rgb, RgbImage};
    use imageproc::definitions::HasWhite;
    use imageproc::drawing::draw_text_mut;
    use imageproc::geometric_transformations::{Interpolation, rotate_about_center};
    use num_traits::abs;
    use once_cell::sync::Lazy;
    use ordered_float::OrderedFloat;
    use rstest::*;
    use std::io::Cursor;

    static GLOBAL_FONT: Lazy<FontVec> = Lazy::new(|| {
        let fonts = reqwest::blocking::get(
            "https://github.com/adobe-fonts/source-han-sans/raw/release/Variable/OTF/SourceHanSansSC-VF.otf",
        ).expect("Failed to download font");
        let fonts_bin = fonts.bytes().expect("Failed to read font from bytes");
        // save
        FontVec::try_from_vec(fonts_bin.to_vec()).expect("Failed to load font")
    });

    #[fixture]
    fn session() -> RettoSession<RettoOrtWorker> {
        let cfg = RettoSessionConfig {
            worker_config: RettoOrtWorkerConfig::default(),
            ..Default::default()
        };
        RettoSession::new(cfg).expect("Failed to create RettoSession")
    }

    fn points_range(lhs: &Point<OrderedFloat<f32>>, x: f32, y: f32) -> f32 {
        let rhs = Point::new(OrderedFloat(x), OrderedFloat(y));
        abs(lhs.range(&rhs))
    }

    fn draw_text(
        font: &FontVec,
        text: &str,
        px: PxScale,
        w: u32,
        h: u32,
        x: i32,
        y: i32,
    ) -> RgbImage {
        let mut image = RgbImage::new(w, h);
        draw_text_mut(&mut image, Rgb::white(), x, y, px, &font, text);
        image.save("test_small_image.png").unwrap();
        image
    }

    fn rotate_text(image: &RgbImage, angle: f32) -> RgbImage {
        let angle = angle.to_radians();
        rotate_about_center(&image, angle, Interpolation::Bilinear, Rgb([0, 0, 0]))
    }

    #[rstest]
    fn test_small_image(mut session: RettoSession<RettoOrtWorker>) -> AnyResult<()> {
        let text = "玩原神玩的";
        let (w, h) = (200.0, 50.0);
        let image = draw_text(
            &GLOBAL_FONT,
            text,
            PxScale::from(20.0),
            w as u32,
            h as u32,
            0,
            0,
        );
        let image = rotate_text(&image, 180.0);
        let mut buf = Vec::new();
        image.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)?;
        let res = session.run(buf)?;
        println!("{:?}", res);
        let point_box = &res.det_result.0[0].boxes;
        assert!(points_range(point_box.br(), w, h) < 10f32);
        assert_eq!(res.cls_result.0[0].label.label, 180);
        assert_eq!(res.rec_result.0[0].text, text);
        Ok(())
    }

    #[rstest]
    // https://github.com/NekoImageLand/retto/commit/7fc4127b
    fn test_large_image(mut session: RettoSession<RettoOrtWorker>) -> AnyResult<()> {
        let text = "玩原神玩的";
        let (w, h) = (7680.0, 4320.0);
        let image = draw_text(
            &GLOBAL_FONT,
            text,
            PxScale::from(300.0),
            w as u32,
            h as u32,
            0,
            0,
        );
        let image = rotate_text(&image, 180.0);
        let mut buf = Vec::new();
        image.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)?;
        let res = session.run(buf)?;
        println!("{:?}", res);
        let point_box = &res.det_result.0[0].boxes;
        assert!(points_range(point_box.br(), w, h) < 100f32);
        assert_eq!(res.cls_result.0[0].label.label, 180);
        assert_eq!(res.rec_result.0[0].text, text);
        Ok(())
    }

    #[test]
    #[allow(clippy::all)]
    #[should_panic]
    fn ab4edca3f85e0c13d6c98009b775a6b3() {
        panic!(
            "Oh no... I look useless, don't I? qwq \n\
            そうは言っても申し訳ないので、一緒に Rust Playground に行きませんか?"
        );
    }
}
