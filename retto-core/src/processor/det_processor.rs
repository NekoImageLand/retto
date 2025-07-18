use crate::error::RettoResult;
use crate::image_helper::ImageHelper;
use crate::points::{Point, PointBox};
use crate::processor::{Processor, ProcessorInner, ProcessorInnerIO, ProcessorInnerRes};
use crate::serde::*;
use geo::prelude::*;
use geo_clipper::{Clipper, EndType, JoinType};
use geo_types::{Coord, LineString, Polygon};
use image::{GrayImage, Luma};
use imageproc::contours::find_contours;
use imageproc::drawing::draw_polygon_mut;
use imageproc::geometry::min_area_rect;
use imageproc::morphology::{Mask, grayscale_dilate};
use imageproc::point::Point as ImagePoint;
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Num, NumCast, Signed};
use ordered_float::OrderedFloat;
use std::fmt::Debug;

#[derive(Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ScoreMode {
    /// Calculate the average score based on all pixels within the original polygon.
    /// This method is relatively slow but more accurate.
    Slow,
    #[default]
    /// Calculate the average score for all pixels within the bounding rectangle of the polygon
    Fast,
}

#[derive(Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LimitType {
    #[default]
    /// Ensure that the shortest side of the image is not less than `limit_side_len`
    Min,
    /// Ensure that the longest side of the image does not exceed `limit_side_len`
    Max,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// DB algorithm
pub struct DetProcessorConfig {
    // Preprocess
    /// Limit side length of input image.
    pub limit_side_len: usize,
    /// Input image side length restriction type.
    pub limit_type: LimitType,
    pub mean: Array1<f32>,
    pub std: Array1<f32>,
    pub scale: f32,
    // PostProcess
    /// In the probability map output by DB, only pixels with scores greater than the threshold
    /// are considered to be text pixels.
    pub threch: f32,
    /// If the average score of all pixels within the border of the measurement result is greater
    /// than the threshold value, the result is considered to be a text area.
    pub box_thresh: f32,
    /// Maximum number of text boxes output.
    pub max_candidates: usize,
    /// Expansion coefficient of the
    /// [Vatti clipping algorithm](https://en.wikipedia.org/wiki/Vatti_clipping_algorithm),
    /// using this method to expand the text area.
    pub unclip_ratio: f32,
    /// Whether to expand the segmentation results.
    pub use_dilation: bool,
    /// DB detection result scoring method.
    pub score_mode: ScoreMode,
    /// Minimum side length threshold for text boxes.
    pub min_mini_box_size: usize,
    pub dilation_kernel: Option<Array2<usize>>,
}

impl Default for DetProcessorConfig {
    fn default() -> Self {
        DetProcessorConfig {
            limit_side_len: 736,
            limit_type: LimitType::default(),
            mean: Array1::from_elem(3, 0.5),
            std: Array1::from_elem(3, 0.5),
            scale: 1f32 / 255.0,
            threch: 0.3,
            box_thresh: 0.5,
            max_candidates: 1000,
            unclip_ratio: 1.6,
            use_dilation: true,
            score_mode: ScoreMode::default(),
            min_mini_box_size: 3,
            dilation_kernel: Some(Array2::from_elem((2, 2), 1)),
        }
    }
}

#[derive(Debug)]
pub(crate) struct DetProcessor<'p> {
    config: &'p DetProcessorConfig,
    dilation_kernel: Option<Mask>,
    // Image size after initial resize
    ori_h: usize,
    ori_w: usize,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DetProcessorInnerResult {
    pub boxes: PointBox<OrderedFloat<f32>>,
    pub score: f32,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DetProcessorResult(pub Vec<DetProcessorInnerResult>);

impl ProcessorInnerRes for DetProcessor<'_> {
    type FinalResult = DetProcessorResult;
}

impl ProcessorInnerIO for DetProcessor<'_> {
    type PreProcessInput<'ppl> = ArrayView3<'ppl, u8>;
    type PreProcessInputExtra<'ppl> = ();
    type PreProcessOutput<'ppl> = Array4<f32>;
    type PostProcessInput<'ppl> = Array4<f32>;
    type PostProcessInputExtra<'pel> = ();
    type PostProcessOutput<'ppl> = DetProcessorResult;
}

fn trans_dilation_kernel(kernel: &Array2<usize>) -> Mask {
    let (h, w) = kernel.dim();
    let mut gray_kernel = GrayImage::new(w as u32, h as u32);
    for ((y, x), &v) in kernel.indexed_iter() {
        let pix = if v != 0 { 255u8 } else { 0u8 };
        gray_kernel.put_pixel(x as u32, y as u32, Luma([pix]));
    }
    let anchor_x = (w / 2) as u8;
    let anchor_y = (h / 2) as u8;
    Mask::from_image(&gray_kernel, anchor_x, anchor_y)
}

/// PreProcess
impl<'a> DetProcessor<'a> {
    pub fn new(config: &'a DetProcessorConfig, ori_h: usize, ori_w: usize) -> RettoResult<Self> {
        Ok(DetProcessor {
            config,
            dilation_kernel: config.dilation_kernel.as_ref().map(trans_dilation_kernel),
            ori_h,
            ori_w,
        })
    }

    fn normalize(&self, input: &Array3<u8>) -> RettoResult<Array3<f32>> {
        let normalized =
            (input.mapv(|x| x as f32) * self.config.scale - &self.config.mean) / &self.config.std;
        Ok(Array3::from(normalized))
    }

    fn permute(&self, input: Array3<f32>) -> RettoResult<Array3<f32>> {
        let permuted = input.permuted_axes((2, 0, 1));
        Ok(permuted)
    }
}

/// PostProcess
impl<'a> DetProcessor<'a> {
    #[inline]
    fn euclid_dist<T>(&self, a: &Point<T>, b: &Point<T>) -> f32
    where
        T: AsPrimitive<f32> + Num + NumCast + Signed + Copy + Ord + Debug,
    {
        let (ax, ay) = (a.x.as_(), a.y.as_());
        let (bx, by) = (b.x.as_(), b.y.as_());
        let (dx, dy) = (ax - bx, ay - by);
        (dx * dx + dy * dy).sqrt()
    }

    fn get_mini_boxes<T>(&self, contour_points: &[ImagePoint<T>]) -> (PointBox<T>, f32)
    where
        T: AsPrimitive<f32> + Num + NumCast + Signed + Copy + Ord + Debug,
    {
        let rect = min_area_rect(contour_points);
        let point_box = PointBox::new_from_clockwise(rect.map(Point::from));
        let side1 = self.euclid_dist(point_box.tl(), point_box.tr());
        let side2 = self.euclid_dist(point_box.bl(), point_box.br());
        let sside = side1.min(side2);
        (point_box, sside)
    }

    fn box_score_fast(&self, bitmap: &ArrayView2<f32>, point_box: &PointBox<i32>) -> f32 {
        let (x_min, x_max, y_min, y_max) = point_box.points().iter().fold(
            (i32::MAX, i32::MIN, i32::MAX, i32::MIN),
            |(xmin, xmax, ymin, ymax), p| {
                (xmin.min(p.x), xmax.max(p.x), ymin.min(p.y), ymax.max(p.y))
            },
        );
        let (h, w) = (bitmap.shape()[0] as i32, bitmap.shape()[1] as i32);
        let x_min = x_min.clamp(0, w - 1) as usize;
        let x_max = x_max.clamp(0, w - 1) as usize;
        let y_min = y_min.clamp(0, h - 1) as usize;
        let y_max = y_max.clamp(0, h - 1) as usize;

        let bw = (x_max - x_min + 1) as u32;
        let bh = (y_max - y_min + 1) as u32;
        let poly: Vec<ImagePoint<i32>> = point_box
            .points()
            .iter()
            .map(|p| ImagePoint::new(p.x - x_min as i32, p.y - y_min as i32))
            .collect();
        let mut mask_img = GrayImage::new(bw, bh);
        draw_polygon_mut(&mut mask_img, &poly, Luma([1u8]));
        let mask = mask_img.into_raw();
        let region = bitmap.slice(s![y_min..=y_max, x_min..=x_max]);
        let (sum, count) =
            region
                .iter()
                .zip(mask.iter())
                .fold((0f32, 0usize), |(acc_sum, acc_cnt), (&v, &m)| {
                    let m = m as usize;
                    (acc_sum + v * m as f32, acc_cnt + m)
                });
        if count > 0 { sum / count as f32 } else { 0.0 }
    }

    fn unclip<T>(&self, point_box: &PointBox<T>) -> Vec<ImagePoint<OrderedFloat<f32>>>
    where
        T: AsPrimitive<f32> + Num + NumCast + Signed + Copy + Ord + Debug,
    {
        let exterior_coords: Vec<Coord<f32>> = point_box
            .points()
            .iter()
            .map(|p| Coord::from((p.x.as_(), p.y.as_())))
            .collect();
        let polygon = Polygon::new(LineString(exterior_coords), vec![]);
        let area = polygon.unsigned_area();
        let perimeter = Euclidean.length(polygon.exterior())
            + polygon
                .interiors()
                .iter()
                .map(|ring| Euclidean.length(ring))
                .sum::<f32>();
        let distance = area * (self.config.unclip_ratio) / perimeter;
        let offset_polys =
            polygon.offset(distance, JoinType::Round(0.5), EndType::ClosedPolygon, 1.0);
        offset_polys
            .into_iter()
            .flat_map(|poly| {
                poly.exterior()
                    .points()
                    .map(|pt| ImagePoint::new(OrderedFloat(pt.x()), OrderedFloat(pt.y())))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

impl ProcessorInner for DetProcessor<'_> {
    fn preprocess<'a>(
        &self,
        input: Self::PreProcessInput<'a>,
        _: Self::PreProcessInputExtra<'a>,
    ) -> RettoResult<Self::PreProcessOutput<'a>> {
        let h = input.shape().first().unwrap();
        let w = input.shape().get(1).unwrap();
        let mut rs_helper = ImageHelper::new_from_rgb_image_flow(
            input.as_standard_layout().as_slice().unwrap(),
            *h,
            *w,
        );
        rs_helper.resize_either(&self.config.limit_type, self.config.limit_side_len)?;
        let input = rs_helper.rgb2bgr()?;
        let input = self.normalize(&input)?;
        let input = self.permute(input)?;
        let input = input.insert_axis(Axis(0));
        Ok(input)
    }

    // TODO: Check for precision alignment issues with the Python implementation,
    // TODO: especially before and after find_contours (since find_contours never uses floating point
    // TODO: numbers internally and is not fully consistent with the opencv implementation).
    fn postprocess<'a>(
        &self,
        input: Self::PostProcessInput<'a>,
        _: Self::PostProcessInputExtra<'a>,
    ) -> RettoResult<Self::PostProcessOutput<'a>> {
        let pred = input.slice(s![0, 0, .., ..]);
        let (h, w) = { (pred.shape()[0] as u32, pred.shape()[1] as u32) };
        let mut mask = GrayImage::from_fn(w, h, |x, y| {
            let v = input[[0, 0, y as usize, x as usize]];
            Luma([if v > self.config.threch { 255 } else { 0 }])
        });
        if let Some(ref k) = self.dilation_kernel {
            mask = grayscale_dilate(&mask, k);
        }
        let mut boxes_res: Vec<_> = find_contours::<i32>(&mask)
            .iter()
            .filter_map(|contour| {
                // #region boxes_from_bitmap
                let (points, sside) = self.get_mini_boxes(&contour.points);
                if sside < self.config.min_mini_box_size as f32 {
                    return None;
                }
                let mean_score = self.box_score_fast(&pred, &points);
                if mean_score < self.config.box_thresh {
                    return None;
                }
                let boxes = self.unclip(&points);
                let (mut point_box, sside) = self.get_mini_boxes(&boxes[..]);
                // TODO: Based on the accuracy issue mentioned earlier, should +2 need fine-tuned?
                if sside < (self.config.min_mini_box_size + 2) as f32 {
                    return None;
                }
                point_box.scale_and_clip(w as f64, h as f64, self.ori_w as f64, self.ori_h as f64);
                // #region filter_det_res
                let (pb_h, pb_w) = (point_box.height_tlc(), point_box.width_tlc());
                if pb_h <= OrderedFloat(3f32) || pb_w <= OrderedFloat(3f32) {
                    return None;
                }
                Some(DetProcessorInnerResult {
                    boxes: point_box,
                    score: mean_score,
                })
            })
            .collect();
        // #region sorted_boxes
        boxes_res.sort_by(|r1, r2| {
            let (c1, c2) = (r1.boxes.center_point(), r2.boxes.center_point());
            let (y1, y2) = (c1.y.into_inner(), c2.y.into_inner());
            if (y1 - y2).abs() < 10f32 {
                let (x1, x2) = (c1.x.into_inner(), c2.x.into_inner());
                x1.partial_cmp(&x2).unwrap()
            } else {
                y1.partial_cmp(&y2).unwrap()
            }
        });
        Ok(DetProcessorResult(boxes_res))
    }
}

impl<'p> Processor for DetProcessor<'p> {
    type Config = DetProcessorConfig;
    type ProcessInput<'pl> = Self::PreProcessInput<'pl>;
    fn process<'a, F>(
        &self,
        input: Self::PreProcessInput<'a>,
        mut worker_fun: F,
    ) -> RettoResult<Self::FinalResult>
    where
        F: FnMut(Self::PreProcessOutput<'a>) -> RettoResult<Self::PostProcessInput<'a>>,
    {
        let pre_processed = self.preprocess(input, ())?;
        let worker_res = worker_fun(pre_processed)?;
        let post_processed = self.postprocess(worker_res, ())?;
        Ok(post_processed)
    }
}
