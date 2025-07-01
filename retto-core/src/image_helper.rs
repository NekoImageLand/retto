use crate::error::RettoResult;
use crate::points::PointBox;
use crate::processor::det_processor::LimitType;
use image::imageops::rotate270;
use image::{ImageBuffer, Rgb, RgbImage, imageops};
use imageproc::definitions::HasWhite;
use imageproc::geometric_transformations::{Interpolation, Projection, warp_into};
use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use paste::paste;
use std::cmp::{max, min};

pub(crate) struct ImageHelper {
    inner: Option<RgbImage>,
    ori_h: usize,
    ori_w: usize,
}

pub(crate) trait ImagesOrder {
    type Item;
    fn ordered_by<K, F>(&mut self, order_f: F)
    where
        F: FnMut(&Self::Item) -> K,
        K: Ord;
    fn get_ordered_index<K, F>(&self, order_f: F) -> Vec<usize>
    where
        F: FnMut(&Self::Item) -> K,
        K: Ord;
}

// TODO: Use fast_image_resize with interpolation methods
impl ImageHelper {
    /// Heavy
    pub fn new_from_raw_img_flow(input: impl AsRef<[u8]>) -> RettoResult<Self> {
        // TODO: cvt_two_to_three + cvt_four_to_three
        let image = image::load_from_memory(input.as_ref())?;
        let image = image.to_rgb8();
        let (ori_w, ori_h) = image.dimensions();
        Ok(Self {
            inner: Some(image),
            ori_h: ori_h as usize,
            ori_w: ori_w as usize,
        })
    }

    /// Heavy
    pub fn new_from_rgb_image_flow(input: impl AsRef<[u8]>, height: usize, weight: usize) -> Self {
        let image: ImageBuffer<Rgb<u8>, &[u8]> =
            ImageBuffer::from_raw(weight as u32, height as u32, input.as_ref())
                .expect("Failed to create ImageBuffer from raw data");
        let (ori_w, ori_h) = image.dimensions();
        let image_owned = RgbImage::from_vec(ori_w, ori_h, input.as_ref().to_vec()).unwrap();
        Self {
            inner: Some(image_owned),
            ori_h: ori_h as usize,
            ori_w: ori_w as usize,
        }
    }

    pub fn new_from_rgb_image(input: RgbImage) -> Self {
        let (ori_w, ori_h) = input.dimensions();
        Self {
            inner: Some(input),
            ori_h: ori_h as usize,
            ori_w: ori_w as usize,
        }
    }

    pub fn take_inner(&mut self) -> Option<RgbImage> {
        self.inner.take()
    }

    #[inline]
    pub fn ori_size(&self) -> (usize, usize) {
        (self.ori_h, self.ori_w)
    }

    #[inline]
    pub fn ori_ratio(&self) -> f64 {
        let (h, w) = self.ori_size();
        (h as f64) / (w as f64)
    }

    #[inline]
    pub fn size(&self) -> (usize, usize) {
        let image = self.inner.as_ref().unwrap();
        let (w, h) = image.dimensions();
        (h as usize, w as usize)
    }

    #[inline]
    pub fn ratio(&self) -> f64 {
        let (h, w) = self.size();
        (h as f64) / (w as f64)
    }

    pub fn array_view(&self) -> RettoResult<ArrayView3<'_, u8>> {
        let image = self.inner.as_ref().unwrap();
        let (w, h) = image.dimensions();
        Ok(ArrayView3::from_shape(
            (h as usize, w as usize, 3),
            image.as_raw().as_slice(),
        )?)
    }

    pub fn resize_both(
        &mut self,
        crop_max_size_len: usize,
        crop_min_size_len: usize,
    ) -> RettoResult<(f32, f32)> {
        let mut image: Option<ImageBuffer<Rgb<u8>, Vec<u8>>> = Some(self.inner.take().unwrap());
        let mut ratio_h = 1.0f32;
        let mut ratio_w = 1.0f32;
        // TODO: use `fast_image_resize::Resizer`
        let (h, w) = (self.ori_h as f32, self.ori_w as f32);
        if max(self.ori_h, self.ori_w) > crop_max_size_len {
            let scale = (crop_max_size_len as f32) / (h.max(w));
            let (resize_h, resize_w) = (
                ((h * scale).floor() as u32 / 32).max(1) * 32,
                ((w * scale).floor() as u32 / 32).max(1) * 32,
            );
            ratio_h = h / resize_h as f32;
            ratio_w = w / resize_w as f32;
            image = Some(imageops::thumbnail(
                &image.unwrap(), // TODO:
                resize_w,
                resize_h,
                // FilterType::Triangle,
            ))
        }
        if min(self.ori_h, self.ori_w) < crop_min_size_len {
            let scale = (crop_min_size_len as f32) / (h.min(w));
            let (resize_h, resize_w) = (
                (((h * scale).floor() / 32.0).round() as u32) * 32,
                (((w * scale).floor() / 32.0).round() as u32) * 32,
            );
            ratio_h = h / resize_h as f32;
            ratio_w = w / resize_w as f32;
            image = Some(imageops::thumbnail(
                &image.unwrap(),
                resize_w,
                resize_h,
                // FilterType::Triangle,
            ))
        }
        self.inner = image;
        Ok((ratio_h, ratio_w))
    }

    pub fn resize_either(&mut self, limit_type: &LimitType, limit_len: usize) -> RettoResult<()> {
        let image: Option<ImageBuffer<Rgb<u8>, Vec<u8>>> = self.inner.take();
        let (w, h) = image.as_ref().unwrap().dimensions();
        let ratio = match limit_type {
            LimitType::Max => match max(w, h) > limit_len as u32 {
                true => limit_len as f32 / max(w, h) as f32,
                false => 1.0f32,
            },
            LimitType::Min => match min(w, h) < limit_len as u32 {
                true => limit_len as f32 / min(w, h) as f32,
                false => 1.0f32,
            },
        };
        let (resize_h, resize_w) = (
            (((h as f32 * ratio).floor() / 32.0).round() as u32) * 32,
            (((w as f32 * ratio).floor() / 32.0).round() as u32) * 32,
        );
        self.inner = image.map(|img| {
            imageops::thumbnail(
                &img, resize_w, resize_h,
                // FilterType::Triangle,
            )
        });
        Ok(())
    }

    pub fn resize_norm_image(&self, shape: [usize; 3], max_wh_ratio: Option<f32>) -> Array3<f32> {
        let [img_c, img_h, img_w] = shape;
        let img_w = match max_wh_ratio {
            Some(mr) => (img_h as f32 * mr) as usize,
            None => img_w,
        };
        let (h, w) = (self.ori_h as u32, self.ori_w as u32);
        let resized_w = min(img_w, (img_h as f64 * w as f64 / h as f64).ceil() as usize);
        let resized_img = imageops::thumbnail(
            self.inner.as_ref().unwrap(),
            resized_w as u32,
            img_h as u32,
            // FilterType::Triangle,
        );
        let mut resized_img_np = match img_c {
            1 => {
                let hwc = Array3::from_shape_fn((img_h, resized_w, 1), |(y, x, _)| {
                    let pixel = resized_img.get_pixel(x as u32, y as u32)[0];
                    pixel as f32 / 255.0
                });
                hwc.permuted_axes([2, 0, 1])
            }
            _ => Array3::from_shape_fn((3, img_h, resized_w), |(c, y, x)| {
                let pixel = resized_img.get_pixel(x as u32, y as u32);
                pixel[c] as f32 / 255.0
            }),
        };
        resized_img_np.mapv_inplace(|v| (v - 0.5) / 0.5);
        let mut padding_im = Array3::<f32>::zeros((img_c, img_h, img_w));
        padding_im
            .slice_mut(s![.., .., 0..resized_w])
            .assign(&resized_img_np);
        padding_im
    }

    pub fn rgb2bgr(&mut self) -> RettoResult<Array3<u8>> {
        let image = self.inner.take().unwrap();
        let (w, h) = image.dimensions();
        let mut raw = image.into_raw();
        // rgb2bgr
        for chunk in raw.chunks_exact_mut(3) {
            chunk.swap(0, 2);
        }
        let arr = Array3::from_shape_vec((h as usize, w as usize, 3), raw)?;
        Ok(arr)
    }

    pub fn get_crop_img(&self, point: &PointBox<OrderedFloat<f64>>) -> RgbImage {
        let img_crop_width = max(point.width_brc(), point.width_tlc()).into_inner() as f32;
        let img_crop_height = max(point.height_brc(), point.height_tlc()).into_inner() as f32;
        let (w, h) = (img_crop_width as u32, img_crop_height as u32);
        let mut out: RgbImage = ImageBuffer::new(w, h);
        let proj = Projection::from_control_points(
            point
                .points()
                .map(|p| (p.x.into_inner() as f32, p.y.into_inner() as f32)),
            [
                (0.0, 0.0),
                (img_crop_width, 0.0),
                (img_crop_width, img_crop_height),
                (0.0, img_crop_height),
            ],
        )
        .unwrap(); // TODO:
        warp_into(
            self.inner.as_ref().unwrap(),
            &proj,
            Interpolation::Bicubic,
            Rgb::white(), // TODO: in opencv impl is cv2.BORDER_REPLICATE, not cv2.BORDER_CONSTANT
            &mut out,
        );
        if (out.height() as f32) / (out.width() as f32) >= 1.5 {
            return rotate270(&out);
        }
        out
    }
}

macro_rules! impl_rotate {
    ($($deg:literal),+ $(,)?) => {
        impl ImageHelper {
            paste! {
                $(
                    pub fn [<rotate_ $deg>](&self) -> RettoResult<RgbImage> {
                        let image = self.inner.as_ref().unwrap();
                        let rotated = image::imageops::[<rotate $deg>](image);
                        Ok(rotated)
                    }
                )+
            }
        }
    };
}

macro_rules! impl_rotate_in_place {
    ($($deg:literal),+ $(,)?) => {
        impl ImageHelper {
            paste! {
                $(
                    pub fn [<rotate_ $deg _in_place>](&mut self) -> RettoResult<()> {
                        let mut img = self.inner.take().unwrap();
                        image::imageops::[<rotate $deg _in_place>](&mut img);
                        self.inner = Some(img);
                        Ok(())
                    }
                )+
            }
        }
    };
}

impl_rotate!(90, 180, 270);
impl_rotate_in_place!(180);

impl ImagesOrder for [ImageHelper] {
    type Item = ImageHelper;

    fn ordered_by<K, F>(&mut self, order_f: F)
    where
        F: FnMut(&Self::Item) -> K,
        K: Ord,
    {
        self.sort_by_key(order_f);
    }

    fn get_ordered_index<K, F>(&self, mut order_f: F) -> Vec<usize>
    where
        F: FnMut(&Self::Item) -> K,
        K: Ord,
    {
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.sort_by_key(|&i| order_f(&self[i]));
        indices
    }
}
