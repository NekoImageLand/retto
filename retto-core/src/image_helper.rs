use crate::error::RettoResult;
use crate::processor::det_processor::LimitType;
use image::{GenericImageView, ImageBuffer, Rgb, RgbImage, imageops};
use ndarray::prelude::*;
use std::cmp::{max, min};

pub struct ImageHelper {
    inner: Option<RgbImage>,
    ori_h: usize,
    ori_w: usize,
}

impl Into<RettoResult<Array3<u8>>> for ImageHelper {
    fn into(self) -> RettoResult<Array3<u8>> {
        let image = self.inner.unwrap();
        let (w, h) = image.dimensions();
        let raw = image.into_raw();
        Ok(Array3::from_shape_vec((h as usize, w as usize, 3), raw)?)
    }
}

impl ImageHelper {
    /// Heavy
    pub fn new_from_raw_img(input: impl AsRef<[u8]>) -> RettoResult<Self> {
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
    pub fn new_from_rgb_image(input: impl AsRef<[u8]>, height: usize, weight: usize) -> Self {
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

    #[inline]
    pub fn ori_size(&self) -> (usize, usize) {
        (self.ori_h, self.ori_w)
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
            image = Some(imageops::resize(
                &image.unwrap(), // TODO:
                resize_w,
                resize_h,
                imageops::FilterType::Triangle,
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
            image = Some(imageops::resize(
                &image.unwrap(),
                resize_w,
                resize_h,
                imageops::FilterType::Triangle,
            ))
        }
        self.inner = image.map(RgbImage::from);
        Ok((ratio_h, ratio_w))
    }

    pub fn resize_either(&mut self, limit_type: &LimitType, limit_len: usize) -> RettoResult<()> {
        let mut image: Option<ImageBuffer<Rgb<u8>, Vec<u8>>> = Some(self.inner.take().unwrap());
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
        self.inner = Some(imageops::resize(
            &image.unwrap(),
            resize_w,
            resize_h,
            imageops::FilterType::Triangle,
        ));
        Ok(())
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
}
