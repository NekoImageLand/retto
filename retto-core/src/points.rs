use crate::serde::*;
use imageproc::point::Point as ImagePoint;
use num_traits::{AsPrimitive, Num, NumCast, Signed};
use std::fmt::Debug;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::Deserialize<'de>"
    ))
)]
/// A point on a two-dimensional plane
pub struct Point<T>
where
    T: Num + NumCast + Signed + AsPrimitive<f32> + Copy + Ord + Debug,
{
    pub x: T,
    pub y: T,
}

impl<T> Point<T>
where
    T: Num + NumCast + Signed + AsPrimitive<f32> + Copy + Ord + Debug,
{
    pub fn new(x: T, y: T) -> Self {
        Point { x, y }
    }

    pub fn dist2(&self) -> T {
        self.x * self.x + self.y * self.y
    }

    pub fn range(&self, other: &Self) -> f32 {
        let dx = self.x.as_() - other.x.as_();
        let dy = self.y.as_() - other.y.as_();
        (dx * dx + dy * dy).sqrt()
    }
}

impl<T> From<ImagePoint<T>> for Point<T>
where
    T: Num + NumCast + Signed + AsPrimitive<f32> + Copy + Ord + Debug,
{
    fn from(p: ImagePoint<T>) -> Self {
        Point { x: p.x, y: p.y }
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::Deserialize<'de>"
    ))
)]
/// A rectangular point frame on a two-dimensional plane <br></br>
/// **Points are ordered as clockwise starting from the top-left corner**
pub struct PointBox<T>
where
    T: Num + NumCast + Signed + AsPrimitive<f32> + Copy + Ord + Debug,
{
    // Points are ordered as clockwise starting from the top-left corner
    inner: [Point<T>; 4],
}

impl<T> Debug for PointBox<T>
where
    T: Num + NumCast + Signed + AsPrimitive<f32> + Copy + Ord + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointBox")
            .field("tl", &self.tl())
            .field("tr", &self.tr())
            .field("br", &self.br())
            .field("bl", &self.bl())
            .finish()
    }
}

impl<T> PointBox<T>
where
    T: Num + NumCast + Signed + AsPrimitive<f32> + Copy + Ord + Debug,
{
    pub fn new_from_clockwise(points: [Point<T>; 4]) -> Self {
        PointBox { inner: points }
    }

    // TODO: Add a constructor that automatically corrects the direction of the point

    #[inline]
    pub fn points(&self) -> &[Point<T>; 4] {
        &self.inner
    }

    /// Top Left point
    #[inline]
    pub fn tl(&self) -> &Point<T> {
        &self.inner[0]
    }

    /// Top Right point
    #[inline]
    pub fn tr(&self) -> &Point<T> {
        &self.inner[1]
    }

    /// Bottom Right point
    #[inline]
    pub fn br(&self) -> &Point<T> {
        &self.inner[2]
    }

    /// Bottom Left point
    #[inline]
    pub fn bl(&self) -> &Point<T> {
        &self.inner[3]
    }

    /// Returns the height of the bounding box (from top-left)
    #[inline]
    pub fn height_tlc(&self) -> T
    where
        T: Num + NumCast + Copy + Ord + Debug + AsPrimitive<f64>,
    {
        let dx: f64 = (self.tl().x - self.bl().x).as_();
        let dy: f64 = (self.tl().y - self.bl().y).as_();
        let dist = (dx * dx + dy * dy).sqrt();
        NumCast::from(dist).unwrap()
    }

    /// Returns the width of the bounding box (from top-left)
    #[inline]
    pub fn width_tlc(&self) -> T
    where
        T: Num + NumCast + Copy + Ord + Debug + AsPrimitive<f64>,
    {
        let dx: f64 = (self.tl().x - self.tr().x).as_();
        let dy: f64 = (self.tl().y - self.tr().y).as_();
        let dist = (dx * dx + dy * dy).sqrt();
        NumCast::from(dist).unwrap()
    }

    /// Returns the height of the bounding box (from bottom-right)
    #[inline]
    pub fn height_brc(&self) -> T
    where
        T: Num + NumCast + Copy + Ord + Debug + AsPrimitive<f64>,
    {
        let dx: f64 = (self.tr().x - self.br().x).as_();
        let dy: f64 = (self.tr().y - self.br().y).as_();
        let dist = (dx * dx + dy * dy).sqrt();
        NumCast::from(dist).unwrap()
    }

    /// Returns the width of the bounding box (from bottom-right)
    #[inline]
    pub fn width_brc(&self) -> T
    where
        T: Num + NumCast + Copy + Ord + Debug + AsPrimitive<f64>,
    {
        let dx: f64 = (self.bl().x - self.br().x).as_();
        let dy: f64 = (self.bl().y - self.br().y).as_();
        let dist = (dx * dx + dy * dy).sqrt();
        NumCast::from(dist).unwrap()
    }

    /// Returns the center point of the bounding box
    #[inline]
    pub fn center_point(&self) -> Point<T> {
        let center_x = (self.tl().x + self.br().x) / NumCast::from(2).unwrap();
        let center_y = (self.tl().y + self.br().y) / NumCast::from(2).unwrap();
        Point::new(center_x, center_y)
    }

    pub(crate) fn scale_and_clip(&mut self, bitmap_w: f64, bitmap_h: f64, ori_w: f64, ori_h: f64)
    where
        T: Num + NumCast + Copy + Ord + Debug + AsPrimitive<f64>,
    {
        let inv_w = ori_w / bitmap_w;
        let inv_h = ori_h / bitmap_h;
        self.inner.iter_mut().for_each(|p| {
            let x0: f64 = p.x.as_();
            let y0: f64 = p.y.as_();
            // also done clip_det_res
            let x1 = (x0 * inv_w).round().clamp(0.0, ori_w - 1f64);
            let y1 = (y0 * inv_h).round().clamp(0.0, ori_h - 1f64);
            p.x = NumCast::from(x1).unwrap();
            p.y = NumCast::from(y1).unwrap();
        })
    }
}
