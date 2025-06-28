#[cfg(feature = "serde")]
pub use serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
pub(crate) trait MaybeSerde: Serialize + for<'a> Deserialize<'a> {}
#[cfg(feature = "serde")]
impl<T> MaybeSerde for T where T: crate::serde::Serialize + for<'de> crate::serde::Deserialize<'de> {}
#[cfg(not(feature = "serde"))]
pub(crate) trait MaybeSerde {}
#[cfg(not(feature = "serde"))]
impl<T> MaybeSerde for T {}
