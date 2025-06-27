#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub mod error;
mod image_helper;
pub mod points;
pub mod processor;
pub mod session;
pub mod worker;

#[cfg(feature = "serde")]
pub(crate) trait MaybeSerde: Serialize + for<'a> Deserialize<'a> {}
#[cfg(feature = "serde")]
impl<T> MaybeSerde for T where T: serde::Serialize + for<'de> serde::Deserialize<'de> {}
#[cfg(not(feature = "serde"))]
pub(crate) trait MaybeSerde {}

pub mod prelude {
    pub use crate::error::{RettoError, RettoResult};
    pub use crate::processor::prelude::*;
    pub use crate::session::*;
    pub use crate::worker::prelude::*;
}
