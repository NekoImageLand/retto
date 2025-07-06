#![no_main]
#![feature(concat_bytes)]
#![feature(linkage)]

mod macros;
#[cfg(all(target_arch = "wasm32", target_os = "emscripten"))]
mod wasm_lib;

#[cfg(all(target_arch = "wasm32", target_os = "emscripten"))]
#[allow(unused_imports)]
use crate::wasm_lib::*;

#[cfg(all(
    any(unix, windows),
    not(all(target_arch = "wasm32", target_os = "emscripten"))
))]
#[allow(dead_code)]
#[unsafe(no_mangle)]
pub extern "C" fn main() -> i32 {
    0
}
