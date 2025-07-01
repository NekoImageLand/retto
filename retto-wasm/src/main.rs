#![no_main]
#![feature(concat_bytes)]
#![feature(linkage)]

mod macros;

use once_cell::sync::Lazy;
use retto_core::prelude::*;
use std::ffi::{CString, c_char, c_uint, c_void};
use std::sync::Mutex;
use std::{alloc, thread};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

static GLOBAL_TRACING: Lazy<Mutex<()>> = Lazy::new(|| {
    let stdout = tracing_subscriber::fmt::layer()
        .with_filter(EnvFilter::new("ort=warn,retto_core=debug,retto_cli=debug"));
    tracing_subscriber::registry().with(stdout).init();
    Mutex::new(())
});

static GLOBAL_SESSION: Lazy<Mutex<RettoSession<RettoOrtWorker>>> = Lazy::new(|| {
    let session: RettoSession<RettoOrtWorker> = RettoSession::new(RettoSessionConfig {
        worker_config: RettoOrtWorkerConfig {
            device: RettoOrtWorkerDevice::CPU,
            models: RettoOrtWorkerModelProvider::from_local_v4_blob_default(),
        },
        ..Default::default()
    })
    .expect("Failed to create RettoSession");
    Mutex::new(session)
});

#[unsafe(no_mangle)]
pub extern "C" fn alloc(size: usize) -> *mut c_void {
    unsafe {
        let layout = alloc::Layout::from_size_align(size, align_of::<u8>())
            .expect("Cannot create memory layout.");
        alloc::alloc(layout) as *mut c_void
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dealloc(ptr: *mut c_void, size: usize) {
    unsafe {
        let layout = alloc::Layout::from_size_align(size, align_of::<u8>())
            .expect("Cannot create memory layout.");
        alloc::dealloc(ptr as *mut u8, layout);
    }
}

em_js!((), retto_notify_done, (msg: *const char), {
    if (Module.onRettoNotifyDone) {
        Module.onRettoNotifyDone(UTF8ToString(msg));
    }
});

/// I'm too lazy qwq
unsafe extern "C" {
    fn emscripten_sync_run_in_main_runtime_thread_(sig: c_uint, func_ptr: *mut c_void, ...) -> i32;
}

#[unsafe(no_mangle)]
pub extern "C" fn retto(image_data_ptr: *const u8, image_data_len: u32) {
    let image_data =
        unsafe { std::slice::from_raw_parts(image_data_ptr, image_data_len as usize).to_vec() };
    thread::spawn(move || {
        Lazy::force(&GLOBAL_TRACING);
        let mut session = GLOBAL_SESSION
            .lock()
            .expect("Failed to lock RettoSession mutex");
        let res = session.run(image_data).expect("Failed to run RettoSession");
        let res_str = serde_json::to_string(&res).expect("Failed to serialize RettoSession");
        let res_cstr = CString::new(res_str).expect("Failed to create CString");
        let ptr: *mut c_char = res_cstr.into_raw();
        unsafe {
            emscripten_sync_run_in_main_runtime_thread_(
                0 | 1 << 25 | 0 << (2 * 0), // aka EM_FUNC_SIG_VI, TODO: use enum
                retto_notify_done as *mut c_void,
                ptr as *const c_char,
            );
        }
    });
}
