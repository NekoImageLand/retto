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

static GLOBAL_SESSION: Lazy<Mutex<Option<RettoSession<RettoOrtWorker>>>> =
    Lazy::new(|| Mutex::new(None));

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

// TODO: add these a session_id (provided by JS) to support multiple sessions
em_js!((), retto_notify_det_done, (msg: *const c_char), {
    if (Module.onRettoNotifyDetDone) {
        Module.onRettoNotifyDetDone(UTF8ToString(msg));
    }
});

em_js!((), retto_notify_cls_done, (msg: *const c_char), {
    if (Module.onRettoNotifyClsDone) {
        Module.onRettoNotifyClsDone(UTF8ToString(msg));
    }
});

em_js!((), retto_notify_rec_done, (msg: *const c_char), {
    if (Module.onRettoNotifyRecDone) {
        Module.onRettoNotifyRecDone(UTF8ToString(msg));
    }
});

// I'm too lazy qwq
unsafe extern "C" {
    fn emscripten_sync_run_in_main_runtime_thread_(sig: c_uint, func_ptr: *mut c_void, ...) -> i32;
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
/// # Safety
/// Make clippy happy!
pub unsafe extern "C" fn retto_init(
    det_ptr: *const u8,
    det_len: usize,
    cls_ptr: *const u8,
    cls_len: usize,
    rec_ptr: *const u8,
    rec_len: usize,
    rec_dict_ptr: *const u8,
    rec_dict_len: usize,
) {
    Lazy::force(&GLOBAL_TRACING);
    let det_model = unsafe { std::slice::from_raw_parts(det_ptr, det_len).to_vec() };
    let cls_model = unsafe { std::slice::from_raw_parts(cls_ptr, cls_len).to_vec() };
    let rec_model = unsafe { std::slice::from_raw_parts(rec_ptr, rec_len).to_vec() };
    let rec_dict = unsafe { std::slice::from_raw_parts(rec_dict_ptr, rec_dict_len).to_vec() };
    let mut guard = GLOBAL_SESSION.lock().unwrap();
    guard.get_or_insert_with(|| {
        RettoSession::new(RettoSessionConfig {
            worker_config: RettoOrtWorkerConfig {
                device: RettoOrtWorkerDevice::CPU,
                models: RettoOrtWorkerModelProvider(RettoWorkerModelProvider {
                    det: RettoWorkerModelSource::Blob(det_model),
                    rec: RettoWorkerModelSource::Blob(rec_model),
                    cls: RettoWorkerModelSource::Blob(cls_model),
                }),
            },
            rec_processor_config: RecProcessorConfig {
                character_source: RecCharacterDictProvider::OutSide(RettoWorkerModelSource::Blob(
                    rec_dict,
                )),
                ..Default::default()
            },
            ..Default::default()
        })
        .expect("Failed to create RettoSession")
    });
}

#[cfg(feature = "download-models")]
#[unsafe(no_mangle)]
/// # Safety
/// Make clippy happy!
pub unsafe extern "C" fn retto_embed_init() {
    Lazy::force(&GLOBAL_TRACING);
    let mut guard = GLOBAL_SESSION.lock().unwrap();
    guard.get_or_insert_with(|| {
        RettoSession::new(RettoSessionConfig {
            worker_config: RettoOrtWorkerConfig {
                device: RettoOrtWorkerDevice::CPU,
                models: RettoOrtWorkerModelProvider::from_local_v4_blob_default(),
            },
            ..Default::default()
        })
        .expect("Failed to create RettoSession")
    });
}

#[allow(clippy::erasing_op, clippy::identity_op)] // simulate C macro behavior
#[unsafe(no_mangle)]
/// # Safety
/// Make clippy happy!
pub unsafe extern "C" fn retto_rec(image_data_ptr: *const u8, image_data_len: u32) {
    let image_data =
        unsafe { std::slice::from_raw_parts(image_data_ptr, image_data_len as usize).to_vec() };
    thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel::<RettoWorkerStageResult>();
        thread::spawn(move || {
            GLOBAL_SESSION
                .lock()
                .unwrap()
                .as_mut()
                .expect("You must call retto_init before retto_rec!")
                .run_stream(image_data, tx)
                .expect("Failed to run RettoSession stream");
        });
        const EMSCRIPTEN_SIG: c_uint = 0 | 1 << 25 | 0 << (2 * 0); // aka EM_FUNC_SIG_VI, TODO: use enum
        for stage in rx {
            let res_str =
                serde_json::to_string(&stage).expect("Failed to serialize RettoWorkerStageResult");
            let res_cstr = CString::new(res_str).expect("Failed to create CString");
            let ptr: *const c_char = res_cstr.as_ptr();
            let _ = unsafe {
                match stage {
                    RettoWorkerStageResult::Det(_) => emscripten_sync_run_in_main_runtime_thread_(
                        EMSCRIPTEN_SIG,
                        retto_notify_det_done as *mut c_void,
                        ptr as *const c_char,
                    ),
                    RettoWorkerStageResult::Cls(_) => emscripten_sync_run_in_main_runtime_thread_(
                        EMSCRIPTEN_SIG,
                        retto_notify_cls_done as *mut c_void,
                        ptr as *const c_char,
                    ),
                    RettoWorkerStageResult::Rec(_) => emscripten_sync_run_in_main_runtime_thread_(
                        EMSCRIPTEN_SIG,
                        retto_notify_rec_done as *mut c_void,
                        ptr as *const c_char,
                    ),
                }
            };
        }
    });
}
