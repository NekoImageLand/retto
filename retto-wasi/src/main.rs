#![no_main]

use retto_core::prelude::*;
use std::alloc;
use std::os::raw::c_void;

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

#[unsafe(no_mangle)]
pub extern "C" fn retto(image_data_ptr: *const u8, image_data_len: u32) {
    let image_data =
        unsafe { std::slice::from_raw_parts(image_data_ptr, image_data_len as usize).to_vec() };
    let mut session: RettoSession<RettoOrtWorker> = RettoSession::new(RettoSessionConfig {
        worker_config: RettoOrtWorkerConfig {
            device: RettoOrtWorkerDeviceConfig::CPU,
            det_model_source: RettoWorkerModelProvider::Blob(
                include_bytes!("../../ch_PP-OCRv4_det_infer.onnx").to_vec(),
            ),
            rec_model_source: RettoWorkerModelProvider::Blob(
                include_bytes!("../../ch_PP-OCRv4_rec_infer.onnx").to_vec(),
            ),
            cls_model_source: RettoWorkerModelProvider::Blob(
                include_bytes!("../../ch_ppocr_mobile_v2.0_cls_infer.onnx").to_vec(),
            ),
        },
        rec_processor_config: RecProcessorConfig {
            character_source: RecCharacterDictProvider::Blob(
                include_bytes!("../../ppocr_keys_v1.txt").to_vec(),
            ),
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("Failed to create RettoSession");
    let res = session.run(image_data).expect("Failed to run RettoSession");
    println!("retto result: {:?}", res);
}
