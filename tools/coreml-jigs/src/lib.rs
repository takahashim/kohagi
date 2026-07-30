//! Shared pieces for the CoreML development jigs.
//!
//! What the jigs share: the MIL blob storage format ([`blob`]), operation
//! inventories for a MIL program ([`mil`]), a model's declared inputs and
//! outputs read from its protobuf ([`spec`]), finding
//! and running the bucket models in a converted directory ([`bundle`]), and a way
//! to call CoreML's completion-handler APIs from a plain synchronous `main`
//! ([`await_handler`]).

pub mod blob;
pub mod bundle;
pub mod mil;
pub mod spec;

use std::sync::mpsc;

use block2::{DynBlock, RcBlock};
use objc2::rc::Retained;
use objc2::Message;
use objc2_foundation::NSError;

/// Drive one of CoreML's `…WithCompletionHandler:` APIs to completion and return
/// its result.
///
/// The handler fires on a queue CoreML picks, so the object cannot be moved out
/// through a channel directly: `Retained` is not `Send`. Instead the handler
/// retains it and passes the raw pointer across as a `usize`, and this side
/// takes ownership of that same `+1` reference. The `Err` case is formatted
/// inside the handler, where the `NSError` is still alive.
pub fn await_handler<T: Message>(
    start: impl FnOnce(&DynBlock<dyn Fn(*mut T, *mut NSError)>),
) -> Result<Retained<T>, String> {
    let (tx, rx) = mpsc::channel::<Result<usize, String>>();
    let block = RcBlock::new(move |obj: *mut T, err: *mut NSError| {
        let sent = match unsafe { obj.as_ref() } {
            Some(_) => {
                let held = unsafe { Retained::retain(obj) }.expect("checked non-null");
                Ok(Retained::into_raw(held) as usize)
            }
            None => Err(unsafe { err.as_ref() }
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "CoreML failed without an error object".to_string())),
        };
        let _ = tx.send(sent);
    });
    start(&block);
    match rx.recv() {
        Ok(Ok(ptr)) => {
            Ok(unsafe { Retained::from_raw(ptr as *mut T) }.expect("retained in the handler"))
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err("CoreML's completion handler never fired".to_string()),
    }
}

/// Human name for a compute device, taken from its Objective-C class rather than
/// by downcasting: `MLNeuralEngineComputeDevice` -> `ANE`.
pub fn device_name(class_name: &str) -> &str {
    match class_name {
        "MLNeuralEngineComputeDevice" => "ANE",
        "MLCPUComputeDevice" => "CPU",
        "MLGPUComputeDevice" => "GPU",
        other => other,
    }
}
