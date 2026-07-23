use std::ffi::CStr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::mpsc;

use crate::error::{self, Error, Result};
use crate::ffi_util::to_cstring;
use crate::types::{ChatCompletionChunk, ChatCompletionRequest};
use crate::Session;
use helix_sys as sys;

pub(crate) struct HelixStream {
    rx: mpsc::UnboundedReceiver<Result<ChatCompletionChunk>>,
    session: Arc<Session>,
    _join: Option<tokio::task::JoinHandle<()>>,
}

unsafe impl Send for HelixStream {}

impl HelixStream {
    pub(crate) fn new(session: Arc<Session>, req: ChatCompletionRequest) -> Pin<Box<Self>> {
        let (tx, rx) = mpsc::unbounded_channel::<Result<ChatCompletionChunk>>();

        let session_clone = Arc::clone(&session);
        let req_json = match serde_json::to_string(&req) {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(Err(e.into()));
                return Box::pin(HelixStream {
                    rx,
                    session,
                    _join: None,
                });
            }
        };

        struct CallbackState {
            tx: mpsc::UnboundedSender<Result<ChatCompletionChunk>>,
        }

        unsafe extern "C" fn stream_cb(
            user_data: *mut std::os::raw::c_void,
            chunk_json: *const std::os::raw::c_char,
        ) -> std::os::raw::c_int {
            let state = &*(user_data as *const CallbackState);
            if chunk_json.is_null() {
                return 0;
            }
            let s = CStr::from_ptr(chunk_json).to_string_lossy();
            let result =
                serde_json::from_str::<ChatCompletionChunk>(&s).map_err(|e| Error::Json(e));
            let closed = state.tx.send(result).is_err();
            if closed {
                1
            } else {
                0
            }
        }

        let handle = tokio::task::spawn_blocking(move || {
            let req_cstr = match to_cstring(&req_json) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };
            let state = Box::new(CallbackState { tx: tx.clone() });
            let state_ptr = Box::into_raw(state) as *mut std::os::raw::c_void;

            let rc = unsafe {
                sys::helix_chat_completions_stream(
                    session_clone.ptr(),
                    req_cstr.as_ptr(),
                    Some(stream_cb),
                    state_ptr,
                )
            };

            // Reclaim state box to avoid leak
            let _state = unsafe { Box::from_raw(state_ptr as *mut CallbackState) };

            if rc != 0 && rc != -9 {
                let err_json = unsafe {
                    CStr::from_ptr(sys::helix_last_error_json())
                        .to_string_lossy()
                        .into_owned()
                };
                let _ = error::check(rc, &err_json).map_err(|e| {
                    let _ = tx.send(Err(e));
                });
            }
        });

        Box::pin(HelixStream {
            rx,
            session,
            _join: Some(handle),
        })
    }
}

impl Stream for HelixStream {
    type Item = Result<ChatCompletionChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for HelixStream {
    fn drop(&mut self) {
        self.session.cancel();
    }
}
