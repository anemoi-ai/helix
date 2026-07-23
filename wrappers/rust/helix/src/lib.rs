/*!
# helix

Idiomatic Rust wrapper for libhelix.

```rust,no_run
use helix::{Runtime, RuntimeOptions, ModelOptions, SessionOptions, ChatCompletionRequest, Message};
use futures::StreamExt;

#[tokio::main]
async fn main() -> helix::Result<()> {
    let runtime = Runtime::new(RuntimeOptions::default())?;
    let model = runtime.load_model("qwen.gguf", ModelOptions::default())?;
    let session = model.session(SessionOptions::default())?;

    let req = ChatCompletionRequest {
        model: "qwen-2.5".into(),
        messages: vec![Message::user("hi")],
        max_tokens: Some(64),
        ..Default::default()
    };

    let resp = session.chat_completions(req)?;
    println!("{}", resp.choices[0].message.content.as_deref().unwrap_or(""));
    Ok(())
}
```
*/

pub mod error;
pub mod ffi_util;
pub mod types;

#[cfg(feature = "tokio")]
mod stream;

pub use error::{Error, Result};
pub use types::*;

use std::ffi::CStr;
use std::ptr;
use std::sync::Arc;

use helix_sys as sys;

use ffi_util::{last_error_json, to_cstring};

// ── RuntimeOptions / ModelOptions / SessionOptions ───────────────────────────

#[derive(Debug, Default)]
pub struct RuntimeOptions {
    pub log_level: LogLevel,
    pub extra: serde_json::Value,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub enum LogLevel {
    Off,
    Error,
    #[default]
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    fn as_u32(&self) -> u32 {
        match self {
            LogLevel::Off => 0,
            LogLevel::Error => 1,
            LogLevel::Warn => 2,
            LogLevel::Info => 3,
            LogLevel::Debug => 4,
            LogLevel::Trace => 5,
        }
    }
}

impl RuntimeOptions {
    fn to_json(&self) -> String {
        let mut m = serde_json::Map::new();
        m.insert("log_level".into(), self.log_level.as_u32().into());
        if let serde_json::Value::Object(extra) = &self.extra {
            m.extend(extra.clone());
        }
        serde_json::Value::Object(m).to_string()
    }
}

#[derive(Debug, Default)]
pub struct ModelOptions {
    pub alias: Option<String>,
    pub n_gpu_layers: Option<i32>,
    pub n_ctx: Option<u32>,
    pub mmproj_path: Option<String>,
    pub reasoning_format: Option<String>,
}

impl ModelOptions {
    fn to_json(&self, model_path: &str) -> String {
        let mut m = serde_json::Map::new();
        m.insert("model_path".into(), model_path.into());
        if let Some(a) = &self.alias {
            m.insert("alias".into(), a.as_str().into());
        }
        if let Some(n) = self.n_gpu_layers {
            m.insert("n_gpu_layers".into(), n.into());
        }
        if let Some(c) = self.n_ctx {
            m.insert("n_ctx".into(), c.into());
        }
        if let Some(p) = &self.mmproj_path {
            m.insert("mmproj_path".into(), p.as_str().into());
        }
        if let Some(r) = &self.reasoning_format {
            m.insert("reasoning_format".into(), r.as_str().into());
        }
        serde_json::Value::Object(m).to_string()
    }
}

#[derive(Debug, Default)]
pub struct SpeculativeOptions {
    /// "none" (default) or "draft-mtp"
    pub spec_type: String,
    /// Path to a separate MTP draft model GGUF (required for Gemma-4)
    pub model_path: Option<String>,
    pub n_max: Option<i32>,
    pub n_min: Option<i32>,
    pub p_min: Option<f32>,
    pub backend_sampling: Option<bool>,
    pub cache_type_k: Option<String>,
    pub cache_type_v: Option<String>,
}

impl SpeculativeOptions {
    pub fn draft_mtp() -> Self {
        SpeculativeOptions {
            spec_type: "draft-mtp".into(),
            ..Default::default()
        }
    }

    fn to_json_value(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        if !self.spec_type.is_empty() && self.spec_type != "none" {
            m.insert("type".into(), self.spec_type.clone().into());
        }
        if let Some(p) = &self.model_path {
            m.insert("model_path".into(), p.clone().into());
        }
        if let Some(v) = self.n_max {
            m.insert("n_max".into(), v.into());
        }
        if let Some(v) = self.n_min {
            m.insert("n_min".into(), v.into());
        }
        if let Some(v) = self.p_min {
            m.insert("p_min".into(), v.into());
        }
        if let Some(v) = self.backend_sampling {
            m.insert("backend_sampling".into(), v.into());
        }
        if let Some(v) = &self.cache_type_k {
            m.insert("cache_type_k".into(), v.clone().into());
        }
        if let Some(v) = &self.cache_type_v {
            m.insert("cache_type_v".into(), v.clone().into());
        }
        serde_json::Value::Object(m)
    }
}

#[derive(Debug, Default)]
pub struct SessionOptions {
    pub n_ctx: Option<u32>,
    pub swa_full: Option<bool>,
    pub speculative: Option<SpeculativeOptions>,
    pub extra: serde_json::Value,
}

impl SessionOptions {
    fn to_json(&self) -> String {
        let mut base = match &self.extra {
            serde_json::Value::Object(m) => m.clone(),
            _ => serde_json::Map::new(),
        };
        if let Some(c) = self.n_ctx {
            base.insert("n_ctx".into(), c.into());
        }
        if let Some(s) = self.swa_full {
            base.insert("swa_full".into(), s.into());
        }
        if let Some(spec) = &self.speculative {
            let spec_val = spec.to_json_value();
            if let serde_json::Value::Object(m) = spec_val {
                if !m.is_empty() {
                    base.insert("speculative".into(), serde_json::Value::Object(m));
                }
            }
        }
        if base.is_empty() {
            "{}".into()
        } else {
            serde_json::Value::Object(base).to_string()
        }
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

pub struct Session {
    ptr: *mut sys::helix_session_t,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    pub fn chat_completions(&self, req: ChatCompletionRequest) -> Result<ChatCompletion> {
        let req_json = to_cstring(&serde_json::to_string(&req)?)?;
        let mut out: *mut std::os::raw::c_char = ptr::null_mut();
        let rc = unsafe { sys::helix_chat_completions(self.ptr, req_json.as_ptr(), &mut out) };
        error::check(rc, &last_error_json())?;
        let resp_str = unsafe { CStr::from_ptr(out).to_string_lossy().into_owned() };
        unsafe { sys::helix_free(out) };
        Ok(serde_json::from_str(&resp_str)?)
    }

    #[cfg(feature = "tokio")]
    pub fn stream_chat_completions(
        self: Arc<Self>,
        req: ChatCompletionRequest,
    ) -> impl futures::Stream<Item = Result<ChatCompletionChunk>> + Send + Unpin {
        stream::HelixStream::new(self, req)
    }

    pub fn cancel(&self) {
        unsafe { sys::helix_session_cancel(self.ptr) };
    }

    pub fn ptr(&self) -> *mut sys::helix_session_t {
        self.ptr
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::helix_session_destroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

pub struct Model {
    ptr: *mut sys::helix_model_t,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn ptr(&self) -> *mut sys::helix_model_t {
        self.ptr
    }

    pub fn session(&self, options: SessionOptions) -> Result<Arc<Session>> {
        let opts_json = to_cstring(&options.to_json())?;
        let mut out: *mut sys::helix_session_t = ptr::null_mut();
        let rc = unsafe { sys::helix_session_create(self.ptr, opts_json.as_ptr(), &mut out) };
        error::check(rc, &last_error_json())?;
        Ok(Arc::new(Session { ptr: out }))
    }

    pub fn describe(&self) -> Result<serde_json::Value> {
        let s = unsafe { CStr::from_ptr(sys::helix_model_describe(self.ptr)).to_string_lossy() };
        Ok(serde_json::from_str(&s)?)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::helix_model_release(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

// ── Runtime ───────────────────────────────────────────────────────────────────

pub struct Runtime {
    ptr: *mut sys::helix_runtime_t,
}

unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    pub fn new(options: RuntimeOptions) -> Result<Self> {
        let opts_json = to_cstring(&options.to_json())?;
        let mut out: *mut sys::helix_runtime_t = ptr::null_mut();
        let rc = unsafe { sys::helix_runtime_create(opts_json.as_ptr(), &mut out) };
        error::check(rc, &last_error_json())?;
        Ok(Runtime { ptr: out })
    }

    pub fn load_model(&self, model_path: &str, options: ModelOptions) -> Result<Model> {
        let model_json = to_cstring(&options.to_json(model_path))?;
        let mut out: *mut sys::helix_model_t = ptr::null_mut();
        let rc = unsafe { sys::helix_model_load(self.ptr, model_json.as_ptr(), &mut out) };
        error::check(rc, &last_error_json())?;
        Ok(Model { ptr: out })
    }

    pub fn describe(&self) -> Result<serde_json::Value> {
        let s = unsafe { CStr::from_ptr(sys::helix_runtime_describe(self.ptr)).to_string_lossy() };
        Ok(serde_json::from_str(&s)?)
    }

    pub fn abi_version(&self) -> u32 {
        unsafe { sys::helix_abi_version() }
    }

    pub fn version_string(&self) -> String {
        unsafe {
            CStr::from_ptr(sys::helix_version_string())
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::helix_runtime_destroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}
