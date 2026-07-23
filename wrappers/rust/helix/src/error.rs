use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Deserialize, Default)]
struct ErrorBody {
    #[serde(default)]
    message: String,
    #[serde(default)]
    param: Option<String>,
    /// Context-length detail (helix ABI 1.7): tokens requested vs the limit.
    #[serde(default)]
    requested: Option<u32>,
    #[serde(default)]
    limit: Option<u32>,
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("invalid argument: {message} (param: {param:?})")]
    InvalidArg {
        message: String,
        param: Option<String>,
    },

    #[error("invalid JSON: {message}")]
    InvalidJson { message: String },

    #[error("validation error: {message} (param: {param:?})")]
    Validation {
        message: String,
        param: Option<String>,
    },

    #[error("model not found: {message}")]
    ModelNotFound { message: String },

    #[error("model load failed: {message}")]
    ModelLoadFailed { message: String },

    #[error("out of memory: {message}")]
    Oom { message: String },

    #[error("VRAM exhausted: {message}")]
    VramExhausted { message: String },

    #[error("context full: {message}")]
    ContextFull {
        message: String,
        /// Tokens requested (helix ABI 1.7; `None` on older libraries).
        requested: Option<u32>,
        /// Context-window limit (helix ABI 1.7; `None` on older libraries).
        limit: Option<u32>,
    },

    #[error("cancelled")]
    Cancelled,

    #[error("backend error: {message}")]
    Backend { message: String },

    #[error("unsupported feature: {message}")]
    UnsupportedFeature { message: String },

    #[error("internal error: {message}")]
    Internal { message: String },

    #[error("JSON serialisation: {0}")]
    Json(#[from] serde_json::Error),

    #[error("unexpected status code {0}")]
    Unknown(i32),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn check(rc: i32, last_error_json: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }

    let body = serde_json::from_str::<ErrorEnvelope>(last_error_json)
        .map(|e| e.error)
        .unwrap_or_default();
    let msg = body.message.clone();
    let param = body.param.clone();
    let requested = body.requested;
    let limit = body.limit;

    // HELIX_E_* constants — mirrored from helix.h
    match rc {
        -1 => Err(Error::InvalidArg {
            message: msg,
            param,
        }),
        -2 => Err(Error::InvalidJson { message: msg }),
        -3 => Err(Error::Validation {
            message: msg,
            param,
        }),
        -4 => Err(Error::ModelNotFound { message: msg }),
        -5 => Err(Error::ModelLoadFailed { message: msg }),
        -6 => Err(Error::Oom { message: msg }),
        -7 => Err(Error::VramExhausted { message: msg }),
        -8 => Err(Error::ContextFull {
            message: msg,
            requested,
            limit,
        }),
        -9 => Err(Error::Cancelled),
        -10 => Err(Error::Backend { message: msg }),
        -11 => Err(Error::UnsupportedFeature { message: msg }),
        -99 => Err(Error::Internal { message: msg }),
        n => Err(Error::Unknown(n)),
    }
}
