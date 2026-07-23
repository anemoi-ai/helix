use crate::error::{Error, Result};
use helix_sys as sys;
use std::ffi::{CStr, CString};

pub fn last_error_json() -> String {
    unsafe {
        CStr::from_ptr(sys::helix_last_error_json())
            .to_string_lossy()
            .into_owned()
    }
}

pub(crate) fn to_cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::InvalidArg {
        message: "string contains interior NUL byte".into(),
        param: None,
    })
}
