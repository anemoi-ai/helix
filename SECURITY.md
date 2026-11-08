# Security Policy

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately via GitHub Security Advisories:
**https://github.com/anemoi-ai/helix/security/advisories/new**

Include in your report:
- A description of the vulnerability and its impact.
- Steps to reproduce or a minimal proof-of-concept.
- The affected versions (check `helix_version_string()`).
- Whether you believe the issue is exploitable remotely or only locally.

### Response SLA

| Milestone | Target |
|-----------|--------|
| Acknowledge receipt | 5 business days |
| Initial triage and severity assessment | 10 business days |
| Public fix or coordinated disclosure | 30 days from acknowledgement |

For critical vulnerabilities (remote code execution, privilege escalation),
we target a coordinated disclosure in 14 days and will work with you on timing.

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.x (current) | Security fixes backported for 18 months from 2026-05-31 |
| 0.8.0-rc.1 | No — upgrade to 1.0.0 |
| < 0.8.0 | No |

## The 18-month ABI commitment and security

The 1.x ABI commitment does not prevent security fixes. A security patch may:
- Fix a bug in an existing function without changing its signature.
- Add a new function (minor version bump, additive and backward-compatible).

A security fix that requires a backward-incompatible change is handled via the
major-version escape (see `docs/abi-policy.md`): we bump to 2.0.0, maintain
1.x security patches for at least 6 additional months, and give users a clear
migration window.

## Scope

In-scope vulnerabilities include:
- Memory safety issues (buffer overflow, use-after-free, etc.) in `libhelix`.
- Issues where crafted model files or JSON inputs cause heap corruption or
  arbitrary code execution in the caller's process.
- Symbol-leakage or ABI-surface issues that expose internal llama.cpp state
  to callers in unexpected ways.

Out-of-scope:
- Vulnerabilities in the underlying model weights (those are the model
  author's responsibility).
- Performance issues or crashes caused by running out of VRAM/RAM with
  valid but very large models.
- `helix-shim-server` (a test tool, not a supported API surface).

## Hardening notes for hosts that accept untrusted requests

Helix runs inside your process and trusts the request JSON it is given. If the
messages come from end users, keep the following in mind.

### Special tokens in message content

Helix renders the whole conversation through the model's chat template and
tokenizes the result as one string with special-token parsing on, matching
llama.cpp's `llama-server`. Once rendered, user text can no longer be told
apart from the template's own markup, so control-token text inside a message
(for example `<|im_end|>`, `<|im_start|>system`, `<|eot_id|>`) is tokenized
as the real control token rather than as plain text. A user can use this to
close their own turn early and forge system, assistant or tool turns.

This is a prompt-injection risk, not a memory-safety issue. If your host
forwards untrusted text, strip or escape the model's special-token strings
from user-supplied content before building the request. The model's special
tokens are listed in its GGUF tokenizer metadata. To check a string, call
`helix_tokenize` on it twice, once with `parse_special` set to 1 and once
with it set to 0; if the first call returns fewer tokens, the string contains
special-token text.

### `file://` image URLs

Image parts may reference a local file with a `file:///absolute/path` URL.
Without `HELIX_FILE_URI_ROOT` set, any file the process can read may be
referenced. Set `HELIX_FILE_URI_ROOT` to the directory images may be read
from, or reject `file://` URLs in your host before calling Helix. Files are
only read for sessions whose model was loaded with an `mmproj_path`; every
failure to read one is reported with the same message, so error responses do
not reveal whether a path exists.
