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
