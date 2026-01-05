//! Shared server configuration constants.

/// Maximum upload size for scan images (10 MB).
pub const MAX_UPLOAD_BYTES: usize = 10 * 1024 * 1024;

/// Requests allowed per second before rate limiting.
pub const RATE_LIMIT_PER_SECOND: u64 = 10;

/// Burst capacity for rate limiting.
pub const RATE_LIMIT_BURST: u32 = 20;
