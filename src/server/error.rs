//! API error types for the Sudoku server.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use thiserror::Error;

/// API error types with corresponding HTTP status codes.
#[derive(Debug, Error)]
pub enum ApiError {
    /// Client sent invalid data.
    #[error("{0}")]
    BadRequest(String),

    /// Upstream service (OCR) failed.
    #[error("{0}")]
    Upstream(String),

    /// Upload payload too large.
    #[error("{0}")]
    PayloadTooLarge(String),

    /// Internal server error.
    #[error("{0}")]
    Internal(String),
}

impl ApiError {
    /// Creates a bad request error.
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::BadRequest(msg.into())
    }

    /// Creates an upstream error.
    pub fn upstream(msg: impl Into<String>) -> Self {
        Self::Upstream(msg.into())
    }

    /// Creates an internal error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Creates a payload too large error.
    pub fn payload_too_large(msg: impl Into<String>) -> Self {
        Self::PayloadTooLarge(msg.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            ApiError::Upstream(msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
            ApiError::PayloadTooLarge(msg) => (StatusCode::PAYLOAD_TOO_LARGE, msg.clone()),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };
        let body = serde_json::json!({ "error": message });
        (status, Json(body)).into_response()
    }
}
