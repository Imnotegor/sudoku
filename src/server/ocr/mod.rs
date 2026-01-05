//! OCR client for communicating with the OCR service.

mod client;

pub use client::request_ocr;

#[cfg(test)]
pub use client::set_mock_ocr_response;
