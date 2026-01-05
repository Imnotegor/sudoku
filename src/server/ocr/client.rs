//! HTTP client for the OCR service.

use crate::error::ApiError;

#[cfg(not(test))]
use std::time::Duration;
#[cfg(not(test))]
use reqwest::multipart;
#[cfg(not(test))]
use serde::Deserialize;
#[cfg(not(test))]
use tracing::error;

#[cfg(not(test))]
#[derive(Debug, Deserialize)]
struct OcrResponse {
    grid: Vec<Vec<u8>>,
}

/// Sends an image to the OCR service and returns the recognized grid.
#[cfg(not(test))]
pub async fn request_ocr(url: &str, image_bytes: Vec<u8>) -> Result<Vec<Vec<u8>>, ApiError> {
    let part = multipart::Part::bytes(image_bytes)
        .file_name("photo.jpg")
        .mime_str("image/jpeg")
        .map_err(|err| ApiError::internal(err.to_string()))?;

    let form = multipart::Form::new().part("image", part);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .http1_only()
        .build()
        .map_err(|err| ApiError::internal(err.to_string()))?;
    let response = client
        .post(url)
        .multipart(form)
        .send()
        .await
        .map_err(|err| {
            error!("OCR request failed: {err:?}");
            ApiError::upstream(err.to_string())
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        error!("OCR error {status}: {text}");
        return Err(ApiError::upstream("OCR service failed"));
    }

    let payload = response
        .json::<OcrResponse>()
        .await
        .map_err(|err| ApiError::upstream(err.to_string()))?;

    Ok(payload.grid)
}

#[cfg(test)]
static MOCK_OCR_RESPONSE: std::sync::Mutex<Option<Vec<Vec<u8>>>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
static MOCK_OCR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
#[allow(dead_code)]
pub struct MockOcrGuard(std::sync::MutexGuard<'static, ()>);

#[cfg(test)]
impl Drop for MockOcrGuard {
    fn drop(&mut self) {
        *MOCK_OCR_RESPONSE.lock().unwrap() = None;
    }
}

#[cfg(test)]
pub fn set_mock_ocr_response(grid: Vec<Vec<u8>>) -> MockOcrGuard {
    let guard = MOCK_OCR_LOCK.lock().unwrap();
    *MOCK_OCR_RESPONSE.lock().unwrap() = Some(grid);
    MockOcrGuard(guard)
}

#[cfg(test)]
pub async fn request_ocr(_url: &str, _image_bytes: Vec<u8>) -> Result<Vec<Vec<u8>>, ApiError> {
    let response = MOCK_OCR_RESPONSE.lock().unwrap().clone();
    response.ok_or_else(|| ApiError::upstream("Mock OCR response not set"))
}
