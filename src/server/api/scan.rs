//! Handler for POST /api/scan

use std::sync::Arc;

use axum::{
    extract::{Multipart, State},
    http::{header::CONTENT_LENGTH, HeaderMap},
    Json,
};

use super::ScanResponse;
use crate::{
    config::MAX_UPLOAD_BYTES,
    error::ApiError,
    grid::{clear_conflicts, grid_to_vec, parse_grid_lenient, GRID_SIZE},
    ocr::request_ocr,
    solver::solve_grid,
    AppState,
};

/// Handles the scan request.
///
/// Takes an uploaded image, sends it to the OCR service, and returns
/// both the recognized puzzle and its solution.
#[utoipa::path(
    post,
    path = "/api/scan",
    request_body(
        content = super::ScanRequest,
        content_type = "multipart/form-data"
    ),
    responses(
        (status = 200, description = "Puzzle and solution", body = ScanResponse),
        (status = 400, description = "Bad request", body = super::ErrorResponse),
        (status = 502, description = "OCR error", body = super::ErrorResponse)
    ),
    tag = "Sudoku"
)]
pub async fn handle_scan(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart_data: Multipart,
) -> Result<Json<ScanResponse>, ApiError> {
    if let Some(length) = headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
    {
        if length > MAX_UPLOAD_BYTES {
            return Err(ApiError::payload_too_large("Image too large"));
        }
    }

    // Extract image from multipart form
    let mut image_bytes = None;
    while let Some(field) = multipart_data
        .next_field()
        .await
        .map_err(|err| ApiError::bad_request(err.to_string()))?
    {
        if field.name() == Some("image") {
            let bytes = field
                .bytes()
                .await
                .map_err(|err| ApiError::bad_request(err.to_string()))?;
            if bytes.len() > MAX_UPLOAD_BYTES {
                return Err(ApiError::payload_too_large("Image too large"));
            }
            image_bytes = Some(bytes);
            break;
        }
    }

    let image_bytes = image_bytes.ok_or_else(|| ApiError::bad_request("Image is required"))?;

    // Send to OCR service
    let ocr_grid = request_ocr(&state.ocr_url, image_bytes.to_vec()).await?;

    // Parse and clean up the grid
    if ocr_grid.len() != GRID_SIZE || !ocr_grid.iter().all(|row| row.len() == GRID_SIZE) {
        return Err(ApiError::bad_request("Invalid grid dimensions"));
    }
    let mut grid = parse_grid_lenient(ocr_grid)?;
    clear_conflicts(&mut grid);

    let puzzle = grid_to_vec(&grid);

    // Solve the grid
    let mut solved = grid;
    if !solve_grid(&mut solved) {
        return Err(ApiError::bad_request("Unsolvable grid"));
    }

    Ok(Json(ScanResponse {
        puzzle,
        solution: grid_to_vec(&solved),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::post,
        Router,
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use crate::ocr::set_mock_ocr_response;

    use crate::config::MAX_UPLOAD_BYTES;

    fn multipart_body(bytes: &[u8]) -> (String, Vec<u8>) {
        let boundary = "sudoku-boundary";
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"image\"; filename=\"scan.jpg\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: image/jpeg\r\n\r\n");
        body.extend_from_slice(bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        (format!("multipart/form-data; boundary={boundary}"), body)
    }

    #[tokio::test]
    async fn test_scan_returns_puzzle_and_solution() {
        let grid = vec![vec![0u8; GRID_SIZE]; GRID_SIZE];
        let _guard = set_mock_ocr_response(grid);

        let state = Arc::new(AppState {
            ocr_url: "mock://scan".to_string(),
            static_dir: ".".to_string(),
        });

        let app = Router::new()
            .route("/api/scan", post(handle_scan))
            .with_state(state);

        let (content_type, body) = multipart_body(&[1, 2, 3, 4]);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/scan")
                    .header("content-type", content_type)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(status, StatusCode::OK, "body: {}", String::from_utf8_lossy(&body));
        let json_body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json_body.get("puzzle").is_some());
        assert!(json_body.get("solution").is_some());
    }

    #[tokio::test]
    async fn test_scan_rejects_invalid_grid_dimensions() {
        let grid = vec![vec![1u8; 3]; 3];
        let _guard = set_mock_ocr_response(grid);

        let state = Arc::new(AppState {
            ocr_url: "mock://scan".to_string(),
            static_dir: ".".to_string(),
        });

        let app = Router::new()
            .route("/api/scan", post(handle_scan))
            .with_state(state);

        let (content_type, body) = multipart_body(&[5, 6, 7]);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/scan")
                    .header("content-type", content_type)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "body: {}",
            String::from_utf8_lossy(&body)
        );
    }

    #[tokio::test]
    async fn test_scan_rejects_oversized_upload() {
        let state = Arc::new(AppState {
            ocr_url: "http://invalid/scan".to_string(),
            static_dir: ".".to_string(),
        });

        let app = Router::new()
            .route("/api/scan", post(handle_scan))
            .with_state(state);

        let oversized = vec![0u8; MAX_UPLOAD_BYTES + 1];
        let (content_type, body) = multipart_body(&oversized);
        let content_length = body.len().to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/scan")
                    .header("content-type", content_type)
                    .header("content-length", content_length)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "body: {}",
            String::from_utf8_lossy(&body)
        );
    }

}
