//! Handler for GET /api/health

use axum::Json;

use super::HealthResponse;

/// Health check endpoint.
#[utoipa::path(
    get,
    path = "/api/health",
    responses(
        (status = 200, description = "Service health", body = HealthResponse)
    ),
    tag = "Sudoku"
)]
pub async fn handle_health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_ok() {
        let response = handle_health().await;
        assert_eq!(response.status, "ok");
    }
}
