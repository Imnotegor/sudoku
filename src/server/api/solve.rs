//! Handler for POST /api/solve

use axum::Json;

use super::{SolveRequest, SolveResponse};
use crate::{
    error::ApiError,
    grid::{grid_to_vec, parse_grid_strict},
    solver::solve_grid,
};

/// Handles the solve request.
///
/// Takes a partially filled Sudoku grid and returns the complete solution.
#[utoipa::path(
    post,
    path = "/api/solve",
    request_body = SolveRequest,
    responses(
        (status = 200, description = "Solved grid", body = SolveResponse),
        (status = 400, description = "Bad request", body = super::ErrorResponse)
    ),
    tag = "Sudoku"
)]
pub async fn handle_solve(Json(payload): Json<SolveRequest>) -> Result<Json<SolveResponse>, ApiError> {
    let mut grid = parse_grid_strict(payload.grid)?;

    if !solve_grid(&mut grid) {
        return Err(ApiError::bad_request("Unsolvable grid"));
    }

    Ok(Json(SolveResponse {
        solution: grid_to_vec(&grid),
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
    use serde_json::json;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_solve_valid_puzzle() {
        let app = Router::new().route("/api/solve", post(handle_solve));

        let payload = json!({
            "grid": [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9],
            ]
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/solve")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json_body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json_body.get("solution").is_some());
    }

    #[tokio::test]
    async fn test_solve_invalid_grid() {
        let app = Router::new().route("/api/solve", post(handle_solve));

        let payload = json!({ "grid": [[1, 2, 3]] });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/solve")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
