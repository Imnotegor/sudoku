//! API handlers for the Sudoku server.

pub mod health;
pub mod scan;
pub mod solve;

pub use health::handle_health;
pub use scan::handle_scan;
pub use solve::handle_solve;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Request body for the solve endpoint.
#[derive(Debug, Deserialize, ToSchema)]
pub struct SolveRequest {
    pub grid: Vec<Vec<u8>>,
}

/// Response for the solve endpoint.
#[derive(Debug, Serialize, ToSchema)]
pub struct SolveResponse {
    pub solution: Vec<Vec<u8>>,
}

/// Response for the scan endpoint.
#[derive(Debug, Serialize, ToSchema)]
pub struct ScanResponse {
    pub puzzle: Vec<Vec<u8>>,
    pub solution: Vec<Vec<u8>>,
}

/// Response for the health endpoint.
#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
}

/// Error response.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    pub error: String,
}

/// Request body for the scan endpoint (multipart form).
#[derive(Debug, Deserialize, ToSchema)]
pub struct ScanRequest {
    #[schema(format = "binary")]
    pub image: String,
}
