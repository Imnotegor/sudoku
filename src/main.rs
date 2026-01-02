use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use reqwest::multipart;
use serde::{Deserialize, Serialize};
use std::{env, sync::Arc};
use tower_http::services::ServeDir;
use tracing::{error, info};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

const GRID_SIZE: usize = 9;

type Grid = [[u8; GRID_SIZE]; GRID_SIZE];

#[derive(Clone)]
struct AppState {
    ocr_url: String,
    static_dir: String,
}

#[derive(Debug, Deserialize, ToSchema)]
struct SolveRequest {
    grid: Vec<Vec<u8>>,
}

#[derive(Debug, Serialize, ToSchema)]
struct SolveResponse {
    solution: Vec<Vec<u8>>,
}

#[derive(Debug, Serialize, ToSchema)]
struct ScanResponse {
    puzzle: Vec<Vec<u8>>,
    solution: Vec<Vec<u8>>,
}

#[derive(Debug, Deserialize)]
struct OcrResponse {
    grid: Vec<Vec<u8>>,
}

#[derive(Debug, Serialize, ToSchema)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Deserialize, ToSchema)]
struct ScanRequest {
    #[schema(format = "binary")]
    image: String,
}

#[derive(OpenApi)]
#[openapi(
    paths(handle_solve, handle_scan),
    components(schemas(SolveRequest, SolveResponse, ScanResponse, ErrorResponse, ScanRequest)),
    tags((name = "Sudoku", description = "Sudoku API"))
)]
struct ApiDoc;

#[derive(Debug)]
enum ApiError {
    BadRequest(String),
    Upstream(String),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Upstream(msg) => (StatusCode::BAD_GATEWAY, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        let body = serde_json::json!({ "error": message });
        (status, Json(body)).into_response()
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("sudoku=info,axum=info")
        .init();

    let ocr_url = env::var("OCR_URL").unwrap_or_else(|_| "http://ocr:8000/scan".to_string());
    let static_dir = resolve_static_dir();

    let state = Arc::new(AppState { ocr_url, static_dir });

    let api = Router::new()
        .route("/api/solve", post(handle_solve))
        .route("/api/scan", post(handle_scan))
        .with_state(state.clone());

    let app = api
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()))
        .nest_service("/", ServeDir::new(state.static_dir.clone()));

    let addr = env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|err| panic!("Failed to bind {addr}: {err}"));

    info!("listening on http://{addr}");
    axum::serve(listener, app).await.unwrap();
}

#[utoipa::path(
    post,
    path = "/api/solve",
    request_body = SolveRequest,
    responses(
        (status = 200, description = "Solved grid", body = SolveResponse),
        (status = 400, description = "Bad request", body = ErrorResponse)
    ),
    tag = "Sudoku"
)]
async fn handle_solve(Json(payload): Json<SolveRequest>) -> Result<Json<SolveResponse>, ApiError> {
    let mut grid = parse_grid_strict(payload.grid)?;
    if !solve_grid(&mut grid) {
        return Err(ApiError::BadRequest("Unsolvable grid".to_string()));
    }
    Ok(Json(SolveResponse {
        solution: grid_to_vec(grid),
    }))
}

#[utoipa::path(
    post,
    path = "/api/scan",
    request_body(
        content = ScanRequest,
        content_type = "multipart/form-data"
    ),
    responses(
        (status = 200, description = "Puzzle and solution", body = ScanResponse),
        (status = 400, description = "Bad request", body = ErrorResponse),
        (status = 502, description = "OCR error", body = ErrorResponse)
    ),
    tag = "Sudoku"
)]
async fn handle_scan(
    State(state): State<Arc<AppState>>,
    mut multipart_data: Multipart,
) -> Result<Json<ScanResponse>, ApiError> {
    let mut image_bytes = None;
    while let Some(field) = multipart_data
        .next_field()
        .await
        .map_err(|err| ApiError::BadRequest(err.to_string()))?
    {
        if field.name() == Some("image") {
            let bytes = field
                .bytes()
                .await
                .map_err(|err| ApiError::BadRequest(err.to_string()))?;
            image_bytes = Some(bytes);
            break;
        }
    }

    let image_bytes = image_bytes.ok_or_else(|| ApiError::BadRequest("Image is required".into()))?;
    let ocr_grid = request_ocr(&state.ocr_url, image_bytes.to_vec()).await?;
    let mut grid = parse_grid_lenient(ocr_grid)?;
    clear_conflicts(&mut grid);
    let puzzle = grid_to_vec(grid);
    let mut solved = grid;
    if !solve_grid(&mut solved) {
        return Err(ApiError::BadRequest("Unsolvable grid".to_string()));
    }

    Ok(Json(ScanResponse {
        puzzle,
        solution: grid_to_vec(solved),
    }))
}

async fn request_ocr(url: &str, image_bytes: Vec<u8>) -> Result<Vec<Vec<u8>>, ApiError> {
    let part = multipart::Part::bytes(image_bytes)
        .file_name("photo.jpg")
        .mime_str("image/jpeg")
        .map_err(|err| ApiError::Internal(err.to_string()))?;
    let form = multipart::Form::new().part("image", part);

    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .multipart(form)
        .send()
        .await
        .map_err(|err| ApiError::Upstream(err.to_string()))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        error!("OCR error {status}: {text}");
        return Err(ApiError::Upstream("OCR service failed".to_string()));
    }

    let payload = response
        .json::<OcrResponse>()
        .await
        .map_err(|err| ApiError::Upstream(err.to_string()))?;
    Ok(payload.grid)
}

fn resolve_static_dir() -> String {
    if let Ok(dir) = env::var("STATIC_DIR") {
        return dir;
    }
    if std::path::Path::new("./static").exists() {
        return "./static".to_string();
    }
    ".".to_string()
}

fn parse_grid_strict(grid: Vec<Vec<u8>>) -> Result<Grid, ApiError> {
    if grid.len() != GRID_SIZE {
        return Err(ApiError::BadRequest("Grid must have 9 rows".to_string()));
    }
    let mut result = [[0u8; GRID_SIZE]; GRID_SIZE];
    for (row_idx, row) in grid.into_iter().enumerate() {
        if row.len() != GRID_SIZE {
            return Err(ApiError::BadRequest("Grid must have 9 columns".to_string()));
        }
        for (col_idx, value) in row.into_iter().enumerate() {
            if value > 9 {
                return Err(ApiError::BadRequest("Values must be 0-9".to_string()));
            }
            result[row_idx][col_idx] = value;
        }
    }
    if !is_grid_valid(&result) {
        return Err(ApiError::BadRequest("Grid has conflicts".to_string()));
    }
    Ok(result)
}

fn parse_grid_lenient(grid: Vec<Vec<u8>>) -> Result<Grid, ApiError> {
    if grid.len() != GRID_SIZE {
        return Err(ApiError::BadRequest("Grid must have 9 rows".to_string()));
    }
    let mut result = [[0u8; GRID_SIZE]; GRID_SIZE];
    for (row_idx, row) in grid.into_iter().enumerate() {
        if row.len() != GRID_SIZE {
            return Err(ApiError::BadRequest("Grid must have 9 columns".to_string()));
        }
        for (col_idx, value) in row.into_iter().enumerate() {
            if value > 9 {
                return Err(ApiError::BadRequest("Values must be 0-9".to_string()));
            }
            result[row_idx][col_idx] = value;
        }
    }
    Ok(result)
}

fn grid_to_vec(grid: Grid) -> Vec<Vec<u8>> {
    grid.iter().map(|row| row.to_vec()).collect()
}

fn is_grid_valid(grid: &Grid) -> bool {
    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            let value = grid[row][col];
            if value == 0 {
                continue;
            }
            if !is_safe(grid, row, col, value) {
                return false;
            }
        }
    }
    true
}

fn clear_conflicts(grid: &mut Grid) {
    let mut conflicts = [[false; GRID_SIZE]; GRID_SIZE];

    for row in 0..GRID_SIZE {
        let mut positions: [Vec<usize>; 10] = std::array::from_fn(|_| Vec::new());
        for col in 0..GRID_SIZE {
            let value = grid[row][col];
            if value > 0 {
                positions[value as usize].push(col);
            }
        }
        for cols in positions.iter().skip(1) {
            if cols.len() > 1 {
                for &col in cols {
                    conflicts[row][col] = true;
                }
            }
        }
    }

    for col in 0..GRID_SIZE {
        let mut positions: [Vec<usize>; 10] = std::array::from_fn(|_| Vec::new());
        for row in 0..GRID_SIZE {
            let value = grid[row][col];
            if value > 0 {
                positions[value as usize].push(row);
            }
        }
        for rows in positions.iter().skip(1) {
            if rows.len() > 1 {
                for &row in rows {
                    conflicts[row][col] = true;
                }
            }
        }
    }

    for box_row in 0..3 {
        for box_col in 0..3 {
            let mut positions: [Vec<(usize, usize)>; 10] = std::array::from_fn(|_| Vec::new());
            for row in (box_row * 3)..(box_row * 3 + 3) {
                for col in (box_col * 3)..(box_col * 3 + 3) {
                    let value = grid[row][col];
                    if value > 0 {
                        positions[value as usize].push((row, col));
                    }
                }
            }
            for cells in positions.iter().skip(1) {
                if cells.len() > 1 {
                    for &(row, col) in cells {
                        conflicts[row][col] = true;
                    }
                }
            }
        }
    }

    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            if conflicts[row][col] {
                grid[row][col] = 0;
            }
        }
    }
}

fn is_safe(grid: &Grid, row: usize, col: usize, value: u8) -> bool {
    for idx in 0..GRID_SIZE {
        if idx != col && grid[row][idx] == value {
            return false;
        }
        if idx != row && grid[idx][col] == value {
            return false;
        }
    }
    let box_row = (row / 3) * 3;
    let box_col = (col / 3) * 3;
    for r in box_row..box_row + 3 {
        for c in box_col..box_col + 3 {
            if (r != row || c != col) && grid[r][c] == value {
                return false;
            }
        }
    }
    true
}

fn solve_grid(grid: &mut Grid) -> bool {
    let next = find_best_empty(grid);
    let Some((row, col, candidates)) = next else {
        return true;
    };

    for value in candidates {
        if is_safe(grid, row, col, value) {
            grid[row][col] = value;
            if solve_grid(grid) {
                return true;
            }
            grid[row][col] = 0;
        }
    }
    false
}

fn find_best_empty(grid: &Grid) -> Option<(usize, usize, Vec<u8>)> {
    let mut best: Option<(usize, usize, Vec<u8>)> = None;
    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            if grid[row][col] != 0 {
                continue;
            }
            let candidates = (1..=9)
                .filter(|&value| is_safe(grid, row, col, value))
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                return Some((row, col, candidates));
            }
            match &best {
                None => best = Some((row, col, candidates)),
                Some((_, _, best_candidates)) => {
                    if candidates.len() < best_candidates.len() {
                        best = Some((row, col, candidates));
                    }
                }
            }
        }
    }
    best
}
