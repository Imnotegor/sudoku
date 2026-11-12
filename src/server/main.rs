//! Sudoku server entry point.
//!
//! This is the main entry point for the Sudoku web application server.
//! It sets up the Axum router with API endpoints and static file serving.

use std::{env, sync::Arc};

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::{cors::CorsLayer, services::ServeDir};
use tracing::info;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

mod api;
mod config;
mod city;
mod error;
mod grid;
mod ocr;
mod solver;

use api::{
    handle_health, handle_scan, handle_solve, ErrorResponse, HealthResponse, ScanRequest,
    ScanResponse, SolveRequest, SolveResponse,
};
use config::{MAX_UPLOAD_BYTES, RATE_LIMIT_BURST, RATE_LIMIT_PER_SECOND};

/// Application state shared across request handlers.
#[derive(Clone)]
pub struct AppState {
    /// URL of the OCR service.
    pub ocr_url: String,
    /// Path to static files directory.
    pub static_dir: String,
}

#[derive(OpenApi)]
#[openapi(
    paths(api::solve::handle_solve, api::scan::handle_scan, api::health::handle_health),
    components(schemas(
        SolveRequest,
        SolveResponse,
        ScanResponse,
        HealthResponse,
        ErrorResponse,
        ScanRequest
    )),
    tags((name = "Sudoku", description = "Sudoku API"))
)]
struct ApiDoc;

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("sudoku=info,axum=info")
        .init();

    // Load configuration from environment
    let ocr_url = env::var("OCR_URL").unwrap_or_else(|_| "http://ocr:8000/scan".to_string());
    let static_dir = resolve_static_dir();

    let state = Arc::new(AppState {
        ocr_url,
        static_dir: static_dir.clone(),
    });

    // Build API routes
    let api = Router::new()
        .route("/api/health", get(handle_health))
        .route("/api/solve", post(handle_solve))
        .route("/api/scan", post(handle_scan))
        .with_state(state.clone());

    let swagger = Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()))
        .with_state(state.clone());

    // Build full application with Swagger UI and static files
    let governor_config = GovernorConfigBuilder::default()
        .per_second(RATE_LIMIT_PER_SECOND)
        .burst_size(RATE_LIMIT_BURST)
        .finish()
        .expect("Failed to build rate limit configuration");

    let app = api
        .merge(swagger)
        .layer(GovernorLayer {
            config: Arc::new(governor_config),
        })
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .nest_service("/", ServeDir::new(static_dir));

    let app = match env::var("DATABASE_URL") {
        Ok(url) => app.nest("/api/city", city::router(&url).await.expect("initialize city database")),
        Err(_) => app,
    };

    // Bind and serve
    let addr = env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap_or_else(|err| {
        tracing::error!("Failed to bind {addr}: {err}");
        std::process::exit(1);
    });

    info!("listening on http://{addr}");
    if let Err(err) = axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        tracing::error!("Server error: {err}");
        std::process::exit(1);
    }
}

/// Resolves the static files directory from environment or defaults.
fn resolve_static_dir() -> String {
    if let Ok(dir) = env::var("STATIC_DIR") {
        return dir;
    }
    if std::path::Path::new("./static").exists() {
        return "./static".to_string();
    }
    ".".to_string()
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::error!("Failed to install CTRL+C handler: {err}");
    }
    tracing::info!("Shutting down gracefully...");
}
