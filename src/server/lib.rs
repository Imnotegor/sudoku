//! Sudoku server library.
//!
//! This library provides the core functionality for the Sudoku web application,
//! including grid manipulation, solving, and API handlers.

pub mod api;
pub mod config;
pub mod error;
pub mod grid;
pub mod ocr;
pub mod solver;

/// Application state shared across request handlers.
#[derive(Clone)]
pub struct AppState {
    /// URL of the OCR service.
    pub ocr_url: String,
    /// Path to static files directory.
    pub static_dir: String,
}
