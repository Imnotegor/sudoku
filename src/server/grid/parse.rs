//! Grid parsing functions.

use super::{is_grid_valid, Grid, GRID_SIZE};
use crate::error::ApiError;

/// Parses a grid with strict validation (checks for conflicts).
pub fn parse_grid_strict(grid: Vec<Vec<u8>>) -> Result<Grid, ApiError> {
    let result = parse_grid_inner(grid)?;
    if !is_grid_valid(&result) {
        return Err(ApiError::bad_request("Grid has conflicts"));
    }
    Ok(result)
}

/// Parses a grid without conflict validation (for OCR results).
pub fn parse_grid_lenient(grid: Vec<Vec<u8>>) -> Result<Grid, ApiError> {
    parse_grid_inner(grid)
}

fn parse_grid_inner(grid: Vec<Vec<u8>>) -> Result<Grid, ApiError> {
    if grid.len() != GRID_SIZE {
        return Err(ApiError::bad_request("Grid must have 9 rows"));
    }

    let mut result = Grid::new();

    for (row_idx, row) in grid.into_iter().enumerate() {
        if row.len() != GRID_SIZE {
            return Err(ApiError::bad_request("Grid must have 9 columns"));
        }
        for (col_idx, value) in row.into_iter().enumerate() {
            if value > 9 {
                return Err(ApiError::bad_request("Values must be 0-9"));
            }
            result[row_idx][col_idx] = value;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_grid() {
        let grid = vec![vec![0u8; 9]; 9];
        let result = parse_grid_strict(grid);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_invalid_row_count() {
        let grid = vec![vec![0u8; 9]; 8];
        let result = parse_grid_strict(grid);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_column_count() {
        let mut grid = vec![vec![0u8; 9]; 9];
        grid[0] = vec![0u8; 8];
        let result = parse_grid_strict(grid);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_value() {
        let mut grid = vec![vec![0u8; 9]; 9];
        grid[0][0] = 10;
        let result = parse_grid_strict(grid);
        assert!(result.is_err());
    }
}
