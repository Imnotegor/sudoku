//! Sudoku grid type and operations.

mod parse;
mod validate;

pub use parse::{parse_grid_lenient, parse_grid_strict};
pub use validate::{clear_conflicts, is_grid_valid, is_safe};

/// Grid dimensions.
pub const GRID_SIZE: usize = 9;

/// A 9x9 Sudoku grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid(pub [[u8; GRID_SIZE]; GRID_SIZE]);

impl Grid {
    /// Creates a new empty grid.
    pub fn new() -> Self {
        Self([[0; GRID_SIZE]; GRID_SIZE])
    }

    /// Converts the grid to a Vec<Vec<u8>> for JSON serialization.
    pub fn to_vec(&self) -> Vec<Vec<u8>> {
        self.0.iter().map(|row| row.to_vec()).collect()
    }
}

impl Default for Grid {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Index<usize> for Grid {
    type Output = [u8; GRID_SIZE];

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Grid {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Converts a Grid to a Vec<Vec<u8>> for JSON serialization.
pub fn grid_to_vec(grid: &Grid) -> Vec<Vec<u8>> {
    grid.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_to_vec() {
        let grid = Grid([[1; 9]; 9]);
        let vec = grid_to_vec(&grid);
        assert_eq!(vec.len(), 9);
        assert_eq!(vec[0].len(), 9);
        assert_eq!(vec[0][0], 1);
    }
}
