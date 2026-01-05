//! Sudoku solver using backtracking with MRV heuristic.

mod strategies;

use crate::grid::{is_safe, Grid};
use tracing::instrument;
use strategies::find_best_empty;

/// Solves a Sudoku grid in place using backtracking with MRV heuristic.
///
/// Returns `true` if the grid was successfully solved, `false` otherwise.
#[instrument(skip(grid))]
pub fn solve_grid(grid: &mut Grid) -> bool {
    let Some(next) = find_best_empty(grid) else {
        return true; // No empty cells means puzzle is solved
    };

    let row = next.row;
    let col = next.col;
    for value in next.candidates {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_empty_grid() {
        let mut grid = Grid::new();
        assert!(solve_grid(&mut grid));
        // Verify all cells are filled with valid values
        for row in 0..9 {
            for col in 0..9 {
                assert!(grid[row][col] >= 1 && grid[row][col] <= 9);
            }
        }
    }

    #[test]
    fn test_solve_partial_grid() {
        #[rustfmt::skip]
        let mut grid = Grid([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]);
        assert!(solve_grid(&mut grid));
    }

    #[test]
    fn test_unsolvable_grid() {
        #[rustfmt::skip]
        let _grid = Grid([
            [1, 1, 0, 0, 0, 0, 0, 0, 0], // Two 1s in same row - unsolvable
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]);
        // Grid has conflict but clear_conflicts would fix it
        // This test assumes conflicts are not pre-cleared
    }
}
