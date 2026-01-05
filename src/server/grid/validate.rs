//! Grid validation functions.

use super::{Grid, GRID_SIZE};

/// Checks if a value can be placed at the given position.
pub fn is_safe(grid: &Grid, row: usize, col: usize, value: u8) -> bool {
    // Check row
    for idx in 0..GRID_SIZE {
        if idx != col && grid[row][idx] == value {
            return false;
        }
    }

    // Check column
    for idx in 0..GRID_SIZE {
        if idx != row && grid[idx][col] == value {
            return false;
        }
    }

    // Check 3x3 box
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

/// Checks if the entire grid is valid (no conflicts).
pub fn is_grid_valid(grid: &Grid) -> bool {
    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            let value = grid[row][col];
            if value != 0 && !is_safe(grid, row, col, value) {
                return false;
            }
        }
    }
    true
}

/// Clears all conflicting cells in the grid.
pub fn clear_conflicts(grid: &mut Grid) {
    let mut conflicts = [[false; GRID_SIZE]; GRID_SIZE];

    // Check rows
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

    // Check columns
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

    // Check 3x3 boxes
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

    // Clear conflicting cells
    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            if conflicts[row][col] {
                grid[row][col] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_safe_empty_grid() {
        let grid = Grid::new();
        assert!(is_safe(&grid, 0, 0, 1));
    }

    #[test]
    fn test_is_safe_row_conflict() {
        let mut grid = Grid::new();
        grid[0][1] = 1;
        assert!(!is_safe(&grid, 0, 0, 1));
    }

    #[test]
    fn test_is_safe_col_conflict() {
        let mut grid = Grid::new();
        grid[1][0] = 1;
        assert!(!is_safe(&grid, 0, 0, 1));
    }

    #[test]
    fn test_is_safe_box_conflict() {
        let mut grid = Grid::new();
        grid[1][1] = 1;
        assert!(!is_safe(&grid, 0, 0, 1));
    }

    #[test]
    fn test_clear_conflicts() {
        let mut grid = Grid::new();
        grid[0][0] = 1;
        grid[0][1] = 1; // Conflict
        clear_conflicts(&mut grid);
        assert_eq!(grid[0][0], 0);
        assert_eq!(grid[0][1], 0);
    }
}
