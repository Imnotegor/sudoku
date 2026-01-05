//! Solver strategies and heuristics.

use crate::grid::{is_safe, Grid, GRID_SIZE};

/// Candidate information for an empty cell.
#[derive(Debug)]
pub struct CellCandidate {
    pub row: usize,
    pub col: usize,
    pub candidates: Vec<u8>,
}

/// Finds the best empty cell using MRV (Minimum Remaining Values) heuristic.
///
/// Returns the cell with the fewest valid candidates, or `None` if no empty cells.
pub fn find_best_empty(grid: &Grid) -> Option<CellCandidate> {
    let mut best: Option<CellCandidate> = None;

    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            if grid[row][col] != 0 {
                continue;
            }

            let candidates: Vec<u8> = (1..=9)
                .filter(|&value| is_safe(grid, row, col, value))
                .collect();

            if candidates.is_empty() {
                return Some(CellCandidate {
                    row,
                    col,
                    candidates,
                });
            }

            match &best {
                None => {
                    best = Some(CellCandidate {
                        row,
                        col,
                        candidates,
                    })
                }
                Some(existing) if candidates.len() < existing.candidates.len() => {
                    best = Some(CellCandidate {
                        row,
                        col,
                        candidates,
                    })
                }
                _ => {}
            }
        }
    }

    best
}
