import type { Grid } from "../types";
import { GRID_SIZE } from "../constants";

/** Checks if placing a value at (row, col) is valid according to Sudoku rules */
export function isSafe(grid: Grid, row: number, col: number, num: number): boolean {
  // Check row
  for (let i = 0; i < GRID_SIZE; i++) {
    if (grid[row][i] === num) return false;
  }

  // Check column
  for (let i = 0; i < GRID_SIZE; i++) {
    if (grid[i][col] === num) return false;
  }

  // Check 3x3 box
  const boxRow = Math.floor(row / 3) * 3;
  const boxCol = Math.floor(col / 3) * 3;
  for (let r = boxRow; r < boxRow + 3; r++) {
    for (let c = boxCol; c < boxCol + 3; c++) {
      if (grid[r][c] === num) return false;
    }
  }

  return true;
}

/** Finds all cells with conflicting values. Returns a Set of cell indices (row * 9 + col). */
export function findConflicts(grid: Grid): Set<number> {
  const conflicts = new Set<number>();

  // Check rows
  for (let row = 0; row < GRID_SIZE; row++) {
    const buckets = new Map<number, number[]>();
    for (let col = 0; col < GRID_SIZE; col++) {
      const value = grid[row][col];
      if (value === 0) continue;
      const indices = buckets.get(value) ?? [];
      indices.push(row * GRID_SIZE + col);
      buckets.set(value, indices);
    }
    for (const indices of buckets.values()) {
      if (indices.length > 1) {
        indices.forEach((idx) => conflicts.add(idx));
      }
    }
  }

  // Check columns
  for (let col = 0; col < GRID_SIZE; col++) {
    const buckets = new Map<number, number[]>();
    for (let row = 0; row < GRID_SIZE; row++) {
      const value = grid[row][col];
      if (value === 0) continue;
      const indices = buckets.get(value) ?? [];
      indices.push(row * GRID_SIZE + col);
      buckets.set(value, indices);
    }
    for (const indices of buckets.values()) {
      if (indices.length > 1) {
        indices.forEach((idx) => conflicts.add(idx));
      }
    }
  }

  // Check 3x3 boxes
  for (let boxRow = 0; boxRow < 3; boxRow++) {
    for (let boxCol = 0; boxCol < 3; boxCol++) {
      const buckets = new Map<number, number[]>();
      for (let r = boxRow * 3; r < boxRow * 3 + 3; r++) {
        for (let c = boxCol * 3; c < boxCol * 3 + 3; c++) {
          const value = grid[r][c];
          if (value === 0) continue;
          const indices = buckets.get(value) ?? [];
          indices.push(r * GRID_SIZE + c);
          buckets.set(value, indices);
        }
      }
      for (const indices of buckets.values()) {
        if (indices.length > 1) {
          indices.forEach((idx) => conflicts.add(idx));
        }
      }
    }
  }

  return conflicts;
}

/** Finds all cells with incorrect values compared to the solution. Returns a Set of cell indices (row * 9 + col). */
export function findIncorrect(current: Grid, solution: Grid | null, puzzle: Grid): Set<number> {
  const incorrect = new Set<number>();

  if (!solution) return incorrect;

  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      const value = current[row][col];
      // Skip empty cells and fixed (puzzle) cells
      if (value === 0) continue;
      if (puzzle[row][col] !== 0) continue;

      // Check if value doesn't match solution
      if (value !== solution[row][col]) {
        incorrect.add(row * GRID_SIZE + col);
      }
    }
  }

  return incorrect;
}

/** Checks if two cells are related (same row or column) */
export function isRelatedCell(row: number, col: number, focusRow: number, focusCol: number): boolean {
  // Same row or column
  return row === focusRow || col === focusCol;
}
