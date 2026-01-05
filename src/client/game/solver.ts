import type { Grid } from "../types";
import { GRID_SIZE } from "../constants";
import { isSafe } from "./validation";

interface CellCandidate {
  row: number;
  col: number;
  candidates: number[];
}

/** Shuffles an array in place using Fisher-Yates algorithm */
function shuffle<T>(array: T[]): T[] {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

/** Finds the best empty cell using MRV (Minimum Remaining Values) heuristic */
function findBestEmpty(grid: Grid): CellCandidate | null {
  let best: CellCandidate | null = null;
  let bestCount = 10;

  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      if (grid[row][col] !== 0) continue;

      const candidates: number[] = [];
      for (let num = 1; num <= 9; num++) {
        if (isSafe(grid, row, col, num)) {
          candidates.push(num);
        }
      }

      // If no candidates, this cell is unsolvable
      if (candidates.length === 0) {
        return { row, col, candidates };
      }

      if (candidates.length < bestCount) {
        best = { row, col, candidates };
        bestCount = candidates.length;
        // Can't do better than 1 candidate
        if (bestCount === 1) return best;
      }
    }
  }

  return best;
}

/** Solves a Sudoku grid in place using backtracking with MRV heuristic */
export function solveGrid(grid: Grid): boolean {
  const next = findBestEmpty(grid);
  if (!next) return true; // No empty cells = solved

  const { row, col, candidates } = next;
  const order = shuffle([...candidates]);

  for (const num of order) {
    if (isSafe(grid, row, col, num)) {
      grid[row][col] = num;
      if (solveGrid(grid)) return true;
      grid[row][col] = 0;
    }
  }

  return false;
}

/** Counts the number of solutions for a grid (up to limit) */
export function countSolutions(grid: Grid, limit: number): number {
  let count = 0;

  function backtrack(): void {
    if (count >= limit) return;

    const next = findBestEmpty(grid);
    if (!next) {
      count++;
      return;
    }

    const { row, col, candidates } = next;
    for (const num of candidates) {
      if (isSafe(grid, row, col, num)) {
        grid[row][col] = num;
        backtrack();
        grid[row][col] = 0;
        if (count >= limit) return;
      }
    }
  }

  backtrack();
  return count;
}
