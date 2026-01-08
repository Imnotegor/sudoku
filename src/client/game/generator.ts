import type { Difficulty, Grid } from "../types";
import { DIFFICULTY_BLANKS, GRID_SIZE } from "../constants";
import { createEmptyGrid, copyGrid } from "./grid";
import { solveGrid, countSolutions } from "./solver";
import { shuffle } from "./utils";

/** Generates a fully solved Sudoku grid */
function generateSolvedGrid(): Grid {
  const grid = createEmptyGrid();
  solveGrid(grid);
  return grid;
}

export interface GeneratedPuzzle {
  puzzle: Grid;
  solution: Grid;
}

/** Generates a Sudoku puzzle with a unique solution for the given difficulty */
export function generatePuzzle(level: Difficulty): GeneratedPuzzle {
  const blanks = level === "manual" ? 0 : DIFFICULTY_BLANKS[level];
  const solutionGrid = generateSolvedGrid();
  const puzzleGrid = copyGrid(solutionGrid);

  // Positions to potentially remove (0-80)
  const positions = shuffle([...Array(81).keys()]);
  let removed = 0;

  // First pass: try to remove cells in random order
  for (const pos of positions) {
    if (removed >= blanks) break;
    const row = Math.floor(pos / GRID_SIZE);
    const col = pos % GRID_SIZE;
    if (puzzleGrid[row][col] === 0) continue;

    const backup = puzzleGrid[row][col];
    puzzleGrid[row][col] = 0;

    // Check if puzzle still has a unique solution
    if (countSolutions(puzzleGrid, 2) !== 1) {
      puzzleGrid[row][col] = backup;
    } else {
      removed++;
    }
  }

  // Second pass: try harder if we haven't removed enough cells
  let attempts = 0;
  const maxAttempts = 1200;
  while (removed < blanks && attempts < maxAttempts) {
    const pos = Math.floor(Math.random() * 81);
    const row = Math.floor(pos / GRID_SIZE);
    const col = pos % GRID_SIZE;

    if (puzzleGrid[row][col] === 0) {
      attempts++;
      continue;
    }

    const backup = puzzleGrid[row][col];
    puzzleGrid[row][col] = 0;

    if (countSolutions(puzzleGrid, 2) !== 1) {
      puzzleGrid[row][col] = backup;
    } else {
      removed++;
    }
    attempts++;
  }

  return { puzzle: puzzleGrid, solution: solutionGrid };
}
