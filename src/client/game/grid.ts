import type { Grid, GridRow, NotesGrid } from "../types";
import { GRID_SIZE } from "../constants";

/** Creates an empty 9x9 grid filled with zeros */
export function createEmptyGrid(): Grid {
  return Array.from({ length: GRID_SIZE }, () =>
    Array(GRID_SIZE).fill(0)
  ) as Grid;
}

/** Creates a deep copy of a grid */
export function copyGrid(grid: Grid): Grid {
  return grid.map((row) => [...row] as GridRow) as Grid;
}

/** Creates an empty notes grid (9x9 with zeros) */
export function createEmptyNotes(): NotesGrid {
  return Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
}

/** Creates a deep copy of a notes grid */
export function copyNotes(notes: NotesGrid): NotesGrid {
  return notes.map((row) => [...row]);
}

/** Restores notes from saved data with validation */
export function restoreNotes(saved: unknown): NotesGrid {
  if (!Array.isArray(saved) || saved.length !== GRID_SIZE) {
    return createEmptyNotes();
  }
  return saved.map((row) => {
    if (!Array.isArray(row) || row.length !== GRID_SIZE) {
      return Array(GRID_SIZE).fill(0);
    }
    return row.map((value) => Number(value) || 0);
  });
}

/** Checks if the grid contains any non-zero values */
export function hasAnyDigits(grid: Grid): boolean {
  return grid.some((row) => row.some((value) => value !== 0));
}

/** Formats a grid as a string (81 characters, dots for empty cells) */
export function formatGrid(grid: Grid): string {
  return grid
    .map((row) =>
      row
        .map((value) => (Number.isInteger(value) && value > 0 ? String(value) : "."))
        .join("")
    )
    .join("\n");
}

/** Parses a manual text input into a grid */
export function parseManualText(text: string): Grid {
  const tokens = String(text).match(/[0-9.]/g) || [];
  if (tokens.length !== 81) {
    throw new Error("Invalid manual input: need exactly 81 characters");
  }
  const grid = createEmptyGrid();
  tokens.forEach((char, index) => {
    const value = char === "." ? 0 : Number(char);
    if (!Number.isInteger(value) || value < 0 || value > 9) {
      throw new Error("Invalid manual input: values must be 0-9 or .");
    }
    const row = Math.floor(index / 9);
    const col = index % 9;
    grid[row][col] = value;
  });
  return grid;
}

/** Sanitizes a grid by clamping values to 0-9 */
export function sanitizeGrid(grid: Grid): Grid {
  return grid.map((row) =>
    row.map((value) => {
      const num = Number(value);
      if (Number.isInteger(num) && num >= 0 && num <= 9) return num;
      return 0;
    }) as GridRow
  ) as Grid;
}

/** Parses and validates a solved grid from API response */
export function parseSolvedGrid(grid: unknown): Grid {
  if (!Array.isArray(grid) || grid.length !== GRID_SIZE) {
    throw new Error("Invalid grid: must be 9x9 array");
  }
  return grid.map((row) => {
    if (!Array.isArray(row) || row.length !== GRID_SIZE) {
      throw new Error("Invalid grid row: must have 9 elements");
    }
    return row.map((value) => {
      const num = Number(value);
      if (!Number.isInteger(num) || num < 1 || num > 9) {
        throw new Error("Invalid value: solved grid values must be 1-9");
      }
      return num;
    }) as GridRow;
  }) as Grid;
}

/** Parses and validates a puzzle grid from API response (allows zeros) */
export function parsePuzzleGrid(grid: unknown): Grid {
  if (!Array.isArray(grid) || grid.length !== GRID_SIZE) {
    throw new Error("Invalid grid: must be 9x9 array");
  }
  return grid.map((row) => {
    if (!Array.isArray(row) || row.length !== GRID_SIZE) {
      throw new Error("Invalid grid row: must have 9 elements");
    }
    return row.map((value) => {
      const num = Number(value);
      if (!Number.isInteger(num) || num < 0 || num > 9) {
        throw new Error("Invalid value: puzzle values must be 0-9");
      }
      return num;
    }) as GridRow;
  }) as Grid;
}
