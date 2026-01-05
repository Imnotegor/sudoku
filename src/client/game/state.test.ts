import { describe, expect, it } from "vitest";
import type { Grid } from "../types";
import { GameState } from "./state";

const puzzle: Grid = [
  [5, 3, 0, 0, 7, 0, 0, 0, 0],
  [6, 0, 0, 1, 9, 5, 0, 0, 0],
  [0, 9, 8, 0, 0, 0, 0, 6, 0],
  [8, 0, 0, 0, 6, 0, 0, 0, 3],
  [4, 0, 0, 8, 0, 3, 0, 0, 1],
  [7, 0, 0, 0, 2, 0, 0, 0, 6],
  [0, 6, 0, 0, 0, 0, 2, 8, 0],
  [0, 0, 0, 4, 1, 9, 0, 0, 5],
  [0, 0, 0, 0, 8, 0, 0, 7, 9],
];

const solution: Grid = [
  [5, 3, 4, 6, 7, 8, 9, 1, 2],
  [6, 7, 2, 1, 9, 5, 3, 4, 8],
  [1, 9, 8, 3, 4, 2, 5, 6, 7],
  [8, 5, 9, 7, 6, 1, 4, 2, 3],
  [4, 2, 6, 8, 5, 3, 7, 9, 1],
  [7, 1, 3, 9, 2, 4, 8, 5, 6],
  [9, 6, 1, 5, 3, 7, 2, 8, 4],
  [2, 8, 7, 4, 1, 9, 6, 3, 5],
  [3, 4, 5, 2, 8, 6, 1, 7, 9],
];

describe("GameState.applyScannedGrid", () => {
  it("locks scanned givens in the puzzle grid", () => {
    const state = new GameState();
    state.applyScannedGrid(puzzle, solution);

    expect(state.puzzle).toEqual(puzzle);
    expect(state.current).toEqual(puzzle);
    expect(state.setCellValue(0, 0, 1)).toBe(false);
    expect(state.setCellValue(0, 2, 4)).toBe(true);
  });
});

describe("GameState.undo", () => {
  it("restores the previous cell state", () => {
    const state = new GameState();
    state.startManualEntry();

    expect(state.setCellValue(0, 0, 4)).toBe(true);
    expect(state.current[0][0]).toBe(4);

    expect(state.undo()).toBe(true);
    expect(state.current[0][0]).toBe(0);
  });
});
