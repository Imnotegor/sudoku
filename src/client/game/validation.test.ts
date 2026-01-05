import { describe, expect, it } from "vitest";
import { createEmptyGrid } from "./grid";
import { findConflicts, isSafe } from "./validation";

describe("validation", () => {
  it("rejects row, column, and box conflicts", () => {
    const grid = createEmptyGrid();
    grid[0][1] = 5;
    grid[1][0] = 6;
    grid[1][1] = 7;

    expect(isSafe(grid, 0, 0, 5)).toBe(false);
    expect(isSafe(grid, 0, 0, 6)).toBe(false);
    expect(isSafe(grid, 0, 0, 7)).toBe(false);
    expect(isSafe(grid, 0, 0, 4)).toBe(true);
  });

  it("finds conflicts across row, column, and box", () => {
    const grid = createEmptyGrid();

    grid[0][0] = 4;
    grid[0][3] = 4;

    grid[2][1] = 7;
    grid[5][1] = 7;

    grid[1][2] = 9;
    grid[2][2] = 9;

    const conflicts = findConflicts(grid);
    const expected = [0, 3, 19, 46, 11, 20];

    expect(conflicts.size).toBe(expected.length);
    expected.forEach((index) => {
      expect(conflicts.has(index)).toBe(true);
    });
  });
});
