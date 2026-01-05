import { describe, expect, it } from "vitest";
import { DIFFICULTY_BLANKS } from "../constants";
import { generatePuzzle } from "./generator";

function countBlanks(grid: number[][]): number {
  return grid.reduce((total, row) => total + row.filter((value) => value === 0).length, 0);
}

describe("generator", () => {
  it("returns a solved grid for manual mode", () => {
    const { puzzle, solution } = generatePuzzle("manual");

    expect(countBlanks(puzzle)).toBe(0);
    expect(puzzle).toEqual(solution);
  });

  it("keeps puzzle values aligned with the solution", () => {
    const { puzzle, solution } = generatePuzzle("easy");
    const blanks = countBlanks(puzzle);

    expect(blanks).toBeLessThanOrEqual(DIFFICULTY_BLANKS.easy);

    puzzle.forEach((row, rowIndex) => {
      row.forEach((value, colIndex) => {
        if (value !== 0) {
          expect(value).toBe(solution[rowIndex][colIndex]);
        }
      });
    });
  });
});
