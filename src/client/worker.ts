import type { Difficulty, Grid } from "./types";
import { generatePuzzle } from "./game";

type PuzzleWorkerMessage = {
  id: number;
  difficulty: Difficulty;
};

type PuzzleWorkerResponse = {
  id: number;
  difficulty: Difficulty;
  puzzle: Grid;
  solution: Grid;
};

declare const self: Worker;

self.addEventListener("message", (event: MessageEvent<PuzzleWorkerMessage>) => {
  const { id, difficulty } = event.data;
  const { puzzle, solution } = generatePuzzle(difficulty);
  const response: PuzzleWorkerResponse = { id, difficulty, puzzle, solution };
  self.postMessage(response);
});
