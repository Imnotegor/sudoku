import type { Difficulty, Grid, NotesGrid, SavedGameState } from "./types";
import { STORAGE_KEY } from "./constants";
import { restoreNotes } from "./game";

/** Save game state to localStorage */
export function saveGame(state: {
  puzzle: Grid;
  solution: Grid | null;
  current: Grid;
  notes: NotesGrid;
  difficulty: Difficulty;
  elapsedMs: number;
}): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (err) {
    console.warn("Failed to save game state to localStorage", err);
  }
}

/** Load game state from localStorage */
export function loadGame(): SavedGameState | null {
  try {
    const savedRaw = localStorage.getItem(STORAGE_KEY);
    if (!savedRaw) return null;

    const saved = JSON.parse(savedRaw) as Partial<SavedGameState>;

    // Validate structure
    if (!saved || !Array.isArray(saved.puzzle) || saved.puzzle.length !== 9) {
      return null;
    }

    return {
      puzzle: saved.puzzle as Grid,
      solution: saved.solution ?? null,
      current: saved.current as Grid,
      notes: restoreNotes(saved.notes),
      difficulty: saved.difficulty ?? "easy",
      elapsedMs: saved.elapsedMs ?? 0,
    };
  } catch (err) {
    console.warn("Failed to load game state from localStorage", err);
    return null;
  }
}

/** Clear saved game from localStorage */
export function clearSavedGame(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (err) {
    console.warn("Failed to clear saved game state", err);
  }
}
