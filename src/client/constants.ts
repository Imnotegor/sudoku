import type { Difficulty } from "./types";

/** Number of blank cells for each difficulty level */
export const DIFFICULTY_BLANKS: Record<Exclude<Difficulty, "manual">, number> = {
  easy: 36,
  medium: 44,
  hard: 52,
  expert: 58,
} as const;

/** localStorage key for game state */
export const STORAGE_KEY = "sudoku:game";

/** Maximum number of history entries for undo */
export const MAX_HISTORY_SIZE = 200;

/** Grid dimensions */
export const GRID_SIZE = 9;

/** Toast display duration in milliseconds */
export const TOAST_DURATION_MS = 2000;

/** Timer update interval in milliseconds */
export const TIMER_INTERVAL_MS = 1000;

/** Delay to allow UI updates before heavy synchronous work */
export const UI_UPDATE_DELAY_MS = 30;
