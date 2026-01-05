import { TIMER_INTERVAL_MS } from "../constants";
import { gameState } from "../game";

let timerElements: HTMLElement[] = [];
let intervalId: ReturnType<typeof setInterval> | null = null;

/** Initialize the timer module with the timer element */
export function initTimer(element: HTMLElement, ...extras: Array<HTMLElement | null | undefined>): void {
  timerElements = [element, ...extras].filter((item): item is HTMLElement => Boolean(item));
}

/** Update the timer display */
export function updateTimer(): void {
  if (timerElements.length === 0) return;

  const totalMs = gameState.getElapsedMs();
  const totalSec = Math.floor(totalMs / 1000);
  const minutes = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const seconds = String(totalSec % 60).padStart(2, "0");
  const text = `${minutes}:${seconds}`;
  timerElements.forEach((element) => {
    element.textContent = text;
  });
}

/** Start the timer interval */
export function startTimer(): void {
  if (intervalId !== null) {
    clearInterval(intervalId);
  }
  intervalId = setInterval(updateTimer, TIMER_INTERVAL_MS);
}

/** Stop the timer interval */
export function stopTimer(): void {
  if (intervalId !== null) {
    clearInterval(intervalId);
    intervalId = null;
  }
}
