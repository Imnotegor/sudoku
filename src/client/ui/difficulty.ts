import type { Difficulty } from "../types";
import { gameState } from "../game";
import { t, getDifficultyLabel } from "../i18n";

let difficultyElement: HTMLElement | null = null;
let difficultyLabelElement: HTMLElement | null = null;

/** Initialize the difficulty module with the difficulty elements */
export function initDifficulty(element: HTMLElement, labelElement: HTMLElement): void {
  difficultyElement = element;
  difficultyLabelElement = labelElement;
}

/** Update the difficulty buttons */
export function updateDifficultyButtons(): void {
  if (!difficultyElement || !difficultyLabelElement) return;

  const buttons = difficultyElement.querySelectorAll<HTMLElement>("[data-level]");
  const currentDifficulty = gameState.difficulty;

  buttons.forEach((button) => {
    const level = button.dataset.level as Difficulty | undefined;
    if (!level) return;

    button.classList.toggle("active", level === currentDifficulty);
    button.textContent = getDifficultyLabel(level);
  });

  const label = getDifficultyLabel(currentDifficulty);
  difficultyLabelElement.textContent = `${t("difficulty_label")}: ${label}`;
}

/** Handle difficulty button click */
export function handleDifficultyClick(callback: (level: Difficulty) => void): void {
  if (!difficultyElement) return;

  difficultyElement.addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest("[data-level]");
    if (!button || !(button instanceof HTMLElement)) return;

    const level = button.dataset.level as Difficulty | undefined;
    if (level) {
      callback(level);
    }
  });
}
