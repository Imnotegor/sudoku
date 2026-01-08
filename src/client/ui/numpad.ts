import { gameState } from "../game";

let numpadElement: HTMLElement | null = null;
let buttons: HTMLButtonElement[] = [];

/** Initialize the numpad module with the numpad element */
export function initNumpad(element: HTMLElement): void {
  numpadElement = element;
  buildNumpad();
}

/** Build the numpad buttons */
function buildNumpad(): void {
  if (!numpadElement) return;

  numpadElement.innerHTML = "";
  buttons = [];

  for (let i = 1; i <= 9; i++) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = String(i);
    btn.dataset.value = String(i);
    numpadElement.appendChild(btn);
    buttons.push(btn);
  }
}

/** Update numpad button availability based on current selection */
export function updateNumpadAvailability(): void {
  if (!numpadElement || buttons.length === 0) return;
  const { row, col } = gameState.selected;
  const current = gameState.current;
  const puzzle = gameState.puzzle;
  const value = current[row]?.[col] ?? 0;

  // If cell is fixed or showing solution, disable all buttons
  if (puzzle[row]?.[col] !== 0 || gameState.showingSolution) {
    buttons.forEach((btn) => {
      btn.disabled = true;
      btn.classList.remove("is-hidden");
    });
    return;
  }

  // Find numbers already used in the 3x3 box
  const boxNumbers = new Set<number>();
  const boxRow = Math.floor(row / 3) * 3;
  const boxCol = Math.floor(col / 3) * 3;

  for (let r = boxRow; r < boxRow + 3; r++) {
    for (let c = boxCol; c < boxCol + 3; c++) {
      const cellValue = current[r][c];
      if (cellValue > 0) boxNumbers.add(cellValue);
    }
  }

  // Exclude current cell value from disabled set
  if (value > 0) boxNumbers.delete(value);

  // Update button states
  buttons.forEach((btn) => {
    const num = Number(btn.dataset.value);
    const hidden = boxNumbers.has(num);
    btn.disabled = hidden;
    btn.classList.toggle("is-hidden", hidden);
  });
}

/** Handle numpad button click */
export function handleNumpadClick(callback: (value: number) => void): void {
  if (!numpadElement) return;

  numpadElement.addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest("button");
    if (!button || button.disabled) return;

    const value = Number(button.dataset.value);
    if (value >= 1 && value <= 9) {
      callback(value);
    }
  });
}
