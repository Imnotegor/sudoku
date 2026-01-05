import { TOAST_DURATION_MS } from "../constants";

let toastElement: HTMLElement | null = null;
let timeoutId: ReturnType<typeof setTimeout> | null = null;

/** Initialize the toast module with the toast element */
export function initToast(element: HTMLElement): void {
  toastElement = element;
}

/** Show a toast notification */
export function showToast(message: string): void {
  if (!toastElement) return;

  toastElement.textContent = message;
  toastElement.classList.add("show");

  if (timeoutId !== null) {
    clearTimeout(timeoutId);
  }

  timeoutId = setTimeout(() => {
    toastElement?.classList.remove("show");
    timeoutId = null;
  }, TOAST_DURATION_MS);
}
