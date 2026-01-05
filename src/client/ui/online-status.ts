import { t } from "../i18n";

let statusElement: HTMLElement | null = null;

/** Initialize the online status module with the status element */
export function initOnlineStatus(element: HTMLElement): void {
  statusElement = element;
  updateOnlineStatus();

  window.addEventListener("online", updateOnlineStatus);
  window.addEventListener("offline", updateOnlineStatus);
}

/** Update the online status display */
export function updateOnlineStatus(): void {
  if (!statusElement) return;

  const online = navigator.onLine;
  statusElement.textContent = online ? t("online") : t("offline");
  statusElement.style.background = online ? "#fff" : "#000";
  statusElement.style.color = online ? "#000" : "#fff";
}
