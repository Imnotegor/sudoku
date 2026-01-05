import type { Difficulty, Locale, Translations } from "../types";
import { translations } from "./translations";

let currentLocale: Locale = "ru";

/** Detects user's preferred locale based on browser settings */
export function detectLocale(): Locale {
  const lang = navigator.language.toLowerCase();
  if (lang.startsWith("ru")) return "ru";
  return "en";
}

/** Sets the current locale */
export function setLocale(locale: Locale): void {
  currentLocale = locale;
  document.documentElement.lang = locale;
}

/** Gets the current locale */
export function getLocale(): Locale {
  return currentLocale;
}

/** Translates a key to the current locale */
export function t(key: keyof Translations): string {
  return translations[currentLocale][key];
}

/** Gets the localized difficulty label */
export function getDifficultyLabel(level: Difficulty): string {
  const key = `difficulty_${level}` as keyof Translations;
  return translations[currentLocale][key];
}

/** Applies translations to static DOM elements with data-i18n attributes */
export function applyTranslations(): void {
  const elements = document.querySelectorAll<HTMLElement>("[data-i18n]");
  elements.forEach((el) => {
    const key = el.dataset.i18n as keyof Translations | undefined;
    if (!key || !(key in translations[currentLocale])) return;

    const value = translations[currentLocale][key];
    const attr = el.dataset.i18nAttr;

    if (attr) {
      if (attr === "placeholder" && (el instanceof HTMLTextAreaElement || el instanceof HTMLInputElement)) {
        el.placeholder = value;
      } else {
        el.setAttribute(attr, value);
      }
      return;
    }

    if (el.dataset.i18nPlaceholder) {
      if (el instanceof HTMLTextAreaElement || el instanceof HTMLInputElement) {
        el.placeholder = value;
      }
      return;
    }

    el.textContent = value;
  });

  const placeholderElements = document.querySelectorAll<HTMLElement>("[data-i18n-placeholder]");
  placeholderElements.forEach((el) => {
    const key = el.dataset.i18nPlaceholder as keyof Translations | undefined;
    if (!key || !(key in translations[currentLocale])) return;
    if (el instanceof HTMLTextAreaElement || el instanceof HTMLInputElement) {
      el.placeholder = translations[currentLocale][key];
    }
  });
}
