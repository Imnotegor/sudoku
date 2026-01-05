import { expect, test } from "@playwright/test";
import { loadApp } from "./helpers";

test("numpad enters value into selected cell", async ({ page }) => {
  await loadApp(page);

  const cell = page.locator("#board .cell:not(.fixed)").first();
  await cell.click();

  const button = page.locator("#numpad button:not([disabled])").first();
  const value = (await button.textContent())?.trim() ?? "";
  await button.click();

  await expect(cell).toHaveText(value);
});

test("timer increments after start", async ({ page }) => {
  await loadApp(page);
  const timer = page.locator("#timer");
  const initial = await timer.textContent();

  await expect
    .poll(async () => timer.textContent(), { timeout: 5000 })
    .not.toBe(initial);
});

test("scan shows API error message", async ({ page }) => {
  await loadApp(page, {
    scanStatus: 500,
    scanErrorText: "OCR service failed",
  });

  await page.setInputFiles("#photo-input", "tests/fixtures/scan.png");
  const toast = page.locator("#toast");

  await expect(toast).toHaveText("OCR service failed");
});
