import { expect, test } from "@playwright/test";
import { loadApp } from "./helpers";

test("fixed cells remain locked", async ({ page }) => {
  await loadApp(page);

  const fixedCell = page.locator("#board .cell.fixed").first();
  const initialValue = await fixedCell.textContent();

  await fixedCell.click();
  await page.keyboard.press("9");

  await expect(fixedCell).toHaveText(initialValue ?? "");
});
