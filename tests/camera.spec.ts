import { expect, test } from "@playwright/test";
import path from "node:path";
import { loadApp, samplePuzzle, sampleSolution } from "./helpers";

test("scan upload locks givens and highlights conflicts", async ({ page }) => {
  await loadApp(page, {
    scanResponse: {
      puzzle: samplePuzzle as unknown as number[][],
      solution: sampleSolution as unknown as number[][],
    },
  });

  const fixturePath = path.join(process.cwd(), "tests", "fixtures", "scan.png");
  await page.setInputFiles("#photo-input", fixturePath);

  const fixedCell = page.locator('[data-row="0"][data-col="0"]');
  await expect(fixedCell).toHaveClass(/fixed/);
  await expect(fixedCell).toHaveText("5");

  const editableCell = page.locator('[data-row="0"][data-col="2"]');
  await editableCell.click();
  await page.keyboard.press("5");

  await expect(editableCell).toHaveClass(/conflict/);
  await expect(fixedCell).toHaveClass(/conflict/);
});
