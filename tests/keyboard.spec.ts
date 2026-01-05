import { expect, test } from "@playwright/test";
import path from "node:path";
import { loadApp, samplePuzzle, sampleSolution } from "./helpers";

test("keyboard navigation and input", async ({ page }) => {
  await loadApp(page, {
    scanResponse: {
      puzzle: samplePuzzle as unknown as number[][],
      solution: sampleSolution as unknown as number[][],
    },
  });

  const fixturePath = path.join(process.cwd(), "tests", "fixtures", "scan.png");
  await page.setInputFiles("#photo-input", fixturePath);

  const startCell = page.locator('[data-row="0"][data-col="2"]');
  await startCell.click();
  await page.keyboard.press("ArrowRight");

  const nextCell = page.locator('[data-row="0"][data-col="3"]');
  await expect(nextCell).toHaveAttribute("aria-selected", "true");

  await page.keyboard.press("1");
  await expect(nextCell).toHaveText("1");
});
