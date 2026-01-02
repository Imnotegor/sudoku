const { test, expect } = require('@playwright/test');

const scannedPuzzle = [
  [5, 3, 0, 0, 7, 0, 0, 0, 0],
  [6, 0, 0, 1, 9, 5, 0, 0, 0],
  [0, 9, 8, 0, 0, 0, 0, 6, 0],
  [8, 0, 0, 0, 6, 0, 0, 0, 3],
  [4, 0, 0, 8, 0, 3, 0, 0, 1],
  [7, 0, 0, 0, 2, 0, 0, 0, 6],
  [0, 6, 0, 0, 0, 0, 2, 8, 0],
  [0, 0, 0, 4, 1, 9, 0, 0, 5],
  [0, 0, 0, 0, 8, 0, 0, 7, 9],
];

const scannedSolution = [
  [5, 3, 4, 6, 7, 8, 9, 1, 2],
  [6, 7, 2, 1, 9, 5, 3, 4, 8],
  [1, 9, 8, 3, 4, 2, 5, 6, 7],
  [8, 5, 9, 7, 6, 1, 4, 2, 3],
  [4, 2, 6, 8, 5, 3, 7, 9, 1],
  [7, 1, 3, 9, 2, 4, 8, 5, 6],
  [9, 6, 1, 5, 3, 7, 2, 8, 4],
  [2, 8, 7, 4, 1, 9, 6, 3, 5],
  [3, 4, 5, 2, 8, 6, 1, 7, 9],
];

test('renders board and numpad for a new game', async ({ page }) => {
  await page.goto('/');

  await expect(page.locator('.cell')).toHaveCount(81);
  await expect(page.locator('#numpad button')).toHaveCount(9);

  const fixedCount = await page.locator('.cell.fixed').count();
  expect(fixedCount).toBeGreaterThan(0);
});

test('manual entry allows typing and clearing digits', async ({ page }) => {
  await page.goto('/');
  await page.click('#manual-entry');

  const cell = page.locator('.cell[data-row="0"][data-col="0"]');
  await cell.click();
  await page.keyboard.press('1');
  await expect(cell).toHaveText('1');

  await page.keyboard.press('Backspace');
  await expect(cell).toHaveText('');
});

test('scan keeps OCR digits editable and shows solution on demand', async ({ page }) => {
  await page.route('**/api/scan', async (route) => {
    await route.fulfill({
      json: {
        puzzle: scannedPuzzle,
        solution: scannedSolution,
      },
    });
  });

  await page.goto('/');
  await page.setInputFiles('#photo-input', 'tests/fixtures/scan.txt');

  const filledCell = page.locator('.cell[data-row="0"][data-col="0"]');
  await expect(filledCell).toHaveText('5');
  await expect(filledCell).not.toHaveClass(/fixed/);

  await filledCell.click();
  await page.keyboard.press('1');
  await expect(filledCell).toHaveText('1');

  const emptyCell = page.locator('.cell[data-row="0"][data-col="2"]');
  await expect(emptyCell).toHaveText('');
  await page.click('#show-solution');
  await expect(emptyCell).toHaveText('4');
});
