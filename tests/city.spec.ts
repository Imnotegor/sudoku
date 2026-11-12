import { test, expect } from "@playwright/test";

const base = process.env.CITY_TEST_URL;
test.skip(!base, "Set CITY_TEST_URL to a disposable city server");

test("campaign notes survive offline and classic play keeps its own save", async ({
  page,
  context,
}) => {
  await page.goto(base! + "/quick.html");
  await page.locator(".cell.fixed").first().waitFor();
  await page.locator(".cell:not(.fixed)").first().click();
  await page.keyboard.press("1");
  await page.getByRole("link", { name: "К выбору режима" }).click();
  const quick = await page.evaluate(() => localStorage.getItem("sudoku:game"));
  await page.getByRole("link", { name: "Мой город", exact: true }).click();
  await page.locator(".map-pin:not([disabled])").first().click();
  await page.locator(".cell:not(.fixed)").first().click();
  await page.getByRole("button", { name: "Заметки" }).click();
  await page.keyboard.press("3");
  await expect(page.locator(".cell.notes")).toHaveCount(1);
  await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
  });
  await context.setOffline(true);
  await page.reload();
  await expect(page.locator(".cell.notes")).toHaveCount(1);
  await page.getByRole("link", { name: "Карта города" }).click();
  await expect(page.locator(".map-pin")).toHaveCount(5);
  await page.getByRole("link", { name: "Судоку", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Продолжить партию" }),
  ).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("sudoku:game"))).toBe(
    quick,
  );
});

test("four browsers join an invited room and see live district activity", async ({
  browser,
}) => {
  const contexts = await Promise.all(
    Array.from({ length: 4 }, () => browser.newContext()),
  );
  try {
    const pages = await Promise.all(contexts.map((c) => c.newPage()));
    await pages[0].goto(base! + "/?mode=battle");
    await pages[0].getByRole("button", { name: "Создать матч 2×2" }).click();
    await pages[0].locator("#share").waitFor();
    const url = pages[0].url();
    for (const [idx, side] of [
      [1, 0],
      [2, 1],
      [3, 1],
    ]) {
      await pages[idx].goto(url);
      await pages[idx].locator("#join-" + side).click();
      await expect(
        pages[idx].getByText("(вы)", { exact: false }).first(),
      ).toBeVisible();
    }
    await expect(pages[0].locator(".map-pin:not([disabled])")).toHaveCount(5);
    await pages[0].locator(".map-pin").first().click();
    await expect(pages[0].locator(".cell")).toHaveCount(81);
    await expect(pages[1].locator(".roster").first()).toContainText("Вокзал");
    await pages[3].goto(base! + "/?mode=admin");
    await expect(pages[3].getByLabel("Ключ администратора")).toBeVisible();
  } finally {
    await Promise.all(contexts.map((c) => c.close()));
  }
});
