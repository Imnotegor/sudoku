import { readFileSync } from "node:fs";
import path from "node:path";
import type { Page } from "@playwright/test";

export const samplePuzzle = [
  [5, 3, 0, 0, 7, 0, 0, 0, 0],
  [6, 0, 0, 1, 9, 5, 0, 0, 0],
  [0, 9, 8, 0, 0, 0, 0, 6, 0],
  [8, 0, 0, 0, 6, 0, 0, 0, 3],
  [4, 0, 0, 8, 0, 3, 0, 0, 1],
  [7, 0, 0, 0, 2, 0, 0, 0, 6],
  [0, 6, 0, 0, 0, 0, 2, 8, 0],
  [0, 0, 0, 4, 1, 9, 0, 0, 5],
  [0, 0, 0, 0, 8, 0, 0, 7, 9],
] as const;

export const sampleSolution = [
  [5, 3, 4, 6, 7, 8, 9, 1, 2],
  [6, 7, 2, 1, 9, 5, 3, 4, 8],
  [1, 9, 8, 3, 4, 2, 5, 6, 7],
  [8, 5, 9, 7, 6, 1, 4, 2, 3],
  [4, 2, 6, 8, 5, 3, 7, 9, 1],
  [7, 1, 3, 9, 2, 4, 8, 5, 6],
  [9, 6, 1, 5, 3, 7, 2, 8, 4],
  [2, 8, 7, 4, 1, 9, 6, 3, 5],
  [3, 4, 5, 2, 8, 6, 1, 7, 9],
] as const;

export async function loadApp(
  page: Page,
  options: {
    scanResponse?: { puzzle: number[][]; solution: number[][] };
    solveResponse?: { solution: number[][] };
    scanStatus?: number;
    scanErrorText?: string;
  } = {}
): Promise<void> {
  const indexPath = path.join(process.cwd(), "static", "quick.html");
  const appPath = path.join(process.cwd(), "static", "dist", "app.js");

  const rawHtml = readFileSync(indexPath, "utf8");
  const html = rawHtml.replace(
    /<script[^>]*src="\/dist\/app\.js"[^>]*><\/script>/,
    ""
  );

  await page.setContent(html, { waitUntil: "domcontentloaded" });

  const scanResponse = options.scanResponse ?? {
    puzzle: samplePuzzle as unknown as number[][],
    solution: sampleSolution as unknown as number[][],
  };
  const scanStatus = options.scanStatus ?? 200;
  const scanErrorText = options.scanErrorText ?? "Scan failed";

  const solveResponse = options.solveResponse ?? {
    solution: sampleSolution as unknown as number[][],
  };

  await page.evaluate(
    ({ scanResponse, solveResponse, scanStatus, scanErrorText }) => {
      const originalFetch = window.fetch.bind(window);
      window.fetch = (input, init) => {
        const url = typeof input === "string" ? input : input.url;
        if (url === "/api/scan") {
          if (scanStatus >= 200 && scanStatus < 300) {
            return Promise.resolve(
              new Response(JSON.stringify(scanResponse), {
                status: scanStatus,
                headers: { "Content-Type": "application/json" },
              })
            );
          }
          return Promise.resolve(new Response(scanErrorText, { status: scanStatus }));
        }
        if (url === "/api/solve") {
          return Promise.resolve(
            new Response(JSON.stringify(solveResponse), {
              status: 200,
              headers: { "Content-Type": "application/json" },
            })
          );
        }
        return originalFetch(input, init);
      };
    },
    { scanResponse, solveResponse, scanStatus, scanErrorText }
  );

  await page.addScriptTag({ path: appPath });
  await page.waitForFunction(() => document.documentElement.dataset.appReady === "true");
  await page.waitForSelector("#board:not(.loading) .cell.fixed");
}
