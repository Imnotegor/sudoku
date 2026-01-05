#!/usr/bin/env node
import * as esbuild from "esbuild";
import { mkdirSync } from "node:fs";

const isWatch = process.argv.includes("--watch");
const outdir = "static/dist";

mkdirSync("static/dist", { recursive: true });

const buildOptions = {
  entryPoints: {
    app: "src/client/main.ts",
    worker: "src/client/worker.ts",
  },
  bundle: true,
  outdir,
  entryNames: "[name]",
  format: "iife",
  target: ["es2020"],
  minify: !isWatch,
  sourcemap: isWatch,
  logLevel: "info",
};

if (isWatch) {
  const ctx = await esbuild.context(buildOptions);
  await ctx.watch();
  console.log("Watching for changes...");
} else {
  await esbuild.build(buildOptions);
  console.log("Build complete!");
}
