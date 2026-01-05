# Architecture

This repository contains a Sudoku web application split into three layers.

## Frontend (TypeScript)
- Entry point: `src/client/main.ts`
- Modules:
  - `src/client/game/*` for state, solver, generator, validation
  - `src/client/ui/*` for board rendering, numpad, timer, camera, toast
  - `src/client/i18n/*` for localization
- Static assets live in `static/` and are served by the Rust server.
- Build output: `static/dist/app.js`

## Backend (Rust)
- Entry point: `src/server/main.rs`
- Modules:
  - `src/server/api/*` HTTP handlers
  - `src/server/grid/*` Grid parsing/validation
  - `src/server/solver/*` Solver logic
  - `src/server/ocr/*` OCR HTTP client
- The server serves static files and provides `/api/solve`, `/api/scan`, and `/api/health`.

## OCR Service (Python)
- Entry point: `ocr/main.py` (FastAPI)
- Modules:
  - `ocr/preprocessing/*` image transforms
  - `ocr/recognition/*` digit recognition
  - `ocr/grid/*` grid detection and cleanup
  - `ocr/scanner.py` high-level scanner

## Data Flow
1. The frontend sends `/api/scan` with an uploaded image.
2. The Rust server forwards the image to the OCR service.
3. The OCR service returns a parsed grid.
4. The Rust server validates and solves the grid, returning puzzle + solution.
5. The frontend renders the grid and locks givens.

## Storage
- Game state is stored in `localStorage` under `sudoku:game`.
