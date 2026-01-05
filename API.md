# Sudoku API

Base URL: `http://<host>:8080`

## GET /api/health
Returns a simple health status.

Response:
```json
{ "status": "ok" }
```

## POST /api/solve
Solves a Sudoku grid.

Request body:
```json
{ "grid": [[0,0,0,0,0,0,0,0,0], ...] }
```

Response:
```json
{ "solution": [[1,2,3,4,5,6,7,8,9], ...] }
```

Errors:
- `400 Bad Request` if the grid is invalid or unsolvable.

## POST /api/scan
Scans an image and returns a puzzle plus its solution.

Request:
- `multipart/form-data` with `image` field (JPEG/PNG).

Response:
```json
{ "puzzle": [[...]], "solution": [[...]] }
```

Errors:
- `400 Bad Request` if the image or grid is invalid.
- `502 Bad Gateway` if the OCR service fails.

## OCR Service (internal)
The OCR service runs separately and is called by the server.

- `POST /scan`: returns `{ "grid": [[...]] }`
- `GET /health`: returns `{ "status": "ok" }`
