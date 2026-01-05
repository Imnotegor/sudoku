"""FastAPI application for the OCR service."""

import os
import tempfile

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import get_settings
from .grid import scan_grid

app = FastAPI(
    title="Sudoku OCR Service",
    description="Extracts Sudoku digits from images using OCR",
    version="1.0.0",
)


@app.post("/scan")
async def scan(image: UploadFile = File(...)):
    """
    Scans a Sudoku puzzle from an uploaded image.

    Args:
        image: Uploaded image file (JPEG, PNG, etc.).

    Returns:
        JSON with the recognized grid: {"grid": [[...], ...]}

    Raises:
        HTTPException: If the image is empty or invalid.
    """
    settings = get_settings()
    contents = await image.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty image")
    if settings.max_upload_size and len(contents) > settings.max_upload_size:
        raise HTTPException(status_code=413, detail="Image too large")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        img = cv2.imread(temp_path)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        grid = scan_grid(img)
        return {"grid": grid}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
