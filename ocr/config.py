"""Configuration settings for the OCR service."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Grid settings
    grid_size: int = 9
    target_size: int = 630

    # Image processing
    min_component_area: int = 40
    min_board_area_ratio: float = 0.08
    target_fg_ratio: float = 0.03

    # OCR confidence thresholds
    min_ocr_confidence: int = 25
    min_cell_confidence: int = 30
    cnn_min_confidence: float = 0.75

    # Upload limits
    max_upload_size: int = 10 * 1024 * 1024

    # Model paths
    cnn_model_path: str = ""

    # Debug settings
    debug_ocr: bool = False
    debug_ocr_dir: str = "/tmp/sudoku-ocr"

    class Config:
        env_prefix = ""
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.cnn_model_path:
            self.cnn_model_path = self._resolve_model_path()

    @staticmethod
    def _resolve_model_path() -> str:
        """Resolves the CNN model path from environment or defaults."""
        if path := os.getenv("CNN_MODEL_PATH"):
            return path

        models_dir = Path(__file__).parent / "models"
        default_model = models_dir / "sudoku-digit-cnn.onnx"
        fallback_model = models_dir / "mnist-8.onnx"

        if default_model.exists():
            return str(default_model)
        if fallback_model.exists():
            return str(fallback_model)
        return ""


@lru_cache
def get_settings() -> Settings:
    """Returns the cached settings instance."""
    return Settings()
