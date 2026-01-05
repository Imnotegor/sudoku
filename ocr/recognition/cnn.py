"""CNN-based digit recognition."""

from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from ..config import get_settings

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class DigitClassifier:
    """CNN-based digit classifier using ONNX Runtime."""

    def __init__(self, model_path: str):
        """
        Initialize the classifier with an ONNX model.

        Args:
            model_path: Path to the ONNX model file.
        """
        if ort is None:
            self._session = None
            self._input_name = None
            self._output_name = None
            return

        try:
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
        except Exception:
            self._session = None
            self._input_name = None
            self._output_name = None

    @property
    def is_available(self) -> bool:
        """Returns True if the model is loaded and ready."""
        return self._session is not None

    def predict(self, digit: np.ndarray) -> tuple[int, int]:
        """
        Predict the digit from a binary image.

        Args:
            digit: Binary image of a digit (white on black).

        Returns:
            Tuple of (digit value, confidence percentage).
            Returns (0, 0) if prediction fails or confidence is too low.
        """
        if not self.is_available:
            return 0, 0

        settings = get_settings()

        # Check ink ratio
        ink_ratio = cv2.countNonZero(digit) / float(digit.size)
        if ink_ratio < 0.01 or ink_ratio > 0.5:
            return 0, 0

        input_tensor = self._prepare_input(digit)
        outputs = self._session.run([self._output_name], {self._input_name: input_tensor})
        scores = np.array(outputs[0]).reshape(-1)

        if scores.size not in (9, 10):
            return 0, 0

        probs = self._softmax(scores)
        index = int(np.argmax(probs))
        confidence = float(probs[index])

        # Handle both 9-class (1-9) and 10-class (0-9) models
        value = index + 1 if scores.size == 9 else index

        if value == 0 or confidence < settings.cnn_min_confidence:
            return 0, 0

        return value, int(confidence * 100)

    @staticmethod
    def _prepare_input(digit: np.ndarray) -> np.ndarray:
        """Prepares digit image for CNN input (NCHW format, 28x28)."""
        height, width = digit.shape
        size = max(height, width)

        # Pad to square
        pad_top = (size - height) // 2
        pad_bottom = size - height - pad_top
        pad_left = (size - width) // 2
        pad_right = size - width - pad_left

        square = cv2.copyMakeBorder(
            digit, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

        # Resize to 28x28 and normalize
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        resized = resized.astype(np.float32) / 255.0
        return resized[np.newaxis, np.newaxis, :, :]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Computes softmax of logits."""
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)


@lru_cache
def get_classifier() -> Optional[DigitClassifier]:
    """Returns the cached digit classifier instance."""
    settings = get_settings()
    if not settings.cnn_model_path:
        return None
    classifier = DigitClassifier(settings.cnn_model_path)
    return classifier if classifier.is_available else None
