import os
import tempfile
from typing import List

import cv2
import numpy as np
import pytesseract
try:
    import onnxruntime as ort
except Exception:
    ort = None
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI()

GRID_SIZE = 9
TARGET_SIZE = 630
MIN_COMPONENT_AREA = 40
MIN_OCR_CONFIDENCE = 25
MIN_CELL_CONFIDENCE = 30
MIN_BOARD_AREA_RATIO = 0.08
TARGET_FG_RATIO = 0.03
CNN_MIN_CONFIDENCE = float(os.getenv("CNN_MIN_CONFIDENCE", "0.75"))
DEFAULT_CNN_MODEL = os.path.join(os.path.dirname(__file__), "models", "sudoku-digit-cnn.onnx")
FALLBACK_CNN_MODEL = os.path.join(os.path.dirname(__file__), "models", "mnist-8.onnx")
CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH")
if not CNN_MODEL_PATH:
    CNN_MODEL_PATH = DEFAULT_CNN_MODEL if os.path.exists(DEFAULT_CNN_MODEL) else FALLBACK_CNN_MODEL

CNN_SESSION = None
CNN_INPUT_NAME = None
CNN_OUTPUT_NAME = None
if ort is not None and os.path.exists(CNN_MODEL_PATH):
    try:
        CNN_SESSION = ort.InferenceSession(CNN_MODEL_PATH, providers=["CPUExecutionProvider"])
        CNN_INPUT_NAME = CNN_SESSION.get_inputs()[0].name
        CNN_OUTPUT_NAME = CNN_SESSION.get_outputs()[0].name
    except Exception:
        CNN_SESSION = None
        CNN_INPUT_NAME = None
        CNN_OUTPUT_NAME = None


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_board(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contour = find_board_contour(gray)
    if contour is None:
        resized = cv2.resize(gray, (TARGET_SIZE, TARGET_SIZE))
        return preprocess_warped(resized)

    rect = order_points(contour.reshape(4, 2))
    dst = np.array(
        [[0, 0], [TARGET_SIZE - 1, 0], [TARGET_SIZE - 1, TARGET_SIZE - 1], [0, TARGET_SIZE - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, matrix, (TARGET_SIZE, TARGET_SIZE))
    maybe_save_debug("warp_raw", warped)
    return preprocess_warped(warped)


def find_board_contour(gray: np.ndarray) -> np.ndarray | None:
    contour = find_board_by_edges(gray)
    if contour is not None:
        return contour
    return find_board_by_threshold(gray)


def find_board_by_edges(gray: np.ndarray) -> np.ndarray | None:
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    maybe_save_debug("edges", edges)
    return select_board_contour(gray, edges)


def find_board_by_threshold(gray: np.ndarray) -> np.ndarray | None:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 4
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    maybe_save_debug("threshold_board", thresh)
    return select_board_contour(gray, thresh)


def select_board_contour(gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = gray.shape[0] * gray.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours[:15]:
        area = cv2.contourArea(contour)
        if area < image_area * MIN_BOARD_AREA_RATIO:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    contour = contours[0]
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.intp(box)


def preprocess_warped(warped: np.ndarray) -> np.ndarray:
    denoise = cv2.fastNlMeansDenoising(warped, None, 12, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoise)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    maybe_save_debug("warp_pre", enhanced)
    return enhanced


def prepare_digit_mask(warped: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(warped, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 3
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    size = thresh.shape[0]
    cell = max(10, size // GRID_SIZE)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, cell))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cell, 1))

    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    grid = cv2.bitwise_or(vertical, horizontal)
    digits = cv2.subtract(thresh, grid)
    digits = cv2.morphologyEx(digits, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    digits = cv2.dilate(digits, np.ones((2, 2), np.uint8), iterations=1)
    return digits


def base_threshold(warped: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(warped, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 4
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return thresh


def maybe_save_debug(stage: str, image: np.ndarray) -> None:
    if os.getenv("DEBUG_OCR") != "1":
        return
    debug_dir = os.getenv("DEBUG_OCR_DIR", "/tmp/sudoku-ocr")
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"{stage}.png"), image)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def prepare_cnn_input(digit: np.ndarray) -> np.ndarray:
    height, width = digit.shape
    size = max(height, width)
    pad_top = (size - height) // 2
    pad_bottom = size - height - pad_top
    pad_left = (size - width) // 2
    pad_right = size - width - pad_left
    square = cv2.copyMakeBorder(
        digit, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32) / 255.0
    return resized[np.newaxis, np.newaxis, :, :]


def predict_digit_cnn(digit: np.ndarray) -> tuple[int, int]:
    if CNN_SESSION is None:
        return 0, 0
    ink_ratio = cv2.countNonZero(digit) / float(digit.size)
    if ink_ratio < 0.01 or ink_ratio > 0.5:
        return 0, 0
    input_tensor = prepare_cnn_input(digit)
    outputs = CNN_SESSION.run([CNN_OUTPUT_NAME], {CNN_INPUT_NAME: input_tensor})
    scores = np.array(outputs[0]).reshape(-1)
    if scores.size not in (9, 10):
        return 0, 0
    probs = softmax(scores)
    index = int(np.argmax(probs))
    confidence = float(probs[index])
    if scores.size == 9:
        value = index + 1
    else:
        value = index
    if value == 0 or confidence < CNN_MIN_CONFIDENCE:
        return 0, 0
    return value, int(confidence * 100)


def ocr_by_boxes(warped: np.ndarray) -> tuple[List[List[int]], List[List[int]]]:
    digit_mask = prepare_digit_mask(warped)
    if cv2.countNonZero(digit_mask) < 1500:
        digit_mask = base_threshold(warped)
    thresh = cv2.bitwise_not(digit_mask)
    maybe_save_debug("ocr_thresh", thresh)

    config = "--oem 3 --psm 6 -c tessedit_char_whitelist=123456789"
    data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

    cell_size = TARGET_SIZE / GRID_SIZE
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    conf_map = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    for i, text in enumerate(data.get("text", [])):
        text = text.strip()
        if not text.isdigit():
            continue
        value = int(text)
        if value < 1 or value > 9:
            continue
        try:
            conf = int(float(data.get("conf", [0])[i]))
        except (ValueError, TypeError):
            conf = 0
        if conf < MIN_OCR_CONFIDENCE:
            continue
        x = data.get("left", [0])[i]
        y = data.get("top", [0])[i]
        w = data.get("width", [0])[i]
        h = data.get("height", [0])[i]
        cx = x + w / 2
        cy = y + h / 2
        row = int(cy / cell_size)
        col = int(cx / cell_size)
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            if conf > conf_map[row][col]:
                conf_map[row][col] = conf
                grid[row][col] = value

    return grid, conf_map


def extract_digit(cell_gray: np.ndarray, cell_mask: np.ndarray) -> tuple[int, int]:
    size = cell_gray.shape[0]
    margin = int(size * 0.12)
    if size - margin <= margin:
        return 0, 0

    gray = cell_gray[margin : size - margin, margin : size - margin]
    mask = cell_mask[margin : size - margin, margin : size - margin]
    if gray.size == 0 or mask.size == 0:
        return 0, 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask_ratio = cv2.countNonZero(mask) / float(mask.size)
    if 0.002 <= mask_ratio <= 0.3:
        value, conf = ocr_from_mask(gray, mask)
        if value:
            return value, conf

    return ocr_from_gray(gray)


def threshold_cell(gray: np.ndarray) -> np.ndarray:
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    otsu_ratio = cv2.countNonZero(otsu) / float(otsu.size)
    adaptive_ratio = cv2.countNonZero(adaptive) / float(adaptive.size)

    if 0.003 <= adaptive_ratio <= 0.18 and (
        abs(adaptive_ratio - TARGET_FG_RATIO) <= abs(otsu_ratio - TARGET_FG_RATIO)
    ):
        return adaptive

    return otsu


def ocr_from_mask(gray: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    min_area = max(MIN_COMPONENT_AREA, int(mask.size * 0.006))
    if area < min_area:
        return 0, 0

    x, y, w, h = cv2.boundingRect(contour)
    digit_gray = gray[y : y + h, x : x + w]
    if digit_gray.size == 0:
        return 0, 0

    return ocr_from_gray(digit_gray)


def ocr_from_gray(gray: np.ndarray) -> tuple[int, int]:
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = threshold_cell(blur)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    thresh = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=1)

    if cv2.countNonZero(thresh) < max(15, int(thresh.size * 0.008)):
        return 0, 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    min_area = max(MIN_COMPONENT_AREA, int(thresh.size * 0.006))
    if area < min_area:
        return 0, 0

    x, y, w, h = cv2.boundingRect(contour)
    digit = thresh[y : y + h, x : x + w]
    if digit.size == 0:
        return 0, 0

    cnn_value, cnn_conf = predict_digit_cnn(digit)
    if cnn_value:
        return cnn_value, cnn_conf

    pad = 6
    digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    digit = cv2.resize(digit, (48, 48), interpolation=cv2.INTER_AREA)
    digit = cv2.bitwise_not(digit)

    configs = [
        "--oem 3 --psm 10 -c tessedit_char_whitelist=123456789 -c classify_bln_numeric_mode=1",
        "--oem 3 --psm 13 -c tessedit_char_whitelist=123456789 -c classify_bln_numeric_mode=1",
        "--oem 3 --psm 8 -c tessedit_char_whitelist=123456789 -c classify_bln_numeric_mode=1",
    ]
    return ocr_digit_image(digit, configs)


def ocr_digit_image(digit: np.ndarray, configs: list[str]) -> tuple[int, int]:
    best_value = 0
    best_conf = -1
    for config in configs:
        data = pytesseract.image_to_data(
            digit, config=config, output_type=pytesseract.Output.DICT
        )
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            text = text.strip()
            if not text.isdigit():
                continue
            try:
                conf_value = int(float(conf))
            except (ValueError, TypeError):
                continue
            if conf_value > best_conf:
                value = int(text)
                if 1 <= value <= 9:
                    best_conf = conf_value
                    best_value = value
        if best_conf >= MIN_CELL_CONFIDENCE:
            return best_value, best_conf
    return 0, 0


def scan_grid(image: np.ndarray) -> List[List[int]]:
    warped = warp_board(image)
    warped = cv2.resize(warped, (TARGET_SIZE, TARGET_SIZE))
    digit_mask = prepare_digit_mask(warped)
    if cv2.countNonZero(digit_mask) < 1500:
        digit_mask = base_threshold(warped)
    maybe_save_debug("warp", warped)
    maybe_save_debug("digit_mask", digit_mask)
    cell_size = TARGET_SIZE // GRID_SIZE

    grid = []
    conf_grid = []
    for row in range(GRID_SIZE):
        row_values = []
        row_conf = []
        for col in range(GRID_SIZE):
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size
            cell_gray = warped[y1:y2, x1:x2]
            cell_mask = digit_mask[y1:y2, x1:x2]
            value, conf = extract_digit(cell_gray, cell_mask)
            row_values.append(value)
            row_conf.append(conf)
        grid.append(row_values)
        conf_grid.append(row_conf)
    if count_digits(grid) < 20:
        box_grid, box_conf = ocr_by_boxes(warped)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if grid[row][col] == 0 and box_grid[row][col] != 0:
                    grid[row][col] = box_grid[row][col]
                    conf_grid[row][col] = box_conf[row][col]
    suppress_conflicts(grid, conf_grid)
    prune_unsolvable(grid, conf_grid)
    return grid


def count_digits(grid: List[List[int]]) -> int:
    return sum(1 for row in grid for value in row if value > 0)


def prune_unsolvable(grid: List[List[int]], conf_grid: List[List[int]]) -> None:
    if is_solvable(grid):
        return

    candidates = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] != 0:
                candidates.append((conf_grid[row][col], row, col))

    candidates.sort()
    for _, row, col in candidates:
        grid[row][col] = 0
        conf_grid[row][col] = 0
        if is_solvable(grid):
            return


def is_solvable(grid: List[List[int]]) -> bool:
    grid_copy = [row[:] for row in grid]
    return solve_grid(grid_copy)


def solve_grid(grid: List[List[int]]) -> bool:
    cell = find_best_cell(grid)
    if cell is None:
        return True
    row, col, options = cell
    if not options:
        return False
    for value in options:
        grid[row][col] = value
        if solve_grid(grid):
            return True
    grid[row][col] = 0
    return False


def find_best_cell(grid: List[List[int]]) -> tuple[int, int, List[int]] | None:
    best = None
    best_options = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] != 0:
                continue
            options = get_candidates(grid, row, col)
            if best is None or len(options) < len(best_options):
                best = (row, col, options)
                best_options = options
            if best is not None and len(best_options) <= 1:
                return best
    return best


def get_candidates(grid: List[List[int]], row: int, col: int) -> List[int]:
    used = set()
    used.update(grid[row])
    used.update(grid[r][col] for r in range(GRID_SIZE))
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            used.add(grid[r][c])
    return [n for n in range(1, 10) if n not in used]


def suppress_conflicts(grid: List[List[int]], conf_grid: List[List[int]]) -> None:
    for row in range(GRID_SIZE):
        remove_conflicts_in_line([(row, col) for col in range(GRID_SIZE)], grid, conf_grid)
    for col in range(GRID_SIZE):
        remove_conflicts_in_line([(row, col) for row in range(GRID_SIZE)], grid, conf_grid)
    for box_row in range(0, GRID_SIZE, 3):
        for box_col in range(0, GRID_SIZE, 3):
            cells = [
                (box_row + r, box_col + c) for r in range(3) for c in range(3)
            ]
            remove_conflicts_in_line(cells, grid, conf_grid)


def remove_conflicts_in_line(
    cells: List[tuple[int, int]], grid: List[List[int]], conf_grid: List[List[int]]
) -> None:
    positions: dict[int, list[tuple[int, int]]] = {}
    for row, col in cells:
        value = grid[row][col]
        if value == 0:
            continue
        positions.setdefault(value, []).append((row, col))

    for spots in positions.values():
        if len(spots) <= 1:
            continue
        best = max(spots, key=lambda rc: conf_grid[rc[0]][rc[1]])
        for row, col in spots:
            if (row, col) != best:
                grid[row][col] = 0


@app.post("/scan")
async def scan(image: UploadFile = File(...)):
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image")

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
