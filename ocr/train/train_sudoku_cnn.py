import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SyntheticSudokuDigits(Dataset):
    def __init__(self, count: int, font_paths: list[str], seed: int) -> None:
        self.count = count
        self.font_paths = font_paths
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + index)
        digit = rng.randint(1, 9)
        image = render_digit(digit, self.font_paths, rng)
        tensor = torch.from_numpy(image).unsqueeze(0)
        label = digit - 1
        return tensor, label


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 9),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_fonts(fonts_dir: Path) -> list[str]:
    if not fonts_dir.exists():
        return []
    paths = []
    for ext in ("*.ttf", "*.otf"):
        paths.extend(str(p) for p in fonts_dir.rglob(ext))
    return sorted(set(paths))


def choose_font(font_paths: list[str], size: int, rng: random.Random) -> ImageFont.FreeTypeFont:
    if not font_paths:
        return ImageFont.load_default()
    path = rng.choice(font_paths)
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def render_digit(digit: int, font_paths: list[str], rng: random.Random) -> np.ndarray:
    canvas = 64
    background = Image.new("L", (canvas, canvas), color=0)
    draw = ImageDraw.Draw(background)

    font_size = rng.randint(34, 54)
    font = choose_font(font_paths, font_size, rng)
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    jitter_x = rng.randint(-6, 6)
    jitter_y = rng.randint(-6, 6)
    x = (canvas - text_w) // 2 + jitter_x
    y = (canvas - text_h) // 2 + jitter_y
    draw.text((x, y), text, fill=rng.randint(200, 255), font=font)

    img = np.array(background, dtype=np.uint8)

    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.9, 1.1)
    center = (canvas / 2, canvas / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    img = cv2.warpAffine(img, matrix, (canvas, canvas), flags=cv2.INTER_LINEAR, borderValue=0)

    if rng.random() < 0.6:
        sigma = rng.uniform(0.5, 1.6)
        img = cv2.GaussianBlur(img, (0, 0), sigma)

    noise_level = rng.uniform(0.0, 18.0)
    if noise_level > 0:
        noise = np.random.default_rng(rng.randint(0, 10_000)).normal(
            0, noise_level, img.shape
        )
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if rng.random() < 0.4:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)
    if rng.random() < 0.3:
        img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)

    if rng.random() < 0.5:
        thickness = 1 if rng.random() < 0.7 else 2
        if rng.random() < 0.5:
            y = rng.randint(4, canvas - 4)
            cv2.line(img, (0, y), (canvas, y), 255, thickness=thickness)
        else:
            x = rng.randint(4, canvas - 4)
            cv2.line(img, (x, 0), (x, canvas), 255, thickness=thickness)

    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    font_paths = load_fonts(Path(args.fonts_dir)) if args.fonts_dir else []

    train_set = SyntheticSudokuDigits(args.train_samples, font_paths, seed=42)
    val_set = SyntheticSudokuDigits(args.val_samples, font_paths, seed=1337)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        accuracy = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"epoch={epoch} loss={avg_loss:.4f} val_acc={accuracy:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, 28, 28, device=device)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        opset_version=11,
    )
    print(f"saved {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sudoku digit CNN and export ONNX.")
    parser.add_argument("--fonts-dir", default="../fonts", help="Directory with TTF/OTF fonts.")
    parser.add_argument("--train-samples", type=int, default=60000)
    parser.add_argument("--val-samples", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="../models/sudoku-digit-cnn.onnx")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
