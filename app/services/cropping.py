from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


Box = tuple[int, int, int, int]


def prediction_to_xyxy(
    prediction: dict[str, Any],
    image_width: int,
    image_height: int,
    padding: int = 0,
) -> Box:
    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])

    left = max(0, int(round(x - width / 2)) - padding)
    top = max(0, int(round(y - height / 2)) - padding)
    right = min(image_width, int(round(x + width / 2)) + padding)
    bottom = min(image_height, int(round(y + height / 2)) + padding)
    return left, top, right, bottom


def sort_reading_order(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(predictions, key=lambda pred: (float(pred["y"]), float(pred["x"])))


def draw_predictions(image: Image.Image, predictions: list[dict[str, Any]], padding: int) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    image_width, image_height = annotated.size
    font = ImageFont.load_default()

    for index, prediction in enumerate(sort_reading_order(predictions), start=1):
        box = prediction_to_xyxy(prediction, image_width, image_height, padding=padding)
        label = f"{index}: {float(prediction.get('confidence', 0)):.2f}"
        draw.rectangle(box, outline="red", width=3)
        label_box = (box[0], max(0, box[1] - 16), box[0] + 62, box[1])
        draw.rectangle(label_box, fill="white")
        draw.text((box[0] + 3, max(0, box[1] - 14)), label, fill="red", font=font)

    return annotated


def crop_predictions(
    image: Image.Image,
    predictions: list[dict[str, Any]],
    crop_dir: Path,
    padding: int,
) -> list[dict[str, Any]]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    image_width, image_height = image.size
    records: list[dict[str, Any]] = []

    for index, prediction in enumerate(sort_reading_order(predictions), start=1):
        box = prediction_to_xyxy(prediction, image_width, image_height, padding=padding)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        crop = image.crop(box)
        crop_path = crop_dir / f"region_{index:03d}.png"
        crop.save(crop_path)
        records.append(
            {
                "region_number": index,
                "crop_path": crop_path,
                "confidence": float(prediction.get("confidence", 0)),
                "x_min": box[0],
                "y_min": box[1],
                "x_max": box[2],
                "y_max": box[3],
            }
        )

    return records
