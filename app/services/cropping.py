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


def load_annotation_font(image_width: int) -> ImageFont.ImageFont:
    font_size = max(26, min(54, image_width // 32))
    for font_name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_predictions(image: Image.Image, predictions: list[dict[str, Any]], padding: int) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    image_width, image_height = annotated.size
    font = load_annotation_font(image_width)
    box_width = max(4, image_width // 350)

    for index, prediction in enumerate(sort_reading_order(predictions), start=1):
        box = prediction_to_xyxy(prediction, image_width, image_height, padding=padding)
        confidence = float(prediction.get("confidence", 0))
        label = f"{index}: {confidence * 100:.0f}%"
        draw.rectangle(box, outline="red", width=box_width)

        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        pad_x = 10
        pad_y = 7
        label_height = text_height + (pad_y * 2)
        label_width = text_width + (pad_x * 2)

        label_left = box[0]
        label_top = box[1] - label_height - 4
        if label_top < 0:
            label_top = box[1] + 4
        label_right = min(image_width, label_left + label_width)
        label_bottom = label_top + label_height

        draw.rectangle(
            (label_left, label_top, label_right, label_bottom),
            fill="white",
            outline="red",
            width=max(2, box_width // 2),
        )
        draw.text(
            (label_left + pad_x, label_top + pad_y - text_box[1]),
            label,
            fill="red",
            font=font,
        )

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
