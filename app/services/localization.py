from __future__ import annotations

from pathlib import Path
from typing import Any

from inference_sdk import InferenceHTTPClient

from app.core.config import Settings


class RoboflowLocalizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = InferenceHTTPClient(
            api_url=settings.roboflow_api_url,
            api_key=settings.roboflow_api_key,
        )

    def infer(self, image_path: Path) -> dict[str, Any]:
        return self.client.infer(str(image_path), model_id=self.settings.roboflow_model_id)

    def filter_predictions(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        raw_predictions = result.get("predictions", [])
        return [
            prediction
            for prediction in raw_predictions
            if float(prediction.get("confidence", 0)) >= self.settings.roboflow_confidence_threshold
        ]
