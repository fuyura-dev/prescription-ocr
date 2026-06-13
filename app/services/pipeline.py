from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.core.config import Settings, get_settings
from app.services.cropping import crop_predictions, draw_predictions
from app.services.localization import RoboflowLocalizer
from app.services.recognition import TrOCRRecognizer


@lru_cache(maxsize=1)
def get_recognizer() -> TrOCRRecognizer:
    return TrOCRRecognizer(get_settings())


class PrescriptionOCRPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.localizer = RoboflowLocalizer(self.settings)
        self.recognizer = get_recognizer()

    def run(self, upload_path: Path) -> dict:
        run_id = uuid4().hex[:12]
        run_dir = self.settings.web_output_dir / "runs" / run_id
        crop_dir = run_dir / "crops"
        run_dir.mkdir(parents=True, exist_ok=True)

        original_path = run_dir / f"original{upload_path.suffix.lower()}"
        shutil.copy2(upload_path, original_path)

        image = Image.open(original_path).convert("RGB")
        roboflow_result = self.localizer.infer(original_path)
        predictions = self.localizer.filter_predictions(roboflow_result)

        roboflow_json_path = run_dir / "roboflow_result.json"
        roboflow_json_path.write_text(json.dumps(roboflow_result, indent=2), encoding="utf-8")

        annotated = draw_predictions(image, predictions, padding=self.settings.crop_padding)
        annotated_path = run_dir / "annotated.png"
        annotated.save(annotated_path)

        crop_records = crop_predictions(image, predictions, crop_dir, padding=self.settings.crop_padding)

        regions = []
        for record in crop_records:
            prediction_text = self.recognizer.predict(record["crop_path"])
            regions.append(
                {
                    "region_number": record["region_number"],
                    "confidence": record["confidence"],
                    "box": {
                        "x_min": record["x_min"],
                        "y_min": record["y_min"],
                        "x_max": record["x_max"],
                        "y_max": record["y_max"],
                    },
                    "crop_url": f"/web-runs/{run_id}/crops/{record['crop_path'].name}",
                    "prediction": prediction_text,
                }
            )

        final_text = "\n".join(region["prediction"] for region in regions if region["prediction"]).strip()
        result = {
            "run_id": run_id,
            "model_dir": str(self.settings.trocr_model_dir),
            "roboflow_model_id": self.settings.roboflow_model_id,
            "confidence_threshold": self.settings.roboflow_confidence_threshold,
            "original_url": f"/web-runs/{run_id}/{original_path.name}",
            "annotated_url": f"/web-runs/{run_id}/annotated.png",
            "region_count": len(regions),
            "regions": regions,
            "final_text": final_text,
        }

        (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (run_dir / "transcription.txt").write_text(final_text, encoding="utf-8")
        return result
