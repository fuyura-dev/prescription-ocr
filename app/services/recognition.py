from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.core.config import Settings


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


class TrOCRRecognizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(settings.trocr_model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(settings.trocr_model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(
            pixel_values,
            max_new_tokens=self.settings.max_new_tokens,
            num_beams=self.settings.num_beams,
        )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return normalize_text(text)
