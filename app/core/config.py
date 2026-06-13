from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    project_root: Path = PROJECT_ROOT
    roboflow_api_key: str | None = os.getenv("ROBOFLOW_API_KEY")
    roboflow_model_id: str | None = os.getenv("ROBOFLOW_MODEL_ID")
    roboflow_api_url: str = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
    roboflow_confidence_threshold: float = float(os.getenv("ROBOFLOW_CONFIDENCE_THRESHOLD", "0.50"))
    trocr_model_dir: Path = (PROJECT_ROOT / os.getenv("TROCR_MODEL_DIR", "models/trocr_prescription_lines-v2")).resolve()
    web_output_dir: Path = PROJECT_ROOT / "outputs" / "web_inference"
    upload_extensions: set[str] = {".jpg", ".jpeg", ".png"}
    crop_padding: int = int(os.getenv("OCR_CROP_PADDING", "4"))
    max_new_tokens: int = int(os.getenv("TROCR_MAX_NEW_TOKENS", "64"))
    num_beams: int = int(os.getenv("TROCR_NUM_BEAMS", "4"))

    def validate(self) -> None:
        if not self.roboflow_api_key:
            raise ValueError("ROBOFLOW_API_KEY is missing in .env")
        if not self.roboflow_model_id:
            raise ValueError("ROBOFLOW_MODEL_ID is missing in .env")
        if not self.trocr_model_dir.exists():
            raise FileNotFoundError(f"TrOCR model folder not found: {self.trocr_model_dir}")
        self.web_output_dir.mkdir(parents=True, exist_ok=True)
        (self.web_output_dir / "runs").mkdir(parents=True, exist_ok=True)
        (self.web_output_dir / "uploads").mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
