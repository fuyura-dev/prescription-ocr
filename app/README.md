# Prescription OCR Web App

Inference-only website for the prescription OCR pipeline.

Pipeline:

1. Upload a full prescription image.
2. Roboflow localizes handwritten text-line regions.
3. Detected regions are cropped.
4. The configured TrOCR model recognizes text from each crop.
5. The app displays detected boxes, crop-level predictions, and final text.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Model

The default model is configured through `.env`:

```env
TROCR_MODEL_DIR=models/trocr_prescription_lines-v2
```

Change that value to switch models without changing code.


## Camera

The web interface supports browser camera capture on `localhost`, `127.0.0.1`, or HTTPS.
Use the `Mirror` toggle if the preview appears reversed. Captured camera images are submitted to the same `/process` endpoint as uploaded files.
