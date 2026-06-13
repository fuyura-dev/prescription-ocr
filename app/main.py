from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings
from app.services.pipeline import PrescriptionOCRPipeline


def template_context(request: Request, result=None, error=None):
    return {
        "request": request,
        "result": result,
        "error": error,
        "model_name": settings.trocr_model_dir.name,
        "roboflow_model_id": settings.roboflow_model_id,
    }


settings = get_settings()
app = FastAPI(title="Prescription OCR", version="0.1.0")

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.mount("/web-runs", StaticFiles(directory=settings.web_output_dir / "runs"), name="web_runs")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(name="index.html", context=template_context(request), request=request)


@app.post("/process", response_class=HTMLResponse)
def process_prescription(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.upload_extensions:
        raise HTTPException(status_code=400, detail="Upload a JPG, JPEG, or PNG image.")

    upload_dir = settings.web_output_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"{uuid4().hex}{suffix}"

    with upload_path.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)

    try:
        result = PrescriptionOCRPipeline(settings).run(upload_path)
    except Exception as exc:
        return templates.TemplateResponse(
            name="index.html",
            context=template_context(request, error=str(exc)),
            request=request,
            status_code=500,
        )
    finally:
        upload_path.unlink(missing_ok=True)

    return templates.TemplateResponse(name="index.html", context=template_context(request, result=result), request=request)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def show_run(request: Request, run_id: str):
    result_path = settings.web_output_dir / "runs" / run_id / "results.json"
    if not result_path.exists():
        return RedirectResponse(url="/", status_code=303)

    import json

    result = json.loads(result_path.read_text(encoding="utf-8"))
    return templates.TemplateResponse(name="index.html", context=template_context(request, result=result), request=request)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "roboflow_model_id": settings.roboflow_model_id,
        "trocr_model_dir": str(settings.trocr_model_dir),
    }
