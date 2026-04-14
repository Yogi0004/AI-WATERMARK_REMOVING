"""
AI-Powered Watermark Removal System for Real Estate Images
FastAPI Backend - Production Ready
"""

import io
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from detection import WatermarkDetector
from inpainting import InpaintingEngine
from postprocessing import PostProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real Estate Watermark Removal API",
    description="AI-powered watermark removal using LaMa inpainting",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import PlainTextResponse
import traceback

detector = WatermarkDetector()
inpainter = InpaintingEngine()
post_processor = PostProcessor()

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return PlainTextResponse(str(traceback.format_exc()), status_code=500)

# Initial log for user feedback
logger.info(f"IMAGE ENGINE: {inpainter.model_name.upper()} active")
if inpainter.model_name == "opencv_ns":
    logger.warning("!!! WARNING: Using Low-Quality OpenCV Fallback. AI features disabled !!!")

BATCH_RESULTS: dict = {}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "detector": "ready",
        "inpainter": inpainter.model_name,
        "ocr_backend": detector.ocr_backend
    }


@app.post("/remove-watermark")
async def remove_watermark(
    file: UploadFile = File(...),
    sharpen: bool = True,
    color_correct: bool = True,
    manual_mask_center: bool = False
):
    """
    Remove watermark from a single real estate image.
    Returns cleaned image as PNG.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_np = np.array(pil_img)

    logger.info(f"Processing image: {file.filename} | size: {pil_img.size}")

    preprocessed, scale = preprocess_image(img_np)

    if manual_mask_center:
        mask = detector.generate_center_mask(preprocessed)
        logger.info("Using manual center mask.")
    else:
        mask = detector.detect_watermark(preprocessed)
        logger.info(f"OCR detection complete. Mask coverage: {mask.mean():.4f}")

    result_np = inpainter.inpaint(preprocessed, mask)

    result_np = post_processor.process(
        result_np,
        original=preprocessed,
        mask=mask,
        sharpen=sharpen,
        color_correct=color_correct
    )

    if scale != 1.0:
        h, w = img_np.shape[:2]
        result_np = cv2.resize(result_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

    result_pil = Image.fromarray(result_np.astype(np.uint8))
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/remove-watermark/mask-preview")
async def get_mask_preview(file: UploadFile = File(...)):
    """
    Returns the detected watermark mask as a grayscale PNG for inspection.
    """
    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_np = np.array(pil_img)

    preprocessed, _ = preprocess_image(img_np)
    mask = detector.detect_watermark(preprocessed)

    mask_pil = Image.fromarray(mask.astype(np.uint8))
    buf = io.BytesIO()
    mask_pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/remove-watermark/batch")
async def batch_remove_watermark(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    sharpen: bool = True,
    color_correct: bool = True
):
    """
    Submit batch watermark removal. Returns a job_id.
    Poll /batch-status/{job_id} for results.
    """
    job_id = str(uuid.uuid4())
    images = []
    for f in files:
        raw = await f.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        images.append((f.filename, np.array(pil_img)))

    BATCH_RESULTS[job_id] = {"status": "processing", "total": len(images), "done": 0, "results": []}
    background_tasks.add_task(process_batch, job_id, images, sharpen, color_correct)
    return {"job_id": job_id, "total": len(images)}


@app.get("/batch-status/{job_id}")
async def batch_status(job_id: str):
    if job_id not in BATCH_RESULTS:
        raise HTTPException(404, "Job not found.")
    info = BATCH_RESULTS[job_id]
    return {
        "job_id": job_id,
        "status": info["status"],
        "total": info["total"],
        "done": info["done"]
    }


@app.get("/batch-result/{job_id}/{index}")
async def batch_result(job_id: str, index: int):
    if job_id not in BATCH_RESULTS:
        raise HTTPException(404, "Job not found.")
    results = BATCH_RESULTS[job_id]["results"]
    if index >= len(results):
        raise HTTPException(404, "Result not ready yet.")
    result_bytes = results[index]
    return StreamingResponse(io.BytesIO(result_bytes), media_type="image/png")


async def process_batch(job_id: str, images, sharpen: bool, color_correct: bool):
    for name, img_np in images:
        try:
            preprocessed, scale = preprocess_image(img_np)
            mask = detector.detect_watermark(preprocessed)
            result = inpainter.inpaint(preprocessed, mask)
            result = post_processor.process(
                result, 
                original=preprocessed, 
                mask=mask, 
                sharpen=sharpen, 
                color_correct=color_correct
            )
            if scale != 1.0:
                h, w = img_np.shape[:2]
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
            pil = Image.fromarray(result.astype(np.uint8))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            BATCH_RESULTS[job_id]["results"].append(buf.getvalue())
        except Exception as e:
            logger.error(f"Batch error on {name}: {e}")
            BATCH_RESULTS[job_id]["results"].append(b"")
        BATCH_RESULTS[job_id]["done"] += 1
        await asyncio.sleep(0)
    BATCH_RESULTS[job_id]["status"] = "done"
    logger.info(f"Batch job {job_id} complete.")


def preprocess_image(img: np.ndarray, max_dim: int = 1024):
    """Resize to max_dim while preserving aspect ratio. Returns (resized, scale)."""
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)