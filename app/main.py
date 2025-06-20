from fastapi import FastAPI, UploadFile, File, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

import pandas as pd
import numpy as np
import uuid
import io
from pathlib import Path
from typing import Dict

from app.services.ml_model import detect_anomalies
from app.services.feature_engineering import preprocess
from app.utils.firebase_auth import (
    get_current_user,
    get_current_user_optional_ws,
)
from app.models import Base
from app.database import engine
from app.routes.results import router as results_router

import logging
import asyncio
import datetime
import random

# ────────────────────────────────────────────────────────────────────────────────
# Initialisation & Config
# ────────────────────────────────────────────────────────────────────────────────

# Create DB tables (Alembic is better, but this works for PoC)
Base.metadata.create_all(bind=engine)

# Logging
logger = logging.getLogger("user‑pattern‑analyzer")
logging.basicConfig(level=logging.INFO)

# Local storage for CSV downloads
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="User Pattern Analyzer API")

# CORS (assumes Vite dev server on :5173) – adjust as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Plug in /results endpoints
app.include_router(results_router)

# ────────────────────────────────────────────────────────────────────────────────
# Auth test helper
# ────────────────────────────────────────────────────────────────────────────────

@app.get("/test-auth")
async def test_auth_route(user: Dict = Depends(get_current_user)):
    return {"message": "Authenticated!", "user": user}


# ────────────────────────────────────────────────────────────────────────────────
# In‑memory anomaly cache (file_id → DataFrame[anomalies])
# ────────────────────────────────────────────────────────────────────────────────

ANOMALY_STORE: Dict[str, pd.DataFrame] = {}

# ────────────────────────────────────────────────────────────────────────────────
# File upload → anomaly detection
# ────────────────────────────────────────────────────────────────────────────────


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    user: Dict = Depends(get_current_user),  # 🔐 login enforced
):
    """Upload a CSV/TSV/JSON file, run anomaly detection, return annotated rows."""

    raw_bytes = await file.read()
    filename = file.filename or "uploaded"

    # ── 1. Parse
    try:
        if filename.endswith(".json"):
            df_raw = pd.read_json(io.BytesIO(raw_bytes))
        else:
            df_raw = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        logger.exception("Failed to parse file")
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}")

    # ── 2. Feature engineering / sanitisation
    df_proc = preprocess(df_raw)

    # ── 3. Detect anomalies + explanations
    preds, reasons = detect_anomalies(
        df_raw=df_proc,
        user_uid=user["uid"],
        file_name=filename,
        return_reasons=True,
    )

    # ── 4. Build response DataFrame (preserve original columns)
    df_out = df_raw.copy()
    df_out["anomaly"] = preds
    df_out["anomaly_reason"] = reasons

    # ── 5. Quick summary for UI cards / notifications
    summary = {
        "total": int(len(df_out)),
        "anomalies": int((df_out["anomaly"] == 1).sum()),
        "normal": int((df_out["anomaly"] == 0).sum()),
    }

    # ── 6. Store anomalies only (for download endpoint)
    file_id = str(uuid.uuid4())
    ANOMALY_STORE[file_id] = df_out[df_out["anomaly"] == 1].copy()

    # ── 7. JSON‑serialisable response
    df_json_safe = df_out.replace({np.nan: None}).to_dict(orient="records")

    return jsonable_encoder(
        {
            "summary": summary,
            "rows": df_json_safe,
            "file_id": file_id,
        }
    )


# ────────────────────────────────────────────────────────────────────────────────
# CSV download endpoint (anomalies only)
# ────────────────────────────────────────────────────────────────────────────────


@app.get("/download/{file_id}")
async def download(
    file_id: str,
    user: Dict = Depends(get_current_user),  # 🔐 login enforced
):
    """Download the anomalies for file_id as CSV."""
    if file_id not in ANOMALY_STORE:
        raise HTTPException(status_code=404, detail="File ID not found")

    tmp_path = DATA_DIR / f"anomalies_{file_id}.csv"
    ANOMALY_STORE[file_id].to_csv(tmp_path, index=False)

    return FileResponse(tmp_path, filename=tmp_path.name, media_type="text/csv")


# ────────────────────────────────────────────────────────────────────────────────
# WebSocket live‑stream demo (simulated)
# ────────────────────────────────────────────────────────────────────────────────


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    # If you want to restrict to logged‑in users only, uncomment:
    # await get_current_user_optional_ws(websocket)

    await websocket.accept()
    try:
        while True:
            is_anomaly = random.random() < 0.1
            row = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "Port": random.randint(1000, 65000),
                "Bytes": random.randint(1000, 10000),
                "Packets": random.randint(1, 500),
                "anomaly": 1 if is_anomaly else 0,
                "anomaly_reason": "Simulated high traffic" if is_anomaly else "",
            }
            await websocket.send_json(row)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
