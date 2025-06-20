import pandas as pd
import numpy as np
import hashlib
import logging
from sklearn.ensemble import IsolationForest
from datetime import datetime
from typing import List, Tuple, Union

from app.database import SessionLocal
from app.models import AnalysisResult

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="logs/anomaly_detection.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

# ────────────────────────────────────────────────────────────────────────────────
# Simple in‑memory cache (hash → prediction list)
# ────────────────────────────────────────────────────────────────────────────────
cache: dict[str, List[int]] = {}

# ────────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────────

def hash_dataframe(df: pd.DataFrame) -> str:
    """Return a SHA‑256 hash for df (including the index)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sanitise the incoming dataframe.

    * Keeps only numeric columns
    * Fills missing values with 0
    * Ensures finiteness
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.select_dtypes(include="number").fillna(0)

    if df.empty:
        raise ValueError("No numeric columns available after filtering.")

    if not np.isfinite(df.values).all():
        raise ValueError("Data contains infinite or non‑numeric values.")

    return df


# ────────────────────────────────────────────────────────────────────────────────
# Anomaly explanation helpers
# ────────────────────────────────────────────────────────────────────────────────

def _compute_reasons(df: pd.DataFrame, preds: List[int], top_n: int = 3) -> List[str]:
    """Return a list of human‑readable reasons for each row.

    For rows flagged as anomalous (pred == 1) the function calculates the feature‑wise
    deviation from the median using a robust MAD denominator. The top_n features
    with the largest absolute Z‑scores are listed.

    Normal rows receive an empty string as reason.
    """
    med = df.median()
    mad = (df - med).abs().median() + 1e-9  # avoid div‑by‑zero

    reasons: List[str] = []
    for i, row in df.iterrows():
        if preds[i] == 0:
            reasons.append("")
            continue

        z_scores = ((row - med).abs() / mad).sort_values(ascending=False)
        top_features = z_scores.head(top_n).index.tolist()
        reason = f"High deviation in {', '.join(top_features)}"
        reasons.append(reason)

    return reasons


def _log_anomalies(df: pd.DataFrame, preds: List[int], reasons: List[str]) -> None:
    """Write each anomaly and its explanation to the logfile."""
    for idx, (row, pred, why) in enumerate(zip(df.iterrows(), preds, reasons)):
        if pred == 1:
            logging.info(
                "Anomaly at index %s | Reason: %s | Row snapshot: %s",
                idx,
                why,
                row[1].to_dict(),
            )


# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────

def detect_anomalies(
    df_raw: pd.DataFrame,
    *,
    user_uid: str,
    file_name: str,
    contamination: float = 0.05,
    return_reasons: bool = False,
) -> Union[List[int], Tuple[List[int], List[str]]]:
    """Detect anomalies in df_raw with Isolation Forest.

    Parameters
    ----------
    df_raw         : Original dataframe coming from the upload / stream.
    user_uid       : Firebase UID of the requesting user.
    file_name      : Friendly name shown in History.
    contamination  : Share of anomalies to assume (IsolationForest hyper‑param).
    return_reasons : If True the function returns a tuple (preds, reasons).
                     Otherwise only the predictions list is returned.

    Returns
    -------
    List[int] or (List[int], List[str]) depending on return_reasons.
    """
    # 1️⃣ Validate & preprocess
    df_proc = validate_schema(df_raw)

    # 2️⃣ Cache lookup
    df_hash = hash_dataframe(df_proc)
    if df_hash in cache:
        preds = cache[df_hash]
    else:
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(df_proc)
        pred_raw = model.predict(df_proc)  # -1 for anomaly, 1 for normal
        preds = [1 if p == -1 else 0 for p in pred_raw]
        cache[df_hash] = preds

    # 3️⃣ Explain anomalies
    reasons = _compute_reasons(df_proc, preds)

    # 4️⃣ Persist per‑file summary
    _store_summary(
        user_uid=user_uid,
        file_name=file_name,
        total_records=len(preds),
        anomaly_count=sum(preds),
    )

    # 5️⃣ Log (only anomalies to keep log size reasonable)
    _log_anomalies(df_proc, preds, reasons)

    return (preds, reasons) if return_reasons else preds


# ────────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────────

def _store_summary(*, user_uid: str, file_name: str, total_records: int, anomaly_count: int) -> None:
    """Insert/commit a row into analysis_results table."""
    db = SessionLocal()
    try:
        db.add(
            AnalysisResult(
                user_id=user_uid,
                file_name=file_name,
                total_records=total_records,
                anomaly_count=anomaly_count,
                timestamp=datetime.utcnow(),
            )
        )
        db.commit()
    finally:
        db.close()