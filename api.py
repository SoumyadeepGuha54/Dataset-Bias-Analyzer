import io
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from main import run_bias_engine

app = FastAPI(
    title="Bias Detection Engine",
    description="Upload a CSV dataset and analyze it for bias across sensitive attributes.",
    version="1.0.0",
)

# Allow CORS for Streamlit or any front-end client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NumpyJSONEncoder(json.JSONEncoder):
    """Handle numpy types that are not JSON-serializable by default."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _serialize(obj):
    """Round-trip through NumpyJSONEncoder so FastAPI returns clean JSON."""
    return json.loads(json.dumps(obj, cls=NumpyJSONEncoder))


@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok", "message": "Bias Detection Engine is running."}


@app.post("/analyze", tags=["Analysis"])
async def analyze(
    file: UploadFile = File(..., description="CSV dataset to analyze"),
    target_col: str = Form(..., description="Name of the target / label column"),
    sensitive_col: str = Form(..., description="Name of the sensitive attribute column"),
):
    """
    Upload a CSV file and receive a structured bias report containing:
    - Model accuracy
    - Per-group metrics (approval rate, TPR, FPR)
    - Fairness scores (demographic parity, equal opportunity, equalized odds, disparate impact)
    - Bias detected flag
    - Proxy variables (features highly correlated with the sensitive attribute)
    """
    # --- Validate file type ---
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # --- Read CSV ---
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # --- Run Engine ---
    try:
        result = run_bias_engine(df, target_col, sensitive_col)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return _serialize(result)
