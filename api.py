"""
Enhanced FastAPI endpoints for Dataset Bias Analyzer
with multi-model support, data quality checks, and cleaning.
"""

import io
import json
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from main import run_bias_engine
from engine.data_cleaner import analyze_data_quality, clean_dataset
from engine.model_trainer import get_available_models

app = FastAPI(
    title="Dataset Bias Analyzer API",
    description="Comprehensive bias detection API with multi-model support, data quality analysis, and cleaning capabilities.",
    version="2.0.0",
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
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Dataset Bias Analyzer API is running",
        "version": "2.0.0"
    }


@app.get("/models", tags=["Models"])
async def list_models():
    """
    Get list of available ML models for bias analysis.
    
    Returns:
        dict: Available models with their display names
    """
    models = get_available_models()
    return {
        "available_models": models,
        "count": len(models)
    }


@app.post("/analyze-quality", tags=["Data Quality"])
async def analyze_quality(
    file: UploadFile = File(..., description="CSV dataset to analyze"),
    target_col: Optional[str] = Form(None, description="Name of target column (for class distribution analysis)"),
):
    """
    Analyze dataset for quality issues including:
    - Missing values
    - Duplicate rows
    - Outliers (using IQR method)
    - Data types
    - Class distribution (if target_col provided)
    
    Returns:
        dict: Comprehensive data quality report
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

    # --- Analyze Quality ---
    try:
        quality_report = analyze_data_quality(df, target_col)
        return _serialize(quality_report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality analysis failed: {e}")


@app.post("/clean", tags=["Data Quality"])
async def clean_data(
    file: UploadFile = File(..., description="CSV dataset to clean"),
    target_col: Optional[str] = Form(None, description="Name of target column"),
    sensitive_col: Optional[str] = Form(None, description="Name of sensitive attribute column"),
    remove_duplicates: bool = Form(True, description="Remove duplicate rows"),
    impute_missing: bool = Form(True, description="Impute missing values"),
    handle_outliers: bool = Form(True, description="Handle outliers using IQR method"),
    scale_features: bool = Form(False, description="Scale numerical features"),
    balance_classes: bool = Form(False, description="Balance classes using SMOTE"),
):
    """
    Clean dataset based on specified options.
    
    Returns:
        dict: Cleaned dataset as CSV string and cleaning report
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

    # --- Clean Dataset ---
    try:
        cleaned_df, cleaning_report = clean_dataset(
            df,
            target_col=target_col,
            sensitive_col=sensitive_col,
            remove_duplicates=remove_duplicates,
            impute_missing=impute_missing,
            handle_outliers=handle_outliers,
            scale_features=scale_features,
            balance_classes=balance_classes
        )
        
        # Convert cleaned dataframe to CSV string
        csv_buffer = io.StringIO()
        cleaned_df.to_csv(csv_buffer, index=False)
        cleaned_csv = csv_buffer.getvalue()
        
        return {
            "cleaned_data": cleaned_csv,
            "cleaning_report": _serialize(cleaning_report)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {e}")


@app.post("/analyze", tags=["Analysis"])
async def analyze(
    file: UploadFile = File(..., description="CSV dataset to analyze"),
    target_col: str = Form(..., description="Name of the target / label column"),
    sensitive_col: str = Form(..., description="Name of the sensitive attribute column"),
    models: Optional[str] = Query(
        None,
        description="Comma-separated list of models to train (e.g., 'logistic_regression,random_forest,xgboost'). If not provided, uses logistic regression only."
    ),
    clean_data: bool = Query(
        False,
        description="Whether to automatically clean the data before analysis"
    ),
):
    """
    Upload a CSV file and receive a comprehensive bias report.
    
    Supports multiple models for comparison:
    - logistic_regression: Logistic Regression
    - random_forest: Random Forest
    - svm: Support Vector Machine
    - xgboost: XGBoost (if installed)
    
    Returns:
        dict: Comprehensive bias analysis including:
        - models: Per-model results with predictions, metrics, fairness scores, bias detection
        - proxy_variables: Features correlated with sensitive attribute
        - cleaning_report: Cleaning statistics (if clean_data=True)
        - data_quality: Data quality analysis
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

    # --- Parse models parameter ---
    model_list = None
    if models:
        model_list = [m.strip() for m in models.split(',') if m.strip()]
        
        # Validate model names
        available_models = get_available_models()
        invalid_models = [m for m in model_list if m not in available_models]
        if invalid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model(s): {invalid_models}. Available: {list(available_models.keys())}"
            )
    
    # --- Prepare cleaning config ---
    cleaning_config = None
    if clean_data:
        cleaning_config = {
            'remove_duplicates': True,
            'impute_missing': True,
            'handle_outliers': True,
            'scale_features': False,
            'balance_classes': False
        }

    # --- Run Engine ---
    try:
        result = run_bias_engine(
            df,
            target_col,
            sensitive_col,
            models=model_list,
            cleaning_config=cleaning_config
        )
        return _serialize(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

