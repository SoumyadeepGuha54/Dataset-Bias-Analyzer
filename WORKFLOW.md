# Dataset Bias Analyzer - Enhanced Workflow

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                          │
├──────────────────────┬──────────────────────────────────────┤
│   Streamlit Web UI   │         FastAPI REST API             │
│     (app.py)         │           (api.py)                   │
└──────────┬───────────┴───────────┬──────────────────────────┘
           │                       │
           └───────────┬───────────┘
                       │
           ┌───────────▼────────────┐
           │   Pipeline Orchestrator │
           │      (main.py)          │
           └───────────┬────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐   ┌────▼─────┐  ┌────▼─────┐
   │  Data   │   │  Model   │  │Fairness  │
   │Cleaning │   │ Training │  │ Metrics  │
   └─────────┘   └──────────┘  └──────────┘
```

## Enhanced User Workflow

### Streamlit Web Interface

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Upload Dataset                                       │
├──────────────────────────────────────────────────────────────┤
│ • Upload CSV file                                            │
│ • System displays: rows × columns                            │
│ • Select target column (binary 0/1)                          │
│ • Select sensitive attribute (demographic column)            │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Data Quality Analysis (Optional but Recommended)    │
├──────────────────────────────────────────────────────────────┤
│ • Click "Check Data Quality"                                 │
│ • System analyzes and reports:                               │
│   ✓ Missing values (column-by-column)                        │
│   ✓ Duplicate rows                                           │
│   ✓ Outliers (using IQR method)                              │
│   ✓ Class distribution (balance check)                       │
│   ✓ Data types                                               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
                  ┌───────┴────────┐
                  │ Issues Found?  │
                  └───┬────────┬───┘
                 YES  │        │ NO
                      ▼        └────────────────┐
┌──────────────────────────────────────────────┐│
│ STEP 3: Data Cleaning                       ││
├──────────────────────────────────────────────┤│
│ Configure cleaning options:                 ││
│ □ Remove duplicates                          ││
│ □ Impute missing values (mean/median/mode)   ││
│ □ Handle outliers (cap using IQR)            ││
│ □ Scale features (standardization)           ││
│ □ Balance classes (SMOTE oversampling)       ││
│                                               ││
│ • Click "Clean Dataset"                      ││
│ • View cleaning summary:                     ││
│   - Duplicates removed: X                    ││
│   - Values imputed: Y                         ││
│   - Outliers capped: Z                        ││
│   - Final row count: N                        ││
└───────────────────┬──────────────────────────┘│
                    │                           │
                    └───────────┬───────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Model Selection                                     │
├──────────────────────────────────────────────────────────────┤
│ Select one or more models to compare:                        │
│ ☑ Logistic Regression (fast linear baseline)                │
│ ☑ Random Forest (ensemble with feature importance)          │
│ ☑ Support Vector Machine (non-linear SVM)                   │
│ ☑ XGBoost (gradient boosting)                               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: Run Analysis                                        │
├──────────────────────────────────────────────────────────────┤
│ • Click "Run Analysis" button                                │
│ • System performs:                                           │
│   1. Proxy variable detection (pre-encoding)                 │
│   2. Data preparation (encoding, train-test split)           │
│   3. Train all selected models                               │
│   4. Compute fairness metrics per model                      │
│   5. Generate visualizations                                 │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 6: Review Results (Tabbed Interface)                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ TAB 1: Overview                                          │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ • Model performance KPIs (accuracy, bias status)         │ │
│ │ • Performance comparison chart (all models)              │ │
│ │ • Fairness metrics comparison chart                      │ │
│ │   - Demographic Parity Difference                        │ │
│ │   - Equal Opportunity Difference                         │ │
│ │   - Equalized Odds Difference                            │ │
│ │   - Disparate Impact Ratio                               │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ TAB 2: Per-Model Details                                 │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ Select model from dropdown:                              │ │
│ │ • Performance metrics (accuracy, precision, recall, F1)  │ │
│ │ • Group-level metrics (approval rate, TPR, FPR)          │ │
│ │ • Confusion matrices per demographic group               │ │
│ │ • ROC curves (overall + per group) with AUC scores       │ │
│ │ • Feature importance (top 15 features)                   │ │
│ │ • Fairness scores detailed table                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ TAB 3: Model Comparisons                                 │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ • Side-by-side comparison table                          │ │
│ │   (all models with metrics and fairness scores)          │ │
│ │ • Best model recommendation:                             │ │
│ │   - Highest accuracy among fair models, OR               │ │
│ │   - Least biased if all models show bias                 │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ TAB 4: Data Insights                                     │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ • Feature distribution plots (histograms, box plots)     │ │
│ │ • Correlation heatmap (numerical features)               │ │
│ │ • Proxy variable detection results                       │ │
│ │   - List of features with high correlation to            │ │
│ │     sensitive attribute (threshold: 0.7)                 │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ • Raw JSON results (expandable section at bottom)           │
└──────────────────────────────────────────────────────────────┘
```

---

## API Workflow

### Endpoint: `/analyze-quality` (POST)

```
INPUT:
┌────────────────────┐
│ CSV File           │
│ target_col (opt)   │
└─────────┬──────────┘
          │
          ▼
┌─────────────────────────────────────┐
│ Data Quality Analysis               │
├─────────────────────────────────────┤
│ • Count missing values per column   │
│ • Detect duplicate rows              │
│ • Identify outliers (IQR method)     │
│ • Analyze class distribution         │
└─────────┬───────────────────────────┘
          │
          ▼
OUTPUT:
┌────────────────────────────────────┐
│ {                                   │
│   "total_rows": 100,                │
│   "total_columns": 10,              │
│   "missing_values": {...},          │
│   "duplicates": 5,                  │
│   "outliers": {...},                │
│   "class_distribution": {...}       │
│ }                                   │
└────────────────────────────────────┘
```

### Endpoint: `/clean` (POST)

```
INPUT:
┌─────────────────────────────────────┐
│ CSV File                             │
│ target_col, sensitive_col (opt)      │
│ remove_duplicates: bool              │
│ impute_missing: bool                 │
│ handle_outliers: bool                │
│ scale_features: bool                 │
│ balance_classes: bool                │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│ Data Cleaning Pipeline              │
├─────────────────────────────────────┤
│ 1. Remove duplicates (if enabled)   │
│ 2. Impute missing values             │
│ 3. Cap outliers (IQR method)         │
│ 4. Scale features (StandardScaler)   │
│ 5. Balance classes (SMOTE)           │
└─────────┬───────────────────────────┘
          │
          ▼
OUTPUT:
┌─────────────────────────────────────┐
│ {                                    │
│   "cleaned_data": "CSV string",      │
│   "cleaning_report": {               │
│     "duplicates_removed": 5,         │
│     "missing_values_imputed": 12,    │
│     "outliers_capped": 8,            │
│     "original_rows": 100,            │
│     "final_rows": 95                 │
│   }                                   │
│ }                                    │
└─────────────────────────────────────┘
```

### Endpoint: `/analyze` (POST)

```
INPUT:
┌─────────────────────────────────────────┐
│ CSV File                                 │
│ target_col: string                       │
│ sensitive_col: string                    │
│ models: "lr,rf,xgb" (comma-separated)   │
│ clean_data: bool (optional)              │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ Optional: Auto-clean if requested       │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ Proxy Detection (pre-encoding)          │
├─────────────────────────────────────────┤
│ • Cramér's V for categorical features   │
│ • Mutual Info for numerical features    │
│ • Threshold: 0.7                         │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ Data Preparation                         │
├─────────────────────────────────────────┤
│ • Drop NA in target/sensitive columns   │
│ • One-hot encode categorical features   │
│ • Train-test split (70-30)               │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ Multi-Model Training                     │
├─────────────────────────────────────────┤
│ For each selected model:                │
│ 1. Train model on training data          │
│ 2. Generate predictions on test data     │
│ 3. Calculate metrics (acc, prec, rec, F1)│
│ 4. Get prediction probabilities          │
│ 5. Extract feature importance            │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│ Fairness Evaluation (per model)         │
├─────────────────────────────────────────┤
│ 1. Compute group metrics:                │
│    - Approval rate per group             │
│    - True Positive Rate (TPR)            │
│    - False Positive Rate (FPR)           │
│                                           │
│ 2. Compute fairness scores:              │
│    - Demographic Parity Difference       │
│    - Equal Opportunity Difference        │
│    - Equalized Odds Difference           │
│    - Disparate Impact Ratio              │
│                                           │
│ 3. Bias detection:                       │
│    - Flag if DP Diff > 0.1               │
│    - Flag if DI Ratio < 0.8              │
└─────────┬───────────────────────────────┘
          │
          ▼
OUTPUT:
┌──────────────────────────────────────────┐
│ {                                         │
│   "models": {                             │
│     "logistic_regression": {              │
│       "predictions": [...],               │
│       "probabilities": [...],             │
│       "metrics": {...},                   │
│       "group_metrics": {...},             │
│       "fairness_scores": {...},           │
│       "bias_detected": true/false,        │
│       "feature_importance": {...}         │
│     },                                     │
│     "random_forest": {...},               │
│     ...                                    │
│   },                                       │
│   "proxy_variables": {...},               │
│   "cleaning_report": {...},               │
│   "data_quality": {...}                   │
│ }                                          │
└──────────────────────────────────────────┘
```

### Endpoint: `/models` (GET)

```
OUTPUT:
┌───────────────────────────────────────┐
│ {                                      │
│   "available_models": {                │
│     "logistic_regression": "Logistic   │
│                            Regression",│
│     "random_forest": "Random Forest",  │
│     "svm": "Support Vector Machine",   │
│     "xgboost": "XGBoost"               │
│   },                                    │
│   "count": 4                            │
│ }                                       │
└───────────────────────────────────────┘
```

---

## Data Flow for Unclean Datasets

```
┌──────────────────────────────────────────────────────────────┐
│ SCENARIO: User uploads dataset with quality issues          │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. Upload Dataset                                            │
│    • CSV has: missing values, duplicates, outliers           │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Check Data Quality                                        │
│    • System detects and reports all issues                   │
│    • UI displays warnings with detailed breakdown            │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. User Configures Cleaning                                 │
│    • Enables desired cleaning options via checkboxes        │
│    • Example configuration:                                  │
│      ✓ Remove duplicates: YES                                │
│      ✓ Impute missing: YES (mean/median/mode)                │
│      ✓ Handle outliers: YES (IQR capping)                    │
│      ✓ Scale features: NO (not needed for tree models)       │
│      ✓ Balance classes: YES (if imbalanced)                  │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Clean Dataset                                             │
│    • System applies selected cleaning operations             │
│    • Displays before/after statistics                        │
│    • Original: 100 rows → Cleaned: 95 rows (5 dupes removed) │
│    • 12 missing values imputed                               │
│    • 8 outliers capped to bounds                             │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Proceed with Analysis                                     │
│    • Analysis runs on cleaned dataset                        │
│    • All models trained on clean data                        │
│    • Results more reliable than with unclean data            │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Features Summary

### ✅ Data Quality & Cleaning

- **Automatic detection** of missing values, duplicates, outliers
- **Configurable cleaning** with user control over each operation
- **Transparency** with before/after statistics
- **Smart imputation** (mean/median for numerical, mode for categorical)
- **Class balancing** using SMOTE for fairness

### ✅ Multi-Model Support

- **4 ML algorithms**: Logistic Regression, Random Forest, SVM, XGBoost
- **Parallel training** and comparison
- **Feature importance** extraction for interpretability
- **Recommendation engine** to suggest best fair model

### ✅ Comprehensive Visualizations

- **Confusion matrices** per demographic group
- **ROC curves** with AUC scores (overall + per group)
- **Feature importance** bar charts
- **Distribution plots** for data exploration
- **Correlation heatmaps** to understand relationships
- **Model comparison charts** for side-by-side analysis

### ✅ Fairness Analysis

- **4 fairness metrics** computed per model
- **Bias detection** with configurable thresholds
- **Proxy variable detection** to identify indirect discrimination
- **Group-level metrics** (approval rate, TPR, FPR)

### ✅ Dual Interface

- **Streamlit Web UI** for interactive analysis
- **FastAPI REST API** for programmatic access and integration

---

## Next Steps After Analysis

### If Bias is Detected:

1. **Review Proxy Variables**
   - Consider removing features with high correlation to sensitive attribute
   - Assess if correlation is justified or problematic

2. **Check Data Balance**
   - Ensure demographic groups are adequately represented
   - Consider collecting more data for underrepresented groups

3. **Examine Feature Importance**
   - Identify which features drive predictions
   - Assess if important features have disparate impact

4. **Compare Models**
   - Some models may be inherently more fair than others
   - Choose least biased model if performance is comparable

5. **Apply Mitigation Techniques** (Future Enhancement)
   - Reweighting training samples
   - Threshold optimization per group
   - Fairness constraints in training

### If No Bias Detected:

1. **Document Results**
   - Save fairness report for compliance/auditing
   - Note which models and metrics were used

2. **Monitor in Production**
   - Bias can emerge over time with data drift
   - Re-run analysis periodically

3. **Consider Edge Cases**
   - Test on different data slices
   - Validate with domain experts

---

## Technical Notes

- **Train-Test Split**: Fixed 70-30 split with random_state=42 for reproducibility
- **Encoding**: Automatic one-hot encoding with drop_first=True
- **Proxy Detection Threshold**: 0.7 correlation (Cramér's V or Mutual Information)
- **Bias Detection Thresholds**: DP Diff > 0.1 OR DI Ratio < 0.8
- **Outlier Method**: IQR (Interquartile Range) with 1.5× IQR bounds
- **SMOTE**: Only applied if target classes are imbalanced (minority < 50% of majority)

---

## File Structure Reference

```
datasetbiasanalyzer/
├── engine/
│   ├── bias_report.py          # Bias flag generation
│   ├── data_cleaner.py         # ✨ NEW: Data quality & cleaning
│   ├── data_processor.py       # Data preparation & encoding
│   ├── fairness_metrics.py     # Fairness metric computation
│   ├── model_trainer.py        # ✨ ENHANCED: Multi-model training
│   ├── proxy_detector.py       # Proxy variable detection
│   └── visualizer.py           # ✨ NEW: Comprehensive visualizations
├── examples/
│   ├── clean_sample.csv        # ✨ NEW: Clean dataset example
│   ├── unclean_sample.csv      # ✨ NEW: Unclean dataset example
│   └── examples_README.md      # ✨ NEW: Example documentation
├── app.py                       # ✨ ENHANCED: Streamlit UI with tabs
├── api.py                       # ✨ ENHANCED: FastAPI with new endpoints
├── main.py                      # ✨ ENHANCED: Pipeline with multi-model
├── styles.css                   # ✨ ENHANCED: Styling for new UI
├── requirements.txt             # ✨ UPDATED: New dependencies
├── README.md                    # ✨ NEW: Comprehensive documentation
├── WORKFLOW.md                  # ✨ NEW: This file
└── LICENSE
```

---

**Version**: 2.0  
**Last Updated**: February 2026
