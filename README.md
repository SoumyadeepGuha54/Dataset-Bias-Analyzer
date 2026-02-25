# Dataset Bias Analyzer

A comprehensive machine learning fairness auditing tool that detects and analyzes bias in datasets across sensitive demographic attributes. This tool helps data scientists and ML practitioners ensure their models are fair and equitable across different demographic groups.

## 🎯 Purpose

The Dataset Bias Analyzer evaluates fairness issues in machine learning datasets by:

- **Detecting proxy variables** that may indirectly encode sensitive attributes
- **Computing fairness metrics** across demographic groups (demographic parity, equal opportunity, equalized odds, disparate impact)
- **Training multiple ML models** to compare bias patterns across different algorithms
- **Cleaning datasets** to handle missing values, outliers, and data quality issues
- **Visualizing bias patterns** through comprehensive charts and statistical reports

## ✨ Features

### Core Capabilities

- 🔍 **Proxy Variable Detection**: Automatically identifies features that correlate strongly with sensitive attributes
- 📊 **Multi-Model Support**: Compare bias across Logistic Regression, Random Forest, SVM, and XGBoost
- 🧹 **Data Cleaning Pipeline**: Handle missing values, duplicates, outliers, feature scaling, and class imbalance
- 📈 **Comprehensive Visualizations**: Confusion matrices, ROC curves, feature importance, distribution plots, correlation heatmaps
- ⚖️ **Fairness Metrics**: Demographic parity, equal opportunity, equalized odds, disparate impact ratio
- 🌐 **Dual Interface**: Interactive Streamlit web UI and RESTful FastAPI endpoints

### Fairness Metrics Explained

- **Demographic Parity Difference**: Measures difference in approval rates between groups (ideal: 0)
- **Equal Opportunity Difference**: Measures difference in true positive rates between groups (ideal: 0)
- **Equalized Odds Difference**: Measures difference in false positive rates between groups (ideal: 0)
- **Disparate Impact Ratio**: Ratio of approval rates (ideal: 1.0, concerning if < 0.8)

### Supported Models

1. **Logistic Regression** - Fast linear baseline model
2. **Random Forest** - Ensemble tree-based model with feature importance
3. **Support Vector Machine (SVM)** - Non-linear classification with RBF kernel
4. **XGBoost** - Gradient boosting for high-performance classification

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the repository**:

```bash
cd datasetbiasanalyzer
```

2. **Create a virtual environment** (recommended):

```bash
python -m venv venv
```

3. **Activate the virtual environment**:

- Windows:

```bash
venv\Scripts\activate
```

- macOS/Linux:

```bash
source venv/bin/activate
```

4. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## 📖 Usage

### Streamlit Web Interface (Recommended for Interactive Analysis)

1. **Start the Streamlit app**:

```bash
streamlit run app.py
```

2. **Open your browser** to `http://localhost:8501`

3. **Follow the workflow**:
   - **Step 1**: Upload your CSV file
   - **Step 2**: Review data quality report and clean if needed
   - **Step 3**: Select models to compare (Logistic Regression, Random Forest, SVM, XGBoost)
   - **Step 4**: Specify target column (binary 0/1) and sensitive attribute column
   - **Step 5**: View comprehensive results with fairness metrics and visualizations

### FastAPI REST API (For Programmatic Access)

1. **Start the API server**:

```bash
uvicorn api:app --reload
```

2. **Access the API** at `http://localhost:8000`

3. **API Endpoints**:

#### Analyze Dataset for Bias

```bash
POST /analyze?models=logistic_regression,random_forest,xgboost
Content-Type: multipart/form-data

Form Data:
- file: CSV file
- target_col: name of target column
- sensitive_col: name of sensitive attribute column
```

#### Check Data Quality

```bash
POST /analyze-quality
Content-Type: multipart/form-data

Form Data:
- file: CSV file
```

#### Clean Dataset

```bash
POST /clean
Content-Type: multipart/form-data

Form Data:
- file: CSV file
- remove_duplicates: true/false
- impute_missing: true/false
- handle_outliers: true/false
- scale_features: true/false
- balance_classes: true/false
```

#### List Available Models

```bash
GET /models
```

#### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI

## 📂 Dataset Format

### Input CSV Requirements

Your dataset must include:

- **Target column**: Binary classification labels (0 or 1)
- **Sensitive attribute column**: Demographic attribute to analyze (e.g., gender, race, age_group)
- **Feature columns**: Any number of numerical or categorical features

### Example Dataset Structure

```csv
age,income,education,gender,approved
25,50000,Bachelor,Female,1
45,75000,Master,Male,1
32,45000,Bachelor,Female,0
28,55000,Bachelor,Male,1
```

- `approved` = target column (0 = denied, 1 = approved)
- `gender` = sensitive attribute
- `age`, `income`, `education` = features

### Example Datasets

See the `examples/` directory for sample datasets:

- `clean_sample.csv` - Clean dataset ready for analysis
- `unclean_sample.csv` - Dataset with quality issues for testing cleaning pipeline

## 📊 Understanding Results

### Web UI Output

The Streamlit interface provides four main result sections:

1. **Overview Tab**
   - Model accuracy for each trained model
   - Bias detection status (Fair/Biased) with color-coded badges
   - Disparate impact ratio with interpretation

2. **Per-Model Details Tab**
   - Confusion matrices for each demographic group
   - ROC curves with AUC scores
   - Feature importance plots (for tree-based models)

3. **Comparisons Tab**
   - Side-by-side model performance comparison
   - Fairness metrics comparison across all models
   - Best model recommendation based on fairness-accuracy tradeoff

4. **Data Insights Tab**
   - Feature distribution plots (histograms, box plots)
   - Correlation heatmap
   - Proxy variables detected with correlation scores

### API JSON Response

```json
{
  "models": {
    "logistic_regression": {
      "accuracy": 0.84,
      "precision": 0.82,
      "recall": 0.85,
      "f1_score": 0.83,
      "group_metrics": {
        "Female": { "approval_rate": 0.65, "tpr": 0.82, "fpr": 0.15 },
        "Male": { "approval_rate": 0.78, "tpr": 0.88, "fpr": 0.22 }
      },
      "fairness_scores": {
        "demographic_parity_difference": 0.13,
        "equal_opportunity_difference": 0.06,
        "equalized_odds_difference": 0.07,
        "disparate_impact_ratio": 0.83
      },
      "bias_detected": true
    }
  },
  "proxy_variables": {
    "zip_code": 0.78,
    "last_name": 0.65
  },
  "cleaning_report": {
    "duplicates_removed": 15,
    "missing_values_imputed": 42,
    "outliers_capped": 8,
    "smote_samples_added": 150
  }
}
```

### Interpreting Bias Detection

**Bias is flagged when**:

- Demographic Parity Difference > 0.1
- Disparate Impact Ratio < 0.8

**What to do if bias is detected**:

1. Examine proxy variables - consider removing highly correlated features
2. Check class distribution across groups - ensure balanced representation
3. Review feature importance - identify which features drive bias
4. Consider bias mitigation techniques (reweighting, threshold optimization, fairness constraints)
5. Compare models - some models may exhibit less bias than others

## 🏗️ Project Structure

```
datasetbiasanalyzer/
├── engine/                      # Core analysis modules
│   ├── __init__.py
│   ├── bias_report.py          # Bias flag generation logic
│   ├── data_cleaner.py         # Data quality analysis and cleaning
│   ├── data_processor.py       # Data preparation and encoding
│   ├── fairness_metrics.py     # Fairness metric computation
│   ├── model_trainer.py        # Multi-model training
│   ├── proxy_detector.py       # Proxy variable detection
│   └── visualizer.py           # Visualization generation
├── examples/                    # Sample datasets
│   ├── clean_sample.csv
│   ├── unclean_sample.csv
│   └── examples_README.md
├── app.py                       # Streamlit web interface
├── api.py                       # FastAPI REST endpoints
├── main.py                      # Core pipeline orchestrator
├── styles.css                   # UI styling
├── requirements.txt             # Python dependencies
├── LICENSE                      # License file
└── README.md                    # This file
```

## 🔧 Configuration

### Model Hyperparameters

Default hyperparameters are set for balanced performance. Modify in `engine/model_trainer.py`:

- **Logistic Regression**: `max_iter=1000`
- **Random Forest**: `n_estimators=100, max_depth=10, random_state=42`
- **SVM**: `kernel='rbf', C=1.0, random_state=42`
- **XGBoost**: `n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42`

### Data Cleaning Options

Configure cleaning pipeline in the UI or via API:

- `remove_duplicates`: Remove exact duplicate rows
- `impute_missing`: Fill missing values (mean for numerical, mode for categorical)
- `handle_outliers`: Cap outliers using IQR method
- `scale_features`: Standardize numerical features (mean=0, std=1)
- `balance_classes`: Apply SMOTE to balance minority class

### Bias Detection Thresholds

Modify in `engine/bias_report.py`:

- Demographic parity threshold: 0.1
- Disparate impact threshold: 0.8

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional fairness metrics (calibration, predictive parity)
- More ML models (neural networks, ensemble methods)
- Advanced bias mitigation strategies
- Multi-attribute fairness analysis
- Continuous sensitive attributes support

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors**:

```bash
pip install -r requirements.txt
```

**Streamlit app won't start**:

- Ensure port 8501 is not in use
- Try: `streamlit run app.py --server.port 8502`

**API returns 422 errors**:

- Verify CSV file has required target and sensitive columns
- Ensure target column contains only 0 and 1 values
- Check that sensitive column has at least 2 unique values

**Memory errors with large datasets**:

- Reduce dataset size or use sampling
- Train fewer models simultaneously
- Disable visualizations for very large datasets

## 📧 Support

For questions or issues, please check:

1. This README documentation
2. Example datasets in `examples/` directory
3. API documentation at `/docs` endpoint
4. Code comments in source files

---

**Version**: 2.0  
**Last Updated**: February 2026  
**Python Version**: 3.8+
