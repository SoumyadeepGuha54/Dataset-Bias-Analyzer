# Example Datasets

This directory contains sample datasets for testing the Dataset Bias Analyzer.

## Files

### 1. clean_sample.csv

**Description**: A clean, ready-to-analyze dataset with no quality issues.

**Scenario**: Loan approval dataset with 50 applicant records.

**Features**:

- `age`: Applicant age (years)
- `income`: Annual income (USD)
- `education`: Education level (High School, Associate, Bachelor, Master, PhD)
- `gender`: Gender (Male, Female) - **Sensitive Attribute**
- `credit_score`: Credit score (300-850)
- `approved`: Loan approval decision (0=denied, 1=approved) - **Target Variable**

**Characteristics**:

- No missing values
- No duplicates
- No outliers
- Balanced male/female representation
- Ready for immediate analysis

**Usage**:

```python
import pandas as pd
from main import run_bias_engine

df = pd.read_csv('examples/clean_sample.csv')
results = run_bias_engine(
    df,
    target_col='approved',
    sensitive_col='gender',
    models=['logistic_regression', 'random_forest', 'xgboost']
)
```

---

### 2. unclean_sample.csv

**Description**: A dataset with intentional quality issues for testing the data cleaning pipeline.

**Scenario**: Same loan approval scenario, but with data quality problems.

**Quality Issues**:

1. **Missing Values**:
   - 3 missing ages (empty cells)
   - 2 missing incomes
   - 1 missing education level
   - 2 missing credit scores

2. **Duplicates**:
   - 2 exact duplicate rows (rows 1 and 51, rows 2 and 52)

3. **Outliers**:
   - Age outliers: 150 years, 19 years
   - Income outliers: $250,000, $15,000, $180,000
   - Credit score outliers: 850, 400, 780

4. **Total Records**: 60 rows (including duplicates)

**Expected Cleaning Results**:

- Duplicates removed: 2 rows
- Missing values imputed: 8 values
- Outliers capped: ~6-8 values (depending on method)
- Final rows: 58 (after duplicate removal)

**Usage**:

```python
import pandas as pd
from main import run_bias_engine

df = pd.read_csv('examples/unclean_sample.csv')

# With automatic cleaning
results = run_bias_engine(
    df,
    target_col='approved',
    sensitive_col='gender',
    models=['logistic_regression', 'random_forest'],
    cleaning_config={
        'remove_duplicates': True,
        'impute_missing': True,
        'handle_outliers': True,
        'scale_features': False,
        'balance_classes': False
    }
)

# Check cleaning report
print(results['cleaning_report'])
```

---

## Testing Scenarios

### Scenario 1: Compare Clean vs Unclean

Compare analysis results between clean and unclean data to see how data quality affects bias detection.

### Scenario 2: Test Cleaning Pipeline

Use `unclean_sample.csv` to verify that the cleaning module correctly:

- Identifies all quality issues
- Removes duplicates
- Imputes missing values appropriately
- Handles outliers without losing too much data

### Scenario 3: Multi-Model Comparison

Use `clean_sample.csv` to compare fairness metrics across different models (Logistic Regression, Random Forest, SVM, XGBoost).

### Scenario 4: Visualizations

Test all visualization functions with both datasets to ensure they handle various data conditions.

---

## Notes

- Both datasets use the same feature schema for consistency
- Target variable (`approved`) is binary: 0 (denied) or 1 (approved)
- Sensitive attribute (`gender`) has two values: Male and Female
- Datasets are small for demonstration purposes; real-world datasets would be larger
- Expected fairness issues may exist due to synthetic data generation
