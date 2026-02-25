"""
Multi-model training module for Dataset Bias Analyzer.

Supports training and comparing multiple ML models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Try to import XGBoost (may not be installed yet)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# Model configurations
MODEL_CONFIGS = {
    'logistic_regression': {
        'class': LogisticRegression,
        'params': {'max_iter': 1000, 'random_state': 42},
        'name': 'Logistic Regression'
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
        'name': 'Random Forest'
    },
    'svm': {
        'class': SVC,
        'params': {'kernel': 'rbf', 'C': 1.0, 'probability': True, 'random_state': 42},
        'name': 'Support Vector Machine'
    },
}

if XGBOOST_AVAILABLE:
    MODEL_CONFIGS['xgboost'] = {
        'class': XGBClassifier,
        'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 
                   'random_state': 42, 'eval_metric': 'logloss'},
        'name': 'XGBoost'
    }


def train_baseline(X_train, y_train, X_test, y_test):
    """
    Train baseline Logistic Regression model (backward compatibility).
    
    Parameters
    ----------
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data
    
    Returns
    -------
    tuple
        (model, y_pred, accuracy)
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, y_pred, accuracy


def train_model(X_train, y_train, X_test, y_test, model_type='logistic_regression'):
    """
    Train a single model of specified type.
    
    Parameters
    ----------
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data
    model_type : str, default='logistic_regression'
        Type of model to train. Options: 'logistic_regression', 'random_forest', 
        'svm', 'xgboost'
    
    Returns
    -------
    dict
        {
            'model': trained model object,
            'predictions': y_pred,
            'metrics': {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1_score': float
            }
        }
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_type]
    model_class = config['class']
    params = config['params']
    
    # Train model
    model = model_class(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Determine if binary or multiclass
    n_classes = len(np.unique(y_test))
    average_method = 'binary' if n_classes == 2 else 'weighted'
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average=average_method, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average=average_method, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, average=average_method, zero_division=0))
    }
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        # For binary classification, use probabilities of positive class
        if n_classes == 2:
            y_pred_proba = y_pred_proba[:, 1]
        else:
            # For multiclass, use max probability across all classes
            y_pred_proba = np.max(y_pred_proba, axis=1)
    else:
        y_pred_proba = None
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'metrics': metrics
    }


def train_multiple_models(X_train, y_train, X_test, y_test, models=None):
    """
    Train multiple models and return results for comparison.
    
    Parameters
    ----------
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data
    models : list of str, optional
        List of model types to train. If None, trains all available models.
        Options: 'logistic_regression', 'random_forest', 'svm', 'xgboost'
    
    Returns
    -------
    dict
        Dictionary with model_type as keys, each containing:
        {
            'model': trained model object,
            'predictions': y_pred,
            'probabilities': y_pred_proba (if available),
            'metrics': {accuracy, precision, recall, f1_score}
        }
    """
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    
    results = {}
    
    for model_type in models:
        if model_type not in MODEL_CONFIGS:
            print(f"Warning: Unknown model type '{model_type}'. Skipping.")
            continue
        
        try:
            result = train_model(X_train, y_train, X_test, y_test, model_type)
            results[model_type] = result
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue
    
    return results


def get_available_models():
    """
    Get list of available models with their display names.
    
    Returns
    -------
    dict
        {model_type: display_name}
    """
    return {model_type: config['name'] for model_type, config in MODEL_CONFIGS.items()}


def get_feature_importance(model, feature_names, model_type):
    """
    Extract feature importance from trained model.
    
    Parameters
    ----------
    model : trained model object
        The trained model
    feature_names : list
        Names of features
    model_type : str
        Type of model
    
    Returns
    -------
    dict or None
        {feature_name: importance_score} sorted by importance, or None if not applicable
    """
    importance = None
    
    if model_type in ['random_forest', 'xgboost']:
        # Tree-based models have feature_importances_
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_))
    elif model_type == 'logistic_regression':
        # Logistic regression has coefficients
        if hasattr(model, 'coef_'):
            # Use absolute values of coefficients
            importance = dict(zip(feature_names, np.abs(model.coef_[0])))
    elif model_type == 'svm':
        # Linear SVM has coefficients
        if hasattr(model, 'coef_'):
            importance = dict(zip(feature_names, np.abs(model.coef_[0])))
    
    if importance:
        # Sort by importance (descending)
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    return importance
