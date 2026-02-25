"""
Visualization module for Dataset Bias Analyzer.

Provides comprehensive visualization functions:
- Confusion matrices per demographic group
- ROC curves (overall and per group)
- Feature importance plots
- Data distribution plots
- Correlation heatmaps
- Model comparison charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import io
import base64


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_confusion_matrices(y_true, y_pred, sensitive, groups=None):
    """
    Create confusion matrices for each demographic group.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    sensitive : array-like
        Sensitive attribute values
    groups : list, optional
        List of specific groups to plot. If None, plots all groups.
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive': sensitive
    })
    
    if groups is None:
        groups = sorted(df['sensitive'].unique())
    
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4))
    
    if n_groups == 1:
        axes = [axes]
    
    for idx, group in enumerate(groups):
        group_df = df[df['sensitive'] == group]
        cm = confusion_matrix(group_df['y_true'], group_df['y_pred'])
        
        # Determine labels based on number of classes
        n_classes = len(np.unique(df['y_true']))
        if n_classes == 2:
            labels = ['Negative', 'Positive']
        else:
            labels = [f'Class {i}' for i in sorted(np.unique(df['y_true']))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=labels,
                   yticklabels=labels)
        axes[idx].set_title(f'Group: {group}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    return fig


def plot_roc_curves(y_true, y_pred_proba, sensitive, groups=None, overall=True):
    """
    Create ROC curves with AUC scores.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    sensitive : array-like
        Sensitive attribute values
    groups : list, optional
        List of specific groups to plot. If None, plots all groups.
    overall : bool, default=True
        Whether to include overall ROC curve
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred_proba': y_pred_proba,
        'sensitive': sensitive
    })
    
    if groups is None:
        groups = sorted(df['sensitive'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Check if binary classification
    n_classes = len(np.unique(df['y_true']))
    
    try:
        # Plot overall ROC if requested
        if overall:
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(df['y_true'], df['y_pred_proba'])
            else:
                # For multiclass, use one-vs-rest for highest class
                positive_class = max(df['y_true'].unique())
                y_true_binary = (df['y_true'] == positive_class).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, df['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f'Overall (AUC = {roc_auc:.3f})', linestyle='--')
        
        # Plot ROC for each group
        colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
        for idx, group in enumerate(groups):
            group_df = df[df['sensitive'] == group]
            if len(group_df) > 0:
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(group_df['y_true'], group_df['y_pred_proba'])
                else:
                    # For multiclass, use one-vs-rest
                    positive_class = max(df['y_true'].unique())
                    y_true_binary = (group_df['y_true'] == positive_class).astype(int)
                    fpr, tpr, _ = roc_curve(y_true_binary, group_df['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, linewidth=2, label=f'{group} (AUC = {roc_auc:.3f})',
                       color=colors[idx])
    except Exception as e:
        # If ROC curve plotting fails, show error message
        ax.text(0.5, 0.5, f'ROC curve plotting failed:\n{str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves by Group', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_dict, top_n=15, title='Feature Importance'):
    """
    Create feature importance bar chart.
    
    Parameters
    ----------
    importance_dict : dict
        {feature_name: importance_score}
    top_n : int, default=15
        Number of top features to display
    title : str
        Plot title
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    if not importance_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Feature importance not available for this model',
               ha='center', va='center', fontsize=12)
        ax.set_title(title)
        ax.axis('off')
        return fig
    
    # Get top N features
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(importances):
        ax.text(v, i, f' {v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_data_distributions(df, columns=None, max_cols=6):
    """
    Create distribution plots for numerical and categorical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    columns : list, optional
        Specific columns to plot. If None, plots all columns.
    max_cols : int, default=6
        Maximum number of columns to plot
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    if columns is None:
        columns = df.columns.tolist()
    
    # Limit number of columns
    columns = columns[:max_cols]
    
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3  # 3 plots per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        if df[col].dtype in [np.float64, np.int64]:
            # Numerical: histogram with KDE
            df[col].hist(bins=30, ax=ax, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_title(f'{col} Distribution', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        else:
            # Categorical: bar chart
            value_counts = df[col].value_counts().head(10)  # Top 10 categories
            value_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
            ax.set_title(f'{col} Distribution', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, method='pearson', figsize=(12, 10)):
    """
    Create correlation heatmap for numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'
    figsize : tuple, default=(12, 10)
        Figure size
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.empty or len(numerical_df.columns) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Insufficient numerical features for correlation analysis',
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    
    # Compute correlation matrix
    corr_matrix = numerical_df.corr(method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(f'Feature Correlation Heatmap ({method.capitalize()})', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_model_comparison(model_results, metrics=None):
    """
    Create comparison chart for multiple models.
    
    Parameters
    ----------
    model_results : dict
        Dictionary with model names as keys, each containing 'metrics' dict
    metrics : list, optional
        List of metrics to compare. If None, uses all available.
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    if not model_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No models to compare', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    
    # Extract metrics
    model_names = list(model_results.keys())
    
    # Determine which metrics to plot
    if metrics is None:
        # Get all available metrics from first model
        first_model = model_results[model_names[0]]
        if 'metrics' in first_model:
            metrics = list(first_model['metrics'].keys())
        else:
            metrics = ['accuracy']
    
    # Collect data
    data = {metric: [] for metric in metrics}
    for model_name in model_names:
        for metric in metrics:
            if 'metrics' in model_results[model_name]:
                value = model_results[model_name]['metrics'].get(metric, 0)
            else:
                value = 0
            data[metric].append(value)
    
    # Create plot
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x_pos = np.arange(len(model_names))
        
        bars = ax.bar(x_pos, data[metric], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_fairness_comparison(model_fairness_results):
    """
    Create comparison chart for fairness metrics across models.
    
    Parameters
    ----------
    model_fairness_results : dict
        Dictionary with model names as keys, each containing 'fairness_scores' dict
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    if not model_fairness_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No fairness results to compare', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    
    model_names = list(model_fairness_results.keys())
    
    # Fairness metrics to compare
    fairness_metrics = [
        'demographic_parity_difference',
        'equal_opportunity_difference',
        'equalized_odds_difference',
        'disparate_impact_ratio'
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for idx, metric in enumerate(fairness_metrics):
        ax = axes[idx]
        values = []
        
        for model_name in model_names:
            if 'fairness_scores' in model_fairness_results[model_name]:
                value = model_fairness_results[model_name]['fairness_scores'].get(metric, 0)
            else:
                value = 0
            values.append(value)
        
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add ideal range visualization
        if 'ratio' in metric:
            ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.8)')
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Ideal (1.0)')
            ax.set_ylim(0, 1.2)
        else:
            ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.1)')
            ax.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Ideal (0)')
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 0.5)
        
        ax.legend(fontsize=8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def fig_to_base64(fig):
    """
    Convert matplotlib figure to base64 string for embedding in HTML/JSON.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
    
    Returns
    -------
    str
        Base64 encoded image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64
