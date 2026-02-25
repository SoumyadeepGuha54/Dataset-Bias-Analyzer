"""
Enhanced Streamlit UI for Dataset Bias Analyzer
with multi-model support, data cleaning, and advanced visualizations.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from main import run_bias_engine
from engine.data_cleaner import analyze_data_quality, clean_dataset, has_quality_issues
from engine.model_trainer import get_available_models
from engine.visualizer import (
    plot_confusion_matrices, plot_roc_curves, plot_feature_importance,
    plot_data_distributions, plot_correlation_heatmap, plot_model_comparison,
    plot_fairness_comparison
)

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dataset Bias Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Load External CSS
# ─────────────────────────────────────────────────────────────
def load_css():
    """Load external CSS file"""
    try:
        with open("styles.css", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file optional

load_css()

# Initialize session state
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'quality_report' not in st.session_state:
    st.session_state.quality_report = None
if 'cleaning_performed' not in st.session_state:
    st.session_state.cleaning_performed = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None


# ─────────────────────────────────────────────────────────────
# Sidebar — Upload + Config
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ Dataset Bias Analyzer")
    st.markdown(
        '<p class="sidebar-subtitle">'
        "Detect and analyze bias in your machine learning datasets using state-of-the-art fairness metrics."
        "</p>",
        unsafe_allow_html=True,
    )
    
    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    st.markdown("##### Dataset Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    target_col = None
    sensitive_col = None
    selected_models = []

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Store original dataframe
        if st.session_state.df_original is None or st.session_state.df_original.shape != df.shape:
            st.session_state.df_original = df.copy()
            st.session_state.df_cleaned = None
            st.session_state.quality_report = None
            st.session_state.cleaning_performed = False
            st.session_state.analysis_results = None
        
        st.markdown(
            f'<div class="upload-success">'
            f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

        st.markdown("##### Configuration")
        columns = df.columns.tolist()
        target_col = st.selectbox("Target Column", columns, index=0)
        sensitive_col = st.selectbox(
            "Sensitive Attribute",
            [c for c in columns if c != target_col],
            index=0,
        )
        
        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
        
        # Model selection
        st.markdown("##### Model Selection")
        available_models = get_available_models()
        model_keys = list(available_models.keys())
        model_labels = [available_models[k] for k in model_keys]
        
        selected_model_labels = st.multiselect(
            "Select models to train",
            model_labels,
            default=model_labels[:2],  # Default: first 2 models
            label_visibility="collapsed"
        )
        
        # Convert back to model keys
        selected_models = [k for k, v in available_models.items() if v in selected_model_labels]

        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
        
        analyze_quality_btn = st.button("📊 Check Data Quality", use_container_width=True)
        run_analysis_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    else:
        analyze_quality_btn = False
        run_analysis_btn = False

# ─────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────
if uploaded_file is None:
    # Landing state
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">Dataset Bias Analyzer</h1>
            <p class="hero-subtitle">
                Detect fairness issues in your machine learning datasets with comprehensive 
                bias analysis, demographic parity metrics, and proxy variable detection. 
                Upload a CSV dataset to get started.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div style="text-align: center; padding: 24px 16px;">
                <div style="font-size: 48px; margin-bottom: 12px;">🧹</div>
                <div class="feature-title">Data Cleaning</div>
                <div class="feature-desc">
                    Automatic detection and cleaning of missing values, outliers, and duplicates
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style="text-align: center; padding: 24px 16px;">
                <div style="font-size: 48px; margin-bottom: 12px;">🤖</div>
                <div class="feature-title">Multi-Model</div>
                <div class="feature-desc">
                    Compare bias across Logistic Regression, Random Forest, SVM, and XGBoost
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div style="text-align: center; padding: 24px 16px;">
                <div style="font-size: 48px; margin-bottom: 12px;">📈</div>
                <div class="feature-title">Visualizations</div>
                <div class="feature-desc">
                    ROC curves, confusion matrices, feature importance, and more
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div style="text-align: center; padding: 24px 16px;">
                <div style="font-size: 48px; margin-bottom: 12px;">⚖️</div>
                <div class="feature-title">Fairness Metrics</div>
                <div class="feature-desc">
                    Comprehensive analysis using demographic parity and equalized odds
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

# ─────────────────────────────────────────────────────────────
# Step 1: Data Quality Analysis
# ─────────────────────────────────────────────────────────────
if analyze_quality_btn:
    with st.spinner("Analyzing data quality..."):
        st.session_state.quality_report = analyze_data_quality(st.session_state.df_original, target_col)

if st.session_state.quality_report is not None:
    st.markdown("## 📊 Data Quality Report")
    
    quality_report = st.session_state.quality_report
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{quality_report['total_rows']:,}")
    with col2:
        missing_count = sum(quality_report['missing_values'].values())
        st.metric("Missing Values", missing_count)
    with col3:
        st.metric("Duplicates", quality_report['duplicates'])
    with col4:
        outlier_count = sum(quality_report['outliers'].values())
        st.metric("Outliers Detected", outlier_count)
    
    # Detailed issues
    if has_quality_issues(quality_report):
        st.warning("⚠️ Data quality issues detected. Consider cleaning the dataset before analysis.")
        
        # Missing values details
        if quality_report['missing_values']:
            st.markdown("### Missing Values")
            missing_df = pd.DataFrame([
                {"Column": col, "Missing Count": count}
                for col, count in quality_report['missing_values'].items()
            ])
            st.dataframe(missing_df, use_container_width=True)
        
        # Outliers details
        if quality_report['outliers']:
            st.markdown("### Outliers")
            outlier_df = pd.DataFrame([
                {"Column": col, "Outlier Count": count}
                for col, count in quality_report['outliers'].items()
            ])
            st.dataframe(outlier_df, use_container_width=True)
        
        # Class distribution
        if quality_report['class_distribution']:
            st.markdown("### Class Distribution")
            class_df = pd.DataFrame([
                {"Class": cls, "Count": count}
                for cls, count in quality_report['class_distribution'].items()
            ])
            st.dataframe(class_df, use_container_width=True)
        
        # Cleaning options
        st.markdown("### Cleaning Options")
        col1, col2 = st.columns(2)
        with col1:
            remove_dups = st.checkbox("Remove duplicates", value=True)
            impute_missing = st.checkbox("Impute missing values", value=True)
            handle_outliers = st.checkbox("Handle outliers (IQR method)", value=True)
        with col2:
            scale_features = st.checkbox("Scale numerical features", value=False)
            balance_classes = st.checkbox("Balance classes (SMOTE)", value=False)
        
        if st.button("🧹 Clean Dataset", type="primary"):
            with st.spinner("Cleaning dataset..."):
                cleaned_df, cleaning_report = clean_dataset(
                    st.session_state.df_original,
                    target_col=target_col,
                    sensitive_col=sensitive_col,
                    remove_duplicates=remove_dups,
                    impute_missing=impute_missing,
                    handle_outliers=handle_outliers,
                    scale_features=scale_features,
                    balance_classes=balance_classes
                )
                st.session_state.df_cleaned = cleaned_df
                st.session_state.cleaning_performed = True
                
                st.success("✅ Dataset cleaned successfully!")
                
                # Show cleaning summary
                st.markdown("### Cleaning Summary")
                summary_cols = st.columns(4)
                summary_cols[0].metric("Duplicates Removed", cleaning_report['duplicates_removed'])
                summary_cols[1].metric("Values Imputed", cleaning_report['missing_values_imputed'])
                summary_cols[2].metric("Outliers Capped", cleaning_report['outliers_capped'])
                summary_cols[3].metric("Final Rows", cleaning_report['final_rows'])
    else:
        st.success("✅ No data quality issues detected. Dataset is ready for analysis.")

# ─────────────────────────────────────────────────────────────
# Step 2: Run Analysis
# ─────────────────────────────────────────────────────────────
if not run_analysis_btn and st.session_state.analysis_results is None:
    if uploaded_file is not None:
        st.markdown(
            '<div class="info-box">👈 Configure your analysis in the sidebar and click <strong>Run Analysis</strong> to begin.</div>',
            unsafe_allow_html=True,
        )
    st.stop()

if run_analysis_btn:
    if not selected_models:
        st.error("Please select at least one model to train.")
        st.stop()
    
    # Determine which dataset to use
    df_to_analyze = st.session_state.df_cleaned if st.session_state.cleaning_performed else st.session_state.df_original
    
    with st.spinner(f"Training {len(selected_models)} model(s) and computing fairness metrics..."):
        try:
            # Prepare cleaning config if cleaning was performed
            cleaning_config = None
            
            result = run_bias_engine(
                df_to_analyze,
                target_col,
                sensitive_col,
                models=selected_models,
                cleaning_config=cleaning_config
            )
            st.session_state.analysis_results = result
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

# Display results if available
if st.session_state.analysis_results is not None:
    result = st.session_state.analysis_results
    
    # ─────────────────────────────────────────────────────────────
    # Header
    # ─────────────────────────────────────────────────────────────
    st.markdown("## 📊 Analysis Results")
    st.markdown("Comprehensive bias analysis across multiple models")
    
    # Extract data
    models_data = result.get('models', {})
    proxy_vars = result.get('proxy_variables', {})
    
    if not models_data:
        st.error("No model results available.")
        st.stop()
    
    # ─────────────────────────────────────────────────────────────
    # Tabs for organized results
    # ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Per-Model Details", "⚖️ Comparisons", "📊 Data Insights"])
    
    # ─────────────────────────────────────────────────────────────
    # TAB 1: Overview
    # ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Model Performance Overview")
        
        # KPI metrics for each model
        metric_cols = st.columns(min(len(models_data), 4))
        for idx, (model_name, model_result) in enumerate(models_data.items()):
            with metric_cols[idx % 4]:
                accuracy = model_result['metrics']['accuracy']
                bias_detected = model_result['bias_detected']
                
                model_display_name = get_available_models().get(model_name, model_name)
                
                st.markdown(f"#### {model_display_name}")
                st.metric("Accuracy", f"{accuracy:.1%}")
                
                if bias_detected:
                    st.markdown('❌ <span style="color: #ef4444;">Bias Detected</span>', unsafe_allow_html=True)
                else:
                    st.markdown('✅ <span style="color: #10b981;">Fair</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model comparison chart
        st.markdown("### Performance Comparison")
        fig = plot_model_comparison(models_data, metrics=['accuracy', 'precision', 'recall', 'f1_score'])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        # Fairness comparison
        st.markdown("### Fairness Metrics Comparison")
        fig = plot_fairness_comparison(models_data)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    # ─────────────────────────────────────────────────────────────
    # TAB 2: Per-Model Details
    # ─────────────────────────────────────────────────────────────
    with tab2:
        # Model selector
        model_display_names = [get_available_models().get(m, m) for m in models_data.keys()]
        selected_model_display = st.selectbox("Select Model", model_display_names)
        
        # Find corresponding model key
        selected_model_key = [k for k, v in get_available_models().items() 
                             if v == selected_model_display and k in models_data][0]
        
        model_result = models_data[selected_model_key]
        
        st.markdown(f"### {selected_model_display} - Detailed Analysis")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{model_result['metrics']['accuracy']:.3f}")
        col2.metric("Precision", f"{model_result['metrics']['precision']:.3f}")
        col3.metric("Recall", f"{model_result['metrics']['recall']:.3f}")
        col4.metric("F1 Score", f"{model_result['metrics']['f1_score']:.3f}")
        
        # Group metrics bar charts
        st.markdown("#### Group-Level Metrics")
        group_metrics = model_result['group_metrics']
        groups = list(group_metrics.keys())
        approval_rates = [group_metrics[g]["approval_rate"] for g in groups]
        tprs = [group_metrics[g]["tpr"] for g in groups]
        fprs = [group_metrics[g]["fpr"] for g in groups]
        
        bar_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"][:len(groups)]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        chart_data = [
            ("Approval Rate", approval_rates),
            ("True Positive Rate", tprs),
            ("False Positive Rate", fprs),
        ]
        
        for ax, (title, values) in zip(axes, chart_data):
            ax.bar(groups, values, color=bar_colors, edgecolor='white', linewidth=1.5)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            
            for i, (g, v) in enumerate(zip(groups, values)):
                ax.text(i, v + 0.02, f"{v:.1%}", ha='center', va='bottom', fontweight='bold')
        
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        # Confusion matrices
        if model_result.get('probabilities'):
            st.markdown("#### Confusion Matrices by Group")
            y_test = np.array(result['test_data']['y_test'])
            y_pred = np.array(model_result['predictions'])
            sensitive_test = np.array(result['test_data']['sensitive_test'])
            
            fig = plot_confusion_matrices(y_test, y_pred, sensitive_test)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
            # ROC curves
            st.markdown("#### ROC Curves")
            y_pred_proba = np.array(model_result['probabilities'])
            fig = plot_roc_curves(y_test, y_pred_proba, sensitive_test)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        # Feature importance
        if model_result.get('feature_importance'):
            st.markdown("#### Feature Importance")
            fig = plot_feature_importance(
                model_result['feature_importance'],
                top_n=15,
                title=f'Top 15 Features - {selected_model_display}'
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        # Fairness scores table
        st.markdown("#### Fairness Scores")
        fairness_scores = model_result['fairness_scores']
        fairness_df = pd.DataFrame([
            {"Metric": "Demographic Parity Difference", 
             "Value": f"{fairness_scores['demographic_parity_difference']:.4f}",
             "Ideal": "< 0.10"},
            {"Metric": "Equal Opportunity Difference",
             "Value": f"{fairness_scores['equal_opportunity_difference']:.4f}",
             "Ideal": "< 0.10"},
            {"Metric": "Equalized Odds Difference",
             "Value": f"{fairness_scores['equalized_odds_difference']:.4f}",
             "Ideal": "< 0.10"},
            {"Metric": "Disparate Impact Ratio",
             "Value": f"{fairness_scores['disparate_impact_ratio']:.4f}",
             "Ideal": "≥ 0.80"},
        ])
        st.dataframe(fairness_df, use_container_width=True, hide_index=True)
    
    # ─────────────────────────────────────────────────────────────
    # TAB 3: Comparisons
    # ─────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Model-to-Model Comparison")
        
        # Create comparison table
        comparison_data = []
        for model_name, model_result in models_data.items():
            model_display_name = get_available_models().get(model_name, model_name)
            comparison_data.append({
                "Model": model_display_name,
                "Accuracy": f"{model_result['metrics']['accuracy']:.3f}",
                "Precision": f"{model_result['metrics']['precision']:.3f}",
                "Recall": f"{model_result['metrics']['recall']:.3f}",
                "F1": f"{model_result['metrics']['f1_score']:.3f}",
                "DP Diff": f"{model_result['fairness_scores']['demographic_parity_difference']:.4f}",
                "DI Ratio": f"{model_result['fairness_scores']['disparate_impact_ratio']:.4f}",
                "Bias": "❌ Yes" if model_result['bias_detected'] else "✅ No"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Recommendation")
        
        # Find best model (highest accuracy with no bias, or least biased)
        fair_models = [(name, data) for name, data in models_data.items() if not data['bias_detected']]
        
        if fair_models:
            best_model = max(fair_models, key=lambda x: x[1]['metrics']['accuracy'])
            best_name = get_available_models().get(best_model[0], best_model[0])
            st.success(f"✅ **Recommended Model: {best_name}** - Highest accuracy ({best_model[1]['metrics']['accuracy']:.1%}) among fair models")
        else:
            # All models have bias - pick least biased
            least_biased = min(models_data.items(), 
                             key=lambda x: x[1]['fairness_scores']['demographic_parity_difference'])
            least_name = get_available_models().get(least_biased[0], least_biased[0])
            st.warning(f"⚠️ All models show bias. **{least_name}** has the lowest demographic parity difference ({least_biased[1]['fairness_scores']['demographic_parity_difference']:.4f})")
    
    # ─────────────────────────────────────────────────────────────
    # TAB 4: Data Insights
    # ─────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Dataset Characteristics")
        
        # Use original or cleaned dataset
        df_to_visualize = st.session_state.df_cleaned if st.session_state.cleaning_performed else st.session_state.df_original
        
        # Feature distributions
        st.markdown("#### Feature Distributions")
        numeric_cols = df_to_visualize.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) > 0:
            fig = plot_data_distributions(df_to_visualize, columns=numeric_cols[:6])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        # Correlation heatmap
        st.markdown("#### Feature Correlation")
        if len(numeric_cols) >= 2:
            fig = plot_correlation_heatmap(df_to_visualize)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Insufficient numerical features for correlation analysis")
        
        # Proxy variables
        st.markdown("#### Proxy Variable Detection")
        if proxy_vars:
            st.warning(f"⚠️ {len(proxy_vars)} potential proxy variable(s) detected")
            proxy_df = pd.DataFrame([
                {"Feature": feat, "Correlation Score": f"{score:.4f}"}
                for feat, score in proxy_vars.items()
            ])
            st.dataframe(proxy_df, use_container_width=True, hide_index=True)
            st.info("💡 Proxy variables show high correlation with the sensitive attribute and may introduce bias.")
        else:
            st.success("✅ No proxy variables detected (correlation threshold: 0.7)")
    
    # ─────────────────────────────────────────────────────────────
    # Raw Data Expander (at bottom)
    # ─────────────────────────────────────────────────────────────
    with st.expander("📄 View Raw JSON Results"):
        st.json(result)
