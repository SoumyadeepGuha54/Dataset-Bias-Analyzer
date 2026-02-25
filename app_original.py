import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from main import run_bias_engine

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
    with open("styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


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

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
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

        st.markdown('<div style="height: 32px;"></div>', unsafe_allow_html=True)
        run_btn = st.button("Run Analysis", use_container_width=True)
    else:
        run_btn = False

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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div style="text-align: center; padding: 32px 20px;">
                <div style="font-size: 48px; margin-bottom: 16px;">📊</div>
                <div class="feature-title">Group Metrics</div>
                <div class="feature-desc">
                    Analyze approval rates, TPR, and FPR across demographic groups
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style="text-align: center; padding: 32px 20px;">
                <div style="font-size: 48px; margin-bottom: 16px;">⚖️</div>
                <div class="feature-title">Fairness Scores</div>
                <div class="feature-desc">
                    Compute disparate impact, demographic parity, and equalized odds
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div style="text-align: center; padding: 32px 20px;">
                <div style="font-size: 48px; margin-bottom: 16px;">🔍</div>
                <div class="feature-title">Proxy Detection</div>
                <div class="feature-desc">
                    Identify features that may serve as proxies for sensitive attributes
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

if not run_btn:
    st.markdown(
        '<div class="info-box">👈 Configure your analysis in the sidebar and click <strong>Run Analysis</strong> to begin.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# Run the engine
# ─────────────────────────────────────────────────────────────
with st.spinner("Analyzing dataset and computing fairness metrics..."):
    try:
        result = run_bias_engine(df, target_col, sensitive_col)
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        st.stop()

accuracy = result["accuracy"]
bias_detected = result["bias_detected"]
group_metrics = result["group_metrics"]
fairness_scores = result["fairness_scores"]
proxy_vars = result.get("proxy_variables", {})

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="result-header" style="font-size: 32px; font-weight: 700; margin-bottom: 8px; margin-top: 0;">Analysis Results</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="result-subtitle" style="font-size: 14px; margin-bottom: 32px;">Comprehensive bias analysis for your dataset</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Model Accuracy</span>
            <div class="metric-value value-blue">{accuracy:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    bias_class = "badge-danger" if bias_detected else "badge-success"
    bias_label = "Bias Detected" if bias_detected else "Fair"
    bias_icon = "⚠" if bias_detected else "✓"
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Bias Status</span>
            <span class="status-badge {bias_class}">{bias_icon} {bias_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    di = fairness_scores["disparate_impact_ratio"]
    di_color = "value-green" if di >= 0.8 else "value-red"
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Disparate Impact</span>
            <div class="metric-value {di_color}">{di:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Group Metrics — Bar Charts
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">Group-Level Metrics</div>', unsafe_allow_html=True
)

groups = list(group_metrics.keys())
approval_rates = [group_metrics[g]["approval_rate"] for g in groups]
tprs = [group_metrics[g]["tpr"] for g in groups]
fprs = [group_metrics[g]["fpr"] for g in groups]

# Modern color palette — high contrast on white background
bar_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"][
    : len(groups)
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#ffffff")

chart_data = [
    ("Approval Rate", approval_rates),
    ("True Positive Rate", tprs),
    ("False Positive Rate", fprs),
]

for ax, (title, values) in zip(axes, chart_data):
    ax.set_facecolor("#ffffff")
    bars = ax.bar(
        groups, values, color=bar_colors, edgecolor="#ffffff", linewidth=1.5, width=0.55,
        zorder=3,
    )

    ax.set_title(title, color="#111827", fontsize=14, fontweight=700, pad=16, fontfamily="sans-serif")

    # Dynamic y-limit based on actual data
    max_val = max(values) if values else 1
    y_upper = max(1.15, max_val * 1.15)
    ax.set_ylim(0, y_upper)
    label_offset = y_upper * 0.025

    # Tick styling
    ax.tick_params(axis="x", colors="#374151", labelsize=11, length=0, pad=8)
    ax.tick_params(axis="y", colors="#6b7280", labelsize=10, length=0, pad=6)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")
    ax.spines["bottom"].set_linewidth(1)

    # Gridlines
    ax.grid(axis="y", color="#e5e7eb", linestyle="-", linewidth=0.8, alpha=1, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Value labels above bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{val:.1%}",
            ha="center", va="bottom",
            color="#111827", fontsize=11, fontweight=700,
        )

fig.tight_layout(pad=2.0)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ─────────────────────────────────────────────────────────────
# Fairness Scores Table
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Fairness Scores</div>', unsafe_allow_html=True)

rows_html = ""
score_labels = {
    "demographic_parity_difference": ("Demographic Parity Difference", "< 0.10"),
    "equal_opportunity_difference": ("Equal Opportunity Difference", "< 0.10"),
    "equalized_odds_difference": ("Equalized Odds Difference", "< 0.10"),
    "disparate_impact_ratio": ("Disparate Impact Ratio", "≥ 0.80"),
}

for key, (label, ideal) in score_labels.items():
    val = fairness_scores[key]
    rows_html += (
        f"<tr><td>{label}</td><td><strong>{val:.4f}</strong></td><td>{ideal}</td></tr>"
    )

st.markdown(
    f"""
    <table class="data-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Ideal Range</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Proxy Variables
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">Proxy Variable Detection</div>', unsafe_allow_html=True
)

if proxy_vars:
    st.markdown(
        f'<p class="warning-text">⚠ {len(proxy_vars)} potential proxy variable(s) detected</p>',
        unsafe_allow_html=True,
    )
    for feat, score in proxy_vars.items():
        st.markdown(
            f"""
            <div class="proxy-alert">
                <div class="proxy-alert-title">
                    <span class="proxy-feature">{feat}</span>
                    <span style="margin-left: 12px; color: #888888;">Correlation: {score:.4f}</span>
                </div>
                <div class="proxy-alert-text">
                    This feature shows high correlation with the sensitive attribute and may act as a proxy for discrimination.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div class="success-box">'
        "✓ No proxy variables detected — all features are below the 0.7 correlation threshold"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Raw Data Expander
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">Raw Analysis Report</div>', unsafe_allow_html=True
)

with st.expander("View JSON Report", expanded=False):
    st.json(result)
