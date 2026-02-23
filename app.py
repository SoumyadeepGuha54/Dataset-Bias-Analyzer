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
# Custom CSS — Sleek Minimalist Design with Light/Dark Mode
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Light Mode Styles (Default) */
    .stApp {
        background: #f8f9fa !important;
    }
    
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }
    
    /* Dark Mode Styles */
    [data-theme="dark"] .stApp {
        background: #0a0a0a !important;
    }
    
    [data-theme="dark"] [data-testid="stSidebar"] {
        background: #111111 !important;
        border-right: 1px solid #222222 !important;
    }
    
    /* Sidebar Text */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label {
        color: #374151 !important;
    }
    
    [data-theme="dark"] [data-testid="stSidebar"] .stMarkdown,
    [data-theme="dark"] [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar Subtitle - Light Mode (Default) */
    .sidebar-subtitle {
        color: #6b7280;
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 32px;
    }
    
    /* Sidebar Subtitle - Dark Mode */
    [data-theme="dark"] .sidebar-subtitle {
        color: #888888;
    }
    
    /* General Text Colors - Light Mode (Default) */
    .stMarkdown, p, div, span {
        color: #1f2937;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #111827;
    }
    
    /* General Text Colors - Dark Mode */
    [data-theme="dark"] .stMarkdown,
    [data-theme="dark"] p,
    [data-theme="dark"] div,
    [data-theme="dark"] span {
        color: #e0e0e0;
    }
    
    [data-theme="dark"] h1,
    [data-theme="dark"] h2,
    [data-theme="dark"] h3,
    [data-theme="dark"] h4,
    [data-theme="dark"] h5,
    [data-theme="dark"] h6 {
        color: #ffffff;
    }
    
    /* Metric Cards - Light Mode (Default) */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 28px;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Metric Cards - Dark Mode */
    [data-theme="dark"] .metric-card {
        background: linear-gradient(145deg, #1a1a1a, #151515);
        border: 1px solid #2a2a2a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border-color: #d1d5db;
    }
    
    [data-theme="dark"] .metric-card:hover {
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
        border-color: #3a3a3a;
    }
    
    /* Metric Labels - Light Mode (Default) */
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6b7280;
        margin-bottom: 12px;
        display: block;
    }
    
    /* Metric Labels - Dark Mode */
    [data-theme="dark"] .metric-label {
        color: #888888;
    }
    
    .metric-value {
        font-size: 42px;
        font-weight: 700;
        line-height: 1;
        margin: 0;
    }
    
    .value-blue { color: #4A9EFF; }
    .value-green { color: #10B981; }
    .value-red { color: #EF4444; }
    .value-amber { color: #F59E0B; }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.5px;
        margin-top: 8px;
    }
    
    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-danger {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Section Header - Light Mode (Default) */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1f2937;
        margin: 48px 0 24px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid #e5e7eb;
        letter-spacing: -0.5px;
    }
    
    /* Section Header - Dark Mode */
    [data-theme="dark"] .section-title {
        color: #ffffff;
        border-bottom: 1px solid #222222;
    }
    
    /* Content Card - Light Mode (Default) */
    .content-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 32px;
        margin-bottom: 24px;
    }
    
    /* Content Card - Dark Mode */
    [data-theme="dark"] .content-card {
        background: #151515;
        border: 1px solid #222222;
    }
    
    /* Table Styling - Light Mode (Default) */
    .data-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        background: #ffffff;
        border: 1px solid #e5e7eb;
    }
    
    .data-table thead {
        background: #f9fafb;
    }
    
    .data-table th {
        padding: 16px 20px;
        text-align: left;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #6b7280;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .data-table td {
        padding: 16px 20px;
        color: #1f2937;
        border-bottom: 1px solid #f3f4f6;
        font-size: 14px;
    }
    
    /* Table Styling - Dark Mode */
    [data-theme="dark"] .data-table {
        background: #1a1a1a;
        border: none;
    }
    
    [data-theme="dark"] .data-table thead {
        background: #202020;
    }
    
    [data-theme="dark"] .data-table th {
        color: #888888;
        border-bottom: 1px solid #2a2a2a;
    }
    
    [data-theme="dark"] .data-table td {
        color: #e0e0e0;
        border-bottom: 1px solid #1a1a1a;
    }
    
    .data-table tbody tr:last-child td {
        border-bottom: none;
    }
    
    .data-table tbody tr {
        transition: background 0.2s ease;
    }
    
    .data-table tbody tr:hover {
        background: #f9fafb;
    }
    
    [data-theme="dark"] .data-table tbody tr:hover {
        background: #1e1e1e;
    }
    
    /* Proxy Alert - Light Mode (Default) */
    .proxy-alert {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    /* Proxy Alert - Dark Mode */
    [data-theme="dark"] .proxy-alert {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .proxy-alert-title {
        font-weight: 600;
        color: #F59E0B;
        margin-bottom: 8px;
        font-size: 14px;
    }
    
    .proxy-alert-text {
        color: #6b7280;
        font-size: 13px;
        line-height: 1.6;
    }
    
    [data-theme="dark"] .proxy-alert-text {
        color: #b0b0b0;
    }
    
    .proxy-feature {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
        font-size: 13px;
        font-weight: 500;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #4A9EFF 0%, #357ABD 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(74, 158, 255, 0.3);
    }
    
    /* File Uploader - Light Mode (Default) */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1px dashed #d1d5db !important;
        border-radius: 8px;
        padding: 20px;
    }
    
    /* File Uploader - Dark Mode */
    [data-theme="dark"] [data-testid="stFileUploader"] {
        background: #1a1a1a !important;
        border: 1px dashed #2a2a2a !important;
    }
    
    /* Select Box - Light Mode (Default) */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px;
        color: #1f2937 !important;
    }
    
    /* Select Box - Dark Mode */
    [data-theme="dark"] .stSelectbox > div > div {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        color: #e0e0e0 !important;
    }
    
    /* Streamlit Labels - Light Mode (Default) */
    label {
        color: #374151 !important;
    }
    
    /* Streamlit Labels - Dark Mode */
    [data-theme="dark"] label {
        color: #e0e0e0 !important;
    }
    
    /* Remove default streamlit padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 80px 20px;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 700;
        background: linear-gradient(135deg, #4A9EFF 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
        letter-spacing: -2px;
    }
    
    .hero-subtitle {
        font-size: 18px;
        color: #6b7280;
        line-height: 1.7;
        font-weight: 400;
    }
    
    [data-theme="dark"] .hero-subtitle {
        color: #888888;
    }
    
    /* Feature Icon Text - Light Mode (Default) */
    .feature-title {
        color: #1f2937;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .feature-desc {
        color: #6b7280;
        font-size: 13px;
        line-height: 1.6;
    }
    
    /* Feature Icon Text - Dark Mode */
    [data-theme="dark"] .feature-title {
        color: #ffffff;
    }
    
    [data-theme="dark"] .feature-desc {
        color: #888888;
    }
    
    /* Divider - Light Mode (Default) */
    .divider {
        height: 1px;
        background: #e5e7eb;
        margin: 40px 0;
    }
    
    /* Divider - Dark Mode */
    [data-theme="dark"] .divider {
        background: #222222;
    }
    
    /* Info Box - Light Mode (Default) */
    .info-box {
        background: rgba(74, 158, 255, 0.08);
        border: 1px solid rgba(74, 158, 255, 0.15);
        border-left: 4px solid #4A9EFF;
        border-radius: 8px;
        padding: 20px;
        color: #4b5563;
        font-size: 14px;
    }
    
    /* Info Box - Dark Mode */
    [data-theme="dark"] .info-box {
        background: rgba(74, 158, 255, 0.1);
        border: 1px solid rgba(74, 158, 255, 0.2);
        color: #b0b0b0;
    }
    
    /* Success Box - Light Mode (Default) */
    .success-box {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 8px;
        padding: 20px;
        color: #059669;
        font-size: 14px;
    }
    
    /* Success Box - Dark Mode */
    [data-theme="dark"] .success-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10B981;
    }
    
    /* Result Header - Light Mode (Default) */
    .result-header {
        color: #1f2937;
    }
    
    .result-subtitle {
        color: #6b7280;
    }
    
    /* Result Header - Dark Mode */
    [data-theme="dark"] .result-header {
        color: #ffffff;
    }
    
    [data-theme="dark"] .result-subtitle {
        color: #888888;
    }
    
    /* Sidebar Upload Success - Light Mode (Default) */
    .upload-success {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 6px;
        padding: 12px;
        margin: 16px 0;
        font-size: 13px;
        color: #059669;
    }
    
    /* Sidebar Upload Success - Dark Mode */
    [data-theme="dark"] .upload-success {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10B981;
    }
    
    /* Warning Text - Light Mode (Default) */
    .warning-text {
        color: #D97706;
        font-weight: 600;
        margin-bottom: 20px;
    }
    
    /* Warning Text - Dark Mode */
    [data-theme="dark"] .warning-text {
        color: #F59E0B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

    st.markdown("##### Dataset Upload")
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

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
    '<h1 class="result-header" style="font-size: 32px; font-weight: 700; margin-bottom: 8px;">Analysis Results</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="result-subtitle" style="font-size: 14px; margin-bottom: 40px;">Comprehensive bias analysis for your dataset</p>',
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

st.markdown('<div class="content-card">', unsafe_allow_html=True)

groups = list(group_metrics.keys())
approval_rates = [group_metrics[g]["approval_rate"] for g in groups]
tprs = [group_metrics[g]["tpr"] for g in groups]
fprs = [group_metrics[g]["fpr"] for g in groups]

# Modern color palette - vibrant colors visible in both themes
colors = ["#4A9EFF", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"][
    : len(groups)
]

# Detect theme from Streamlit config (defaults to transparent/auto-adjust)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("none")  # Transparent to adapt to theme
fig.patch.set_alpha(0.0)

chart_data = [
    ("Approval Rate", approval_rates),
    ("True Positive Rate", tprs),
    ("False Positive Rate", fprs),
]

for ax, (title, values) in zip(axes, chart_data):
    ax.set_facecolor("none")  # Transparent to adapt to theme
    ax.patch.set_alpha(0.0)
    bars = ax.bar(groups, values, color=colors, edgecolor="none", width=0.6)

    # Title and labels - using darker/lighter colors that work on both backgrounds
    ax.set_title(
        title,
        color="#4A9EFF",
        fontsize=15,
        fontweight=600,
        pad=16,
        fontfamily="sans-serif",
    )
    ax.set_ylim(0, 1.08)
    ax.tick_params(colors="#888888", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(axis="y", color="#888888", linestyle="-", linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Value labels on bars - high contrast color
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            color="#1f2937",  # Dark color visible on light bars
            fontsize=10,
            fontweight=700,
        )

fig.tight_layout(pad=2.5)
st.pyplot(fig)
plt.close(fig)

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Fairness Scores Table
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Fairness Scores</div>', unsafe_allow_html=True)

st.markdown('<div class="content-card">', unsafe_allow_html=True)

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

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Proxy Variables
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">Proxy Variable Detection</div>', unsafe_allow_html=True
)

st.markdown('<div class="content-card">', unsafe_allow_html=True)

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

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Raw Data Expander
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">Raw Analysis Report</div>', unsafe_allow_html=True
)

with st.expander("View JSON Report", expanded=False):
    st.json(result)
