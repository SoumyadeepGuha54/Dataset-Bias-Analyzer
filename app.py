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
    page_title="Bias Detection Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS — premium dark-themed look
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.92);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ---------- Cards ---------- */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .metric-card h3 {
        margin: 0 0 8px 0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #a0a0c0;
    }
    .metric-card .value {
        font-size: 36px;
        font-weight: 700;
    }
    .accent-green  { color: #00e676; }
    .accent-red    { color: #ff5252; }
    .accent-blue   { color: #448aff; }
    .accent-amber  { color: #ffd740; }

    /* ---------- Badge ---------- */
    .badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.6px;
    }
    .badge-danger  { background: rgba(255,82,82,0.18); color: #ff5252; border: 1px solid rgba(255,82,82,0.35); }
    .badge-success { background: rgba(0,230,118,0.18); color: #00e676; border: 1px solid rgba(0,230,118,0.35); }

    /* ---------- Section header ---------- */
    .section-header {
        font-size: 22px;
        font-weight: 600;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(255,255,255,0.08);
        color: #e0e0ff;
    }

    /* ---------- Table ---------- */
    .styled-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .styled-table th {
        background: rgba(255,255,255,0.07);
        color: #a0a0c0;
        padding: 12px 16px;
        text-align: left;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .styled-table td {
        padding: 12px 16px;
        color: #e0e0ff;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    /* ---------- Proxy warning card ---------- */
    .proxy-card {
        background: rgba(255,215,64,0.08);
        border: 1px solid rgba(255,215,64,0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 8px;
    }
    .proxy-card code {
        color: #ffd740;
    }

    /* ---------- Misc ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #7c4dff 0%, #448aff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 15px;
        letter-spacing: 0.4px;
        transition: opacity .2s ease, transform .15s ease;
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Sidebar — Upload + Config
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️  Bias Detection Engine")
    st.caption("Upload any tabular CSV to detect model bias across sensitive attributes.")
    st.markdown("---")

    uploaded_file = st.file_uploader("📁  Upload CSV Dataset", type=["csv"])

    target_col = None
    sensitive_col = None

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")

        columns = df.columns.tolist()
        target_col = st.selectbox("🎯  Target Column (label)", columns, index=0)
        sensitive_col = st.selectbox(
            "🔒  Sensitive Attribute",
            [c for c in columns if c != target_col],
            index=0,
        )

        st.markdown("---")
        run_btn = st.button("🚀  Run Bias Analysis", use_container_width=True)
    else:
        run_btn = False

# ─────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────
if uploaded_file is None:
    # Landing state
    st.markdown(
        """
        <div style='text-align:center; padding: 120px 20px 40px 20px;'>
            <h1 style='font-size:48px; font-weight:800; background: linear-gradient(135deg, #7c4dff, #448aff, #00e676);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Bias Detection Engine
            </h1>
            <p style='color:#a0a0c0; font-size:18px; max-width:560px; margin:16px auto 0;'>
                Upload a CSV dataset, choose a target and a sensitive attribute, and get an
                instant fairness audit with group-level metrics, proxy variable detection, and
                actionable insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if not run_btn:
    st.info("👈  Configure the analysis in the sidebar and click **Run Bias Analysis**.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Run the engine
# ─────────────────────────────────────────────────────────────
with st.spinner("Training model & computing fairness metrics…"):
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
# KPI Cards
# ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>Model Accuracy</h3>
            <div class="value accent-blue">{accuracy:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    bias_class = "badge-danger" if bias_detected else "badge-success"
    bias_label = "⚠️ BIAS DETECTED" if bias_detected else "✅ FAIR"
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>Bias Status</h3>
            <div style="margin-top:8px;">
                <span class="badge {bias_class}">{bias_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    di = fairness_scores["disparate_impact_ratio"]
    di_color = "accent-green" if di >= 0.8 else "accent-red"
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>Disparate Impact Ratio</h3>
            <div class="value {di_color}">{di:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Group Metrics — Bar Charts
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊  Group-Level Metrics</div>', unsafe_allow_html=True)

groups = list(group_metrics.keys())
approval_rates = [group_metrics[g]["approval_rate"] for g in groups]
tprs = [group_metrics[g]["tpr"] for g in groups]
fprs = [group_metrics[g]["fpr"] for g in groups]

# Color palette
bar_colors = ["#7c4dff", "#448aff", "#00e676", "#ffd740", "#ff5252", "#18ffff", "#ea80fc"]
colors = [bar_colors[i % len(bar_colors)] for i in range(len(groups))]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#1a1a2e")

chart_data = [
    ("Approval Rate", approval_rates),
    ("True Positive Rate", tprs),
    ("False Positive Rate", fprs),
]

for ax, (title, values) in zip(axes, chart_data):
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(groups, values, color=colors, edgecolor="white", linewidth=0.3, width=0.55)
    ax.set_title(title, color="#e0e0ff", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#a0a0c0", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3a3a5c")
    ax.spines["bottom"].set_color("#3a3a5c")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2%}",
            ha="center",
            va="bottom",
            color="#e0e0ff",
            fontsize=9,
            fontweight="bold",
        )

fig.tight_layout(pad=3)
st.pyplot(fig)
plt.close(fig)

# ─────────────────────────────────────────────────────────────
# Fairness Scores Table
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📐  Fairness Scores</div>', unsafe_allow_html=True)

rows_html = ""
score_labels = {
    "demographic_parity_difference": ("Demographic Parity Difference", "< 0.10"),
    "equal_opportunity_difference": ("Equal Opportunity Difference", "< 0.10"),
    "equalized_odds_difference": ("Equalized Odds Difference", "< 0.10"),
    "disparate_impact_ratio": ("Disparate Impact Ratio", "≥ 0.80"),
}

for key, (label, ideal) in score_labels.items():
    val = fairness_scores[key]
    rows_html += f"<tr><td>{label}</td><td><strong>{val:.4f}</strong></td><td>{ideal}</td></tr>"

st.markdown(
    f"""
    <table class="styled-table">
        <thead><tr><th>Metric</th><th>Value</th><th>Ideal</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Proxy Variables
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔍  Proxy Variable Detection</div>', unsafe_allow_html=True)

if proxy_vars:
    st.warning(f"**{len(proxy_vars)}** potential proxy variable(s) detected!")
    for feat, score in proxy_vars.items():
        st.markdown(
            f"""
            <div class="proxy-card">
                <strong><code>{feat}</code></strong> — correlation score: <strong>{score:.4f}</strong><br>
                <span style="color:#a0a0c0; font-size:13px;">
                    This feature is highly correlated with the sensitive attribute and may act as a proxy for discrimination.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.success("No proxy variables detected — no feature exceeds the 0.7 correlation threshold.")

# ─────────────────────────────────────────────────────────────
# Raw Data Expander
# ─────────────────────────────────────────────────────────────
with st.expander("📋  View Raw JSON Report"):
    st.json(result)
