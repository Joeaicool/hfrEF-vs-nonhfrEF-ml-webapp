import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Prediction Model with SHAP Visualization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom SCI-style CSS
# =========================
st.markdown("""
<style>
    .main {
        background-color: #f6f9fc;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }
    .title-box {
        background: linear-gradient(90deg, #0f4c81 0%, #1b6ca8 100%);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .subtitle-box {
        background: white;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border-left: 5px solid #1b6ca8;
        margin-bottom: 1rem;
    }
    .card {
        background: white;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 1px solid #e5edf5;
        margin-bottom: 0.8rem;
    }
    .footer {
        margin-top: 1.5rem;
        padding-top: 0.6rem;
        border-top: 1px solid #dbe5f0;
        color: #4a6072;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown(
    """
    <div class="title-box">
        <h2 style="margin:0;">Prediction Model with SHAP Visualization</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle-box">
    <b>Clinical Objective:</b> Machine-learning classification of heart failure phenotype:
    <b>HFrEF (positive class = 1)</b> vs <b>non-HFrEF (HFmrEF + HFpEF)</b>.
    </div>
    """,
    unsafe_allow_html=True
)

# Hero image (your uploaded image in repo root)
st.image("Heart Failure and Symptoms.jpg", caption="Heart Failure and Symptoms", use_container_width=True)

# =========================
# Load model and data
# =========================
MODEL_PATH = "RF_best.pkl"  # 你现在放在仓库根目录
DATA_FILE = "Final_Cleaned_Data.xlsx"
TARGET_COL = "status"
ID_COL = "ID"

FEATURES = ['proBNP', 'HCT', 'Glb', 'GGT', 'BUN', 'IBil', 'CRP', 'Mono_Percent', 'B2_MG']

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_excel(DATA_FILE)

model = load_model()
df = load_data()

if ID_COL in df.columns:
    df_feat = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
else:
    df_feat = df.drop(columns=[TARGET_COL], errors="ignore")

# =========================
# Input panel
# =========================
st.markdown('<div class="card"><b>Patient Feature Input</b></div>', unsafe_allow_html=True)

feature_ranges = {}
for f in FEATURES:
    col = df_feat[f]
    if pd.api.types.is_numeric_dtype(col):
        mn = float(np.nanmin(col.values))
        mx = float(np.nanmax(col.values))
        dv = float(np.nanmedian(col.values))
        if mn == mx:
            mx = mn + 1.0
        feature_ranges[f] = {"type": "numerical", "min": mn, "max": mx, "default": dv}
    else:
        opts = [str(x) for x in col.dropna().unique().tolist()] or ["0", "1"]
        feature_ranges[f] = {"type": "categorical", "options": opts, "default": opts[0]}

left, right = st.columns(2)
vals = []

for i, (feat, p) in enumerate(feature_ranges.items()):
    box = left if i % 2 == 0 else right
    with box:
        if p["type"] == "numerical":
            v = st.number_input(
                f"{feat} ({p['min']:.3f} - {p['max']:.3f})",
                min_value=float(p["min"]),
                max_value=float(p["max"]),
                value=float(p["default"])
            )
        else:
            v = st.selectbox(f"{feat}", p["options"])
            try:
                v = float(v)
            except:
                pass
        vals.append(v)

X_input = pd.DataFrame([vals], columns=FEATURES)

# =========================
# Prediction + Visualization
# =========================
if st.button("Predict", type="primary", use_container_width=True):
    pred = model.predict(X_input)[0]
    proba_hfref = model.predict_proba(X_input)[0][1] * 100 if hasattr(model, "predict_proba") else 0.0

    # Top result cards
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="card"><b>Model Classification</b></div>', unsafe_allow_html=True)
        if pred == 1:
            st.success("Predicted phenotype: **HFrEF**")
        else:
            st.warning("Predicted phenotype: **non-HFrEF (HFmrEF + HFpEF)**")

        st.info(f"Predicted probability of **HFrEF**: **{proba_hfref:.2f}%**")

    with c2:
        st.markdown('<div class="card"><b>Risk Gauge</b></div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_hfref,
            number={"suffix": "%"},
            title={"text": "HFrEF Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f4c81"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#f8d7da"},
                ],
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # SHAP block
    st.markdown('<div class="card"><b>Explainability (SHAP)</b></div>', unsafe_allow_html=True)

    try:
        # RandomForest -> TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        # 兼容返回格式
        if isinstance(shap_values, list):
            sv_class1 = shap_values[1][0]
            base_vals = explainer.expected_value
            base_class1 = base_vals[1] if isinstance(base_vals, (list, np.ndarray)) else float(base_vals)
        elif len(np.array(shap_values).shape) == 3:
            sv_class1 = shap_values[0, :, 1]
            base_vals = explainer.expected_value
            base_class1 = base_vals[1] if isinstance(base_vals, (list, np.ndarray)) else float(base_vals)
        else:
            sv_class1 = shap_values[0]
            base_class1 = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]

        # Waterfall + Force
        p1, p2 = st.columns(2)

        with p1:
            st.markdown("**SHAP Waterfall Plot**")
            exp = shap.Explanation(
                values=sv_class1,
                base_values=base_class1,
                data=X_input.iloc[0].values,
                feature_names=FEATURES
            )
            fig_wf = plt.figure(figsize=(8, 4.2), dpi=200)
            shap.plots.waterfall(exp, max_display=min(10, len(FEATURES)), show=False)
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)

        with p2:
            st.markdown("**SHAP Force Plot**")
            fig_force = plt.figure(figsize=(8, 4.2), dpi=200)
            shap.force_plot(
                base_class1,
                sv_class1,
                X_input.iloc[0],
                feature_names=FEATURES,
                matplotlib=True,
                show=False
            )
            st.pyplot(fig_force, use_container_width=True)
            plt.close(fig_force)

        # Contribution table
        st.markdown("**Feature Contribution Table**")
        abs_sv = np.abs(sv_class1)
        total = abs_sv.sum() if abs_sv.sum() != 0 else 1.0
        pct = abs_sv / total * 100

        contribution_df = pd.DataFrame({
            "Feature": FEATURES,
            "Input Value": X_input.iloc[0].values,
            "SHAP Value": sv_class1,
            "Direction": ["Increase HFrEF risk" if v > 0 else "Decrease HFrEF risk" for v in sv_class1],
            "Contribution (%)": pct
        }).sort_values("Contribution (%)", ascending=False)

        st.dataframe(
            contribution_df.style.format({
                "Input Value": "{:.4f}",
                "SHAP Value": "{:.4f}",
                "Contribution (%)": "{:.2f}"
            }),
            use_container_width=True
        )

        # Contribution bar chart
        st.markdown("**Contribution Bar Chart**")
        fig_bar, ax = plt.subplots(figsize=(9, 4), dpi=220)
        bar_colors = ["#E53935" if v > 0 else "#1E88E5" for v in contribution_df["SHAP Value"]]
        ax.barh(contribution_df["Feature"], contribution_df["SHAP Value"], color=bar_colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("SHAP value")
        ax.set_title("Red: increase HFrEF risk | Blue: decrease HFrEF risk")
        ax.invert_yaxis()
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    except Exception as e:
        st.warning(f"SHAP visualization failed: {e}")

# =========================
# Footer
# =========================
st.markdown(
    """
    <div class="footer">
        <b>Author:</b> Zhiping Meng<br>
        <b>Affiliation:</b> Guigang People's Hospital, Guigang, China
    </div>
    """,
    unsafe_allow_html=True
)
