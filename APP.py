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
    page_title="Explainable ML Model for Heart Failure Phenotype Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom SCI-style CSS
# =========================
st.markdown("""
<style>
    .main { background-color: #f6f9fc; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    .title-box { background: linear-gradient(90deg, #0f4c81 0%, #1b6ca8 100%); padding: 1rem 1.2rem; border-radius: 12px; color: white; margin-bottom: 1rem; }
    .subtitle-box { background: white; padding: 0.8rem 1rem; border-radius: 10px; border-left: 5px solid #1b6ca8; margin-bottom: 1rem; }
    .card { background: white; padding: 0.8rem 1rem; border-radius: 10px; border: 1px solid #e5edf5; margin-bottom: 0.8rem; }
    .footer { margin-top: 1.5rem; padding-top: 0.6rem; border-top: 1px solid #dbe5f0; color: #4a6072; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown(
    """
    <div class="title-box">
        <h2 style="margin:0;">Explainable ML Model for Heart Failure Phenotype Classification (HFrEF vs HFmrEF/HFpEF)</h2>
    </div>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle-box">
    <b>Clinical Objective:</b> Machine-learning classification of heart failure phenotype:
    <b>HFrEF (positive class = 1)</b> vs <b>non-HFrEF (HFmrEF + HFpEF)</b>.
    </div>
    """, unsafe_allow_html=True
)

try:
    st.image("Heart Failure and Symptoms.jpg", caption="Heart Failure and Symptoms", use_container_width=True)
except:
    pass

# =========================
# 核心：加载模型并自动侦测特征
# =========================
MODEL_PATH = "RF_best.pkl"
DATA_FILE = "Final_Cleaned_Data.xlsx"
TARGET_COL = "status"
ID_COL = "ID"

# 移除 st.cache_resource，强制每次加载最新的模型文件，避免缓存Bug
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    try:
        return pd.read_excel(DATA_FILE)
    except Exception:
        return pd.DataFrame()

try:
    model = load_model()
    # 【黑科技】自动从模型中读取它到底是用哪些特征训练的
    if hasattr(model, "feature_names_in_"):
        FEATURES = list(model.feature_names_in_)
    else:
        # 兼容旧版本sklearn的备选方案
        FEATURES = ['proBNP', 'TBil', 'HCT', 'BUN', 'B2_MG', 'CRP', 'DBil', 'UA', 'Glb', 'GGT', 'MCH', 'LDL_C', 'FIB']
except Exception as e:
    st.error(f"模型加载失败，请确保 {MODEL_PATH} 存在。错误信息: {e}")
    st.stop()
    
df = load_data()

# =========================
# 提取特征参考范围
# =========================
if not df.empty:
    df_feat = df.drop(columns=[TARGET_COL, ID_COL, 'HFstatus'], errors="ignore")
else:
    df_feat = pd.DataFrame()

st.markdown(f'<div class="card"><b>Patient Feature Input ({len(FEATURES)} Variables Detected)</b></div>', unsafe_allow_html=True)

feature_ranges = {}
for f in FEATURES:
    if f in df_feat.columns:
        col = df_feat[f]
        if pd.api.types.is_numeric_dtype(col):
            mn = float(np.nanmin(col.values))
            mx = float(np.nanmax(col.values))
            dv = float(np.nanmedian(col.values))
            if mn == mx: mx = mn + 1.0
            feature_ranges[f] = {"type": "numerical", "min": mn, "max": mx, "default": dv}
        else:
            opts = [str(x) for x in col.dropna().unique().tolist()] or ["0", "1"]
            feature_ranges[f] = {"type": "categorical", "options": opts, "default": opts[0]}
    else:
        feature_ranges[f] = {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 50.0}

left, right = st.columns(2)
vals = []

for i, (feat, p) in enumerate(feature_ranges.items()):
    box = left if i % 2 == 0 else right
    with box:
        if p["type"] == "numerical":
            v = st.number_input(f"{feat} ({p['min']:.2f} - {p['max']:.2f})", min_value=float(p["min"]), max_value=float(p["max"]), value=float(p["default"]))
        else:
            v = st.selectbox(f"{feat}", p["options"])
            try: v = float(v)
            except: pass
        vals.append(v)

# 确保列名与模型训练时完全一致
X_input = pd.DataFrame([vals], columns=FEATURES)

# =========================
# Prediction + Visualization
# =========================
if st.button("Predict", type="primary", use_container_width=True):
    try:
        pred = model.predict(X_input)[0]
        proba_hfref = model.predict_proba(X_input)[0][1] * 100
    except Exception as e:
        st.error(f"预测失败！请检查数据。错误详情: {e}")
        st.stop()

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

    st.markdown('<div class="card"><b>Explainability (SHAP)</b></div>', unsafe_allow_html=True)

    try:
        # 处理模型是 Pipeline 的情况 (拆解预处理和分类器)
        if hasattr(model, "named_steps"):
            preprocessor = model.named_steps['preprocessor']
            clf = model.named_steps['classifier']
            X_trans = preprocessor.transform(X_input)
            X_trans_df = pd.DataFrame(X_trans, columns=FEATURES)
        else:
            clf = model
            X_trans_df = X_input

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans_df)

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

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**SHAP Waterfall Plot**")
            exp = shap.Explanation(
                values=sv_class1,
                base_values=base_class1,
                data=X_input.iloc[0].values, # 用原始输入值展示，更直观
                feature_names=FEATURES
            )
            fig_wf = plt.figure(figsize=(8, 5), dpi=200)
            shap.plots.waterfall(exp, max_display=10, show=False)
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)

        with p2:
            st.markdown("**SHAP Force Plot**")
            fig_force = plt.figure(figsize=(8, 5), dpi=200)
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
    """, unsafe_allow_html=True
)
