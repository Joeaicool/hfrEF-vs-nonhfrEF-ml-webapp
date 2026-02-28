import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title='Best Model Predictor', layout='wide')
st.title("Prediction Model with SHAP Visualization")

model = joblib.load('RF_best.pkl')
FEATURES = ['proBNP', 'HCT', 'Glb', 'GGT', 'BUN', 'IBil', 'CRP', 'Mono_Percent', 'B2_MG']
DATA_FILE = 'Final_Cleaned_Data.xlsx'
TARGET_COL = 'status'
ID_COL = 'ID'

df = pd.read_excel(DATA_FILE)
if ID_COL in df.columns:
    df_feat = df.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
else:
    df_feat = df.drop(columns=[TARGET_COL], errors='ignore')

feature_ranges = {}
for f in FEATURES:
    col = df_feat[f]
    if pd.api.types.is_numeric_dtype(col):
        mn = float(np.nanmin(col.values))
        mx = float(np.nanmax(col.values))
        dv = float(np.nanmedian(col.values))
        if mn == mx:
            mx = mn + 1.0
        feature_ranges[f] = {"type":"numerical","min":mn,"max":mx,"default":dv}
    else:
        opts = [str(x) for x in col.dropna().unique().tolist()] or ["0","1"]
        feature_ranges[f] = {"type":"categorical","options":opts,"default":opts[0]}

st.header("Enter the following feature values:")
vals = []
for feat, p in feature_ranges.items():
    if p["type"] == "numerical":
        v = st.number_input(f"{feat} ({p['min']:.3f} - {p['max']:.3f})", float(p["min"]), float(p["max"]), float(p["default"]))
    else:
        v = st.selectbox(f"{feat}", p["options"])
        try:
            v = float(v)
        except:
            pass
    vals.append(v)

X_input = pd.DataFrame([vals], columns=FEATURES)

if st.button("Predict"):
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1] * 100 if hasattr(model, 'predict_proba') else 0.0
    st.write(f"Predicted risk of positive class (status=1): {proba:.2f}%")

    try:
        explainer = shap.Explainer(model, df_feat[FEATURES])
        sv = explainer(X_input)
        fig = plt.figure(figsize=(10, 3))
        shap.plots.waterfall(sv[0], max_display=min(10, len(FEATURES)), show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP visualization failed: {e}")
