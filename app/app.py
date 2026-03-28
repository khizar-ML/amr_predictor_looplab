import streamlit as st
import pandas as pd
import numpy as np
import category_encoders as ce
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import io
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AMR Phenotype Predictor By The Elites",
    page_icon="🧬",
    layout="wide",
)

# ─────────────────────────────────────────
# Resource Loading (Cached for Performance)
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_ml_assets():
    """Loads all PKL files once and stores them in memory."""
    try:
        model = joblib.load(os.path.join(BASE_DIR, 'xgb_model.pkl'))
        ohe = joblib.load(os.path.join(BASE_DIR, 'ohe_encoder.pkl'))
        te = joblib.load(os.path.join(BASE_DIR, 'target_encoder.pkl'))
        ab_list = joblib.load(os.path.join(BASE_DIR, 'top_ab_list.pkl'))
        return model, ohe, te, ab_list
    except FileNotFoundError as e:
        st.error(f"Critical Error: ML assets not found in {BASE_DIR}. Please ensure .pkl files are in the same folder as app.py.")
        st.stop()

# Initialize global assets
XGB_MODEL, OHE_ENCODER, TARGET_ENCODER, TOP_ANTIBIOTICS = load_ml_assets()

# ─────────────────────────────────────────
# UI Header
# ─────────────────────────────────────────
st.title("🧬 Antimicrobial Resistance Phenotype Predictor \n **By The Elites**")
st.markdown(
    "Upload a CSV file from the **BVBRC genome AMR** dataset. "
    "The app will preprocess your data, load an XGBoost model, "
    "and return a downloadable predictions file."
)
st.info("👈 **Important:** Please refer to the sidebar to select whether your dataset has a target variable or not.")

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
REQUIRED_COLS_WITH_TARGET = [
    "Genome ID", "Genome Name", "Antibiotic", "Measurement Value",
    "Evidence", "Resistant Phenotype", "Measurement Sign",
]
REQUIRED_COLS_NO_TARGET = [
    "Genome ID", "Genome Name", "Antibiotic", "Measurement Value",
    "Evidence", "Measurement Sign",
]
SIGN_MAP = {"<=": -2, "<": -1, "=": 0, ">": 1, ">=": 2}
EVIDENCE_MAP = {"Laboratory Method": 0, "Computational Method": 1}
CUSTOM_THRESHOLD = 0.65 

# ─────────────────────────────────────────
# Helper: convert measurement values
# ─────────────────────────────────────────
def convert_measurement(value):
    if pd.isna(value):
        return np.nan
    try:
        # Safety: eval can be risky, but used here for fractions like '2/73'
        result = eval(str(value))  
        return float(result)
    except Exception:
        return np.nan

# ─────────────────────────────────────────
# Preprocessing pipeline
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame, has_target: bool):
    cols = REQUIRED_COLS_WITH_TARGET if has_target else REQUIRED_COLS_NO_TARGET
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in uploaded file: {missing}")

    df = df[cols].copy()
    genome_ids = df["Genome ID"].values
    df.drop(columns=["Genome ID"], inplace=True)

    # --- Step 1: Target logic ---
    if has_target:
        y = df["Resistant Phenotype"].replace({"Nonsusceptible": "Resistant"}).map({
            "Susceptible": 0, 
            "Resistant": 1
        })
        df.drop(columns=["Resistant Phenotype"], inplace=True)
    else:
        y = None

    # --- Step 2: One-hot encode antibiotic using cached TOP_ANTIBIOTICS ---
    df["Antibiotic"] = df["Antibiotic"].apply(
        lambda x: x if x in TOP_ANTIBIOTICS else "Other"
    )
    
    # Use cached OHE_ENCODER
    encoded_array = OHE_ENCODER.transform(df[['Antibiotic']])
    column_names = OHE_ENCODER.get_feature_names_out(['Antibiotic'])
    encoded_df = pd.DataFrame(encoded_array, columns=column_names, index=df.index)

    column_to_drop = column_names[0]
    encoded_df = encoded_df.drop(columns=[column_to_drop])

    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=['Antibiotic'])
    
    # --- Step 3: Measurement Sign ---
    df["sign"] = df["Measurement Sign"].map(SIGN_MAP).fillna(0).astype(int)
    df.drop(columns=["Measurement Sign"], inplace=True)

    # --- Step 4: Measurement Value ---
    df["Measurement Value"] = df["Measurement Value"].apply(convert_measurement)
    df["has_measurement"] = df["Measurement Value"].notnull().astype(int)

    # --- Step 5: Target-encode Genome Name using cached TARGET_ENCODER ---
    df["Genome Name"] = df["Genome Name"].astype(object) 
    df["Genome Name_Encoded"] = TARGET_ENCODER.transform(df[["Genome Name"]])
    df.drop(columns=["Genome Name"], inplace=True)

    # --- Step 6: Label-encode Evidence ---
    df["Evidence_Encoded"] = df["Evidence"].map(EVIDENCE_MAP)
    df.drop(columns=["Evidence"], inplace=True)

    return df, y, genome_ids

# ─────────────────────────────────────────
# Inference
# ─────────────────────────────────────────
def predict(X: pd.DataFrame):
    """Uses the globally loaded XGB_MODEL."""
    X = X.copy()
    X['Measurement_Numeric'] = X['Measurement Value']
    
    expected_cols = [
        'Measurement Value', 'has_measurement', 'Measurement_Numeric', 
        'Antibiotic_ampicillin', 'Antibiotic_aztreonam', 'Antibiotic_cefotaxime', 
        'Antibiotic_ceftazidime', 'Antibiotic_chloramphenicol', 'Antibiotic_ciprofloxacin', 
        'Antibiotic_meropenem', 'Antibiotic_sulfamethoxazole', 'Antibiotic_tetracycline', 
        'Antibiotic_tobramycin', 'sign', 'Genome Name_Encoded', 'Evidence_Encoded'
    ]
    
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
            
    X = X[expected_cols] 
    
    y_prob = XGB_MODEL.predict_proba(X)[:, 1]
    y_pred = (y_prob >= CUSTOM_THRESHOLD).astype(int)
    y_label = np.where(y_pred == 1, "Resistant", "Susceptible")

    return y_label, y_prob

# ─────────────────────────────────────────
# UI Controls
# ─────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

has_target_selection = st.sidebar.radio(
    "Does your CSV file have the target variable (`Resistant Phenotype`)?",
    options=["Yes — it has the target variable", "No — target variable is absent"],
    index=1,
)
has_target = (has_target_selection == "Yes — it has the target variable")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()

    st.subheader("📋 Data Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    if has_target:
        if "Resistant Phenotype" not in raw_df.columns:
            st.warning("⚠️ `Resistant Phenotype` column not found. Automatically switching to 'No Target' mode.")
            has_target = False
        elif raw_df["Resistant Phenotype"].isna().any():
            valid_count = raw_df["Resistant Phenotype"].notna().sum()
            missing_count = raw_df["Resistant Phenotype"].isna().sum()
            st.warning(f"⚠️ Detected {missing_count} missing labels. Metrics will be calculated on {valid_count} rows.")

    run_btn = st.button("🚀 Run Preprocessing & Predict", type="primary")

    if run_btn:
        with st.spinner("Processing..."):
            try:
                X, y_true, genome_ids = preprocess(raw_df, has_target=has_target)
                y_labels, y_probs = predict(X)
                
                out_df = pd.DataFrame({
                    "Genome ID": raw_df["Genome ID"],
                    "Antibiotic": raw_df["Antibiotic"],
                    "Predicted Phenotype": y_labels,
                    "Resistance Probability": np.round(y_probs, 4),
                })
                
                if has_target:
                    out_df.insert(2, "Actual Phenotype", raw_df["Resistant Phenotype"])

                st.success("✅ Model predictions generated!")
                st.subheader("📊 Predictions")
                st.dataframe(out_df.head(50), use_container_width=True)

                # Download button
                csv_bytes = out_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Predictions CSV", csv_bytes, "amr_predictions.csv", "text/csv")

                # Metrics
                if has_target:
                    st.subheader("📈 Classification Metrics")
                    valid_mask = y_true.notna()
                    y_eval_true = y_true[valid_mask]
                    y_eval_prob = y_probs[valid_mask]
                    y_eval_pred = (y_eval_prob >= CUSTOM_THRESHOLD).astype(int)

                    if len(y_eval_true) > 0:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Weighted F1", f"{f1_score(y_eval_true, y_eval_pred, average='weighted'):.4f}")
                        col2.metric("AUC-ROC", f"{roc_auc_score(y_eval_true, y_eval_prob):.4f}")
                        col3.metric("Threshold", f"{CUSTOM_THRESHOLD}")
                        
                        # Feature Importance
                        st.subheader("🏆 Top 10 Feature Importances")
                        feat_imp = pd.Series(XGB_MODEL.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        feat_imp.sort_values().plot(kind="barh", ax=ax, color="#2196F3")
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during execution: {e}")

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.caption("⚙️ **Engine:** XGBoost · Target Encoding · One-Hot Encoding")
with col2:
    st.caption("👨‍💻 **Credits:** Khizar Abbas Khan, Zeeshan Ahmad, Muhammad Humais")