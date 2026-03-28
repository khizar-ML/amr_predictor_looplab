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
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AMR Phenotype Predictor By The Elites",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Antimicrobial Resistance Phenotype Predictor \n **By The Elites**")
st.markdown(
    "Upload a CSV file from the **BVBRC genome AMR** dataset. "
    "The app will preprocess your data, load an XGBoost model, "
    "and return a downloadable predictions file."
)

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
# ADDED 'Genome ID' to the required columns
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
CUSTOM_THRESHOLD = 0.65  # Adjusted based on footer
TOP_N_ANTIBIOTICS = 10

# ─────────────────────────────────────────
# Helper: convert measurement values
# ─────────────────────────────────────────
def convert_measurement(value):
    if pd.isna(value):
        return np.nan
    try:
        result = eval(str(value))  # handles fractions like '2/73'
        return float(result)
    except Exception:
        return np.nan

# ─────────────────────────────────────────
# Preprocessing pipeline
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame, has_target: bool):
    """
    Prepares features for the XGBoost model. 
    Strictly preserves all rows.
    """
    cols = REQUIRED_COLS_WITH_TARGET if has_target else REQUIRED_COLS_NO_TARGET
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in uploaded file: {missing}")

    df = df[cols].copy()
    
    # Extract Genome ID and then drop it so it doesn't interfere with the ML features
    genome_ids = df["Genome ID"].values
    df.drop(columns=["Genome ID"], inplace=True)

    # --- Step 1: Target logic (NO DROPPING ROWS) ---
    if has_target:
        # Map targets, leaving 'Intermediate' as NaN so it doesn't break binary metrics
        # but rows are NOT dropped from the dataframe.
        y = df["Resistant Phenotype"].replace({"Nonsusceptible": "Resistant"}).map({
            "Susceptible": 0, 
            "Resistant": 1
        })
        df.drop(columns=["Resistant Phenotype"], inplace=True)
    else:
        y = None

    # --- Step 2: One-hot encode antibiotic ---
    top_antibiotics = joblib.load('top_ab_list.pkl')
    df["Antibiotic"] = df["Antibiotic"].apply(
        lambda x: x if x in top_antibiotics else "Other"
    )
    
    ohe = joblib.load('ohe_encoder.pkl')
    encoded_array = ohe.transform(df[['Antibiotic']])
    column_names = ohe.get_feature_names_out(['Antibiotic'])
    encoded_df = pd.DataFrame(encoded_array, columns=column_names, index=df.index)

    # Drop the first encoded column to prevent multicollinearity
    column_to_drop = column_names[0]
    encoded_df = encoded_df.drop(columns=[column_to_drop])

    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=['Antibiotic'])
    
    # --- Step 3: Measurement Sign → ordinal ---
    df["sign"] = df["Measurement Sign"].map(SIGN_MAP).fillna(0).astype(int)
    df.drop(columns=["Measurement Sign"], inplace=True)

    # --- Step 4: Convert Measurement Value & Flag ---
    df["Measurement Value"] = df["Measurement Value"].apply(convert_measurement)
    df["has_measurement"] = df["Measurement Value"].notnull().astype(int)

    # --- Step 5: Target-encode Genome Name ---
    # Optional safety cast to avoid Pandas 3.14 StringDtype issues
    df["Genome Name"] = df["Genome Name"].astype(object) 
    
    target_encoder = joblib.load('target_encoder.pkl')
    df["Genome Name_Encoded"] = target_encoder.transform(df[["Genome Name"]])
    df.drop(columns=["Genome Name"], inplace=True)

    # --- Step 6: Label-encode Evidence ---
    df["Evidence_Encoded"] = df["Evidence"].map(EVIDENCE_MAP)
    df.drop(columns=["Evidence"], inplace=True)

    return df, y, genome_ids

# ─────────────────────────────────────────
# Inference
# ─────────────────────────────────────────
def predict(X: pd.DataFrame):
    """Loads model and generates predictions and probabilities."""
    xgb_model = joblib.load('xgb_model.pkl')
    
    # 1. Recreate the missing column
    X['Measurement_Numeric'] = X['Measurement Value']
    
    # 2. Force the exact column order that XGBoost expects
    expected_cols = [
        'Measurement Value', 'has_measurement', 'Measurement_Numeric', 
        'Antibiotic_ampicillin', 'Antibiotic_aztreonam', 'Antibiotic_cefotaxime', 
        'Antibiotic_ceftazidime', 'Antibiotic_chloramphenicol', 'Antibiotic_ciprofloxacin', 
        'Antibiotic_meropenem', 'Antibiotic_sulfamethoxazole', 'Antibiotic_tetracycline', 
        'Antibiotic_tobramycin', 'sign', 'Genome Name_Encoded', 'Evidence_Encoded'
    ]
    
    # Fill any unexpectedly missing one-hot columns with 0
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
            
    X = X[expected_cols] # Reorder strictly
    
    # 3. Predict
    y_prob = xgb_model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= CUSTOM_THRESHOLD).astype(int)
    y_label = np.where(y_pred == 1, "Resistant", "Susceptible")

    return y_label, y_prob

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

# Defaulted to index 1 ("No target variable is absent")
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
    st.caption(f"Shape: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

    # ── Auto-Fallback if Target is Missing ────────────────
    if has_target:
        if "Resistant Phenotype" not in raw_df.columns:
            st.warning("⚠️ `Resistant Phenotype` column not found. Automatically switching to 'No Target' mode.")
            has_target = False
        elif raw_df["Resistant Phenotype"].isna().any():
            # Count valid vs missing for the user
            valid_count = raw_df["Resistant Phenotype"].notna().sum()
            missing_count = raw_df["Resistant Phenotype"].isna().sum()
            
            st.warning(
                f"⚠️ Detected {missing_count} missing values in `Resistant Phenotype`. "
                f"The app will predict phenotypes for ALL rows, but the Classification Metrics "
                f"will only be calculated using the {valid_count} rows with known labels."
            )
            # We purposely DO NOT set has_target = False here, allowing metrics to run on valid rows.

    # ── Validate columns ──────────────────────────────────
    required = REQUIRED_COLS_WITH_TARGET if has_target else REQUIRED_COLS_NO_TARGET
    missing_cols = [c for c in required if c not in raw_df.columns]

    if missing_cols:
        st.error(
            f"**Missing required columns:** {missing_cols}\n\n"
            f"Your file must contain: `{required}`"
        )
        st.stop()

    # ── Run pipeline ─────────────────────────────────────
    run_btn = st.button("🚀 Run Preprocessing & Predict", type="primary")

    if run_btn:
        with st.spinner("Preprocessing data…"):
            try:
                X, y_true, genome_ids = preprocess(raw_df, has_target=has_target)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        with st.spinner("Predicting on XGBoost model..."):
            y_labels, y_probs = predict(X)
            
        # Output dataframe generation: CHANGED Genome Name to Genome ID
        out_df = pd.DataFrame({
            "Genome ID": raw_df["Genome ID"],
            "Antibiotic": raw_df["Antibiotic"],
            "Predicted Phenotype": y_labels,
            "Resistance Probability": np.round(y_probs, 4),
        })
        
        if has_target:
            out_df.insert(2, "Actual Phenotype", raw_df["Resistant Phenotype"])

        st.success("✅ Model predictions generated!")

        # ── Predictions table ──────────────────────────
        st.subheader("📊 Predictions")
        st.dataframe(out_df.head(50), use_container_width=True)
        st.caption(f"Showing first 50 of {len(out_df):,} rows")

        # ── Download predictions ───────────────────────
        csv_bytes = out_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv_bytes,
            file_name="amr_predictions.csv",
            mime="text/csv",
        )

        # ── Classification Metrics (Only if target exists) ──
        if has_target:
            st.subheader("📈 Classification Metrics (Based on Valid Actuals)")
            
            # Filter out NaNs (e.g., "Intermediate" or missing phenotypes) just for metrics calculation
            valid_mask = y_true.notna()
            y_eval_true = y_true[valid_mask]
            y_eval_prob = y_probs[valid_mask]
            y_eval_pred = (y_eval_prob >= CUSTOM_THRESHOLD).astype(int)
            
            if len(y_eval_true) > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric("Weighted F1-Score", f"{f1_score(y_eval_true, y_eval_pred, average='weighted'):.4f}")
                col2.metric("AUC-ROC", f"{roc_auc_score(y_eval_true, y_eval_prob):.4f}")
                col3.metric("Threshold Used", f"{CUSTOM_THRESHOLD}")

                # Report table
                report = classification_report(
                    y_eval_true, y_eval_pred,
                    target_names=["Susceptible (0)", "Resistant (1)"],
                    output_dict=True
                )
                report_df = pd.DataFrame(report).transpose().round(4)
                st.dataframe(report_df, use_container_width=True)

                # Confusion matrix
                st.subheader("🔲 Confusion Matrix")
                cm = confusion_matrix(y_eval_true, y_eval_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Susceptible", "Resistant"])
                disp.plot(ax=ax, colorbar=False, cmap="Blues")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No valid binary labels found to generate metrics (e.g., all were 'Intermediate' or missing).")
                
        # Feature importance calculation logic
        st.subheader("🏆 Top 10 Feature Importances")
        xgb_model = joblib.load('xgb_model.pkl')
        feat_imp = pd.Series(
            xgb_model.feature_importances_,
            index=X.columns,
        ).sort_values(ascending=False).head(10)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        feat_imp.sort_values().plot(kind="barh", ax=ax2, color="#2196F3")
        ax2.set_title("Top 10 Feature Importances (XGBoost)")
        ax2.set_xlabel("Importance Score")
        st.pyplot(fig2)
        plt.close(fig2)

else:
    st.info(
        "👈 Upload a CSV file to get started. "
        "The file should be in the **BVBRC genome AMR** format "
        "with columns such as `Genome ID`, `Genome Name`, `Antibiotic`, `Measurement Value`, "
        "`Evidence`, `Measurement Sign`, and optionally `Resistant Phenotype`."
    )

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.caption("⚙️ **Engine:** XGBoost · Target Encoding · One-Hot Encoding")
    st.caption("📊 **Pipeline:** Built to mirror the AMR Prediction Notebook")

with col2:
    st.caption("👨‍💻 **Credits:** Khizar Abbas Khan, Zeeshan Ahmad, Muhammad Humais")
    st.caption("🚀 **Project:** CUI · LoopVerse 2.0 ML Module")