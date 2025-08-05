# ─────────────────── page_12_regression.py ───────────────────
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)

import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ──────────────────────────────────────────────────────────────

def show_page() -> None:
    st.header("📈 Step 12 · Logistic Regression Analysis")

    # Prerequisite checks
    if "factor_scores_df" not in st.session_state or st.session_state.factor_scores_df is None:
        st.error("⚠️ No factor scores available. Please complete factor analysis first.")
        return
    if "selected_target_col" not in st.session_state:
        st.error("⚠️ No target variable selected. Please complete previous steps.")
        return

    # 1 Prepare data
    prepare_regression_data()

    # 2 Show preparation results
    display_data_summary()

    # 3 VIF
    st.subheader("🔍 Multicollinearity Check (VIF Analysis)")
    if st.button("Calculate VIF", type="secondary"):
        calculate_vif_analysis()

    # 4 Variable selection UI
    st.subheader("🎛️ Variable Selection")
    variable_selection_interface()

    # 5 Correlation Matrix (NEW)
    st.subheader("📈 Correlation Matrix (Selected Variables)")
    if st.button("Show Correlation Matrix"):
        display_correlation_matrix()

    # 6 Model
    st.subheader("🚀 Model Training & Evaluation")
    if st.button("Train Logistic Regression Model", type="primary"):
        train_and_evaluate_model()

# ──────────────────────────────────────────────────────────────

def prepare_regression_data() -> None:
    """Collect everything needed for modelling + work out raw features."""
    # Data already built in earlier steps
    factor_scores_df = st.session_state.factor_scores_df # factor scores
    model_df = st.session_state.model_df # ALL 37 numeric features
    feature_list = st.session_state.feature_list # list of 37 names
    selected_features = st.session_state.selected_features # 22 chosen for FA
    target_col = st.session_state.selected_target_col

    # Split target & predictors
    X_factors = factor_scores_df.reset_index(drop=True) # k factor columns
    y_target = model_df[target_col].reset_index(drop=True)

    # ---- NEW: find RAW FEATURES ----
    raw_features = [f for f in feature_list if f not in selected_features]

    # Store for later pages
    st.session_state.X_factors = X_factors
    st.session_state.y_target = y_target
    st.session_state.factor_names = list(X_factors.columns) # factored variable names
    st.session_state.raw_features = raw_features # 15 names
    st.session_state.model_df_full = model_df # keep full df

    # Debug line so you can verify counts
    st.info(
        f"Step 5 features: {len(feature_list)} · "
        f"Selected for FA: {len(selected_features)} · "
        f"Raw features now available: {len(raw_features)}"
    )

# ──────────────────────────────────────────────────────────────

def display_data_summary() -> None:
    st.subheader("📊 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Factored Vars", len(st.session_state.factor_names))
    col2.metric("Raw Vars", len(st.session_state.raw_features))
    col3.metric("Sample Size", len(st.session_state.X_factors))
    col4.metric("Target Var", st.session_state.selected_target_name)

# ──────────────────────────────────────────────────────────────

def variable_selection_interface() -> None:
    """Two-tab selector: Factored & Raw."""
    factor_names = st.session_state.factor_names
    raw_features = st.session_state.raw_features

    # initialise session selections
    if "sel_factored" not in st.session_state:
        st.session_state.sel_factored = factor_names.copy()
    if "sel_raw" not in st.session_state:
        st.session_state.sel_raw = []

    tab_f, tab_r = st.tabs(["🔬 Factored Variables", "📊 Raw Variables"])

    # ---- Factored tab ----
    with tab_f:
        col1, col2 = st.columns(2)
        if col1.button("Select All Factored"):
            st.session_state.sel_factored = factor_names.copy()
            st.rerun()
        if col2.button("Deselect All Factored"):
            st.session_state.sel_factored = []
            st.rerun()

        for v in factor_names:
            checked = st.checkbox(v, v in st.session_state.sel_factored, key=f"f_{v}")
            if checked and v not in st.session_state.sel_factored:
                st.session_state.sel_factored.append(v)
            if not checked and v in st.session_state.sel_factored:
                st.session_state.sel_factored.remove(v)

    # ---- Raw tab ----
    with tab_r:
        if not raw_features:
            st.info("All original features were selected for factor analysis.")
        else:
            col1, col2 = st.columns(2)
            if col1.button("Select All Raw"):
                st.session_state.sel_raw = raw_features.copy()
                st.rerun()
            if col2.button("Deselect All Raw"):
                st.session_state.sel_raw = []
                st.rerun()

            for v in raw_features:
                checked = st.checkbox(v, v in st.session_state.sel_raw, key=f"r_{v}")
                if checked and v not in st.session_state.sel_raw:
                    st.session_state.sel_raw.append(v)
                if not checked and v in st.session_state.sel_raw:
                    st.session_state.sel_raw.remove(v)

    # quick summary
    st.write(
        f"**Selected:** "
        f"{len(st.session_state.sel_factored)} factored + "
        f"{len(st.session_state.sel_raw)} raw = "
        f"{len(st.session_state.sel_factored)+len(st.session_state.sel_raw)}"
    )

# ──────────────────────────────────────────────────────────────
# NEW: Correlation Matrix Function
# ──────────────────────────────────────────────────────────────

def display_correlation_matrix() -> None:
    """Show correlation matrix for all selected variables (factored + raw)."""
    sel_factored = st.session_state.get('sel_factored', [])
    sel_raw = st.session_state.get('sel_raw', [])
    
    if not sel_factored and not sel_raw:
        st.error("⚠️ Please select at least one variable to show correlation matrix.")
        return
    
    # Build combined dataset
    X_combined = pd.DataFrame()
    
    # Add factored variables
    if sel_factored:
        X_factors_selected = st.session_state.X_factors[sel_factored]
        X_combined = pd.concat([X_combined, X_factors_selected], axis=1)
    
    # Add raw variables
    if sel_raw:
        raw_df = st.session_state.model_df_full[sel_raw].fillna(
            st.session_state.model_df_full[sel_raw].median()
        )
        X_combined = pd.concat([X_combined, raw_df], axis=1)
    
    if X_combined.shape[1] < 2:
        st.warning("Need at least 2 variables to compute correlation matrix.")
        return
    
    # Calculate correlation matrix
    corr_matrix = X_combined.corr()
    
    # Create interactive heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=f"Correlation Matrix ({X_combined.shape[1]} Variables)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation table in expander
    with st.expander("📊 View Correlation Table"):
        st.dataframe(corr_matrix.round(3), use_container_width=True)

# ──────────────────────────────────────────────────────────────

def calculate_vif_analysis() -> None:
    """VIF on the currently selected variables (factored + raw)."""
    if not st.session_state.sel_factored and not st.session_state.sel_raw:
        st.error("Select at least one variable first.")
        return

    # build combined X
    X = pd.DataFrame()
    if st.session_state.sel_factored:
        X = pd.concat([X, st.session_state.X_factors[st.session_state.sel_factored]], axis=1)
    if st.session_state.sel_raw:
        raw_df = st.session_state.model_df_full[st.session_state.sel_raw].fillna(
            st.session_state.model_df_full[st.session_state.sel_raw].median()
        )
        X = pd.concat([X, raw_df], axis=1)

    X_const = sm.add_constant(X)
    vif = pd.DataFrame(
        {
            "Variable": X_const.columns,
            "VIF": [variance_inflation_factor(X_const.values, i)
                    for i in range(X_const.shape[1])]
        }
    ).sort_values("VIF", ascending=False)

    st.dataframe(vif, use_container_width=True)
    st.session_state.vif_results = vif

# ──────────────────────────────────────────────────────────────

def train_and_evaluate_model() -> None:
    """Train logistic regression using the selected variables."""
    sel_f = st.session_state.sel_factored
    sel_r = st.session_state.sel_raw
    if not sel_f and not sel_r:
        st.error("Select at least one variable.")
        return

    try:
        # Build X matrix
        X = pd.DataFrame()
        if sel_f:
            X = pd.concat([X, st.session_state.X_factors[sel_f]], axis=1)
        if sel_r:
            raw_df = st.session_state.model_df_full[sel_r].fillna(
                st.session_state.model_df_full[sel_r].median()
            )
            X = pd.concat([X, raw_df], axis=1)

        y = st.session_state.y_target
        
        # FIX: Ensure X and y have the same number of rows by combining and dropping NAs
        # Combine X and y into single dataframe
        combined_data = pd.concat([X, y], axis=1)
        
        # Drop rows with any missing values
        combined_clean = combined_data.dropna(axis=0)
        
        # Separate back into X and y
        y_clean = combined_clean.iloc[:, -1]  # Last column (target)
        X_clean = combined_clean.iloc[:, :-1]  # All columns except last
        
        st.info(f"Data cleaning: {len(combined_data)} → {len(combined_clean)} rows (dropped {len(combined_data) - len(combined_clean)} rows with missing values)")
        
        # Check if we have enough data
        if len(X_clean) < 10:
            st.error("⚠️ Insufficient data after cleaning. Need at least 10 rows.")
            return
            
        # Check class distribution
        class_counts = y_clean.value_counts()
        if len(class_counts) < 2:
            st.error("⚠️ Target variable has only one class. Cannot perform classification.")
            return
            
        min_class_count = class_counts.min()
        if min_class_count < 2:
            st.error(f"⚠️ Smallest class has only {min_class_count} samples. Need at least 2 per class.")
            return

        # Train-test split with cleaned data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42, 
            stratify=y_clean if min_class_count >= 3 else None  # Only stratify if enough samples
        )

        with st.spinner("Training model…"):
            model = LogisticRegression(max_iter=1_000, random_state=42)
            model.fit(X_train, y_train)

        # predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        # summary
        st.subheader("📊 Performance Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall", f"{rec:.3f}")
        c4.metric("F1-Score", f"{f1:.3f}")
        c5.metric("AUC-ROC", f"{auc:.3f}")

        # coefficients
        coef_df = pd.DataFrame({
            "Variable": X_clean.columns,
            "Coefficient": model.coef_[0],
            "AbsCoef": np.abs(model.coef_[0]),
            "Type": ["Factored" if v in sel_f else "Raw" for v in X_clean.columns]
        }).sort_values("AbsCoef", ascending=False)

        fig = px.bar(
            coef_df, y="Variable", x="Coefficient",
            orientation="h", color="Type",
            color_discrete_map={"Factored": "#2E86AB", "Raw": "#F24236"},
            title="Variable Importance (Logistic Regression Coefficients)"
        )
        fig.update_layout(height=max(400, len(coef_df) * 28))
        st.plotly_chart(fig, use_container_width=True)

        # confusion matrix + ROC
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"),
                           color_continuous_scale="Blues")
        st.plotly_chart(fig_cm, use_container_width=True)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"ROC curve (AUC = {auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash"), name="Random"))
        fig_roc.update_layout(title="ROC Curve",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

        # classification report
        st.subheader("📋 Classification Report")
        rep = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
        st.dataframe(rep.round(3), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error during model training: {str(e)}")
        st.info("This might be due to data inconsistencies. Please check your data preparation steps.")

# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    show_page()
