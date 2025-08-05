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

    # ── prerequisite checks ───────────────────────────────────
    if "factor_scores_df" not in st.session_state or st.session_state.factor_scores_df is None:
        st.error("⚠️ No factor scores available. Please complete factor analysis first.")
        return
    if "selected_target_col" not in st.session_state:
        st.error("⚠️ No target variable selected. Please complete previous steps.")
        return

    # 1 ▸ prepare data
    prepare_regression_data()

    # 2 ▸ dataset summary
    display_data_summary()

    # 3 ▸ VIF
    st.subheader("🔍 Multicollinearity Check (VIF Analysis)")
    if st.button("Calculate VIF", type="secondary"):
        calculate_vif_analysis()

    # 4 ▸ variable selection
    st.subheader("🎛️ Variable Selection")
    variable_selection_interface()

    # 5 ▸ correlation matrix  (NEW)
    st.subheader("📈 Correlation Matrix (Selected Variables)")
    if st.button("Show Correlation Matrix"):
        display_correlation_matrix()

    # 6 ▸ model training
    st.subheader("🚀 Model Training & Evaluation")
    if st.button("Train Logistic Regression Model", type="primary"):
        train_and_evaluate_model()


# ──────────────────────────────────────────────────────────────
# helper blocks
# ──────────────────────────────────────────────────────────────
def prepare_regression_data() -> None:
    """Pull everything needed for modelling and split out raw-feature pool."""
    factor_scores_df = st.session_state.factor_scores_df
    model_df          = st.session_state.model_df
    feature_list      = st.session_state.feature_list
    selected_features = st.session_state.selected_features    # those sent to FA
    target_col        = st.session_state.selected_target_col

    X_factors = factor_scores_df.reset_index(drop=True)
    y_target  = model_df[target_col].reset_index(drop=True)

    raw_features = [f for f in feature_list if f not in selected_features]

    st.session_state.X_factors    = X_factors
    st.session_state.y_target     = y_target
    st.session_state.factor_names = list(X_factors.columns)
    st.session_state.raw_features = raw_features
    st.session_state.model_df_full = model_df


def display_data_summary() -> None:
    st.subheader("📊 Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Factored Vars", len(st.session_state.factor_names))
    c2.metric("Raw Vars",     len(st.session_state.raw_features))
    c3.metric("Sample Size",  len(st.session_state.X_factors))
    c4.metric("Target Var",   st.session_state.selected_target_name)


# ───────────────────────── variable selection ─────────────────────────
def variable_selection_interface() -> None:
    factor_names = st.session_state.factor_names
    raw_features = st.session_state.raw_features

    if "sel_factored" not in st.session_state:
        st.session_state.sel_factored = factor_names.copy()
    if "sel_raw" not in st.session_state:
        st.session_state.sel_raw = []

    tab_f, tab_r = st.tabs(["🔬 Factored Variables", "📊 Raw Variables"])

    # factored
    with tab_f:
        a, b = st.columns(2)
        if a.button("Select All Factored"):
            st.session_state.sel_factored = factor_names.copy()
            st.rerun()
        if b.button("Deselect All Factored"):
            st.session_state.sel_factored = []
            st.rerun()

        for v in factor_names:
            chk = st.checkbox(v, v in st.session_state.sel_factored, key=f"f_{v}")
            if chk and v not in st.session_state.sel_factored:
                st.session_state.sel_factored.append(v)
            if not chk and v in st.session_state.sel_factored:
                st.session_state.sel_factored.remove(v)

    # raw
    with tab_r:
        if not raw_features:
            st.info("All original features were selected for factor analysis.")
        else:
            a, b = st.columns(2)
            if a.button("Select All Raw"):
                st.session_state.sel_raw = raw_features.copy()
                st.rerun()
            if b.button("Deselect All Raw"):
                st.session_state.sel_raw = []
                st.rerun()

            for v in raw_features:
                chk = st.checkbox(v, v in st.session_state.sel_raw, key=f"r_{v}")
                if chk and v not in st.session_state.sel_raw:
                    st.session_state.sel_raw.append(v)
                if not chk and v in st.session_state.sel_raw:
                    st.session_state.sel_raw.remove(v)

    st.write(
        f"**Selected:** {len(st.session_state.sel_factored)} factored + "
        f"{len(st.session_state.sel_raw)} raw = "
        f"{len(st.session_state.sel_factored) + len(st.session_state.sel_raw)}"
    )


# ───────────────────────── correlation matrix ─────────────────────────
def display_correlation_matrix() -> None:
    """Interactive heat-map for all currently-selected variables."""
    sel_f = st.session_state.sel_factored
    sel_r = st.session_state.sel_raw

    if not sel_f and not sel_r:
        st.error("Select at least one variable first.")
        return

    # build combined X
    X = pd.DataFrame()
    if sel_f:
        X = pd.concat([X, st.session_state.X_factors[sel_f]], axis=1)
    if sel_r:
        raw_df = st.session_state.model_df_full[sel_r]\
                                   .fillna(st.session_state.model_df_full[sel_r].median())
        X = pd.concat([X, raw_df], axis=1)

    if X.shape[1] < 2:
        st.warning("Need at least two variables to compute correlations.")
        return

    corr = X.corr()

    fig = px.imshow(
        corr, text_auto=True, zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r",
        title=f"Correlation Matrix ({X.shape[1]} Variables)"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Correlation Table"):
        st.dataframe(corr.round(3), use_container_width=True)


# ───────────────────────── VIF analysis ─────────────────────────
def calculate_vif_analysis() -> None:
    if not st.session_state.sel_factored and not st.session_state.sel_raw:
        st.error("Select at least one variable first.")
        return

    X = pd.DataFrame()
    if st.session_state.sel_factored:
        X = pd.concat([X,
                       st.session_state.X_factors[st.session_state.sel_factored]],
                      axis=1)
    if st.session_state.sel_raw:
        raw_df = st.session_state.model_df_full[st.session_state.sel_raw]\
                                   .fillna(st.session_state.model_df_full[st.session_state.sel_raw].median())
        X = pd.concat([X, raw_df], axis=1)

    X_const = sm.add_constant(X)
    vif = pd.DataFrame({
        "Variable": X_const.columns,
        "VIF": [variance_inflation_factor(X_const.values, i)
                for i in range(X_const.shape[1])]
    }).sort_values("VIF", ascending=False)

    st.dataframe(vif, use_container_width=True)
    st.session_state.vif_results = vif


# ───────────────────────── model training ─────────────────────────
def train_and_evaluate_model() -> None:
    sel_f = st.session_state.sel_factored
    sel_r = st.session_state.sel_raw
    if not sel_f and not sel_r:
        st.error("Select at least one variable.")
        return

    # X matrix
    X = pd.DataFrame()
    if sel_f:
        X = pd.concat([X, st.session_state.X_factors[sel_f]], axis=1)
    if sel_r:
        raw_df = st.session_state.model_df_full[sel_r]\
                                   .fillna(st.session_state.model_df_full[sel_r].median())
        X = pd.concat([X, raw_df], axis=1)

    y = st.session_state.y_target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    with st.spinner("Training model…"):
        model = LogisticRegression(max_iter=1_000, random_state=42)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    st.subheader("📊 Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall",    f"{rec:.3f}")
    c4.metric("F1-Score",  f"{f1:.3f}")
    c5.metric("AUC-ROC",   f"{auc:.3f}")

    coef_df = pd.DataFrame({
        "Variable": X.columns,
        "Coefficient": model.coef_[0],
        "Abs": np.abs(model.coef_[0]),
        "Type": ["Factored" if v in sel_f else "Raw" for v in X.columns]
    }).sort_values("Abs", ascending=False)

    fig_coef = px.bar(
        coef_df, y="Variable", x="Coefficient",
        orientation="h", color="Type",
        color_discrete_map={"Factored": "#2E86AB", "Raw": "#F24236"},
        title="Variable Importance (Logistic Regression Coefficients)"
    )
    fig_coef.update_layout(height=max(400, len(coef_df)*28))
    st.plotly_chart(fig_coef, use_container_width=True)

    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(
        px.imshow(cm, text_auto=True, title="Confusion Matrix",
                  labels=dict(x="Predicted", y="Actual"),
                  color_continuous_scale="Blues"),
        use_container_width=True
    )

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"ROC curve (AUC = {auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash"), name="Random"))
    fig_roc.update_layout(title="ROC Curve",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("📋 Classification Report")
    st.dataframe(
        pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.round(3),
        use_container_width=True
    )


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    show_page()
