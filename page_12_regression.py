import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)

import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ────────────────────────────────────────────────────────────────────────────────
#  Main Page
# ────────────────────────────────────────────────────────────────────────────────
def show_page():
    st.header("📈 Step 12: Logistic Regression Analysis")

    # --- Prerequisite checks ----------------------------------------------------
    if 'factor_scores_df' not in st.session_state or st.session_state.factor_scores_df is None:
        st.error("⚠️ No factor scores available. Please complete factor analysis first.")
        return

    if 'selected_target_col' not in st.session_state:
        st.error("⚠️ No target variable selected. Please complete previous steps.")
        return

    # --- Data preparation & basic summary --------------------------------------
    prepare_regression_data()
    display_data_summary()

    # --- VIF analysis ----------------------------------------------------------
    st.subheader("🔍 Multicollinearity Check (VIF Analysis)")
    if st.button("Calculate VIF", type="secondary"):
        calculate_vif_analysis()

    # --- Variable selection interface -----------------------------------------
    st.subheader("🎛️ Variable Selection")
    variable_selection_interface()

    # --- NEW: Correlation matrix ----------------------------------------------
    st.subheader("📈 Correlation Matrix (Selected Variables)")
    if st.button("Show Correlation Matrix"):
        display_correlation_matrix()

    # --- Model training & evaluation ------------------------------------------
    st.subheader("🚀 Model Training & Evaluation")
    if st.button("Train Logistic Regression Model", type="primary"):
        train_and_evaluate_model()


# ────────────────────────────────────────────────────────────────────────────────
#  Helper Functions
# ────────────────────────────────────────────────────────────────────────────────
def prepare_regression_data():
    """Combine factor scores with target variable and store in session_state."""
    X_factors = st.session_state.factor_scores_df.reset_index(drop=True)
    y_target = st.session_state.final_model_df[st.session_state.selected_target_col].reset_index(drop=True)

    st.session_state.X_factors = X_factors
    st.session_state.y_target = y_target
    st.session_state.feature_names = list(X_factors.columns)


def display_data_summary():
    """Basic dataset KPIs & target distribution."""
    X_factors = st.session_state.X_factors
    y_target = st.session_state.y_target
    feature_names = st.session_state.feature_names

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Independent Variables", len(feature_names))
    with col2:
        st.metric("Sample Size", len(X_factors))
    with col3:
        st.metric("Target Variable", st.session_state.selected_target_name)

    # Target pie-chart
    target_counts = y_target.value_counts()
    fig = px.pie(values=target_counts.values,
                 names=target_counts.index,
                 title="Target Variable Distribution",
                 color_discrete_sequence=['lightcoral', 'lightblue'])
    st.plotly_chart(fig, use_container_width=True)


def calculate_vif_analysis():
    """Variance-Inflation-Factor table and recommendations."""
    X_with_const = sm.add_constant(st.session_state.X_factors)
    vif_data = pd.DataFrame({
        "Factor": X_with_const.columns,
        "VIF": [variance_inflation_factor(X_with_const.values, i)
                for i in range(X_with_const.shape[1])]
    }).sort_values('VIF', ascending=False)

    st.write("📊 **VIF Results:**")
    st.dataframe(vif_data, use_container_width=True)

    # Store for later use
    st.session_state.vif_results = vif_data


def variable_selection_interface():
    """Checkbox UI for choosing which factors go into the model."""
    feature_names = st.session_state.feature_names

    # Initialise selection state
    if 'selected_regression_features' not in st.session_state:
        st.session_state.selected_regression_features = feature_names.copy()

    # Bulk selection buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Select All"):
            st.session_state.selected_regression_features = feature_names.copy()
            st.rerun()
    with col2:
        if st.button("Deselect All"):
            st.session_state.selected_regression_features = []
            st.rerun()
    with col3:
        if 'vif_results' in st.session_state:
            if st.button("Remove High VIF (>10)"):
                high_vif_vars = st.session_state.vif_results.query("VIF > 10 and Factor != 'const'")['Factor']
                st.session_state.selected_regression_features = [f for f in feature_names if f not in high_vif_vars]
                st.rerun()

    # Individual check-boxes
    selected_features = []
    for var in feature_names:
        checked = st.checkbox(var,
                              value=var in st.session_state.selected_regression_features,
                              key=f"chk_{var}")
        if checked:
            selected_features.append(var)

    st.session_state.selected_regression_features = selected_features
    st.info(f"**Selected Variables:** {len(selected_features)} / {len(feature_names)}")


# ────────────────────────────────────────────────────────────────────────────────
#  NEW: Correlation Matrix
# ────────────────────────────────────────────────────────────────────────────────
def display_correlation_matrix():
    """Show correlation heat-map & table for selected variables."""
    selected_features = st.session_state.selected_regression_features
    if len(selected_features) < 2:
        st.warning("Select at least two variables to compute correlations.")
        return

    corr_matrix = st.session_state.X_factors[selected_features].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show correlation table"):
        st.dataframe(corr_matrix.round(3), use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
#  Modelling
# ────────────────────────────────────────────────────────────────────────────────
def train_and_evaluate_model():
    """Fit logistic regression and present performance metrics."""
    selected_features = st.session_state.selected_regression_features
    if len(selected_features) == 0:
        st.error("⚠️ Please select at least one variable for modeling.")
        return

    X = st.session_state.X_factors[selected_features]
    y = st.session_state.y_target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    with st.spinner("Training logistic regression model..."):
        model = LogisticRegression(random_state=42, max_iter=1_000)
        model.fit(X_train, y_train)

    # Predictions & probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Store results
    st.session_state.regression_model = model

    # ── Key driver (coefficient) plot ─────────────────────────────────────────
    coeffs = pd.DataFrame({
        'Factor': selected_features,
        'Coefficient': model.coef_[0],
        'Abs': np.abs(model.coef_[0])
    }).sort_values('Abs', ascending=False)

    fig_coeffs = px.bar(
        coeffs,
        y='Factor', x='Coefficient',
        orientation='h',
        title='Factor Importance (Logistic Regression Coefficients)',
        color='Coefficient',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0
    )
    fig_coeffs.update_layout(height=max(400, len(selected_features)*30))
    st.plotly_chart(fig_coeffs, use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1-Score", f"{f1:.3f}")
    col5.metric("AUC-ROC", f"{auc:.3f}")

    # ── Confusion Matrix & ROC ───────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       title="Confusion Matrix",
                       labels=dict(x="Predicted", y="Actual"))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f"ROC Curve (AUC = {auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 name='Random', line=dict(dash='dash')))
    fig_roc.update_layout(title="ROC Curve",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_cm, use_container_width=True)
    col2.plotly_chart(fig_roc, use_container_width=True)

    # ── Classification report table ──────────────────────────────────────────
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    st.subheader("📋 Detailed Classification Report")
    st.dataframe(report_df.round(3), use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    show_page()
