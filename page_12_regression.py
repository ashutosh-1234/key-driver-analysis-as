import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ─────────────────────────────────────────────────────────────────────────────
def show_page():
    st.header("📈 Step 12: Logistic Regression Analysis")

    # Check prerequisites
    if 'factor_scores_df' not in st.session_state or st.session_state.factor_scores_df is None:
        st.error("⚠️ No factor scores available. Please complete factor analysis first.")
        return
    if 'selected_target_col' not in st.session_state:
        st.error("⚠️ No target variable selected. Please complete previous steps.")
        return

    # Prepare data
    prepare_regression_data()

    # Show data preparation results
    display_data_summary()

    # VIF Analysis
    st.subheader("🔍 Multicollinearity Check (VIF Analysis)")
    if st.button("Calculate VIF", type="secondary"):
        calculate_vif_analysis()

    # Variable selection interface
    st.subheader("🎛️ Variable Selection")
    variable_selection_interface()

    # Model training and evaluation
    st.subheader("🚀 Model Training & Evaluation")
    if st.button("Train Logistic Regression Model", type="primary"):
        train_and_evaluate_model()

# ─────────────────────────────────────────────────────────────────────────────
def prepare_regression_data():
    """Prepare data for logistic regression"""
    factor_scores_df   = st.session_state.factor_scores_df
    final_model_df     = st.session_state.final_model_df
    selected_target_col = st.session_state.selected_target_col

    # Combine factor scores with target variable
    X_factors = factor_scores_df.reset_index(drop=True)
    y_target  = final_model_df[selected_target_col].reset_index(drop=True)

    # Store in session state
    st.session_state.X_factors     = X_factors
    st.session_state.y_target      = y_target
    st.session_state.feature_names = list(factor_scores_df.columns)

# ─────────────────────────────────────────────────────────────────────────────
def display_data_summary():
    """Display data preparation summary"""
    st.subheader("📊 Dataset Summary")

    X_factors     = st.session_state.X_factors
    y_target      = st.session_state.y_target
    feature_names = st.session_state.feature_names

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Independent Variables", len(feature_names))
    with col2:
        st.metric("Sample Size", len(X_factors))
    with col3:
        st.metric("Target Variable", st.session_state.selected_target_name)

    # ✂️ REMOVED: Target Variable Distribution visual (pie chart + details)
    # --------------------------------------------------------------------
    # The section that rendered the pie chart and textual distribution
    # details has been intentionally removed as requested.
    # --------------------------------------------------------------------

    # Factor overview
    st.subheader("🔍 Factor Variables Overview")
    with st.expander("View all factor variables"):
        for i, factor in enumerate(feature_names, 1):
            st.write(f"{i}. {factor}")

# ─────────────────────────────────────────────────────────────────────────────
def calculate_vif_analysis():
    """Calculate and display VIF analysis"""
    X_factors = st.session_state.X_factors

    # Add constant for VIF calculation
    X_with_const = sm.add_constant(X_factors)

    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Factor"] = X_with_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(X_with_const.shape[1])
    ]

    vif_data = vif_data.sort_values('VIF', ascending=False)
    st.write("📊 **VIF Results:**")
    st.dataframe(vif_data, use_container_width=True)

    # Interpretation
    high_vif     = vif_data[vif_data['VIF'] > 10]
    moderate_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)]
    low_vif      = vif_data[vif_data['VIF'] <= 5]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High VIF (>10)", len(high_vif))
    with col2:
        st.metric("Moderate VIF (5-10)", len(moderate_vif))
    with col3:
        st.metric("Low VIF (≤5)", len(low_vif))

    if len(high_vif) > 0:
        st.warning("⚠️ Consider removing factors with VIF > 10.")
        for _, row in high_vif.iterrows():
            if row['Factor'] != 'const':
                st.write(f"• {row['Factor']}: {row['VIF']:.2f}")
    else:
        st.success("✅ No high multicollinearity detected.")

    st.session_state.vif_results = vif_data

# ─────────────────────────────────────────────────────────────────────────────
# (All remaining functions: variable_selection_interface, train_and_evaluate_model,
#  display_model_results, etc., remain UNCHANGED from your original code.)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    show_page()
