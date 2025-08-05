import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ─────────────────────────────────────────────────────────────────────────────
def show_page() -> None:
    st.header("📈 Step 12: Logistic Regression Analysis")

    # Prerequisite checks
    if "factor_scores_df" not in st.session_state or st.session_state.factor_scores_df is None:
        st.error("⚠️ No factor scores available. Please complete factor analysis first.")
        return
    if "selected_target_col" not in st.session_state:
        st.error("⚠️ No target variable selected. Please complete previous steps.")
        return

    # Prepare and display data
    prepare_regression_data()
    display_data_summary()

    # VIF analysis
    st.subheader("🔍 Multicollinearity Check (VIF Analysis)")
    if st.button("Calculate VIF", type="secondary"):
        calculate_vif_analysis()

    # Variable selection
    st.subheader("🎛️ Variable Selection")
    variable_selection_interface()

    # Model training
    st.subheader("🚀 Model Training & Evaluation")
    if st.button("Train Logistic Regression Model", type="primary"):
        train_and_evaluate_model()


# ─────────────────────────────────────────────────────────────────────────────
def prepare_regression_data() -> None:
    """Create X_factors, y_target, feature_names in session_state."""
    factor_scores_df = st.session_state.factor_scores_df
    final_model_df = st.session_state.final_model_df
    target_col = st.session_state.selected_target_col

    X_factors = factor_scores_df.reset_index(drop=True)
    y_target = final_model_df[target_col].reset_index(drop=True)

    st.session_state.X_factors = X_factors
    st.session_state.y_target = y_target
    st.session_state.feature_names = list(factor_scores_df.columns)


# ─────────────────────────────────────────────────────────────────────────────
def display_data_summary() -> None:
    """Show basic dataset info (pie-chart block removed)."""
    st.subheader("📊 Dataset Summary")

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

    # 🔻 PIE-CHART SECTION REMOVED AS REQUESTED 🔻
    # -------------------------------------------------------------------------
    # (No target-distribution pie chart or percentage text here anymore.)
    # -------------------------------------------------------------------------

    # Factor overview
    st.subheader("🔍 Factor Variables Overview")
    with st.expander("View all factor variables"):
        for i, fac in enumerate(feature_names, 1):
            st.write(f"{i}. {fac}")


# ─────────────────────────────────────────────────────────────────────────────
def calculate_vif_analysis() -> None:
    """Compute VIF for all factor variables."""
    X_factors = st.session_state.X_factors
    X_const = sm.add_constant(X_factors)

    vif_df = pd.DataFrame(
        {
            "Factor": X_const.columns,
            "VIF": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
        }
    ).sort_values("VIF", ascending=False)

    st.write("📊 **VIF Results:**")
    st.dataframe(vif_df, use_container_width=True)

    high_vif = vif_df[vif_df["VIF"] > 10]
    moderate_vif = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]
    low_vif = vif_df[vif_df["VIF"] <= 5]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High VIF (>10)", len(high_vif))
    with col2:
        st.metric("Moderate VIF (5-10)", len(moderate_vif))
    with col3:
        st.metric("Low VIF (≤5)", len(low_vif))

    if len(high_vif) > 0:
        st.warning("⚠️ Consider removing factors with VIF > 10")
        for _, row in high_vif.iterrows():
            if row["Factor"] != "const":
                st.write(f"• {row['Factor']}: {row['VIF']:.2f}")
    else:
        st.success("✅ No high multicollinearity detected")

    st.session_state.vif_results = vif_df


# ─────────────────────────────────────────────────────────────────────────────
def variable_selection_interface() -> None:
    """Checkbox interface for selecting factor variables."""
    feature_names = st.session_state.feature_names

    if "selected_regression_features" not in st.session_state:
        st.session_state.selected_regression_features = feature_names.copy()

    st.write("Select variables to include in the logistic regression model:")

    # Bulk buttons
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
        if "vif_results" in st.session_state:
            if st.button("Remove High VIF (>10)"):
                high_vif_vars = (
                    st.session_state.vif_results[st.session_state.vif_results["VIF"] > 10]["Factor"]
                    .tolist()
                )
                high_vif_vars = [v for v in high_vif_vars if v != "const"]
                st.session_state.selected_regression_features = [
                    v for v in feature_names if v not in high_vif_vars
                ]
                st.rerun()

    # Category-based checkboxes
    categories: dict[str, list[str]] = {}
    for feat in feature_names:
        if "_Factor_" in feat:
            cat = feat.split("_Factor_")[0]
        else:
            cat = "Other"
        categories.setdefault(cat, []).append(feat)

    selected = []
    for cat, vars_in_cat in categories.items():
        st.write(f"**{cat}:**")
        for var in vars_in_cat:
            checked = st.checkbox(
                var,
                value=var in st.session_state.selected_regression_features,
                key=f"chk_{var}",
            )
            if checked:
                selected.append(var)

    st.session_state.selected_regression_features = selected
    st.write(f"**Selected Variables:** {len(selected)} of {len(feature_names)}")


# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate_model() -> None:
    """Train logistic regression and show results."""
    selected_vars = st.session_state.selected_regression_features
    if not selected_vars:
        st.error("⚠️ Please select at least one variable.")
        return

    X = st.session_state.X_factors[selected_vars]
    y = st.session_state.y_target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    with st.spinner("Training logistic regression model…"):
        model = LogisticRegression(max_iter=1_000, random_state=42)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    display_model_results(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_proba,
        selected_vars,
    )


# ─────────────────────────────────────────────────────────────────────────────
def display_model_results(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    y_proba,
    selected_vars,
) -> None:
    """Visualise coefficients and performance metrics."""
    st.subheader("🎯 Model Performance Results")

    # Coefficients
    st.subheader("🔑 Key Driver Analysis – Factor Importance")
    coef_df = (
        pd.DataFrame(
            {
                "Factor": selected_vars,
                "Coefficient": model.coef_[0],
                "Abs_Coefficient": np.abs(model.coef_[0]),
            }
        )
        .sort_values("Abs_Coefficient", ascending=False)
        .reset_index(drop=True)
    )

    fig = px.bar(
        coef_df,
        y="Factor",
        x="Coefficient",
        orientation="h",
        color="Coefficient",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Factor Importance (Logistic Regression Coefficients)",
    )
    fig.update_layout(height=max(400, len(selected_vars) * 30))
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Detailed Coefficients:**")
    st.dataframe(coef_df[["Factor", "Coefficient"]].round(4), use_container_width=True)

    # Metrics
    st.subheader("📊 Model Performance Metrics")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1-Score", f"{f1:.3f}")
    m5.metric("AUC-ROC", f"{auc:.3f}")

    # Confusion matrix & ROC
    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_test, y_pred)
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            color_continuous_scale="Blues",
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    with c2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC (AUC = {auc:.3f})",
                line=dict(color="blue", width=2),
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="red", dash="dash"),
            )
        )
        roc_fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    # Classification report
    st.subheader("📋 Detailed Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    st.dataframe(report_df.round(3), use_container_width=True)

    # Summary
    st.subheader("📈 Model Summary")
    st.write(f"• **{len(selected_vars)} factors** used in final model")
    st.write(f"• **Training set:** {len(X_train):,} rows")
    st.write(f"• **Test set:** {len(X_test):,} rows")
    if auc > 0.8:
        st.success("🎉 Excellent model performance (AUC > 0.8)")
    elif auc > 0.7:
        st.info("👍 Good model performance (AUC > 0.7)")
    else:
        st.warning("⚠️ Model performance could be improved (AUC ≤ 0.7)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    show_page()
