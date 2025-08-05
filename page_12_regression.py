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

    # Enhanced variable selection (factored + raw)
    st.subheader("🎛️ Enhanced Variable Selection")
    enhanced_variable_selection_interface()

    # Model training
    st.subheader("🚀 Model Training & Evaluation")
    if st.button("Train Logistic Regression Model", type="primary"):
        train_and_evaluate_model()


# ─────────────────────────────────────────────────────────────────────────────
def prepare_regression_data() -> None:
    """Create X_factors, y_target, feature_names, and identify raw features."""
    factor_scores_df = st.session_state.factor_scores_df
    final_model_df = st.session_state.final_model_df
    target_col = st.session_state.selected_target_col

    # Factor data
    X_factors = factor_scores_df.reset_index(drop=True)
    y_target = final_model_df[target_col].reset_index(drop=True)

    # Enhanced raw feature detection
    # Method 1: Get ALL numeric columns from final_model_df (broader approach)
    all_numeric_cols = final_model_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Method 2: Also check the original feature_list (if available)
    original_features = st.session_state.get('feature_list', [])
    
    # Combine both approaches
    all_potential_features = list(set(all_numeric_cols + original_features))
    
    # Remove target column
    if target_col in all_potential_features:
        all_potential_features.remove(target_col)

    # Find which features were actually used in successful factor analysis
    used_in_factors = set()
    fa_results = st.session_state.get('fa_results', {})
    
    if fa_results:
        for category_name, results in fa_results.items():
            if results and isinstance(results, dict) and results.get('success', False):
                used_in_factors.update(results.get('features', []))

    # Raw features = all potential features that weren't successfully factored
    raw_features = []
    for feature in all_potential_features:
        if (feature in final_model_df.columns and  # Must exist in dataframe
            feature not in used_in_factors):        # Not used in factor analysis
            raw_features.append(feature)

    # Debug information
    st.write(f"**🔍 Debug Information:**")
    st.write(f"- Total numeric columns in data: {len(all_numeric_cols)}")
    st.write(f"- Original feature_list length: {len(original_features)}")
    st.write(f"- Features used in factors: {len(used_in_factors)}")
    st.write(f"- Raw features found: {len(raw_features)}")
    
    if raw_features:
        st.write(f"- Sample raw features: {raw_features[:5]}")

    # Store everything in session state
    st.session_state.X_factors = X_factors
    st.session_state.y_target = y_target
    st.session_state.feature_names = list(factor_scores_df.columns)
    st.session_state.raw_features = raw_features
    st.session_state.final_model_df = final_model_df


# ─────────────────────────────────────────────────────────────────────────────
def display_data_summary() -> None:
    """Show basic dataset info including raw features count."""
    st.subheader("📊 Dataset Summary")

    X_factors = st.session_state.X_factors
    y_target = st.session_state.y_target
    feature_names = st.session_state.feature_names
    raw_features = st.session_state.get('raw_features', [])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Factored Variables", len(feature_names))
    with col2:
        st.metric("Raw Variables", len(raw_features))
    with col3:
        st.metric("Sample Size", len(X_factors))
    with col4:
        st.metric("Target Variable", st.session_state.selected_target_name)

    # Feature overview
    st.subheader("🔍 Available Variables Overview")
    
    tab1, tab2 = st.tabs(["🔬 Factored Variables", "📊 Raw Variables"])
    
    with tab1:
        if feature_names:
            st.write(f"**{len(feature_names)} factor variables available:**")
            with st.expander("View all factor variables"):
                for i, fac in enumerate(feature_names, 1):
                    st.write(f"{i}. {fac}")
        else:
            st.info("No factored variables available.")
    
    with tab2:
        if raw_features:
            st.write(f"**{len(raw_features)} raw variables available:**")
            with st.expander("View all raw variables"):
                for i, raw in enumerate(raw_features, 1):
                    st.write(f"{i}. {raw}")
        else:
            st.info("No raw variables available (all features were factored).")


# ─────────────────────────────────────────────────────────────────────────────
def enhanced_variable_selection_interface() -> None:
    """Enhanced interface for selecting both factored and raw variables."""
    feature_names = st.session_state.feature_names
    raw_features = st.session_state.get('raw_features', [])
    
    # Initialize selections
    if "selected_factored_features" not in st.session_state:
        st.session_state.selected_factored_features = feature_names.copy()
    if "selected_raw_features" not in st.session_state:
        st.session_state.selected_raw_features = []

    st.write("Select variables to include in the logistic regression model:")

    # Create tabs for factored and raw variables
    tab1, tab2, tab3 = st.tabs(["🔬 Factored Variables", "📊 Raw Variables", "📋 Selection Summary"])
    
    with tab1:
        st.write("**Select factored variables from factor analysis:**")
        
        if feature_names:
            # Bulk buttons for factored variables
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Select All Factored", key="select_all_factored"):
                    st.session_state.selected_factored_features = feature_names.copy()
                    st.rerun()
            with col2:
                if st.button("❌ Deselect All Factored", key="deselect_all_factored"):
                    st.session_state.selected_factored_features = []
                    st.rerun()
            with col3:
                if "vif_results" in st.session_state:
                    if st.button("🧹 Remove High VIF Factored", key="remove_vif_factored"):
                        high_vif_vars = (
                            st.session_state.vif_results[st.session_state.vif_results["VIF"] > 10]["Variable"]
                            .tolist()
                        )
                        high_vif_vars = [v for v in high_vif_vars if v != "const"]
                        st.session_state.selected_factored_features = [
                            v for v in feature_names if v not in high_vif_vars
                        ]
                        st.rerun()
            
            # Category-based checkboxes for factored variables
            factored_categories: dict[str, list[str]] = {}
            for feat in feature_names:
                if "_Factor_" in feat:
                    cat = feat.split("_Factor_")[0]
                else:
                    cat = "Other"
                factored_categories.setdefault(cat, []).append(feat)

            selected_factored = []
            for cat, vars_in_cat in factored_categories.items():
                st.write(f"**{cat}:**")
                for var in vars_in_cat:
                    checked = st.checkbox(
                        var,
                        value=var in st.session_state.selected_factored_features,
                        key=f"factored_{var}",
                    )
                    if checked:
                        selected_factored.append(var)

            st.session_state.selected_factored_features = selected_factored
        else:
            st.info("No factored variables available.")
    
    with tab2:
        st.write("**Select raw variables (not included in factor analysis):**")
        
        if raw_features:
            # Bulk buttons for raw variables
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Select All Raw", key="select_all_raw"):
                    st.session_state.selected_raw_features = raw_features.copy()
                    st.rerun()
            with col2:
                if st.button("❌ Deselect All Raw", key="deselect_all_raw"):
                    st.session_state.selected_raw_features = []
                    st.rerun()
            
            # Categorize raw features for better organization
            raw_categories = {
                'Rep Attributes': [f for f in raw_features if "Rep Attributes" in f],
                'Product Perceptions': [f for f in raw_features if "Perceptions" in f],
                'Message Delivery': [f for f in raw_features if "Delivery of topic" in f],
                'Topics/Messages': [f for f in raw_features if any(keyword in f.lower() for keyword in ["topic", "message"])],
                'Miscellaneous': [f for f in raw_features if not any(cat in f for cat in 
                                 ["Rep Attributes", "Perceptions", "Delivery of topic"]) and 
                                 not any(keyword in f.lower() for keyword in ["topic", "message"])]
            }
            
            # Remove empty categories
            raw_categories = {k: v for k, v in raw_categories.items() if v}
            
            selected_raw = []
            for cat, vars_in_cat in raw_categories.items():
                if vars_in_cat:
                    st.write(f"**{cat} ({len(vars_in_cat)} variables):**")
                    for var in vars_in_cat:
                        checked = st.checkbox(
                            var,
                            value=var in st.session_state.selected_raw_features,
                            key=f"raw_{var}",
                        )
                        if checked:
                            selected_raw.append(var)
            
            st.session_state.selected_raw_features = selected_raw
        else:
            st.info("No raw variables available (all original features were factored).")
    
    with tab3:
        display_selection_summary()


# ─────────────────────────────────────────────────────────────────────────────
def display_selection_summary() -> None:
    """Display summary of selected variables."""
    selected_factored = st.session_state.get('selected_factored_features', [])
    selected_raw = st.session_state.get('selected_raw_features', [])
    
    st.write("**📊 Variable Selection Summary**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Factored Variables", len(selected_factored))
    with col2:
        st.metric("Raw Variables", len(selected_raw))
    with col3:
        total_selected = len(selected_factored) + len(selected_raw)
        st.metric("Total Selected", total_selected)
    
    # Show selected variables
    if selected_factored:
        st.write("**✅ Selected Factored Variables:**")
        for i, var in enumerate(selected_factored, 1):
            st.write(f"{i}. {var}")
    
    if selected_raw:
        st.write("**✅ Selected Raw Variables:**")
        for i, var in enumerate(selected_raw, 1):
            st.write(f"{i}. {var}")
    
    if not selected_factored and not selected_raw:
        st.warning("⚠️ No variables selected for modeling!")


# ─────────────────────────────────────────────────────────────────────────────
def calculate_vif_analysis() -> None:
    """Compute VIF for all selected variables (factored + raw)."""
    selected_factored = st.session_state.get('selected_factored_features', [])
    selected_raw = st.session_state.get('selected_raw_features', [])
    
    if not selected_factored and not selected_raw:
        st.error("⚠️ Please select at least one variable for VIF analysis.")
        return
    
    try:
        # Combine data from different sources
        combined_data = pd.DataFrame()
        
        # Add factored variables
        if selected_factored:
            X_factors = st.session_state.X_factors
            factored_subset = X_factors[selected_factored].reset_index(drop=True)
            combined_data = pd.concat([combined_data, factored_subset], axis=1)
        
        # Add raw variables
        if selected_raw:
            final_model_df = st.session_state.final_model_df
            raw_subset = final_model_df[selected_raw].reset_index(drop=True)
            # Handle missing values in raw data
            raw_subset = raw_subset.fillna(raw_subset.median())
            combined_data = pd.concat([combined_data, raw_subset], axis=1)
        
        if combined_data.empty:
            st.error("❌ No data available for VIF calculation.")
            return
        
        # Add constant and calculate VIF
        X_const = sm.add_constant(combined_data)
        
        vif_df = pd.DataFrame(
            {
                "Variable": X_const.columns,
                "VIF": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
            }
        ).sort_values("VIF", ascending=False)

        st.write("📊 **VIF Results:**")
        st.dataframe(vif_df, use_container_width=True)

        # VIF interpretation
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
            st.warning("⚠️ Consider removing variables with VIF > 10")
            st.write("**High VIF Variables:**")
            for _, row in high_vif.iterrows():
                if row["Variable"] != "const":
                    st.write(f"• {row['Variable']}: {row['VIF']:.2f}")
        else:
            st.success("✅ No high multicollinearity detected")

        st.session_state.vif_results = vif_df
        
    except Exception as e:
        st.error(f"❌ Error calculating VIF: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate_model() -> None:
    """Train logistic regression with selected factored and raw variables."""
    selected_factored = st.session_state.get('selected_factored_features', [])
    selected_raw = st.session_state.get('selected_raw_features', [])
    
    if not selected_factored and not selected_raw:
        st.error("⚠️ Please select at least one variable.")
        return

    try:
        # Combine data from different sources
        X_combined = pd.DataFrame()
        
        # Add factored variables
        if selected_factored:
            X_factors = st.session_state.X_factors
            factored_data = X_factors[selected_factored].reset_index(drop=True)
            X_combined = pd.concat([X_combined, factored_data], axis=1)
        
        # Add raw variables
        if selected_raw:
            final_model_df = st.session_state.final_model_df
            raw_data = final_model_df[selected_raw].reset_index(drop=True)
            # Handle missing values in raw data
            raw_data = raw_data.fillna(raw_data.median())
            X_combined = pd.concat([X_combined, raw_data], axis=1)
        
        if X_combined.empty:
            st.error("⚠️ No data available for modeling.")
            return

        y = st.session_state.y_target
        
        # Ensure same length
        min_length = min(len(X_combined), len(y))
        X_combined = X_combined.iloc[:min_length]
        y = y.iloc[:min_length]

        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.3, random_state=42, stratify=y
        )

        with st.spinner("Training logistic regression model…"):
            model = LogisticRegression(max_iter=1_000, random_state=42)
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Store results for potential use in next steps
        st.session_state.regression_model = model
        st.session_state.model_results = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred': y_pred, 'y_pred_proba': y_proba,
            'selected_features': list(X_combined.columns),
            'selected_factored_features': selected_factored,
            'selected_raw_features': selected_raw,
            'regression_model': model
        }

        display_model_results(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            y_pred,
            y_proba,
            list(X_combined.columns),
            selected_factored,
            selected_raw,
        )
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
def display_model_results(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    y_proba,
    all_features,
    selected_factored,
    selected_raw,
) -> None:
    """Visualise coefficients and performance metrics with variable type distinction."""
    st.subheader("🎯 Model Performance Results")

    # Enhanced coefficients with variable type
    st.subheader("🔑 Key Driver Analysis – Variable Importance")
    coef_df = pd.DataFrame(
        {
            "Variable": all_features,
            "Coefficient": model.coef_[0],
            "Abs_Coefficient": np.abs(model.coef_[0]),
            "Variable_Type": ["Factored" if var in selected_factored else "Raw" for var in all_features]
        }
    ).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)

    # Enhanced bar chart with variable type coloring
    fig = px.bar(
        coef_df,
        y="Variable",
        x="Coefficient",
        orientation="h",
        color="Variable_Type",
        title="Variable Importance (Logistic Regression Coefficients)",
        color_discrete_map={"Factored": "#2E86AB", "Raw": "#F24236"}
    )
    fig.update_layout(height=max(400, len(all_features) * 25))
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Detailed Coefficients by Variable Type:**")
    st.dataframe(coef_df[["Variable", "Coefficient", "Variable_Type"]].round(4), use_container_width=True)

    # Performance metrics
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

    # Top drivers by variable type
    st.subheader("🏆 Top Key Drivers by Variable Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔬 Top Factored Variables:**")
        factored_drivers = coef_df[coef_df['Variable_Type'] == 'Factored'].head(5)
        if not factored_drivers.empty:
            for i, (_, row) in enumerate(factored_drivers.iterrows(), 1):
                impact = "↗️ Positive" if row['Coefficient'] > 0 else "↘️ Negative"
                st.write(f"{i}. **{row['Variable']}**: {row['Coefficient']:.4f} {impact}")
        else:
            st.write("No factored variables selected")
    
    with col2:
        st.write("**📊 Top Raw Variables:**")
        raw_drivers = coef_df[coef_df['Variable_Type'] == 'Raw'].head(5)
        if not raw_drivers.empty:
            for i, (_, row) in enumerate(raw_drivers.iterrows(), 1):
                impact = "↗️ Positive" if row['Coefficient'] > 0 else "↘️ Negative"
                st.write(f"{i}. **{row['Variable']}**: {row['Coefficient']:.4f} {impact}")
        else:
            st.write("No raw variables selected")

    # Enhanced summary
    st.subheader("📈 Enhanced Model Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Performance:**")
        st.write(f"• **Accuracy**: {acc:.1%}")
        st.write(f"• **AUC-ROC**: {auc:.3f}")
        st.write(f"• **Precision**: {prec:.1%}")
        st.write(f"• **Recall**: {rec:.1%}")
    
    with col2:
        st.write("**Variable Composition:**")
        st.write(f"• **Total Variables**: {len(all_features)}")
        st.write(f"• **Factored Variables**: {len(selected_factored)}")
        st.write(f"• **Raw Variables**: {len(selected_raw)}")
        st.write(f"• **Training Samples**: {len(X_train):,}")

    if auc > 0.8:
        st.success("🎉 Excellent model performance (AUC > 0.8)")
    elif auc > 0.7:
        st.info("👍 Good model performance (AUC > 0.7)")
    else:
        st.warning("⚠️ Model performance could be improved (AUC ≤ 0.7)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    show_page()
