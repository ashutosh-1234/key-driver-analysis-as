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

def prepare_regression_data():
    """Prepare data for logistic regression"""

    factor_scores_df = st.session_state.factor_scores_df
    final_model_df = st.session_state.final_model_df
    selected_target_col = st.session_state.selected_target_col

    # Combine factor scores with target variable
    X_factors = factor_scores_df.reset_index(drop=True)
    y_target = final_model_df[selected_target_col].reset_index(drop=True)

    # Store in session state
    st.session_state.X_factors = X_factors
    st.session_state.y_target = y_target
    st.session_state.feature_names = list(factor_scores_df.columns)

def display_data_summary():
    """Display data preparation summary"""

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

    # Target distribution
    target_counts = y_target.value_counts()
    total_count = len(y_target)

    st.subheader("🎯 Target Variable Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Create pie chart
        fig = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title="Target Variable Distribution",
            color_discrete_sequence=['lightcoral', 'lightblue']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Distribution Details:**")
        for class_val, count in target_counts.items():
            percentage = (count / total_count) * 100
            st.write(f"• Class {class_val}: {count:,} ({percentage:.1f}%)")

        classification_threshold = target_counts.get(1, 0) / total_count
        st.write(f"• Classification Threshold: {classification_threshold:.3f}")

    # Feature overview
    st.subheader("🔍 Factor Variables Overview")

    with st.expander("View all factor variables"):
        for i, factor in enumerate(feature_names, 1):
            st.write(f"{i}. {factor}")

def calculate_vif_analysis():
    """Calculate and display VIF analysis"""

    X_factors = st.session_state.X_factors

    # Add constant for VIF calculation
    X_with_const = sm.add_constant(X_factors)

    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Factor"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                       for i in range(X_with_const.shape[1])]

    # Sort by VIF value
    vif_data = vif_data.sort_values('VIF', ascending=False)

    st.write("📊 **VIF Results:**")
    st.dataframe(vif_data, use_container_width=True)

    # VIF interpretation
    high_vif = vif_data[vif_data['VIF'] > 10]
    moderate_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)]
    low_vif = vif_data[vif_data['VIF'] <= 5]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("High VIF (>10)", len(high_vif), help="May indicate multicollinearity")

    with col2:
        st.metric("Moderate VIF (5-10)", len(moderate_vif), help="Moderate multicollinearity")

    with col3:
        st.metric("Low VIF (≤5)", len(low_vif), help="Low multicollinearity")

    # Recommendations
    if len(high_vif) > 0:
        st.warning("⚠️ **Recommendation:** Consider removing factors with VIF > 10 to reduce multicollinearity")
        st.write("**High VIF Factors:**")
        for _, row in high_vif.iterrows():
            if row['Factor'] != 'const':
                st.write(f"• {row['Factor']}: {row['VIF']:.2f}")
    else:
        st.success("✅ **Good:** No high multicollinearity detected among factors")

    # Store VIF results
    st.session_state.vif_results = vif_data

def variable_selection_interface():
    """Interactive variable selection interface"""

    feature_names = st.session_state.feature_names

    # Initialize selected features if not exists
    if 'selected_regression_features' not in st.session_state:
        st.session_state.selected_regression_features = feature_names.copy()

    st.write("Select variables to include in the logistic regression model:")

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
        # Remove high VIF variables if VIF analysis was done
        if 'vif_results' in st.session_state:
            if st.button("Remove High VIF (>10)"):
                vif_results = st.session_state.vif_results
                high_vif_vars = vif_results[vif_results['VIF'] > 10]['Factor'].tolist()
                high_vif_vars = [var for var in high_vif_vars if var != 'const']
                st.session_state.selected_regression_features = [
                    var for var in feature_names if var not in high_vif_vars
                ]
                st.rerun()

    # Individual variable selection
    selected_features = []

    # Group variables by category if possible
    categories = {}
    for feature in feature_names:
        if '_Factor_' in feature:
            category = feature.split('_Factor_')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(feature)
        else:
            if 'Other' not in categories:
                categories['Other'] = []
            categories['Other'].append(feature)

    # Display checkboxes by category
    for category, vars_in_category in categories.items():
        st.write(f"**{category}:**")
        for var in vars_in_category:
            selected = st.checkbox(
                var,
                value=var in st.session_state.selected_regression_features,
                key=f"var_{var}"
            )
            if selected:
                selected_features.append(var)

    # Update selected features
    st.session_state.selected_regression_features = selected_features

    st.write(f"**Selected Variables:** {len(selected_features)} out of {len(feature_names)}")

def train_and_evaluate_model():
    """Train logistic regression model and display results"""

    selected_features = st.session_state.selected_regression_features

    if len(selected_features) == 0:
        st.error("⚠️ Please select at least one variable for modeling.")
        return

    X_factors = st.session_state.X_factors[selected_features]
    y_target = st.session_state.y_target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_factors, y_target, test_size=0.3, random_state=42, stratify=y_target
    )

    # Train model
    with st.spinner("Training logistic regression model..."):
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Store results
    st.session_state.regression_model = model
    st.session_state.model_results = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred, 'y_pred_proba': y_pred_proba,
        'selected_features': selected_features
    }

    # Display results
    display_model_results(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, selected_features)

def display_model_results(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, selected_features):
    """Display comprehensive model results"""

    st.subheader("🎯 Model Performance Results")

    # Model coefficients (Key Drivers)
    st.subheader("🔑 Key Driver Analysis - Factor Importance")

    coefficients = pd.DataFrame({
        'Factor': selected_features,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)

    # Create horizontal bar chart for coefficients
    fig = px.bar(
        coefficients,
        y='Factor',
        x='Coefficient',
        orientation='h',
        title='Factor Importance (Logistic Regression Coefficients)',
        color='Coefficient',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0
    )

    fig.update_layout(height=max(400, len(selected_features) * 30))
    st.plotly_chart(fig, use_container_width=True)

    # Coefficients table
    st.write("**Detailed Coefficients:**")
    st.dataframe(coefficients[['Factor', 'Coefficient']].round(4), use_container_width=True)

    # Model performance metrics
    st.subheader("📊 Model Performance Metrics")

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Precision", f"{precision:.3f}")
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    with col4:
        st.metric("F1-Score", f"{f1:.3f}")
    with col5:
        st.metric("AUC-ROC", f"{auc_score:.3f}")

    # Confusion Matrix and ROC Curve
    col1, col2 = st.columns(2)

    with col1:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            color_continuous_scale='Blues'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    # Classification Report
    st.subheader("📋 Detailed Classification Report")

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    # Feature importance summary
    st.subheader("🏆 Top Key Drivers Summary")

    top_drivers = coefficients.head(5)

    st.write("**Top 5 Most Important Factors:**")
    for i, (_, row) in enumerate(top_drivers.iterrows(), 1):
        impact = "Positive" if row['Coefficient'] > 0 else "Negative"
        st.write(f"{i}. **{row['Factor']}**: {row['Coefficient']:.4f} ({impact} impact)")

    # Model summary
    st.subheader("📈 Model Summary")

    st.write(f"**Model Performance:**")
    st.write(f"• **Accuracy**: {accuracy:.1%} - Percentage of correct predictions")
    st.write(f"• **AUC-ROC**: {auc_score:.3f} - Model's ability to distinguish between classes")
    st.write(f"• **Precision**: {precision:.1%} - Of predicted positives, how many were actually positive")
    st.write(f"• **Recall**: {recall:.1%} - Of actual positives, how many were correctly identified")

    st.write(f"\n**Key Insights:**")
    st.write(f"• **{len(selected_features)} factors** used in final model")
    st.write(f"• **Training set**: {len(X_train):,} observations")
    st.write(f"• **Test set**: {len(X_test):,} observations")

    if auc_score > 0.8:
        st.success("🎉 **Excellent model performance** (AUC > 0.8)")
    elif auc_score > 0.7:
        st.info("👍 **Good model performance** (AUC > 0.7)")
    else:
        st.warning("⚠️ **Model performance could be improved** (AUC ≤ 0.7)")

if __name__ == "__main__":
    show_page()
