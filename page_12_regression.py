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
    if 'selected_target_col' not in st.session_state:
        st.error("⚠️ No target variable selected. Please complete previous steps.")
        return
    
    # Prepare data sources
    prepare_regression_data()
    
    # Show data preparation results
    display_data_summary()
    
    # Enhanced Variable Selection
    st.subheader("🎛️ Enhanced Variable Selection")
    enhanced_variable_selection_interface()
    
    # VIF Analysis
    st.subheader("🔍 Multicollinearity Check (VIF Analysis)")
    if st.button("Calculate VIF", type="secondary"):
        calculate_vif_analysis()
    
    # Model training
    st.subheader("🚀 Model Training & Evaluation")
    if st.button("Train Logistic Regression Model", type="primary"):
        train_and_evaluate_model()

def prepare_regression_data():
    """Prepare regression data with improved raw variable detection"""
    
    final_model_df = st.session_state.get("final_model_df")
    selected_target_col = st.session_state.get("selected_target_col")
    
    if final_model_df is None or selected_target_col is None:
        st.error("⚠️ Required data missing – please complete previous steps.")
        return
    
    # Get factor scores (if available)
    factor_scores_df = st.session_state.get("factor_scores_df", pd.DataFrame())
    
    # Get all original features from Step 5
    feature_list = st.session_state.get("feature_list", [])
    
    # Find features that were actually used in successful factor analysis
    used_in_factors = set()
    fa_results = st.session_state.get("fa_results", {})
    
    if fa_results:
        for category_name, results in fa_results.items():
            if results and isinstance(results, dict) and results.get("success", False):
                used_in_factors.update(results.get("features", []))
    
    # Find raw features: all numeric features that weren't successfully factored
    available_columns = set(final_model_df.columns.tolist())
    raw_features_not_factored = []
    
    for feature in feature_list:
        if (feature in available_columns and 
            feature != selected_target_col and 
            feature not in used_in_factors):
            raw_features_not_factored.append(feature)
    
    # Also check selected_features for any that didn't get factored
    selected_features = st.session_state.get("selected_features", [])
    for feature in selected_features:
        if (feature in available_columns and 
            feature != selected_target_col and 
            feature not in used_in_factors and 
            feature not in raw_features_not_factored):
            raw_features_not_factored.append(feature)
    
    # Target variable
    y_target = final_model_df[selected_target_col].reset_index(drop=True)
    
    # Store results
    st.session_state.factor_scores_df = factor_scores_df
    st.session_state.raw_features_not_factored = raw_features_not_factored
    st.session_state.y_target = y_target
    st.session_state.final_model_df = final_model_df
    
    # Debug info
    st.write(f"**🔍 Debug:** Found {len(raw_features_not_factored)} raw features available")

def display_data_summary():
    """Display data summary"""
    
    st.subheader("📊 Dataset Summary")
    
    factor_scores_df = st.session_state.get('factor_scores_df', pd.DataFrame())
    raw_features_not_factored = st.session_state.get('raw_features_not_factored', [])
    y_target = st.session_state.get('y_target')
    
    if y_target is None:
        st.error("⚠️ Target variable not available.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Factored Variables", len(factor_scores_df.columns) if not factor_scores_df.empty else 0)
    with col2:
        st.metric("Raw Variables Available", len(raw_features_not_factored))
    with col3:
        st.metric("Sample Size", len(y_target))
    with col4:
        st.metric("Target Variable", st.session_state.get('selected_target_name', 'Unknown'))
    
    # Target distribution
    target_counts = y_target.value_counts()
    total_count = len(y_target)
    
    st.subheader("🎯 Target Variable Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
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

def enhanced_variable_selection_interface():
    """Simplified variable selection interface"""
    
    factor_scores_df = st.session_state.get('factor_scores_df', pd.DataFrame())
    raw_features_not_factored = st.session_state.get('raw_features_not_factored', [])
    
    # Initialize selection lists in session state
    if 'selected_factored_vars' not in st.session_state:
        st.session_state.selected_factored_vars = list(factor_scores_df.columns) if not factor_scores_df.empty else []
    
    if 'selected_raw_vars' not in st.session_state:
        st.session_state.selected_raw_vars = []
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Factored Variables", "📊 Raw Variables", "📋 Selection Summary"])
    
    with tab1:
        st.write("**Select factored variables from factor analysis:**")
        
        if not factor_scores_df.empty:
            # Bulk operations
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Select All Factored"):
                    st.session_state.selected_factored_vars = list(factor_scores_df.columns)
                    st.rerun()
            with col2:
                if st.button("❌ Deselect All Factored"):
                    st.session_state.selected_factored_vars = []
                    st.rerun()
            
            # Individual selection using multiselect
            st.session_state.selected_factored_vars = st.multiselect(
                "Choose factored variables:",
                options=list(factor_scores_df.columns),
                default=st.session_state.selected_factored_vars,
                key="factored_multiselect"
            )
        else:
            st.info("ℹ️ No factored variables available.")
    
    with tab2:
        st.write("**Select raw variables that were not included in factor analysis:**")
        
        if raw_features_not_factored:
            st.write(f"**Found {len(raw_features_not_factored)} raw variables**")
            
            # Categorize raw features for better organization
            raw_categories = {
                'Rep Attributes': [f for f in raw_features_not_factored if "Rep Attributes" in f],
                'Product Perceptions': [f for f in raw_features_not_factored if "Perceptions" in f],
                'Message Delivery': [f for f in raw_features_not_factored if "Delivery of topic" in f],
                'Miscellaneous': [f for f in raw_features_not_factored if not any(cat in f for cat in 
                                 ["Rep Attributes", "Perceptions", "Delivery of topic"])]
            }
            
            # Remove empty categories
            raw_categories = {k: v for k, v in raw_categories.items() if v}
            
            # Bulk operations
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Select All Raw"):
                    st.session_state.selected_raw_vars = raw_features_not_factored.copy()
                    st.rerun()
            with col2:
                if st.button("❌ Deselect All Raw"):
                    st.session_state.selected_raw_vars = []
                    st.rerun()
            
            # Category-wise selection
            for category, features in raw_categories.items():
                if features:
                    st.write(f"**{category} ({len(features)} variables):**")
                    with st.expander(f"Select from {category}", expanded=True):
                        # Use multiselect for each category
                        current_selection = [f for f in st.session_state.selected_raw_vars if f in features]
                        new_selection = st.multiselect(
                            f"Choose {category} variables:",
                            options=features,
                            default=current_selection,
                            key=f"raw_multiselect_{category}"
                        )
                        
                        # Update session state
                        # Remove old selections from this category
                        st.session_state.selected_raw_vars = [f for f in st.session_state.selected_raw_vars if f not in features]
                        # Add new selections
                        st.session_state.selected_raw_vars.extend(new_selection)
        else:
            st.info("ℹ️ All original features were included in factor analysis.")
    
    with tab3:
        display_selection_summary()

def display_selection_summary():
    """Display selection summary"""
    
    selected_factored = st.session_state.get('selected_factored_vars', [])
    selected_raw = st.session_state.get('selected_raw_vars', [])
    
    st.write("**📊 Variable Selection Summary**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Factored Variables", len(selected_factored))
    with col2:
        st.metric("Raw Variables", len(selected_raw))
    with col3:
        st.metric("Total Selected", len(selected_factored) + len(selected_raw))
    
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

def calculate_vif_analysis():
    """Calculate VIF analysis"""
    
    st.write("🔄 Starting VIF analysis...")
    
    selected_factored = st.session_state.get('selected_factored_vars', [])
    selected_raw = st.session_state.get('selected_raw_vars', [])
    
    if not selected_factored and not selected_raw:
        st.error("⚠️ Please select at least one variable for VIF analysis.")
        return
    
    try:
        combined_data = pd.DataFrame()
        
        # Add factored variables
        if selected_factored:
            factor_scores_df = st.session_state.get('factor_scores_df', pd.DataFrame())
            if not factor_scores_df.empty:
                valid_factored = [col for col in selected_factored if col in factor_scores_df.columns]
                if valid_factored:
                    factored_subset = factor_scores_df[valid_factored].reset_index(drop=True)
                    combined_data = pd.concat([combined_data, factored_subset], axis=1)
                    st.write(f"✅ Added {len(valid_factored)} factored variables")
        
        # Add raw variables
        if selected_raw:
            final_model_df = st.session_state.get('final_model_df')
            if final_model_df is not None:
                valid_raw = [col for col in selected_raw if col in final_model_df.columns]
                if valid_raw:
                    raw_subset = final_model_df[valid_raw].reset_index(drop=True)
                    raw_subset = raw_subset.select_dtypes(include=[np.number])
                    raw_subset = raw_subset.fillna(raw_subset.median())
                    combined_data = pd.concat([combined_data, raw_subset], axis=1)
                    st.write(f"✅ Added {len(valid_raw)} raw variables")
        
        if combined_data.empty:
            st.error("❌ No valid data for VIF calculation.")
            return
        
        # Handle missing values and constants
        combined_data = combined_data.fillna(combined_data.median())
        
        # Remove constant columns
        constant_cols = [col for col in combined_data.columns if combined_data[col].nunique() <= 1]
        if constant_cols:
            st.warning(f"⚠️ Removing constant columns: {constant_cols}")
            combined_data = combined_data.drop(columns=constant_cols)
        
        if combined_data.shape[1] < 2:
            st.error("❌ Need at least 2 variables for VIF calculation.")
            return
        
        # Calculate VIF
        X_with_const = sm.add_constant(combined_data)
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_const.columns
        
        vif_values = []
        for i in range(X_with_const.shape[1]):
            try:
                vif_val = variance_inflation_factor(X_with_const.values, i)
                if np.isnan(vif_val) or np.isinf(vif_val) or vif_val > 1000:
                    vif_val = 999.9
                vif_values.append(vif_val)
            except:
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        vif_data_clean = vif_data.dropna().sort_values('VIF', ascending=False)
        
        st.success("✅ VIF calculation completed!")
        st.dataframe(vif_data_clean, use_container_width=True)
        
        # Interpretation
        valid_vif = vif_data_clean[vif_data_clean['Variable'] != 'const']
        high_vif = valid_vif[valid_vif['VIF'] > 10]
        
        if len(high_vif) > 0:
            st.warning("⚠️ Variables with high multicollinearity (VIF > 10):")
            for _, row in high_vif.iterrows():
                st.write(f"• {row['Variable']}: {row['VIF']:.2f}")
        else:
            st.success("✅ No high multicollinearity detected")
        
        st.session_state.vif_results = vif_data_clean
        
    except Exception as e:
        st.error(f"❌ Error in VIF calculation: {str(e)}")

def train_and_evaluate_model():
    """Train and evaluate logistic regression model"""
    
    selected_factored = st.session_state.get('selected_factored_vars', [])
    selected_raw = st.session_state.get('selected_raw_vars', [])
    
    if not selected_factored and not selected_raw:
        st.error("⚠️ Please select at least one variable for modeling.")
        return
    
    try:
        # Combine data
        X_combined = pd.DataFrame()
        
        # Add factored variables
        if selected_factored:
            factor_scores_df = st.session_state.get('factor_scores_df', pd.DataFrame())
            if not factor_scores_df.empty:
                factored_data = factor_scores_df[selected_factored].reset_index(drop=True)
                X_combined = pd.concat([X_combined, factored_data], axis=1)
        
        # Add raw variables
        if selected_raw:
            final_model_df = st.session_state.get('final_model_df')
            if final_model_df is not None:
                raw_data = final_model_df[selected_raw].reset_index(drop=True)
                raw_data = raw_data.fillna(raw_data.median())
                X_combined = pd.concat([X_combined, raw_data], axis=1)
        
        if X_combined.empty:
            st.error("⚠️ No data available for modeling.")
            return
        
        # Get target
        y_target = st.session_state.get('y_target')
        if y_target is None:
            st.error("⚠️ Target variable not available.")
            return
        
        # Ensure same length
        min_length = min(len(X_combined), len(y_target))
        X_combined = X_combined.iloc[:min_length]
        y_target = y_target.iloc[:min_length]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_target, test_size=0.3, random_state=42, stratify=y_target
        )
        
        # Train model
        with st.spinner("Training logistic regression model..."):
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Store results
        st.session_state.regression_model = model
        st.session_state.model_results = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred': y_pred, 'y_pred_proba': y_pred_proba,
            'selected_features': list(X_combined.columns),
            'selected_factored_features': selected_factored,
            'selected_raw_features': selected_raw,
            'regression_model': model
        }
        
        # Mark step completed
        if 'step_completed' not in st.session_state:
            st.session_state.step_completed = {}
        st.session_state.step_completed[12] = True
        
        # Display results
        display_model_results(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, 
                             list(X_combined.columns), selected_factored, selected_raw)
        
    except Exception as e:
        st.error(f"❌ Error in model training: {str(e)}")

def display_model_results(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, 
                         all_features, selected_factored, selected_raw):
    """Display comprehensive model results"""
    
    st.subheader("🎯 Model Performance Results")
    
    # Variable importance
    st.subheader("🔑 Key Driver Analysis")
    
    coefficients = pd.DataFrame({
        'Variable': all_features,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0]),
        'Variable_Type': ['Factored' if var in selected_factored else 'Raw' for var in all_features]
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Bar chart
    fig = px.bar(
        coefficients,
        y='Variable',
        x='Coefficient',
        orientation='h',
        title='Variable Importance (Coefficients)',
        color='Variable_Type',
        color_discrete_map={'Factored': '#2E86AB', 'Raw': '#F24236'}
    )
    fig.update_layout(height=max(400, len(all_features) * 25))
    st.plotly_chart(fig, use_container_width=True)
    
    # Coefficients table
    st.dataframe(coefficients[['Variable', 'Coefficient', 'Variable_Type']].round(4), use_container_width=True)
    
    # Performance metrics
    st.subheader("📊 Model Performance")
    
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
    
    # Confusion Matrix and ROC
    col1, col2 = st.columns(2)
    
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc_score:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary
    st.subheader("📈 Model Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance:**")
        st.write(f"• Accuracy: {accuracy:.1%}")
        st.write(f"• AUC-ROC: {auc_score:.3f}")
        st.write(f"• Precision: {precision:.1%}")
        st.write(f"• Recall: {recall:.1%}")
    
    with col2:
        st.write("**Model Composition:**")
        st.write(f"• Total Variables: {len(all_features)}")
        st.write(f"• Factored Variables: {len(selected_factored)}")
        st.write(f"• Raw Variables: {len(selected_raw)}")
        st.write(f"• Training Samples: {len(X_train):,}")
    
    if auc_score > 0.8:
        st.success("🎉 Excellent model performance (AUC > 0.8)")
    elif auc_score > 0.7:
        st.info("👍 Good model performance (AUC > 0.7)")
    else:
        st.warning("⚠️ Model performance could be improved (AUC ≤ 0.7)")
    
    st.info("📌 Logistic regression completed! Click Next ➡️ to proceed.")

if __name__ == "__main__":
    show_page()
