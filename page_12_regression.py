import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def show_page():
    st.header("📈 Step 12: Logistic Regression Analysis")
    
    # Check prerequisites
    if 'selected_target_col' not in st.session_state:
        st.error("⚠️ No target variable selected. Please complete previous steps.")
        return
    
    # Simple data preparation
    prepare_data_simple()
    
    # Show what we found
    show_available_variables()
    
    # Variable selection form
    variable_selection_form()
    
    # VIF and modeling buttons
    if 'final_selected_vars' in st.session_state and st.session_state.final_selected_vars:
        st.subheader("🔍 Analysis Options")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Calculate VIF", type="secondary"):
                calculate_vif_simple()
        
        with col2:
            if st.button("Train Model", type="primary"):
                train_model_simple()

def prepare_data_simple():
    """Very simple data preparation"""
    
    # Get basic data
    final_model_df = st.session_state.get("final_model_df")
    if final_model_df is None:
        st.error("⚠️ No model data available")
        return
    
    # Get factor scores
    factor_scores_df = st.session_state.get("factor_scores_df", pd.DataFrame())
    
    # Get target
    selected_target_col = st.session_state.get("selected_target_col")
    y_target = final_model_df[selected_target_col].reset_index(drop=True)
    
    # Find ALL numeric columns that aren't the target
    all_numeric_cols = final_model_df.select_dtypes(include=[np.number]).columns.tolist()
    if selected_target_col in all_numeric_cols:
        all_numeric_cols.remove(selected_target_col)
    
    # Find which columns were used for factors
    factored_columns = set()
    fa_results = st.session_state.get("fa_results", {})
    if fa_results:
        for category_name, results in fa_results.items():
            if results and isinstance(results, dict) and results.get("success", False):
                factored_columns.update(results.get("features", []))
    
    # Raw features = all numeric columns that weren't factored
    raw_features = [col for col in all_numeric_cols if col not in factored_columns]
    
    # Store everything simply
    st.session_state.available_factors = list(factor_scores_df.columns) if not factor_scores_df.empty else []
    st.session_state.available_raw = raw_features
    st.session_state.y_target_simple = y_target
    st.session_state.factor_data = factor_scores_df
    st.session_state.raw_data = final_model_df

def show_available_variables():
    """Show what variables we found"""
    
    available_factors = st.session_state.get('available_factors', [])
    available_raw = st.session_state.get('available_raw', [])
    
    st.subheader("📊 Available Variables")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Factored Variables", len(available_factors))
    with col2:
        st.metric("Raw Variables", len(available_raw))
    with col3:
        st.metric("Total Available", len(available_factors) + len(available_raw))
    
    # Show lists
    if available_factors:
        with st.expander(f"📋 Factored Variables ({len(available_factors)})", expanded=False):
            for i, var in enumerate(available_factors, 1):
                st.write(f"{i}. {var}")
    
    if available_raw:
        with st.expander(f"📋 Raw Variables ({len(available_raw)})", expanded=False):
            for i, var in enumerate(available_raw, 1):
                st.write(f"{i}. {var}")

def variable_selection_form():
    """Simple form for variable selection"""
    
    available_factors = st.session_state.get('available_factors', [])
    available_raw = st.session_state.get('available_raw', [])
    
    st.subheader("🎛️ Variable Selection")
    
    with st.form("variable_selection_form"):
        st.write("**Select variables for your model:**")
        
        # Factor variables
        selected_factors = []
        if available_factors:
            st.write("**Factored Variables:**")
            selected_factors = st.multiselect(
                "Choose factored variables:",
                options=available_factors,
                default=available_factors  # Select all by default
            )
        
        # Raw variables
        selected_raw = []
        if available_raw:
            st.write("**Raw Variables:**")
            selected_raw = st.multiselect(
                "Choose raw variables:",
                options=available_raw,
                default=[]  # Select none by default
            )
        
        # Submit button
        submitted = st.form_submit_button("✅ Confirm Selection")
        
        if submitted:
            total_selected = len(selected_factors) + len(selected_raw)
            
            if total_selected == 0:
                st.error("⚠️ Please select at least one variable!")
            else:
                # Store selections
                st.session_state.selected_factors_final = selected_factors
                st.session_state.selected_raw_final = selected_raw
                st.session_state.final_selected_vars = selected_factors + selected_raw
                
                st.success(f"✅ Selected {total_selected} variables!")
                st.write("**Your Selection:**")
                if selected_factors:
                    st.write(f"• Factored: {len(selected_factors)} variables")
                if selected_raw:
                    st.write(f"• Raw: {len(selected_raw)} variables")
                
                st.rerun()

def calculate_vif_simple():
    """Simple VIF calculation"""
    
    st.subheader("🔍 VIF Analysis Results")
    
    try:
        # Get selections
        selected_factors = st.session_state.get('selected_factors_final', [])
        selected_raw = st.session_state.get('selected_raw_final', [])
        
        # Combine data
        combined_data = pd.DataFrame()
        
        # Add factors
        if selected_factors:
            factor_data = st.session_state.get('factor_data', pd.DataFrame())
            if not factor_data.empty:
                factor_subset = factor_data[selected_factors].reset_index(drop=True)
                combined_data = pd.concat([combined_data, factor_subset], axis=1)
        
        # Add raw variables
        if selected_raw:
            raw_data = st.session_state.get('raw_data')
            if raw_data is not None:
                raw_subset = raw_data[selected_raw].reset_index(drop=True)
                raw_subset = raw_subset.fillna(raw_subset.median())
                combined_data = pd.concat([combined_data, raw_subset], axis=1)
        
        if combined_data.empty:
            st.error("No data to analyze")
            return
        
        # Calculate VIF
        X_with_const = sm.add_constant(combined_data)
        
        vif_results = []
        for i in range(X_with_const.shape[1]):
            try:
                vif_val = variance_inflation_factor(X_with_const.values, i)
                if np.isnan(vif_val) or np.isinf(vif_val):
                    vif_val = 999.9
                vif_results.append({
                    'Variable': X_with_const.columns[i],
                    'VIF': vif_val
                })
            except:
                vif_results.append({
                    'Variable': X_with_const.columns[i],
                    'VIF': 999.9
                })
        
        vif_df = pd.DataFrame(vif_results).sort_values('VIF', ascending=False)
        
        st.success("✅ VIF calculation completed!")
        st.dataframe(vif_df, use_container_width=True)
        
        # Simple interpretation
        high_vif = vif_df[vif_df['VIF'] > 10]
        if len(high_vif) > 0:
            st.warning(f"⚠️ {len(high_vif)} variables have high multicollinearity (VIF > 10)")
        else:
            st.success("✅ No high multicollinearity detected")
        
    except Exception as e:
        st.error(f"VIF calculation error: {str(e)}")

def train_model_simple():
    """Simple model training"""
    
    st.subheader("🚀 Model Training Results")
    
    try:
        # Get selections
        selected_factors = st.session_state.get('selected_factors_final', [])
        selected_raw = st.session_state.get('selected_raw_final', [])
        
        # Combine data
        X_combined = pd.DataFrame()
        
        # Add factors
        if selected_factors:
            factor_data = st.session_state.get('factor_data', pd.DataFrame())
            if not factor_data.empty:
                factor_subset = factor_data[selected_factors].reset_index(drop=True)
                X_combined = pd.concat([X_combined, factor_subset], axis=1)
        
        # Add raw variables
        if selected_raw:
            raw_data = st.session_state.get('raw_data')
            if raw_data is not None:
                raw_subset = raw_data[selected_raw].reset_index(drop=True)
                raw_subset = raw_subset.fillna(raw_subset.median())
                X_combined = pd.concat([X_combined, raw_subset], axis=1)
        
        if X_combined.empty:
            st.error("No data for modeling")
            return
        
        # Get target
        y_target = st.session_state.get('y_target_simple')
        if y_target is None:
            st.error("No target variable")
            return
        
        # Make sure same length
        min_len = min(len(X_combined), len(y_target))
        X_combined = X_combined.iloc[:min_len]
        y_target = y_target.iloc[:min_len]
        
        st.write(f"📊 Training with {X_combined.shape[0]} samples and {X_combined.shape[1]} features")
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_target, test_size=0.3, random_state=42, stratify=y_target
        )
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Show results
        show_simple_results(model, X_test, y_test, y_pred, y_pred_proba, 
                           list(X_combined.columns), selected_factors, selected_raw)
        
        # Mark complete
        if 'step_completed' not in st.session_state:
            st.session_state.step_completed = {}
        st.session_state.step_completed[12] = True
        
        st.success("🎉 Model training completed successfully!")
        
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

def show_simple_results(model, X_test, y_test, y_pred, y_pred_proba, 
                       all_features, selected_factors, selected_raw):
    """Show simple model results"""
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    st.subheader("📊 Model Performance")
    
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
    
    # Variable importance
    st.subheader("🔑 Variable Importance")
    
    importance_df = pd.DataFrame({
        'Variable': all_features,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0]),
        'Type': ['Factored' if var in selected_factors else 'Raw' for var in all_features]
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Bar chart
    fig = px.bar(
        importance_df,
        y='Variable',
        x='Coefficient',
        orientation='h',
        color='Type',
        title='Variable Importance (Coefficients)',
        color_discrete_map={'Factored': 'blue', 'Raw': 'red'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.dataframe(importance_df[['Variable', 'Coefficient', 'Type']], use_container_width=True)
    
    # Summary
    st.subheader("📋 Summary")
    st.write(f"**Model trained with {len(all_features)} variables:**")
    st.write(f"• {len(selected_factors)} factored variables")
    st.write(f"• {len(selected_raw)} raw variables")
    st.write(f"• AUC-ROC: {auc_score:.3f}")
    
    if auc_score > 0.8:
        st.success("🎉 Excellent performance!")
    elif auc_score > 0.7:
        st.info("👍 Good performance!")
    else:
        st.warning("⚠️ Room for improvement")

if __name__ == "__main__":
    show_page()
