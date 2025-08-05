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
    
    # Enhanced Variable Selection with Raw + Factored Variables
    st.subheader("🎛️ Enhanced Variable Selection")
    enhanced_variable_selection_interface()
    
    # VIF Analysis
    st.subheader("🔍 Multicollinearity Check (VIF Analysis)")
    if st.button("Calculate VIF", type="secondary"):
        calculate_vif_analysis()
    
    # Model training and evaluation
    st.subheader("🚀 Model Training & Evaluation")
    if st.button("Train Logistic Regression Model", type="primary"):
        train_and_evaluate_model()

def prepare_regression_data():
    """Prepare data for logistic regression with both factored and raw variables"""
    
    final_model_df = st.session_state.get('final_model_df')
    selected_target_col = st.session_state.get('selected_target_col')
    
    if final_model_df is None or selected_target_col is None:
        st.error("⚠️ Required data not available. Please complete previous steps.")
        return
    
    # Get factored variables (if available)
    factor_scores_df = st.session_state.get('factor_scores_df', pd.DataFrame())
    
    # Get all original features that were available for modeling
    original_features = st.session_state.get('feature_list', [])
    selected_features_for_fa = st.session_state.get('selected_features', [])
    
    # Identify raw features that were NOT included in factor analysis
    raw_features_not_factored = [f for f in original_features if f not in selected_features_for_fa]
    
    # ✅ CRITICAL FIX: Validate that raw features exist in final_model_df
    if not final_model_df.empty:
        raw_features_not_factored = [f for f in raw_features_not_factored if f in final_model_df.columns]
    
    # Prepare target variable
    y_target = final_model_df[selected_target_col].reset_index(drop=True)
    
    # Store all available variable sources in session state
    st.session_state.factor_scores_df = factor_scores_df
    st.session_state.raw_features_not_factored = raw_features_not_factored
    st.session_state.y_target = y_target
    st.session_state.final_model_df = final_model_df

def display_data_summary():
    """Display comprehensive data preparation summary"""
    
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
        st.metric("Raw Variables (Not Factored)", len(raw_features_not_factored))
    
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

def enhanced_variable_selection_interface():
    """Enhanced variable selection with both factored and raw variables"""
    
    factor_scores_df = st.session_state.get('factor_scores_df', pd.DataFrame())
    raw_features_not_factored = st.session_state.get('raw_features_not_factored', [])
    final_model_df = st.session_state.get('final_model_df')
    
    # Initialize selections
    if 'selected_factored_features' not in st.session_state:
        st.session_state.selected_factored_features = list(factor_scores_df.columns) if not factor_scores_df.empty else []
    
    if 'selected_raw_features' not in st.session_state:
        st.session_state.selected_raw_features = []
    
    # Create tabs for different variable types
    tab1, tab2, tab3 = st.tabs(["🔬 Factored Variables", "📊 Raw Variables", "📋 Selection Summary"])
    
    with tab1:
        st.write("**Select factored variables from factor analysis:**")
        
        if not factor_scores_df.empty:
            # Group factored variables by category
            factored_categories = {}
            for factor in factor_scores_df.columns:
                if '_Factor_' in factor:
                    category = factor.split('_Factor_')[0]
                    if category not in factored_categories:
                        factored_categories[category] = []
                    factored_categories[category].append(factor)
                else:
                    if 'Other' not in factored_categories:
                        factored_categories['Other'] = []
                    factored_categories['Other'].append(factor)
            
            # Bulk selection for factored variables
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Select All Factored", key="select_all_factored"):
                    st.session_state.selected_factored_features = list(factor_scores_df.columns)
                    st.rerun()
            
            with col2:
                if st.button("❌ Deselect All Factored", key="deselect_all_factored"):
                    st.session_state.selected_factored_features = []
                    st.rerun()
            
            with col3:
                # Remove high VIF if analysis was done
                if 'vif_results' in st.session_state:
                    if st.button("🧹 Remove High VIF Factored", key="remove_vif_factored"):
                        vif_results = st.session_state.vif_results
                        high_vif_vars = vif_results[vif_results['VIF'] > 10]['Variable'].tolist()
                        high_vif_vars = [var for var in high_vif_vars if var != 'const']
                        st.session_state.selected_factored_features = [
                            var for var in st.session_state.selected_factored_features 
                            if var not in high_vif_vars
                        ]
                        st.rerun()
            
            # Individual factored variable selection by category
            selected_factored = []
            for category, factors in factored_categories.items():
                st.write(f"**{category} Factors:**")
                for factor in factors:
                    selected = st.checkbox(
                        factor,
                        value=factor in st.session_state.selected_factored_features,
                        key=f"factored_{factor}"
                    )
                    if selected:
                        selected_factored.append(factor)
            
            st.session_state.selected_factored_features = selected_factored
            
        else:
            st.info("ℹ️ No factored variables available. Factor analysis may not have been completed.")
    
    with tab2:
        st.write("**Select raw variables that were not included in factor analysis:**")
        
        if raw_features_not_factored:
            # Categorize raw features
            raw_categories = {
                'Rep Attributes': [f for f in raw_features_not_factored if "Rep Attributes" in f],
                'Product Perceptions': [f for f in raw_features_not_factored if "Perceptions" in f],
                'Message Delivery': [f for f in raw_features_not_factored if "Delivery of topic" in f],
                'Miscellaneous': [f for f in raw_features_not_factored if not any(cat in f for cat in 
                                 ["Rep Attributes", "Perceptions", "Delivery of topic"])]
            }
            
            # Remove empty categories
            raw_categories = {k: v for k, v in raw_categories.items() if v}
            
            # Bulk selection for raw variables
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Select All Raw", key="select_all_raw"):
                    st.session_state.selected_raw_features = raw_features_not_factored.copy()
                    st.rerun()
            
            with col2:
                if st.button("❌ Deselect All Raw", key="deselect_all_raw"):
                    st.session_state.selected_raw_features = []
                    st.rerun()
            
            # Individual raw variable selection by category
            selected_raw = []
            for category, features in raw_categories.items():
                if features:
                    st.write(f"**{category} ({len(features)} variables):**")
                    with st.expander(f"View {category} variables"):
                        for feature in features:
                            selected = st.checkbox(
                                feature,
                                value=feature in st.session_state.selected_raw_features,
                                key=f"raw_{feature}"
                            )
                            if selected:
                                selected_raw.append(feature)
            
            st.session_state.selected_raw_features = selected_raw
            
        else:
            st.info("ℹ️ All original features were included in factor analysis. No raw features available.")
    
    with tab3:
        display_selection_summary()

def display_selection_summary():
    """Display comprehensive selection summary"""
    
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

def calculate_vif_analysis():
    """Calculate VIF for selected variables with proper error handling"""
    
    selected_factored = st.session_state.get('selected_factored_features', [])
    selected_raw = st.session_state.get('selected_raw_features', [])
    
    if not selected_factored and not selected_raw:
        st.error("⚠️ Please select at least one variable for VIF analysis.")
        return
    
    try:
        # Combine data from different sources with validation
        combined_data = pd.DataFrame()
        
        # ✅ CRITICAL FIX: Validate factored variables exist in factor_scores_df
        factor_scores_df = st.session_state.get('factor_scores_df', pd.DataFrame())
        if not factor_scores_df.empty and selected_factored:
            # Filter to only existing columns
            valid_factored = [col for col in selected_factored if col in factor_scores_df.columns]
            
            if valid_factored:
                factored_data = factor_scores_df[valid_factored].reset_index(drop=True)
                combined_data = pd.concat([combined_data, factored_data], axis=1)
                
                if len(valid_factored) != len(selected_factored):
                    missing_factored = [col for col in selected_factored if col not in valid_factored]
                    st.warning(f"⚠️ Some factored variables not found: {missing_factored}")
            else:
                st.warning("⚠️ None of the selected factored variables were found in factor scores.")
        
        # ✅ CRITICAL FIX: Validate raw variables exist in final_model_df
        final_model_df = st.session_state.get('final_model_df')
        if final_model_df is not None and not final_model_df.empty and selected_raw:
            # Filter to only existing columns
            valid_raw = [col for col in selected_raw if col in final_model_df.columns]
            
            if valid_raw:
                raw_data = final_model_df[valid_raw].reset_index(drop=True)
                # Fill missing values in raw data
                raw_data = raw_data.fillna(raw_data.median())
                combined_data = pd.concat([combined_data, raw_data], axis=1)
                
                if len(valid_raw) != len(selected_raw):
                    missing_raw = [col for col in selected_raw if col not in valid_raw]
                    st.warning(f"⚠️ Some raw variables not found: {missing_raw}")
            else:
                st.warning("⚠️ None of the selected raw variables were found in the dataset.")
        
        if combined_data.empty:
            st.error("⚠️ No valid data available for VIF calculation. Please check your variable selections.")
            return
        
        # Ensure all data is numeric and handle any remaining missing values
        combined_data = combined_data.select_dtypes(include=[np.number])
        combined_data = combined_data.fillna(combined_data.median())
        
        if combined_data.empty:
            st.error("⚠️ No numeric data available for VIF calculation.")
            return
        
        # Add constant for VIF calculation
        X_with_const = sm.add_constant(combined_data)
        
        # Calculate VIF
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_const.columns
        
        vif_values = []
        for i in range(X_with_const.shape[1]):
            try:
                vif_val = variance_inflation_factor(X_with_const.values, i)
                # Handle infinite or extremely large VIF values
                if np.isinf(vif_val) or vif_val > 1000:
                    vif_val = 1000  # Cap at 1000 for display
                vif_values.append(vif_val)
            except Exception as e:
                st.warning(f"⚠️ Could not calculate VIF for variable {X_with_const.columns[i]}: {str(e)}")
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        
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
            st.warning("⚠️ **Recommendation:** Consider removing variables with VIF > 10 to reduce multicollinearity")
            st.write("**High VIF Variables:**")
            for _, row in high_vif.iterrows():
                if row['Variable'] != 'const' and not pd.isna(row['VIF']):
                    st.write(f"• {row['Variable']}: {row['VIF']:.2f}")
        else:
            st.success("✅ **Good:** No high multicollinearity detected among selected variables")
        
        # Store VIF results
        st.session_state.vif_results = vif_data
        
    except Exception as e:
        st.error(f"❌ Error calculating VIF: {str(e)}")
        st.write("**Debug Information:**")
        st.write(f"- Selected factored features: {len(selected_factored)}")
        st.write(f"- Selected raw features: {len(selected_raw)}")
        st.write("Please ensure all selected variables exist in the respective datasets.")

def train_and_evaluate_model():
    """Train logistic regression with combined variable set"""
    
    selected_factored = st.session_state.get('selected_factored_features', [])
    selected_raw = st.session_state.get('selected_raw_features', [])
    
    if not selected_factored and not selected_raw:
        st.error("⚠️ Please select at least one variable for modeling.")
        return
    
    try:
        # Combine data from different sources with validation
        X_combined = pd.DataFrame()
        
        # Add factored variables with validation
        factor_scores_df = st.session_state.get('factor_scores_df', pd.DataFrame())
        if not factor_scores_df.empty and selected_factored:
            valid_factored = [col for col in selected_factored if col in factor_scores_df.columns]
            if valid_factored:
                factored_data = factor_scores_df[valid_factored].reset_index(drop=True)
                X_combined = pd.concat([X_combined, factored_data], axis=1)
        
        # Add raw variables with validation
        final_model_df = st.session_state.get('final_model_df')
        if final_model_df is not None and not final_model_df.empty and selected_raw:
            valid_raw = [col for col in selected_raw if col in final_model_df.columns]
            if valid_raw:
                raw_data = final_model_df[valid_raw].reset_index(drop=True)
                # Fill missing values in raw data
                raw_data = raw_data.fillna(raw_data.median())
                X_combined = pd.concat([X_combined, raw_data], axis=1)
        
        if X_combined.empty:
            st.error("⚠️ No valid data available for modeling.")
            return
        
        # Get target variable
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
        
        # Make predictions
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
            'regression_model': model  # Add model to results for Step 13
        }
        
        # Mark step as completed
        if 'step_completed' not in st.session_state:
            st.session_state.step_completed = {}
        st.session_state.step_completed[12] = True
        
        # Display results
        display_model_results(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, 
                             list(X_combined.columns), selected_factored, selected_raw)
        
    except Exception as e:
        st.error(f"❌ Error in model training: {str(e)}")
        st.write("Please check your variable selections and data quality.")

def display_model_results(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, 
                         all_features, selected_factored, selected_raw):
    """Display comprehensive model results"""
    
    st.subheader("🎯 Model Performance Results")
    
    # Model coefficients (Key Drivers) with variable type identification
    st.subheader("🔑 Key Driver Analysis - Variable Importance")
    
    coefficients = pd.DataFrame({
        'Variable': all_features,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0]),
        'Variable_Type': ['Factored' if var in selected_factored else 'Raw' for var in all_features]
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Create enhanced horizontal bar chart
    fig = px.bar(
        coefficients,
        y='Variable',
        x='Coefficient',
        orientation='h',
        title='Variable Importance (Logistic Regression Coefficients)',
        color='Variable_Type',
        color_discrete_map={'Factored': '#2E86AB', 'Raw': '#F24236'}
    )
    fig.update_layout(height=max(400, len(all_features) * 25))
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced coefficients table
    st.write("**Detailed Coefficients by Variable Type:**")
    st.dataframe(coefficients[['Variable', 'Coefficient', 'Variable_Type']].round(4), use_container_width=True)
    
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
    
    # Enhanced Feature importance summary by type
    st.subheader("🏆 Top Key Drivers Summary by Variable Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔬 Top Factored Variables:**")
        factored_drivers = coefficients[coefficients['Variable_Type'] == 'Factored'].head(5)
        if not factored_drivers.empty:
            for i, (_, row) in enumerate(factored_drivers.iterrows(), 1):
                impact = "↗️ Positive" if row['Coefficient'] > 0 else "↘️ Negative"
                st.write(f"{i}. **{row['Variable']}**: {row['Coefficient']:.4f} {impact}")
        else:
            st.write("No factored variables selected")
    
    with col2:
        st.write("**📊 Top Raw Variables:**")
        raw_drivers = coefficients[coefficients['Variable_Type'] == 'Raw'].head(5)
        if not raw_drivers.empty:
            for i, (_, row) in enumerate(raw_drivers.iterrows(), 1):
                impact = "↗️ Positive" if row['Coefficient'] > 0 else "↘️ Negative"
                st.write(f"{i}. **{row['Variable']}**: {row['Coefficient']:.4f} {impact}")
        else:
            st.write("No raw variables selected")
    
    # Enhanced Model summary
    st.subheader("📈 Enhanced Model Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Performance:**")
        st.write(f"• **Accuracy**: {accuracy:.1%}")
        st.write(f"• **AUC-ROC**: {auc_score:.3f}")
        st.write(f"• **Precision**: {precision:.1%}")
        st.write(f"• **Recall**: {recall:.1%}")
    
    with col2:
        st.write("**Variable Composition:**")
        st.write(f"• **Total Variables**: {len(all_features)}")
        st.write(f"• **Factored Variables**: {len(selected_factored)}")
        st.write(f"• **Raw Variables**: {len(selected_raw)}")
        st.write(f"• **Training Samples**: {len(X_train):,}")
    
    # Performance assessment
    if auc_score > 0.8:
        st.success("🎉 **Excellent model performance** (AUC > 0.8)")
    elif auc_score > 0.7:
        st.info("👍 **Good model performance** (AUC > 0.7)")
    else:
        st.warning("⚠️ **Model performance could be improved** (AUC ≤ 0.7)")
    
    st.info("📌 Logistic regression completed! Click **Next ➡️** to proceed to final results dashboard.")

if __name__ == "__main__":
    show_page()
