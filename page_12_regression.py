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
    """
    Build everything Step 12 needs:

    • factor_scores_df  – the factors actually created
    • raw_features_not_factored – every numeric feature that never became a factor
    • y_target          – the binary target column
    """

    final_model_df      = st.session_state.get("final_model_df")
    selected_target_col = st.session_state.get("selected_target_col")

    # Basic guard-rails
    if final_model_df is None or selected_target_col is None:
        st.error("⚠️  Required data missing – please finish the previous steps first.")
        return

    # 1⃣  Factors that DO exist
    factor_scores_df = st.session_state.get("factor_scores_df", pd.DataFrame())

    # 2⃣  Features that really fed any successful factor analysis
    used_in_factors = set()
    fa_results = st.session_state.get("fa_results", {})
    for res in fa_results.values():
        if res and res.get("success", False):
            used_in_factors.update(res.get("features", []))

    # 3⃣  Full numeric feature list detected back in Step 5
    feature_list = st.session_state.get("feature_list", [])

    # 4⃣  Anything numeric that never became a factor = "raw"
    raw_features_not_factored = [f for f in feature_list if f not in used_in_factors]

    # (optional) keep only columns that really exist in the modelling dataframe
    raw_features_not_factored = [f for f in raw_features_not_factored
                                 if f in final_model_df.columns]

    # 5⃣  Target column (already binary)
    y_target = final_model_df[selected_target_col].reset_index(drop=True)

    # 6⃣  Store everything for later steps
    st.session_state.factor_scores_df          = factor_scores_df
    st.session_state.raw_features_not_factored = raw_features_not_factored
    st.session_state.y_target                  = y_target
    st.session_state.final_model_df            = final_model_df

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
            
            # Show debug info
            st.write(f"**Debug Info:** Found {len(raw_features_not_factored)} raw features")
            
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
    """Calculate VIF with bulletproof error handling and validation"""
    
    st.write("🔄 Starting VIF analysis...")
    
    # Get selected variables
    selected_factored = st.session_state.get('selected_factored_features', [])
    selected_raw = st.session_state.get('selected_raw_features', [])
    
    if not selected_factored and not selected_raw:
        st.error("⚠️ Please select at least one variable for VIF analysis.")
        return
    
    try:
        # Initialize empty dataframe for combined data
        combined_data = pd.DataFrame()
        data_sources = []
        
        # Process factored variables
        if selected_factored:
            factor_scores_df = st.session_state.get('factor_scores_df', pd.DataFrame())
            st.write(f"📊 Processing {len(selected_factored)} factored variables...")
            
            if not factor_scores_df.empty:
                # Get available factored columns
                available_factored_cols = factor_scores_df.columns.tolist()
                valid_factored = []
                
                for col in selected_factored:
                    if col in available_factored_cols:
                        valid_factored.append(col)
                    else:
                        st.warning(f"⚠️ Factored variable not found: {col}")
                
                if valid_factored:
                    try:
                        factored_subset = factor_scores_df[valid_factored].reset_index(drop=True)
                        # Ensure all data is numeric
                        factored_subset = factored_subset.select_dtypes(include=[np.number])
                        
                        if not factored_subset.empty:
                            combined_data = pd.concat([combined_data, factored_subset], axis=1)
                            data_sources.append(f"✅ Added {len(factored_subset.columns)} factored variables")
                        else:
                            st.warning("⚠️ No numeric factored variables found")
                    except Exception as e:
                        st.error(f"❌ Error processing factored variables: {str(e)}")
            else:
                st.warning("⚠️ Factor scores dataframe is empty")
        
        # Process raw variables
        if selected_raw:
            final_model_df = st.session_state.get('final_model_df')
            st.write(f"📊 Processing {len(selected_raw)} raw variables...")
            
            if final_model_df is not None and not final_model_df.empty:
                # Get available raw columns
                available_raw_cols = final_model_df.columns.tolist()
                valid_raw = []
                
                for col in selected_raw:
                    if col in available_raw_cols:
                        valid_raw.append(col)
                    else:
                        st.warning(f"⚠️ Raw variable not found: {col}")
                
                if valid_raw:
                    try:
                        raw_subset = final_model_df[valid_raw].reset_index(drop=True)
                        # Ensure all data is numeric
                        raw_subset = raw_subset.select_dtypes(include=[np.number])
                        
                        if not raw_subset.empty:
                            # Handle missing values
                            raw_subset = raw_subset.fillna(raw_subset.median())
                            combined_data = pd.concat([combined_data, raw_subset], axis=1)
                            data_sources.append(f"✅ Added {len(raw_subset.columns)} raw variables")
                        else:
                            st.warning("⚠️ No numeric raw variables found")
                    except Exception as e:
                        st.error(f"❌ Error processing raw variables: {str(e)}")
            else:
                st.warning("⚠️ Final model dataframe is empty")
        
        # Display data source summary
        st.write("**Data Sources:**")
        for source in data_sources:
            st.write(source)
        
        # Check if we have any data
        if combined_data.empty:
            st.error("❌ No valid numeric data available for VIF calculation. Please check your variable selections.")
            return
        
        st.write(f"📈 Combined dataset shape: {combined_data.shape}")
        st.write(f"📊 Variables for VIF: {list(combined_data.columns)}")
        
        # Handle any remaining missing values
        missing_count = combined_data.isnull().sum().sum()
        if missing_count > 0:
            st.write(f"⚠️ Found {missing_count} missing values, filling with median...")
            combined_data = combined_data.fillna(combined_data.median())
        
        # Check for constant columns (zero variance)
        constant_cols = []
        for col in combined_data.columns:
            if combined_data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            st.warning(f"⚠️ Removing constant columns: {constant_cols}")
            combined_data = combined_data.drop(columns=constant_cols)
        
        if combined_data.empty:
            st.error("❌ No valid data remaining after preprocessing.")
            return
        
        # Check minimum requirements for VIF
        if combined_data.shape[1] < 2:
            st.error("❌ Need at least 2 variables for VIF calculation.")
            return
        
        if combined_data.shape[0] < combined_data.shape[1] + 1:
            st.error(f"❌ Insufficient observations ({combined_data.shape[0]}) for {combined_data.shape[1]} variables.")
            return
        
        # Add constant term for VIF calculation
        st.write("🔢 Adding constant term and calculating VIF...")
        X_with_const = sm.add_constant(combined_data, has_constant='add')
        
        # Calculate VIF values
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_const.columns.tolist()
        
        vif_values = []
        for i, col in enumerate(X_with_const.columns):
            try:
                vif_val = variance_inflation_factor(X_with_const.values, i)
                
                # Handle problematic VIF values
                if np.isnan(vif_val) or np.isinf(vif_val):
                    vif_val = 999.9  # Large value to indicate problem
                elif vif_val > 1000:
                    vif_val = 999.9  # Cap extremely large values
                
                vif_values.append(vif_val)
                st.write(f"  ✓ {col}: {vif_val:.2f}")
                
            except Exception as e:
                st.warning(f"  ⚠️ Could not calculate VIF for {col}: {str(e)}")
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        
        # Remove rows with NaN VIF values for sorting
        vif_data_clean = vif_data.dropna()
        vif_data_clean = vif_data_clean.sort_values('VIF', ascending=False)
        
        # Display results
        st.success("✅ VIF calculation completed!")
        st.write("📊 **VIF Results:**")
        st.dataframe(vif_data_clean, use_container_width=True)
        
        # VIF interpretation
        valid_vif = vif_data_clean[vif_data_clean['Variable'] != 'const']
        high_vif = valid_vif[valid_vif['VIF'] > 10]
        moderate_vif = valid_vif[(valid_vif['VIF'] > 5) & (valid_vif['VIF'] <= 10)]
        low_vif = valid_vif[valid_vif['VIF'] <= 5]
        
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
                st.write(f"• {row['Variable']}: {row['VIF']:.2f}")
        else:
            st.success("✅ **Good:** No high multicollinearity detected among selected variables")
        
        # Store VIF results
        st.session_state.vif_results = vif_data_clean
        
    except Exception as e:
        st.error(f"❌ Unexpected error in VIF calculation: {str(e)}")
        
        # Debug information
        st.write("**🔍 Debug Information:**")
        try:
            st.write(f"- Selected factored features: {len(st.session_state.get('selected_factored_features', []))}")
            st.write(f"- Selected raw features: {len(st.session_state.get('selected_raw_features', []))}")
            
            factor_scores_df = st.session_state.get('factor_scores_df', pd.DataFrame())
            if not factor_scores_df.empty:
                st.write(f"- Factor scores shape: {factor_scores_df.shape}")
                st.write(f"- Factor scores columns: {list(factor_scores_df.columns)}")
            
            final_model_df = st.session_state.get('final_model_df')
            if final_model_df is not None and not final_model_df.empty:
                st.write(f"- Final model df shape: {final_model_df.shape}")
                st.write(f"- Sample columns: {list(final_model_df.columns)[:10]}")
            
        except Exception as debug_e:
            st.write(f"Could not generate debug info: {str(debug_e)}")

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
