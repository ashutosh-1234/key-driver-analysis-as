import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def render_feature_prep_page():
    """Render the feature engineering and data preparation page"""
    
    # Check prerequisites
    if st.session_state.selected_target_col is None:
        st.error("❌ No target variable selected. Please complete Step 4 first.")
        return
    
    filtered_df = st.session_state.filtered_df
    bin_df = st.session_state.bin_df
    selected_target_col = st.session_state.selected_target_col
    selected_target_name = st.session_state.selected_target_name
    
    st.markdown(f"""
    ## 🔧 Feature Engineering & Data Preparation
    
    Prepare features for analysis with selected target variable: **{selected_target_name}**
    """)
    
    # Show target variable info
    st.subheader("🎯 Selected Target Variable")
    col1, col2, col3 = st.columns(3)
    
    if selected_target_col in bin_df.columns:
        value_counts = bin_df[selected_target_col].value_counts().sort_index()
        total = bin_df[selected_target_col].count()
        positive_pct = (value_counts.get(1, 0) / total * 100) if total > 0 else 0
        
        with col1:
            st.metric("Target Variable", selected_target_name)
        with col2:
            st.metric("Positive Rate", f"{positive_pct:.1f}%")
        with col3:
            st.metric("Sample Size", f"{total:,}")
    
    st.markdown("---")
    
    # Feature preparation button
    if st.button("🔄 Prepare Features for Analysis", type="primary"):
        prepare_features_for_analysis()
        st.session_state.step_completed[4] = True

def prepare_features_for_analysis():
    """Prepare features for key driver analysis"""
    
    filtered_df = st.session_state.filtered_df
    bin_df = st.session_state.bin_df
    selected_target_col = st.session_state.selected_target_col
    
    st.subheader("⚙️ Feature Preparation Process")
    
    # Combine original features with binary target
    with st.spinner("Processing features..."):
        analysis_df = filtered_df.copy()
        analysis_df[selected_target_col] = bin_df[selected_target_col]
        
        # Get all feature columns (excluding metadata)
        metadata_cols = ['Product', 'users_wave_id', 'wave_id', 'wave_number', 'user_id', 'user_type', 'status', 
                       'completed_date', 'completed_date_user_tz', 'npi', 'time_period']
        feature_cols = [col for col in analysis_df.columns if col not in metadata_cols and col != selected_target_col]
        
        # Select only numeric features
        numeric_features = analysis_df[feature_cols].select_dtypes(include=[np.number])
        
        # Handle missing values using median imputation
        numeric_features_filled = numeric_features.fillna(numeric_features.median())
        
        # Combine with target
        final_df = numeric_features_filled.copy()
        final_df[selected_target_col] = analysis_df[selected_target_col]
        
        # Store in session state
        st.session_state.model_df = final_df
        st.session_state.feature_list = list(numeric_features_filled.columns)
    
    st.success("✅ Feature preparation completed successfully!")
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Analysis Dataset Shape", f"{final_df.shape[0]} × {final_df.shape[1]}")
        st.metric("Number of Features", len(st.session_state.feature_list))
    
    with col2:
        missing_count = final_df.isnull().sum().sum()
        st.metric("Missing Values", missing_count)
        st.metric("Target Variable", selected_target_col)
    
    # Categorize and display features
    st.subheader("📋 Feature Categories")
    
    feature_list = st.session_state.feature_list
    
    # Categorize features
    rep_features = [f for f in feature_list if "Rep Attributes" in f]
    perception_features = [f for f in feature_list if "Perceptions" in f]
    delivery_features = [f for f in feature_list if "Delivery of topic" in f]
    misc_features = [f for f in feature_list if not any(cat in f for cat in ["Rep Attributes", "Perceptions", "Delivery of topic"])]
    
    # Display in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        f"📈 Rep Attributes ({len(rep_features)})",
        f"📊 Perceptions ({len(perception_features)})", 
        f"📋 Message Delivery ({len(delivery_features)})",
        f"📦 Miscellaneous ({len(misc_features)})"
    ])
    
    with tab1:
        if rep_features:
            for i, feature in enumerate(rep_features, 1):
                st.write(f"{i}. {feature}")
        else:
            st.write("No Rep Attributes features found.")
    
    with tab2:
        if perception_features:
            for i, feature in enumerate(perception_features, 1):
                st.write(f"{i}. {feature}")
        else:
            st.write("No Perceptions features found.")
    
    with tab3:
        if delivery_features:
            for i, feature in enumerate(delivery_features, 1):
                st.write(f"{i}. {feature}")
        else:
            st.write("No Message Delivery features found.")
    
    with tab4:
        if misc_features:
            for i, feature in enumerate(misc_features, 1):
                st.write(f"{i}. {feature}")
        else:
            st.write("No Miscellaneous features found.")
    
    # Feature distribution analysis
    st.subheader("📊 Feature Analysis")
    
    # Basic statistics
    stats_df = final_df[st.session_state.feature_list].describe().round(3)
    
    with st.expander("📈 Descriptive Statistics"):
        st.dataframe(stats_df, use_container_width=True)
    
    # Correlation with target
    with st.expander("🎯 Correlation with Target Variable"):
        correlations = []
        for feature in st.session_state.feature_list:
            try:
                corr = final_df[feature].corr(final_df[selected_target_col])
                correlations.append({
                    'Feature': feature,
                    'Correlation': corr,
                    'Abs_Correlation': abs(corr)
                })
            except:
                pass
        
        if correlations:
            corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
            
            # Show top 10 correlations
            st.write("**Top 10 Correlations with Target Variable:**")
            st.dataframe(corr_df.head(10)[['Feature', 'Correlation']], use_container_width=True)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            top_corrs = corr_df.head(15)
            colors = ['green' if x > 0 else 'red' for x in top_corrs['Correlation']]
            
            bars = ax.barh(range(len(top_corrs)), top_corrs['Correlation'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_corrs)))
            ax.set_yticklabels([f.split('_')[-1] if '_' in f else f for f in top_corrs['Feature']], fontsize=8)
            ax.set_xlabel('Correlation with Target')
            ax.set_title('Top 15 Feature Correlations with Target Variable')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.info("📌 Features prepared successfully! Click 'Next ➡️' to proceed to feature selection.")

if __name__ == "__main__":
    render_feature_prep_page()