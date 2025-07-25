import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

def render_export_page():
    """Render the export results page"""
    
    # Check prerequisites
    if st.session_state.fa_results is None or st.session_state.factor_scores_df is None:
        st.error("❌ No factor analysis results to export. Please complete Step 9 first.")
        return
    
    fa_results = st.session_state.fa_results
    factor_scores_df = st.session_state.factor_scores_df
    
    st.markdown("""
    ## 💾 Export Results
    
    Export your factor analysis results and factor scores for further analysis.
    """)
    
    # Show export summary
    st.subheader("📊 Export Summary")
    
    successful_analyses = sum(1 for result in fa_results.values() if result.get('success', False))
    total_factors = factor_scores_df.shape[1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Successful Analyses", successful_analyses)
    with col2:
        st.metric("Total Factors", total_factors)
    with col3:
        st.metric("Factor Scores Shape", f"{factor_scores_df.shape[0]} × {factor_scores_df.shape[1]}")
    
    st.markdown("---")
    
    # Export options
    st.subheader("📤 Export Options")
    
    export_options = st.multiselect(
        "Select what to export:",
        options=[
            "Factor Analysis Summary",
            "Factor Loadings",
            "Factor Scores",
            "Factor Scores with Target Variable",
            "Variance Explained",
            "Suitability Test Results"
        ],
        default=[
            "Factor Analysis Summary",
            "Factor Loadings", 
            "Factor Scores with Target Variable"
        ]
    )
    
    # Preview selected exports
    if export_options:
        st.subheader("👀 Export Preview")
        
        for option in export_options:
            with st.expander(f"Preview: {option}"):
                if option == "Factor Analysis Summary":
                    summary_df = create_summary_dataframe(fa_results)
                    st.dataframe(summary_df, use_container_width=True)
                
                elif option == "Factor Scores":
                    st.dataframe(factor_scores_df.head(), use_container_width=True)
                    st.write(f"Shape: {factor_scores_df.shape}")
                
                elif option == "Factor Scores with Target Variable":
                    if st.session_state.selected_target_col and st.session_state.final_model_df is not None:
                        combined_df = pd.concat([
                            factor_scores_df,
                            st.session_state.final_model_df[st.session_state.selected_target_col].reset_index(drop=True)
                        ], axis=1)
                        st.dataframe(combined_df.head(), use_container_width=True)
                        st.write(f"Shape: {combined_df.shape}")
                    else:
                        st.warning("Target variable not available")
                
                elif option == "Factor Loadings":
                    for category, results in fa_results.items():
                        if results.get('success', False):
                            st.write(f"**{category} Loadings:**")
                            st.dataframe(results['loadings'].head(), use_container_width=True)
                
                elif option == "Variance Explained":
                    variance_df = create_variance_dataframe(fa_results)
                    st.dataframe(variance_df, use_container_width=True)
                
                elif option == "Suitability Test Results":
                    suitability_df = create_suitability_dataframe(fa_results)
                    st.dataframe(suitability_df, use_container_width=True)
    
    # Export button
    if export_options:
        if st.button("📥 Generate Excel Export", type="primary"):
            excel_data = generate_excel_export(fa_results, factor_scores_df, export_options)
            
            if excel_data:
                st.session_state.step_completed[10] = True
                
                # Create download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                filename = f'Factor_Analysis_Results_{timestamp}.xlsx'
                
                st.download_button(
                    label="⬇️ Download Excel File",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Excel file generated successfully!")
                st.info("📌 Results exported! Click 'Next ➡️' to proceed to logistic regression analysis.")
            else:
                st.error("❌ Error generating Excel file.")

def create_summary_dataframe(fa_results):
    """Create summary dataframe of factor analysis results"""
    summary_data = []
    
    for category, results in fa_results.items():
        if results.get('success', False):
            summary_data.append({
                'Category': category,
                'Features_Count': len(results['features']),
                'Factors_Count': results['n_factors'],
                'Cumulative_Variance': f"{results['cumulative_variance']:.1%}",
                'Top_Factor_Variance': f"{results['variance_explained'][0]*100:.1f}%" if results['variance_explained'] else "N/A",
                'KMO_Measure': f"{results['suitability'].get('kmo_measure', 0):.3f}",
                'KMO_Rating': results['suitability'].get('kmo_rating', 'Unknown'),
                'Bartlett_Significant': 'Yes' if results['suitability'].get('bartlett_suitable', False) else 'No',
                'Status': 'Success'
            })
        else:
            summary_data.append({
                'Category': category,
                'Features_Count': 0,
                'Factors_Count': 0,
                'Cumulative_Variance': 'N/A',
                'Top_Factor_Variance': 'N/A',
                'KMO_Measure': 'N/A',
                'KMO_Rating': 'Error',
                'Bartlett_Significant': 'N/A',
                'Status': 'Failed'
            })
    
    return pd.DataFrame(summary_data)

def create_variance_dataframe(fa_results):
    """Create variance explained dataframe"""
    variance_data = []
    
    for category, results in fa_results.items():
        if results.get('success', False):
            for i, var_exp in enumerate(results['variance_explained']):
                variance_data.append({
                    'Category': category,
                    'Factor': f'Factor_{i+1}',
                    'Variance_Explained': var_exp,
                    'Variance_Percentage': f"{var_exp*100:.2f}%",
                    'Cumulative_Variance': sum(results['variance_explained'][:i+1])
                })
    
    return pd.DataFrame(variance_data)

def create_suitability_dataframe(fa_results):
    """Create suitability test results dataframe"""
    suitability_data = []
    
    for category, results in fa_results.items():
        if results.get('success', False):
            suit = results['suitability']
            suitability_data.append({
                'Category': category,
                'KMO_Measure': suit.get('kmo_measure', 0),
                'KMO_Rating': suit.get('kmo_rating', 'Unknown'),
                'Bartlett_Chi_Square': suit.get('bartlett_chi_square', 0),
                'Bartlett_P_Value': suit.get('bartlett_p_value', 1),
                'Bartlett_Significant': 'Yes' if suit.get('bartlett_suitable', False) else 'No',
                'Overall_Suitable': 'Yes' if suit.get('overall_suitable', False) else 'No'
            })
    
    return pd.DataFrame(suitability_data)

def generate_excel_export(fa_results, factor_scores_df, export_options):
    """Generate Excel file with selected export options"""
    try:
        # Create Excel writer object
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # Export based on selected options
            if "Factor Analysis Summary" in export_options:
                summary_df = create_summary_dataframe(fa_results)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            if "Factor Scores" in export_options:
                factor_scores_df.to_excel(writer, sheet_name='Factor_Scores', index=False)
            
            if "Factor Scores with Target Variable" in export_options:
                if st.session_state.selected_target_col and st.session_state.final_model_df is not None:
                    combined_df = pd.concat([
                        factor_scores_df,
                        st.session_state.final_model_df[st.session_state.selected_target_col].reset_index(drop=True)
                    ], axis=1)
                    combined_df.to_excel(writer, sheet_name='Scores_with_Target', index=False)
            
            if "Factor Loadings" in export_options:
                for category, results in fa_results.items():
                    if results.get('success', False):
                        sheet_name = f'{category}_Loadings'[:31]  # Excel sheet name limit
                        results['loadings'].to_excel(writer, sheet_name=sheet_name)
            
            if "Variance Explained" in export_options:
                variance_df = create_variance_dataframe(fa_results)
                variance_df.to_excel(writer, sheet_name='Variance_Explained', index=False)
            
            if "Suitability Test Results" in export_options:
                suitability_df = create_suitability_dataframe(fa_results)
                suitability_df.to_excel(writer, sheet_name='Suitability_Tests', index=False)
        
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error generating Excel file: {str(e)}")
        return None

if __name__ == "__main__":
    render_export_page()