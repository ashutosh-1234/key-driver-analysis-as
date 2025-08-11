import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    categorize_features, prepare_factor_data, run_suitability_tests,
    determine_optimal_factors, perform_factor_analysis, create_loadings_heatmap,
    create_variance_chart, interpret_factor_loadings
)

def render_factor_execution_page():
    """Render the factor analysis execution page"""
    
    # Check prerequisites
    if st.session_state.fa_config is None:
        st.error("❌ Factor analysis not configured. Please complete Step 7 first.")
        return
    
    fa_config = st.session_state.fa_config
    final_model_df = st.session_state.final_model_df
    selected_features = st.session_state.selected_features
    
    st.markdown("""
    ## 🚀 Factor Analysis Execution
    
    Execute factor analysis based on your configuration and review the results.
    """)
    
    # Show configuration summary
    st.subheader("📋 Configuration Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Analysis Type:** {fa_config['analysis_type']}")
        st.write(f"**Factor Method:** {fa_config['factor_method']}")
        if fa_config['manual_factors']:
            st.write(f"**Number of Factors:** {fa_config['manual_factors']}")
        else:
            st.write(f"**Coverage Threshold:** {fa_config['coverage_threshold']:.0%}")
    
    with col2:
        st.write(f"**Rotation Method:** {fa_config['rotation']}")
        st.write(f"**Total Features:** {len(selected_features)}")
        st.write(f"**Sample Size:** {len(final_model_df):,}")
    
    st.markdown("---")
    
    # Execute factor analysis
    if st.button("🚀 Execute Factor Analysis", type="primary"):
        execute_factor_analysis()
        st.session_state.step_completed[8] = True

def execute_factor_analysis():
    """Execute the factor analysis process"""
    
    fa_config = st.session_state.fa_config
    final_model_df = st.session_state.final_model_df
    selected_features = st.session_state.selected_features
    
    st.subheader("⚙️ Factor Analysis Execution")
    
    fa_results = {}
    
    with st.spinner("Running factor analysis..."):
        
        if "Within Categories" in fa_config['analysis_type']:
            # Category-wise analysis
            categories = {
                'Rep Attributes': fa_config['rep_features'],
                'Product Perceptions': fa_config['perception_features'],
                'Message Delivery': fa_config['delivery_features'],
                'Miscellaneous': fa_config['misc_features']
            }
            
            for category, features in categories.items():
                if len(features) >= 3:  # Minimum features for factor analysis
                    st.write(f"**Analyzing {category}...**")
                    result = analyze_category(category, features, final_model_df, fa_config)
                    fa_results[category] = result
                else:
                    st.write(f"**Skipping {category}** (insufficient features: {len(features)})")
        
        else:
            # All features analysis
            st.write("**Analyzing All Features...**")
            result = analyze_category('All Features', selected_features, final_model_df, fa_config)
            fa_results['All Features'] = result
    
    # Store results
    st.session_state.fa_results = fa_results
    
    # Display execution summary
    display_execution_summary(fa_results)

def analyze_category(category_name, features, data, fa_config):
    """Analyze a specific category of features"""
    
    try:
        # Prepare data
        prepared_data, scaler = prepare_factor_data(data, features)
        
        # Run suitability tests
        suitability = run_suitability_tests(prepared_data)
        
        st.write(f"- KMO Measure: {suitability.get('kmo_measure', 'Error'):.3f} ({suitability.get('kmo_rating', 'Error')})")
        st.write(f"- Bartlett's Test: {'Significant' if suitability.get('bartlett_suitable', False) else 'Not Significant'}")
        
        if not suitability.get('overall_suitable', False):
            st.warning(f"⚠️ {category_name} may not be suitable for factor analysis")
        
        # Determine number of factors
        if fa_config['manual_factors']:
            n_factors = min(fa_config['manual_factors'], len(features) - 1)
            st.write(f"- Using manual factor count: {n_factors}")
        else:
            # Find optimal factors
            coverage_results = determine_optimal_factors(
                prepared_data, 
                fa_config['coverage_threshold']
            )
            
            if coverage_results:
                # Find first result that meets threshold
                optimal_result = next(
                    (r for r in coverage_results if r['cumulative_variance'] >= fa_config['coverage_threshold']),
                    coverage_results[-1]  # Use last if none meet threshold
                )
                n_factors = optimal_result['n_factors']
                st.write(f"- Optimal factors for {fa_config['coverage_threshold']:.0%} coverage: {n_factors}")
            else:
                n_factors = min(5, len(features) - 1)
                st.write(f"- Using default factor count: {n_factors}")
        
        # Perform factor analysis
        fa_result = perform_factor_analysis_with_proper_rotation(prepared_data, n_factors, fa_config['rotation'])
        if fa_result['success']:
        # Check factor correlation matrix
           factor_corr = fa_result['factor_scores'].corr()
           st.write("Factor Correlation Matrix:")
           st.dataframe(factor_corr.round(3))
    
           # Check if rotation was applied
           st.write(f"Rotation used: {fa_config['rotation']}")
        
        if fa_result['success']:
            st.success(f"✅ {category_name}: {n_factors} factors, {fa_result['cumulative_variance']:.1%} variance explained")
            
            # Add category prefix to factor scores
            factor_scores = fa_result['factor_scores'].copy()
            factor_scores.columns = [f'{category_name}_{col}' for col in factor_scores.columns]
            fa_result['factor_scores'] = factor_scores
            
            return {
                'success': True,
                'category': category_name,
                'features': features,
                'n_factors': n_factors,
                'loadings': fa_result['loadings'],
                'factor_scores': factor_scores,
                'variance_explained': fa_result['variance_explained'],
                'cumulative_variance': fa_result['cumulative_variance'],
                'suitability': suitability,
                'scaler': scaler
            }
        else:
            st.error(f"❌ {category_name}: {fa_result.get('error', 'Unknown error')}")
            return {'success': False, 'error': fa_result.get('error')}
            
    except Exception as e:
        st.error(f"❌ Error analyzing {category_name}: {str(e)}")
        return {'success': False, 'error': str(e)}

def display_execution_summary(fa_results):
    """Display comprehensive execution summary"""
    
    st.subheader("📊 Execution Summary")
    
    # Overall metrics
    successful_analyses = sum(1 for result in fa_results.values() if result.get('success', False))
    total_factors_created = sum(result.get('n_factors', 0) for result in fa_results.values() if result.get('success', False))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Successful Analyses", successful_analyses)
    with col2:
        st.metric("Total Factors Created", total_factors_created)
    with col3:
        st.metric("Categories Analyzed", len(fa_results))
    
    # Detailed results
    st.write("**📈 Category Results:**")
    
    summary_data = []
    all_factor_scores = pd.DataFrame()
    
    for category, results in fa_results.items():
        if results.get('success', False):
            # Add to summary
            summary_data.append({
                'Category': category,
                'Features': len(results['features']),
                'Factors': results['n_factors'],
                'Variance Explained': f"{results['cumulative_variance']:.1%}",
                'KMO Rating': results['suitability'].get('kmo_rating', 'Unknown'),
                'Status': '✅ Success'
            })
            
            # Combine factor scores
            if all_factor_scores.empty:
                all_factor_scores = results['factor_scores']
            else:
                all_factor_scores = pd.concat([all_factor_scores, results['factor_scores']], axis=1)
                
        else:
            summary_data.append({
                'Category': category,
                'Features': 0,
                'Factors': 0,
                'Variance Explained': 'N/A',
                'KMO Rating': 'Error',
                'Status': '❌ Failed'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # Store combined factor scores
    if not all_factor_scores.empty:
        st.session_state.factor_scores_df = all_factor_scores
        
        st.success(f"✅ Factor scores ready: {all_factor_scores.shape[0]} observations × {all_factor_scores.shape[1]} factors")
        
        # Preview of factor scores
        with st.expander("👀 Preview Factor Scores"):
            st.dataframe(all_factor_scores.head(), use_container_width=True)
    
    # Next steps
    if successful_analyses > 0:
        st.info("📌 Factor analysis completed successfully! Click 'Next ➡️' to proceed to results visualization.")
    else:
        st.error("❌ No successful factor analyses. Please review your data and configuration.")

import numpy as np
from scipy.linalg import inv
from sklearn.decomposition import FactorAnalysis

def perform_factor_analysis_with_proper_rotation(data, n_factors, rotation='varimax'):
    """
    Perform factor analysis with proper rotation applied to both loadings and scores
    """
    
    # Step 1: Initial factor analysis (unrotated)
    fa = FactorAnalysis(n_components=n_factors)
    fa.fit(data)
    
    # Get unrotated loadings and factor scores
    unrotated_loadings = fa.components_.T  # Shape: (n_variables, n_factors)
    unrotated_scores = fa.transform(data)   # Shape: (n_observations, n_factors)
    
    # Step 2: Apply rotation to loadings
    if rotation.lower() == 'varimax':
        rotated_loadings, rotation_matrix = varimax_rotation(unrotated_loadings)
    elif rotation.lower() == 'promax':
        rotated_loadings, rotation_matrix = promax_rotation(unrotated_loadings)
    else:
        rotated_loadings = unrotated_loadings
        rotation_matrix = np.eye(n_factors)
    
    # Step 3: Transform factor scores using rotation matrix
    # The key insight: F_rotated = F_unrotated * R^(-1)
    rotation_matrix_inv = inv(rotation_matrix)
    rotated_scores = unrotated_scores @ rotation_matrix_inv
    
    # Step 4: Alternative method - compute scores from rotated loadings
    # This is more mathematically sound
    properly_rotated_scores = compute_factor_scores_from_loadings(
        data, rotated_loadings, fa.noise_variance_
    )
    
    return {
        'success': True,
        'loadings': rotated_loadings,
        'factor_scores': pd.DataFrame(
            properly_rotated_scores,
            columns=[f'Factor_{i+1}' for i in range(n_factors)],
            index=data.index
        ),
        'rotation_matrix': rotation_matrix,
        'variance_explained': np.var(properly_rotated_scores, axis=0),
        'cumulative_variance': np.var(properly_rotated_scores, axis=0).sum() / data.shape[1]
    }

def varimax_rotation(loadings, gamma=1.0, q=20, tol=1e-6):
    """
    Perform Varimax rotation on factor loadings
    """
    p, k = loadings.shape
    R = np.eye(k)
    d = 0
    
    for i in range(q):
        d_old = d
        Lambda = loadings @ R
        u, s, vh = np.linalg.svd(loadings.T @ (Lambda ** 3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))))
        R = u @ vh
        d = np.sum(s)
        
        if d_old != 0 and d / d_old < 1 + tol:
            break
    
    return loadings @ R, R

def compute_factor_scores_from_loadings(data, loadings, noise_variance):
    """
    Compute factor scores from rotated loadings using regression method
    This maintains the proper mathematical relationship between data, loadings, and scores
    """
    # Regression method: F = X * L * (L'L + Ψ)^(-1)
    # Where Ψ is the diagonal matrix of uniquenesses
    
    # Convert noise_variance to diagonal matrix
    if np.isscalar(noise_variance):
        psi = np.eye(data.shape[1]) * noise_variance
    else:
        psi = np.diag(noise_variance)
    
    # Compute factor scores
    L = loadings
    LTL_plus_psi = L.T @ L + inv(psi)
    factor_scores = data @ L @ inv(LTL_plus_psi)
    
    return factor_scores

def promax_rotation(loadings, k=4):
    """
    Perform Promax (oblique) rotation
    """
    # First apply varimax
    varimax_loadings, varimax_R = varimax_rotation(loadings)
    
    # Then apply promax transformation
    P = varimax_loadings * np.abs(varimax_loadings) ** (k - 1)
    P = P / np.sqrt(np.sum(P**2, axis=0))
    
    # Solve for transformation matrix
    T = np.linalg.lstsq(varimax_loadings, P, rcond=None)[0]
    
    # Get final loadings
    promax_loadings = varimax_loadings @ T
    
    return promax_loadings, varimax_R @ T

if __name__ == "__main__":
    render_factor_execution_page()
