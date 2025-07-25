import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def render_binary_page():
    """Render the binary conversion (Top-2 Box) page"""
    
    # Check prerequisites
    if st.session_state.filtered_df is None:
        st.error("❌ No filtered data available. Please complete Step 2 first.")
        return
    
    filtered_df = st.session_state.filtered_df
    
    st.markdown("""
    ## ✅ Binary Conversion (Top-2 Box)
    
    Convert key outcome variables to binary format using the Top-2 Box approach.
    
    **Top-2 Box Rule:** Values > 5 = 1 (positive), Values ≤ 5 = 0 (negative)
    """)
    
    # Detect outcome columns dynamically
    ltip_col = next((col for col in filtered_df.columns if 'ltip' in col.lower()), None)
    rep_col = next((col for col in filtered_df.columns if 'overall quality' in col.lower()), None)
    percep_col = next((col for col in filtered_df.columns if 'overall perception' in col.lower()), None)
    
    # Check if all columns are found
    outcome_cols = {'LTIP': ltip_col, 'Rep Quality': rep_col, 'Perception': percep_col}
    missing_cols = [name for name, col in outcome_cols.items() if col is None]
    
    if missing_cols:
        st.error(f"⚠️ Missing outcome columns: {', '.join(missing_cols)}")
        st.write("**Available columns containing outcome keywords:**")
        for col in filtered_df.columns:
            if any(keyword in col.lower() for keyword in ['ltip', 'overall', 'quality', 'perception']):
                st.write(f"- {col}")
        return
    
    st.success("✅ All required outcome columns detected:")
    for name, col in outcome_cols.items():
        st.write(f"**{name}:** {col}")
    
    st.markdown("---")
    
    # Apply Top-2 Box conversion
    if st.button("🔄 Apply Top-2 Box Conversion", type="primary"):
        apply_binary_conversion(filtered_df, ltip_col, rep_col, percep_col)
        st.session_state.step_completed[2] = True

def apply_binary_conversion(filtered_df, ltip_col, rep_col, percep_col):
    """Apply binary conversion and generate visualizations"""
    
    # Create binary dataframe
    bin_df = filtered_df[[ltip_col, rep_col, percep_col]].copy()
    bin_df['Binary_LTIB'] = bin_df[ltip_col].apply(lambda x: 1 if x > 5 else 0)
    bin_df['Binary_Rep'] = bin_df[rep_col].apply(lambda x: 1 if x > 5 else 0)
    bin_df['Binary_Perception'] = bin_df[percep_col].apply(lambda x: 1 if x > 5 else 0)
    
    # Store in session state
    st.session_state.bin_df = bin_df
    
    st.subheader("📊 Binary Conversion Results")
    
    # Create visualizations
    binary_cols = ['Binary_LTIB', 'Binary_Rep', 'Binary_Perception']
    original_cols = [ltip_col, rep_col, percep_col]
    titles = ['LTIP (Likelihood to Increase Prescription)', 'Rep Performance (Overall Quality)', 'Product Perception (Overall Perception)']
    
    # Method 1: Matplotlib subplots
    st.write("**📈 Distribution Plots with Counts and Percentages**")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for i, (binary_col, original_col, title) in enumerate(zip(binary_cols, original_cols, titles)):
        total = bin_df[binary_col].count()
        value_counts = bin_df[binary_col].value_counts().sort_index()
        
        # Create bar plot
        bars = axes[i].bar([0, 1], [value_counts.get(0, 0), value_counts.get(1, 0)], 
                          alpha=0.7, color=['lightcoral', 'lightgreen'])
        
        axes[i].set_title(f'Top-2 Box: {title}', fontsize=12, pad=20)
        axes[i].set_xlabel('Binary Value')
        axes[i].set_ylabel('Count')
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['0 (≤5)', '1 (>5)'])
        
        # Add percentage labels on bars
        for j, bar in enumerate(bars):
            count = int(bar.get_height())
            if total > 0:
                pct = 100 * count / total
                axes[i].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + total * 0.01,
                           f'{count}\n({pct:.1f}%)', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Method 2: Interactive Plotly charts
    st.write("**📊 Interactive Distribution Charts**")
    
    # Create tabs for each variable
    tab1, tab2, tab3 = st.tabs(["LTIP", "Rep Performance", "Product Perception"])
    
    tabs = [tab1, tab2, tab3]
    
    for i, (tab, binary_col, title) in enumerate(zip(tabs, binary_cols, titles)):
        with tab:
            # Calculate percentages
            value_counts = bin_df[binary_col].value_counts().sort_index()
            total = bin_df[binary_col].count()
            
            percentages = [100 * value_counts.get(j, 0) / total for j in [0, 1]]
            counts = [value_counts.get(j, 0) for j in [0, 1]]
            
            # Create plotly bar chart
            fig = px.bar(
                x=['0 (≤5)', '1 (>5)'],
                y=counts,
                title=f'{title} - Binary Distribution',
                labels={'x': 'Binary Value', 'y': 'Count'},
                color=counts,
                color_continuous_scale='RdYlGn',
                text=[f'{count}<br>({pct:.1f}%)' for count, pct in zip(counts, percentages)]
            )
            
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Responses", total)
            with col2:
                st.metric("Negative (0)", f"{counts[0]} ({percentages[0]:.1f}%)")
            with col3:
                st.metric("Positive (1)", f"{counts[1]} ({percentages[1]:.1f}%)")
            with col4:
                balance_ratio = min(percentages) / max(percentages) if max(percentages) > 0 else 0
                st.metric("Balance Ratio", f"{balance_ratio:.2f}")
    
    # Summary table
    st.write("**📋 Binary Conversion Summary Table**")
    
    summary_data = []
    for binary_col, original_col, title in zip(binary_cols, original_cols, titles):
        value_counts = bin_df[binary_col].value_counts().sort_index()
        total = bin_df[binary_col].count()
        
        summary_data.append({
            'Variable': title,
            'Original Column': original_col,
            'Binary Column': binary_col,
            'Total Count': total,
            'Negative (0)': value_counts.get(0, 0),
            'Positive (1)': value_counts.get(1, 0),
            'Positive %': f"{100 * value_counts.get(1, 0) / total:.1f}%" if total > 0 else "0%",
            'Balance Ratio': f"{min(value_counts.get(0, 0), value_counts.get(1, 0)) / max(value_counts.get(0, 0), value_counts.get(1, 0)):.2f}" if max(value_counts.get(0, 0), value_counts.get(1, 0)) > 0 else "0"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Analysis insights
    st.subheader("🔍 Analysis Insights")
    
    insights = []
    for i, (binary_col, title) in enumerate(zip(binary_cols, titles)):
        value_counts = bin_df[binary_col].value_counts().sort_index()
        total = bin_df[binary_col].count()
        positive_pct = 100 * value_counts.get(1, 0) / total if total > 0 else 0
        
        if positive_pct > 70:
            insights.append(f"🟢 **{title}**: High positive sentiment ({positive_pct:.1f}%) - Good distribution for analysis")
        elif positive_pct < 30:
            insights.append(f"🔴 **{title}**: Low positive sentiment ({positive_pct:.1f}%) - May need review")
        else:
            insights.append(f"🟡 **{title}**: Balanced distribution ({positive_pct:.1f}%) - Ideal for modeling")
    
    for insight in insights:
        st.write(insight)
    
    # Data quality check
    st.write("**🔍 Data Quality Check:**")
    col1, col2 = st.columns(2)
    
    with col1:
        missing_original = sum(filtered_df[col].isnull().sum() for col in [ltip_col, rep_col, percep_col])
        st.metric("Missing Values (Original)", missing_original)
    
    with col2:
        missing_binary = sum(bin_df[col].isnull().sum() for col in binary_cols)
        st.metric("Missing Values (Binary)", missing_binary)
    
    st.success("✅ Binary conversion completed successfully!")
    st.info("📌 Binary variables are ready for target selection. Click 'Next ➡️' to proceed.")

if __name__ == "__main__":
    render_binary_page()