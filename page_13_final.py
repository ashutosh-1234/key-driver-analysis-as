# ---------------------- page_13_final.py ----------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ---------- CONFIG ----------
MIN_OTHER_BUCKET = 5.0        # % threshold below which drivers are grouped as "Others"
TOP_N_LIST = [5, 10]          # for the waterfall scenarios


def show_page():
    st.header("🎯 Step 13: Final Results Dashboard")

    # Prerequisite checks ----------------------------------------------------
    required_states = ["regression_model", "model_results"]
    missing = [key for key in required_states if key not in st.session_state]
    if missing:
        st.error(f"⚠️ Missing previous steps: {', '.join(missing)}")
        st.info("Please complete Step 12 (Logistic Regression) first.")
        return

    # ✅ CRITICAL: Define BOTH variables from session state
    try:
        regression_model = st.session_state.regression_model
        model_results = st.session_state.model_results
    except Exception as e:
        st.error(f"❌ Error accessing session data: {str(e)}")
        return

    # Build impact dataframe
    try:
        coef_df = build_impact_df(model_results, regression_model)
    except Exception as e:
        st.error(f"❌ Error building impact data: {str(e)}")
        return

    if coef_df.empty:
        st.warning("⚠️ No impact data available to display.")
        return

    # Display overview metrics
    st.subheader("📊 Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Variables", len(coef_df))
    with col2:
        positive_vars = len(coef_df[coef_df["Impact_%"] > 0])
        st.metric("Positive Impact", positive_vars)
    with col3:
        total_impact = coef_df["Impact_%"].sum()
        st.metric("Total Impact", f"{total_impact:.1f}%")
    with col4:
        if not coef_df.empty:
            top_driver_impact = coef_df.iloc[0]["Impact_%"]
            st.metric("Top Driver", f"{top_driver_impact:.1f}%")

    # -------- Feature selector ---------------------------------------------
    st.subheader("✅ Select Final Set of Drivers")

    all_vars = coef_df["Variable"].tolist()
    if "final_vars" not in st.session_state:
        # default: top variables with positive impact
        positive_vars = coef_df[coef_df["Impact_%"] > 0]["Variable"].tolist()
        st.session_state.final_vars = positive_vars[:10]  # top 10 by default

    final_vars = st.multiselect(
        label="Choose variables to include in final analysis:",
        options=all_vars,
        default=st.session_state.final_vars,
        help="These will be reflected in the bar & waterfall charts."
    )
    st.session_state.final_vars = final_vars

    if len(final_vars) == 0:
        st.warning("⚠️ Select at least one variable to proceed.")
        return

    st.success(f"✅ {len(final_vars)} variables selected for final analysis")

    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("📊 Impact Bar Chart")
    
    try:
        bar_fig = make_impact_bar(coef_df, final_vars)
        st.plotly_chart(bar_fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error creating impact bar chart: {str(e)}")

    # -----------------------------------------------------------------------
    st.subheader("🚀 Waterfall Analysis - Uplift Scenarios")
    
    try:
        wf_fig = make_waterfall_chart(coef_df, final_vars, model_results)
        st.plotly_chart(wf_fig, use_container_width=True)
        
        # Add explanation
        with st.expander("📖 How to Interpret the Waterfall Chart"):
            st.write("""
            **Current Level**: The baseline performance of your target variable
            
            **Top 5 Metrics**: Shows projected improvement if you optimize the top 5 drivers by 10 percentage points each
            
            **Top 10 Metrics**: Shows projected improvement if you optimize the top 10 drivers by 10 percentage points each
            
            This helps you understand the potential business impact of focusing improvement efforts on your key drivers.
            """)
    except Exception as e:
        st.error(f"❌ Error creating waterfall chart: {str(e)}")

    # Summary insights
    st.subheader("💡 Key Insights & Recommendations")
    display_insights(coef_df, final_vars)


# === Helper Functions ======================================================


def build_impact_df(model_results: dict, regression_model) -> pd.DataFrame:
    """
    Combine coefficients + normalized impacts into one dataframe.
    """
    try:
        # Get coefficients from model
        selected_features = model_results.get("selected_features", [])
        if not selected_features:
            st.error("❌ No selected features found in model results")
            return pd.DataFrame()
        
        betas = regression_model.coef_[0]
        
        if len(selected_features) != len(betas):
            st.error("❌ Mismatch between features and coefficients")
            return pd.DataFrame()
        
        # Create base dataframe
        base_df = pd.DataFrame({
            "Variable": selected_features, 
            "Beta": betas
        })

        # Calculate normalized impact percentages
        base_df["Abs_Beta"] = base_df["Beta"].abs()
        total_abs_impact = base_df["Abs_Beta"].sum()
        
        if total_abs_impact == 0:
            st.warning("⚠️ All coefficients are zero - no impact to display")
            return pd.DataFrame()
        
        base_df["Impact_%"] = (base_df["Abs_Beta"] / total_abs_impact) * 100

        # Check for external normalized impacts
        if "normalized_impacts" in st.session_state:
            try:
                ext_imp = st.session_state.normalized_impacts
                ext_df = pd.DataFrame(list(ext_imp.items()), columns=["Variable", "External_Impact_%"])
                base_df = base_df.merge(ext_df, on="Variable", how="left")
                # Use external impacts if available, otherwise use calculated
                base_df["Impact_%"] = base_df["External_Impact_%"].fillna(base_df["Impact_%"])
                base_df.drop(columns=["External_Impact_%"], inplace=True)
            except Exception:
                pass  # Continue with calculated impacts

        # Keep sign information for waterfall
        base_df["Signed_Impact"] = np.where(
            base_df["Beta"] > 0, 
            base_df["Impact_%"], 
            -base_df["Impact_%"]
        )
        
        return base_df.sort_values("Impact_%", ascending=False).reset_index(drop=True)
    
    except Exception as e:
        st.error(f"❌ Error in build_impact_df: {str(e)}")
        return pd.DataFrame()


def make_impact_bar(impact_df: pd.DataFrame, picked_vars: list):
    """
    Create horizontal bar chart with main drivers + Others bucket.
    """
    if impact_df.empty or not picked_vars:
        return go.Figure()
    
    # Filter to selected variables and positive impacts only
    df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
    df = df[df["Impact_%"] > 0]
    
    if df.empty:
        st.warning("⚠️ No positive impact variables to display")
        return go.Figure()
    
    df = df.sort_values("Impact_%", ascending=False)

    # Split into main drivers and others
    main = df[df["Impact_%"] >= MIN_OTHER_BUCKET]
    others = df[df["Impact_%"] < MIN_OTHER_BUCKET]
    
    plot_df = main.copy()
    
    # Add Others bucket if there are small drivers
    if not others.empty:
        others_row = pd.DataFrame({
            "Variable": [f"Others ({len(others)} drivers)"],
            "Impact_%": [others["Impact_%"].sum()],
            "Beta": [0],  # placeholder
            "Abs_Beta": [0],  # placeholder
            "Signed_Impact": [others["Impact_%"].sum()]
        })
        plot_df = pd.concat([plot_df, others_row], ignore_index=True)

    # Create colors
    colors = ["#2E86AB" if not var.startswith("Others") else "#A23B72" 
              for var in plot_df["Variable"]]

    # Create the bar chart
    fig = px.bar(
        plot_df,
        x="Impact_%", 
        y="Variable",
        orientation="h",
        color=colors,
        color_discrete_map="identity",
        height=max(400, 50 * len(plot_df))
    )
    
    # Add percentage labels
    fig.update_traces(
        texttemplate="%{x:.1f}%", 
        textposition="outside",
        showlegend=False
    )
    
    fig.update_layout(
        title="Driver Impact Analysis - Normalized Importance (%)",
        xaxis_title="Impact Contribution (%)",
        yaxis_title="",
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        margin=dict(l=200)  # More space for long variable names
    )
    
    return fig


def make_waterfall_chart(impact_df: pd.DataFrame, picked_vars: list, model_results: dict):
    """
    Create waterfall chart showing current level vs optimized scenarios.
    """
    if impact_df.empty or not picked_vars:
        return go.Figure()
    
    try:
        # Get baseline from test data
        y_test = model_results.get("y_test")
        if y_test is None:
            st.error("❌ No test data available for baseline calculation")
            return go.Figure()
        
        current_level = y_test.mean()
        
        # Filter and sort selected variables
        df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
        df = df[df["Impact_%"] > 0]  # Only positive impacts
        df = df.sort_values("Impact_%", ascending=False)
        
        if df.empty:
            st.warning("⚠️ No positive impact variables for waterfall")
            return go.Figure()
        
        # Calculate uplift scenarios
        # Assumption: 10% improvement in top drivers translates to impact proportional to their coefficients
        impacts = []
        for n in TOP_N_LIST:
            top_n = df.head(min(n, len(df)))
            # Simple approach: sum of top N impacts as proportion
            uplift = (top_n["Impact_%"].sum() / 100) * 0.1  # 10% of normalized impact
            impacts.append(uplift)
        
        # Calculate projected levels
        levels = [
            current_level,
            current_level + impacts[0],
            current_level + impacts[1] if len(impacts) > 1 else current_level + impacts[0]
        ]
        
        labels = [
            "Current Level",
            f"Optimizing Top {TOP_N_LIST[0]} Drivers",
            f"Optimizing Top {TOP_N_LIST[1]} Drivers" if len(TOP_N_LIST) > 1 else f"Optimizing Top {TOP_N_LIST[0]} Drivers"
        ]
        
        # Create waterfall chart
        fig = go.Figure()
        
        colors = ["#3E4B8B", "#28A745", "#17A2B8"]
        
        for i, (label, level) in enumerate(zip(labels, levels)):
            fig.add_trace(go.Bar(
                x=[label],
                y=[level * 100],  # Convert to percentage
                marker_color=colors[i % len(colors)],
                text=[f"{level*100:.1f}%"],
                textposition="outside",
                width=0.5,
                showlegend=False
            ))
        
        # Add connecting arrows
        for i in range(len(levels) - 1):
            fig.add_annotation(
                x=i + 0.4, y=levels[i] * 100,
                ax=i + 0.6, ay=levels[i+1] * 100,
                arrowhead=2, arrowsize=1.5, arrowwidth=2,
                arrowcolor="gray",
                showarrow=True
            )
        
        fig.update_layout(
            title="Impact Waterfall: Current vs. Optimized Performance",
            yaxis_title="Target Variable Level (%)",
            xaxis_title="Scenario",
            height=500,
            yaxis=dict(range=[0, max(levels) * 100 * 1.2])
        )
        
        return fig
    
    except Exception as e:
        st.error(f"❌ Error creating waterfall chart: {str(e)}")
        return go.Figure()


def display_insights(impact_df: pd.DataFrame, final_vars: list):
    """Display key insights and recommendations."""
    
    if impact_df.empty or not final_vars:
        st.info("No insights available - please select variables first.")
        return
    
    # Filter to selected variables
    selected_df = impact_df[impact_df["Variable"].isin(final_vars)]
    positive_df = selected_df[selected_df["Impact_%"] > 0].sort_values("Impact_%", ascending=False)
    
    if positive_df.empty:
        st.info("No positive impact drivers selected.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏆 Top 3 Key Drivers:**")
        for i, (_, row) in enumerate(positive_df.head(3).iterrows(), 1):
            st.write(f"{i}. **{row['Variable']}**: {row['Impact_%']:.1f}% impact")
    
    with col2:
        st.write("**📈 Optimization Potential:**")
        top_5_impact = positive_df.head(5)["Impact_%"].sum()
        st.write(f"• Top 5 drivers represent {top_5_impact:.1f}% of total impact")
        
        if len(positive_df) > 5:
            remaining_impact = positive_df.iloc[5:]["Impact_%"].sum()
            st.write(f"• Remaining drivers: {remaining_impact:.1f}% of total impact")
    
    # Recommendations
    st.write("**🎯 Strategic Recommendations:**")
    
    if top_5_impact > 60:
        st.success("✅ **Focus Strategy**: The top 5 drivers account for most impact. Concentrate efforts here for maximum ROI.")
    elif top_5_impact > 40:
        st.info("📊 **Balanced Strategy**: Impact is moderately concentrated. Focus on top drivers while monitoring others.")
    else:
        st.warning("⚠️ **Broad Strategy**: Impact is distributed across many drivers. Consider a comprehensive improvement approach.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    show_page()
