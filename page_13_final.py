# ---------------------- page_13_final.py ----------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------- CONFIG ----------
MIN_OTHER_BUCKET = 5.0  # % threshold below which drivers are grouped as "Others"
TOP_N_LIST = [5, 10]    # for the waterfall scenarios

def show_page():
    st.header("🎯 Step 13: Final Key Driver Summary")
    
    # More flexible prerequisite checks
    required_data = check_prerequisites()
    if not required_data:
        return
        
    model, X_test, y_test, selected_features = required_data
    
    # Build impact dataframe
    coef_df = build_impact_df(model, selected_features)
    
    # Feature selector
    st.subheader("✅ Select the final set of drivers to include")
    all_vars = coef_df["Variable"].tolist()
    
    if "final_vars" not in st.session_state:
        # Default: all variables with positive impact
        st.session_state.final_vars = coef_df[coef_df["Impact_%"] > 0]["Variable"].tolist()
    
    final_vars = st.multiselect(
        label="Choose variables",
        options=all_vars,
        default=st.session_state.final_vars,
        help="These will be reflected in the bar & waterfall charts."
    )
    
    st.session_state.final_vars = final_vars
    
    if len(final_vars) == 0:
        st.warning("Select at least one variable to proceed.")
        return
    
    # Charts
    st.divider()
    st.subheader("📊 Impact Bar Chart")
    bar_fig = make_impact_bar(coef_df, final_vars)
    st.plotly_chart(bar_fig, use_container_width=True)
    
    st.subheader("🚀 Waterfall – 'What-if' uplift scenarios")
    wf_fig = make_waterfall_chart(coef_df, final_vars, y_test)
    st.plotly_chart(wf_fig, use_container_width=True)
    
    st.caption(
        "Bar = % contribution when all selected drivers improve by 10 p.p.; "
        "Waterfall = projected Top-5 and Top-10 uplift vs. current level."
    )

def check_prerequisites():
    """Check for required data from Step 12 in various possible locations."""
    
    # Try multiple possible sources for the model and results
    model = None
    selected_features = None
    X_test = None
    y_test = None
    
    # Option 1: Direct from session state (new format)
    if hasattr(st.session_state, 'last_trained_model'):
        model = st.session_state.last_trained_model
    elif hasattr(st.session_state, 'regression_model'):
        model = st.session_state.regression_model
    
    # Option 2: From model_results dict
    if hasattr(st.session_state, 'model_results') and st.session_state.model_results:
        results = st.session_state.model_results
        if 'regression_model' in results:
            model = results['regression_model']
        if 'selected_features' in results:
            selected_features = results['selected_features']
        if 'X_test' in results:
            X_test = results['X_test']
        if 'y_test' in results:
            y_test = results['y_test']
    
    # Option 3: Try to reconstruct from available data
    if model is None:
        st.error("⚠️ Missing trained model from Step 12.")
        st.info("Please complete Step 12 (Logistic Regression Analysis) first.")
        return None
    
    # Get selected features if not available
    if selected_features is None:
        if hasattr(st.session_state, 'sel_factored') and hasattr(st.session_state, 'sel_raw'):
            selected_features = st.session_state.sel_factored + st.session_state.sel_raw
        else:
            st.error("⚠️ Missing selected features information.")
            st.info("Please complete Step 12 variable selection first.")
            return None
    
    # Get test data if not available - reconstruct if needed
    if y_test is None:
        if hasattr(st.session_state, 'y_target'):
            y_test = st.session_state.y_target
        else:
            st.error("⚠️ Missing target variable data.")
            return None
    
    return model, X_test, y_test, selected_features

def build_impact_df(model, selected_features: list) -> pd.DataFrame:
    """Build impact dataframe from model coefficients."""
    try:
        # Get coefficients
        if hasattr(model, 'coef_'):
            betas = model.coef_[0]
        else:
            st.error("Invalid model object - no coefficients found.")
            return pd.DataFrame()
        
        # Ensure we have the right number of features
        if len(selected_features) != len(betas):
            st.warning(f"Feature count mismatch: {len(selected_features)} features vs {len(betas)} coefficients")
            min_len = min(len(selected_features), len(betas))
            selected_features = selected_features[:min_len]
            betas = betas[:min_len]
        
        base_df = pd.DataFrame({
            "Variable": selected_features, 
            "Beta": betas
        })
        
        # Calculate impact percentages
        base_df["Abs_Beta"] = base_df["Beta"].abs()
        total_abs_impact = base_df["Abs_Beta"].sum()
        
        if total_abs_impact == 0:
            base_df["Impact_%"] = 0
        else:
            base_df["Impact_%"] = (base_df["Abs_Beta"] / total_abs_impact) * 100
        
        # Check for external normalized impacts
        if hasattr(st.session_state, 'normalized_impacts') and st.session_state.normalized_impacts:
            ext_imp = st.session_state.normalized_impacts
            ext_df = pd.DataFrame(list(ext_imp.items()), columns=["Variable", "Impact_%"])
            base_df = (
                base_df.drop(columns=["Impact_%"])
                .merge(ext_df, on="Variable", how="left")
                .fillna(0)
            )
        
        # Signed impact for waterfall
        base_df["Signed_Impact"] = np.where(
            base_df["Beta"] > 0, 
            base_df["Impact_%"], 
            -base_df["Impact_%"]
        )
        
        return base_df.sort_values("Impact_%", ascending=False).reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Error building impact dataframe: {str(e)}")
        return pd.DataFrame()

def make_impact_bar(impact_df: pd.DataFrame, picked_vars: list):
    """Horizontal bar chart: top drivers ≥ MIN_OTHER_BUCKET plus 'Others'."""
    try:
        df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
        df = df[df["Impact_%"] > 0]  # only positive
        df = df.sort_values("Impact_%", ascending=False)
        
        if df.empty:
            st.warning("No positive impact variables to display.")
            return go.Figure()
        
        main = df[df["Impact_%"] >= MIN_OTHER_BUCKET]
        others = df[df["Impact_%"] < MIN_OTHER_BUCKET]
        
        if not others.empty:
            others_row = pd.DataFrame({
                "Variable": [f"Others ({len(others)})"],
                "Impact_%": [others["Impact_%"].sum()]
            })
            plot_df = pd.concat([main, others_row], ignore_index=True)
        else:
            plot_df = main.copy()
        
        # Create color mapping
        colors = []
        for var in plot_df["Variable"]:
            if var.startswith("Others"):
                colors.append("#A23B72")  # Others color
            else:
                colors.append("#2E86AB")  # Driver color
        
        fig = px.bar(
            plot_df,
            x="Impact_%", y="Variable",
            orientation="h",
            text="Impact_%",
            height=max(400, 40 * len(plot_df)),
            title="Normalized Impact of Selected Drivers"
        )
        
        # Update colors
        fig.update_traces(
            marker_color=colors,
            texttemplate="%{text:.1f}%", 
            textposition="outside"
        )
        
        fig.update_layout(
            xaxis_title="Impact (%)",
            yaxis_title="",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating bar chart: {str(e)}")
        return go.Figure()

def make_waterfall_chart(impact_df: pd.DataFrame, picked_vars: list, y_target):
    """Build a 3-column waterfall: current level vs +Top5 vs +Top10."""
    try:
        # Calculate current level
        if hasattr(y_target, 'mean'):
            current = y_target.mean()
        else:
            current = np.mean(y_target)
        
        df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
        df = df.sort_values("Impact_%", ascending=False)
        
        if df.empty:
            st.warning("No variables available for waterfall chart.")
            return go.Figure()
        
        # Calculate impacts for top N
        impacts = []
        for n in TOP_N_LIST:
            top_n = df.head(n)
            uplift = top_n["Impact_%"].sum() / 100  # Convert % to proportion
            # Scale the uplift (assuming 10% improvement in drivers leads to this impact)
            scaled_uplift = uplift * 0.1  # 10% improvement factor
            impacts.append(scaled_uplift)
        
        levels = [current, current + impacts[0], current + impacts[1]]
        labels = [
            "Current Level",
            f"10% ↑ in Top {TOP_N_LIST[0]}",
            f"10% ↑ in Top {TOP_N_LIST[1]}"
        ]
        
        fig = go.Figure()
        colors = ["#3E4B8B", "#28A745", "#17A2B8"]
        
        for i, (lab, val) in enumerate(zip(labels, levels)):
            fig.add_trace(go.Bar(
                x=[lab], 
                y=[val * 100],  # Show as percentage
                marker_color=colors[i],
                text=[f"{val*100:.1f}%"],
                textposition="outside",
                width=0.5
            ))
        
        # Add connecting lines
        for i in range(len(levels) - 1):
            fig.add_annotation(
                x=i + 0.5, y=levels[i] * 100,
                ax=i, ay=levels[i] * 100,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowcolor="gray"
            )
        
        fig.update_layout(
            title="Projected Lift with Driver Optimization",
            yaxis_title="Top-2-Box Level (%)",
            xaxis_title="Scenario",
            showlegend=False,
            bargap=0.6,
            height=500
        )
        
        max_val = max([v*100 for v in levels])
        fig.update_yaxes(range=[0, max_val * 1.15])
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating waterfall chart: {str(e)}")
        return go.Figure()

if __name__ == "__main__":
    show_page()
