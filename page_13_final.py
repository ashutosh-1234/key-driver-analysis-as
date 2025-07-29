# ------------------- page_13_final.py -------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Constants
MIN_OTHER_BUCKET = 5.0  # Impact % threshold for grouping as "Others"
TOP_N_LIST = [5, 10]   # For waterfall impact scenarios

def show_page():
    st.header("🎯 Step 13 | Final Results Dashboard")

    # Check for previous step dependencies
    required_keys = ['regression_model', 'model_results']
    missing_keys = [k for k in required_keys if k not in st.session_state]
    if missing_keys:
        st.error(f"⚠️ Missing previous computations: {', '.join(missing_keys)}. Please complete prior steps.")
        return

    # Retrieve model and results
    regression_model = st.session_state['regression_model']
    model_results = st.session_state['model_results']

    # Build impact DataFrame from model coefficients
    coef_df = build_impact_df(model_results, regression_model)

    # Let user select final drivers
    st.subheader("✅ Select Final Set of Drivers for Reporting")
    all_vars = coef_df["Variable"].tolist()
    if 'final_driver_vars' not in st.session_state:
        st.session_state['final_driver_vars'] = all_vars

    final_vars = st.multiselect(
        "Choose variables to include in final report:",
        options=all_vars,
        default=st.session_state['final_driver_vars'],
        help="Select variables that you consider key drivers."
    )
    st.session_state['final_driver_vars'] = final_vars

    if len(final_vars) == 0:
        st.warning("Please select at least one driver to proceed.")
        return

    st.divider()

    # Impact Bar Chart
    st.subheader("📊 Impact of Selected Drivers")
    bar_fig = make_impact_bar(coef_df, final_vars)
    st.plotly_chart(bar_fig, use_container_width=True)

    # Waterfall Chart
    st.subheader("🚀 Impact Waterfall Analysis")
    wf_fig = make_waterfall_chart(coef_df, final_vars, model_results)
    st.plotly_chart(wf_fig, use_container_width=True)

    st.caption("**Note:** Impact bars show % contribution; waterfall visualizes incremental uplift.")

# Helper functions

def build_impact_df(model_results, regression_model):
    """
    Compose variable impact DataFrame from model and optionally normalized impacts.
    """
    # Use the selected features from prior step
    vars_ = model_results['selected_features']
    betas = regression_model.coef_[0]
    df_impacts = pd.DataFrame({"Variable": vars_, "Beta": betas})

    # Calculate absolute impacts for normalization
    df_impacts["Abs_Beta"] = abs(df_impacts["Beta"])
    total_impact = df_impacts["Abs_Beta"].sum()
    df_impacts["Impact_%"] = (df_impacts["Abs_Beta"] / total_impact) * 100

    # Store sign for impact direction in waterfall
    df_impacts["Signed_Impact"] = np.where(df_impacts["Beta"] > 0, df_impacts["Impact_%"], -df_impacts["Impact_%"])
    # Sort by impact for clarity
    df_impacts = df_impacts.sort_values("Impact_%", ascending=False).reset_index(drop=True)
    return df_impacts

def make_impact_bar(impact_df, picked_vars):
    """
    Create impact bar plot with 'Others' bucket if applicable.
    """
    df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
    df = df[df["Impact_%"] > 0]  # focus on positive impacts for bar
    df = df.sort_values("Impact_%", ascending=False)

    main_vars = df[df["Impact_%"] >= MIN_OTHER_BUCKET]
    other_vars = df[df["Impact_%"] < MIN_OTHER_BUCKET]
    if len(other_vars) > 0:
        other_sum = other_vars["Impact_%"].sum()
        other_row = pd.DataFrame({"Variable": [f"Others ({len(other_vars)})"], "Impact_%": [other_sum]})
        plot_df = pd.concat([main_vars, other_row], ignore_index=True)
    else:
        plot_df = main_vars.copy()

    fig = px.bar(
        plot_df,
        x="Impact_%",
        y="Variable",
        orientation="h",
        text="Impact_%",
        color=np.where(plot_df["Variable"].str.startswith("Others"), "Others", "Driver"),
        color_discrete_map={"Driver": "#2E86AB", "Others": "#A23B72"},
        height=max(400, 40 * len(plot_df))
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        xaxis_title="Impact (%)",
        yaxis_title="",
        title="Impact Contribution of Drivers",
        showlegend=False,
        margin=dict(l=150)
    )
    return fig

def make_waterfall_chart(impact_df, picked_vars, model_results):
    """
    Generate a waterfall chart for impact scenarios: Top 5 & 10 drivers uplift.
    """
    # Baseline (mean of baseline target)
    y_test = model_results['y_test']
    baseline_level = y_test.mean()

    df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
    df = df.sort_values("Impact_%", ascending=False)

    impacts = []
    for n in TOP_N_LIST:
        impacts.append(df.head(n)["Impact_%"].sum() / 100)  # back to proportion

    levels = [baseline_level, baseline_level + impacts[0], baseline_level + impacts[1]]
    labels = [
        "Current Level",
        f"Top {TOP_N_LIST[0]} (+10% uplift in impact)",
        f"Top {TOP_N_LIST[1]} (+10
