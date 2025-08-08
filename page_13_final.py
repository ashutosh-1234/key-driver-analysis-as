# ---------------------- page_13_final.py ----------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

MIN_OTHER_BUCKET = 5.0  # % threshold for grouping as "Others"
TOP_N_LIST = [5, 10]    # for waterfall

def show_page():
    st.header("🎯 Step 13: Final Key Driver Summary")

    model, X_test, y_test, selected_features = check_prerequisites()
    if model is None:
        return

    # Map factor impacts to raw features using loadings from Step 9
    impact_df = get_raw_feature_impact_df(model, selected_features)

    st.subheader("✅ Select drivers for summary")
    all_vars = impact_df["Variable"].tolist()

    if "final_vars" not in st.session_state:
        st.session_state.final_vars = impact_df[impact_df["Impact_%"] > 0]["Variable"].tolist()

    final_vars = st.multiselect(
        "Choose variables",
        options=all_vars, default=st.session_state.final_vars,
        help="Selected drivers are reflected in the bar & waterfall charts."
    )
    st.session_state.final_vars = final_vars

    if not final_vars:
        st.warning("Select at least one variable to proceed.")
        return

    st.divider()
    st.subheader("📊 Impact Bar Chart")
    bar_fig = make_impact_bar(impact_df, final_vars)
    st.plotly_chart(bar_fig, use_container_width=True)

    st.subheader("🚀 Waterfall: 'What-if' uplift scenarios")
    wf_fig = make_waterfall_chart(impact_df, final_vars, y_test)
    st.plotly_chart(wf_fig, use_container_width=True)
    st.caption(
        "Bar = % contribution when all selected drivers improve by 10 p.p.; "
        "Waterfall = projected Top-5 and Top-10 uplift vs. current level."
    )

def check_prerequisites():
    model = None
    selected_features = None
    X_test = None
    y_test = None

    if hasattr(st.session_state, "last_trained_model"):
        model = st.session_state.last_trained_model
    elif hasattr(st.session_state, "regression_model"):
        model = st.session_state.regression_model

    if hasattr(st.session_state, "model_results") and st.session_state.model_results:
        results = st.session_state.model_results
        if 'regression_model' in results:
            model = results['regression_model']
        if 'selected_features' in results:
            selected_features = results['selected_features']
        if 'X_test' in results:
            X_test = results['X_test']
        if 'y_test' in results:
            y_test = results['y_test']

    if model is None:
        st.error("⚠️ Missing trained model from Step 12.")
        st.info("Please complete Step 12 (Logistic Regression Analysis) first.")
        return None, None, None, None

    if selected_features is None:
        if hasattr(st.session_state, "sel_factored") and hasattr(st.session_state, "sel_raw"):
            selected_features = st.session_state.sel_factored + st.session_state.sel_raw
        else:
            st.error("⚠️ Missing selected features information.")
            st.info("Please complete Step 12 variable selection.")
            return None, None, None, None

    if y_test is None:
        if hasattr(st.session_state, "y_target"):
            y_test = st.session_state.y_target

    return model, X_test, y_test, selected_features

def get_raw_feature_impact_df(model, selected_features):
    model_coefs = np.asarray(model.coef_).flatten()
    feat_types = ["Factored" if "Factor" in f else "Raw" for f in selected_features]

    df = pd.DataFrame({"Variable": selected_features, "Beta": model_coefs, "Type": feat_types})
    df["Abs_Beta"] = df["Beta"].abs()
    total_abs = df["Abs_Beta"].sum()
    df["Impact_%"] = 0.0 if total_abs == 0 else (df["Abs_Beta"] / total_abs) * 100

    fa_results = st.session_state.get("fa_results", {})
    raw_impact_dict = {}

    for _, row in df[df.Type == "Factored"].iterrows():
        factor_var = row["Variable"]
        factor_impact = row["Impact_%"]
        found = False
        for cat, res in fa_results.items():
            if not res.get('success'):
                continue
            loadings = res.get('loadings')
            if loadings is None or not isinstance(loadings, pd.DataFrame):
                continue
            factor_cols = [col for col in loadings.columns if "Factor" in str(col)]
            for fc in factor_cols:
                if factor_var == f"{cat}_{fc}":
                    feature_col = loadings.columns[0]  # Raw feature names column
                    for i, rfeat in enumerate(loadings[feature_col]):
                        loading_val = abs(loadings.iloc[i][fc])
                        if loading_val > 0:
                            raw_impact_dict[rfeat] = raw_impact_dict.get(rfeat, 0) + factor_impact * loading_val
                    found = True
                    break
            if found:
                break
        if not found:
            st.warning(f"Loadings not found for {factor_var}, skipping.")

    for _, row in df[df.Type == "Raw"].iterrows():
        raw_feat = row["Variable"]
        impact = row["Impact_%"]
        raw_impact_dict[raw_feat] = raw_impact_dict.get(raw_feat, 0) + impact

    if not raw_impact_dict:
        st.warning("⚠️ No raw-feature mapping found: defaulting to model input variables.")
        return df[["Variable", "Impact_%"]].sort_values("Impact_%", ascending=False).reset_index(drop=True)

    raw_df = pd.DataFrame(list(raw_impact_dict.items()), columns=["Variable", "Impact_%"])
    tot = raw_df["Impact_%"].sum()
    if tot != 0:
        raw_df["Impact_%"] = raw_df["Impact_%"] / tot * 100
    else:
        raw_df["Impact_%"] = 0.0
    raw_df = raw_df.sort_values("Impact_%", ascending=False).reset_index(drop=True)
    return raw_df

def make_impact_bar(impact_df: pd.DataFrame, picked_vars: list):
    try:
        df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
        df = df[df["Impact_%"] > 0]
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
        colors = ["#2E86AB" if not var.startswith("Others") else "#A23B72"
                  for var in plot_df["Variable"]]
        fig = px.bar(
            plot_df,
            x="Impact_%", y="Variable",
            orientation="h",
            text="Impact_%",
            height=max(400, 40 * len(plot_df)),
            title="Normalized Impact of Selected Drivers (Raw Features)"
        )
        fig.update_traces(marker_color=colors, texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(xaxis_title="Impact (%)", yaxis_title="", showlegend=False)
        return fig
    except Exception as e:
        st.error(f"Error creating bar chart: {str(e)}")
        return go.Figure()

def make_waterfall_chart(impact_df: pd.DataFrame, picked_vars: list, y_target):
    try:
        current = y_target.mean() if hasattr(y_target, 'mean') else np.mean(y_target)
        df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
        df = df.sort_values("Impact_%", ascending=False)
        if df.empty:
            st.warning("No variables available for waterfall chart.")
            return go.Figure()
        impacts = []
        for n in TOP_N_LIST:
            uplift = df.head(n)["Impact_%"].sum() / 100
            scaled_uplift = uplift * 0.1  # 10% improvement assumption
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
                y=[val * 100],
                marker_color=colors[i],
                text=[f"{val*100:.1f}%"],
                textposition="outside",
                width=0.5
            ))
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
            title="Projected Lift with Driver Optimization (Raw Features)",
            yaxis_title="Top-2-Box Level (%)",
            xaxis_title="Scenario",
            showlegend=False,
            bargap=0.6,
            height=500
        )
        max_val = max([v * 100 for v in levels])
        fig.update_yaxes(range=[0, max_val * 1.15])
        return fig
    except Exception as e:
        st.error(f"Error creating waterfall chart: {str(e)}")
        return go.Figure()

if __name__ == "__main__":
    show_page()
