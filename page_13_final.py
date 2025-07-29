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
    st.header("🎯 Step 13 | Final Results Dashboard")

    # Prerequisite checks ----------------------------------------------------
    required_states = ["regression_model", "model_results"]
    missing = [key for key in required_states if key not in st.session_state]
    if missing:
        st.error(f"⚠️  Missing previous steps: {', '.join(missing)}")
        st.stop()

    model_results = st.session_state.model_results
    coef_df = build_impact_df(model_results)

    # -------- Feature selector ---------------------------------------------
    st.subheader("✅ Select the final set of drivers to include")

    all_vars = coef_df["Variable"].tolist()
    if "final_vars" not in st.session_state:
        # default: everything that has positive impact
        st.session_state.final_vars = all_vars

    final_vars = st.multiselect(
        label="Choose variables",
        options=all_vars,
        default=st.session_state.final_vars,
        help="These will be reflected in the bar & waterfall charts."
    )
    st.session_state.final_vars = final_vars

    if len(final_vars) == 0:
        st.warning("Select at least one variable to proceed.")
        st.stop()

    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("📊 Impact Bar Chart")
    bar_fig = make_impact_bar(coef_df, final_vars)
    st.plotly_chart(bar_fig, use_container_width=True)

    # -----------------------------------------------------------------------
    st.subheader("🚀 Waterfall – ‘What-if’ uplift scenarios")
    wf_fig = make_waterfall_chart(coef_df, final_vars, model_results)
    st.plotly_chart(wf_fig, use_container_width=True)

    st.caption(
        "Bar = % contribution when all selected drivers improve by 10 p.p.; "
        "Waterfall = projected Top-5 and Top-10 uplift vs. current level."
    )


# === Helpers ===============================================================


def build_impact_df(model_results: dict) -> pd.DataFrame:
    """
    Combine coefficients + st.session_state.normalized_impacts (if present)
    into one tidy dataframe with positive-only ‘Impact_%’.
    """
    # 1/ start from coefficients
    coef = model_results["selected_features"]
    betas = model_results["regression_model"].coef_[0]
    base_df = pd.DataFrame({"Variable": coef, "Beta": betas})

    # 2/ normalise to % share of absolute impact
    base_df["Abs_Beta"] = base_df["Beta"].abs()
    base_df["Impact_%"] = (base_df["Abs_Beta"] / base_df["Abs_Beta"].sum()) * 100

    # (optional) replace by external normalised impacts if provided
    if "normalized_impacts" in st.session_state:
        ext_imp = st.session_state.normalized_impacts
        ext_df = pd.DataFrame(list(ext_imp.items()), columns=["Variable", "Impact_%"])
        base_df = (
            base_df.drop(columns=["Impact_%"])
            .merge(ext_df, on="Variable", how="left")
            .fillna(0)
        )

    # Retain sign for later water-fall but bar uses only positives
    base_df["Signed_Impact"] = np.where(base_df["Beta"] > 0, base_df["Impact_%"], -base_df["Impact_%"])
    return base_df.sort_values("Impact_%", ascending=False).reset_index(drop=True)


def make_impact_bar(impact_df: pd.DataFrame, picked_vars: list):
    """
    Horizontal bar chart: top drivers ≥ MIN_OTHER_BUCKET plus ‘Others’.
    """
    df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
    df = df[df["Impact_%"] > 0]                          # only positive
    df = df.sort_values("Impact_%", ascending=False)

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

    fig = px.bar(
        plot_df,
        x="Impact_%", y="Variable",
        orientation="h",
        text="Impact_%",  # auto-labels
        color=np.where(plot_df["Variable"].str.startswith("Others"), "Others", "Driver"),
        color_discrete_map={"Driver": "#2E86AB", "Others": "#A23B72"},
        height=max(400, 40 * len(plot_df))
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        xaxis_title="Impact (%)",
        yaxis_title="",
        title="Normalized Impact of Selected Drivers",
        showlegend=False
    )
    return fig


def make_waterfall_chart(impact_df: pd.DataFrame, picked_vars: list, model_results: dict):
    """
    Build a 3-column waterfall: current level vs +Top5 vs +Top10 (10 % lift each).
    """
    # Baseline = current top-2-box level passed to regression earlier
    y_target = model_results["y_test"]   # we just need mean -> any split ok
    current = y_target.mean()

    df = impact_df[impact_df["Variable"].isin(picked_vars)].copy()
    df = df.sort_values("Impact_%", ascending=False)

    impacts = []
    for n in TOP_N_LIST:
        uplift = df.head(n)["Impact_%"].sum() / 100          # back to proportion
        impacts.append(uplift)

    levels = [current, current + impacts[0], current + impacts[1]]
    labels = [
        "Current Level",
        f"10 % ↑ in Top {TOP_N_LIST[0]}",
        f"10 % ↑ in Top {TOP_N_LIST[1]}"
    ]

    fig = go.Figure()
    colors = ["#3E4B8B", "#28A745", "#17A2B8"]
    for i, (lab, val) in enumerate(zip(labels, levels)):
        fig.add_trace(go.Bar(
            x=[lab], y=[val * 100],   # show as %
            marker_color=colors[i],
            text=[f"{val*100:.0f}%"],
            textposition="outside",
            width=0.5
        ))

    # connectors
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
        title="Projected Lift with Driver Optimisation",
        yaxis_title="Top-2-Box Level (%)",
        xaxis_title="Scenario",
        showlegend=False,
        bargap=0.6,
        height=500
    )
    fig.update_yaxes(range=[0, max([v*100 for v in levels]) * 1.15])
    return fig


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    show_page()
