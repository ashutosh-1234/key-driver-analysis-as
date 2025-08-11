# ---------------------- page_13_final.py ----------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------- CONFIG ----------
MIN_OTHER_BUCKET = 0.0  # % threshold below which drivers are grouped as "Others"
TOP_N_LIST = [5, 10]    # for the waterfall scenarios

def show_page():
    st.header("🎯 Step 13: Final Key Driver Summary")
    
    # Check prerequisites
    required_data = check_prerequisites()
    if not required_data:
        return
        
    model, X, y, selected_features = required_data
    
    # Build impact dataframe - mapping to raw features using Step 10 loadings
    coef_df = build_impact_df_with_raw_features(model, selected_features)
    
    if coef_df.empty:
        st.error("❌ No impact data available. Please check that factor analysis was completed successfully.")
        return
    
    # Filter only selected variables (Impact_% > 0 after exclusion)
    final_vars = coef_df.loc[coef_df["Impact_%"] > 0, "Variable"].tolist()
    
    if len(final_vars) == 0:
        st.warning("Select at least one variable to proceed.")
        return
    
    # Charts
    st.divider()
    st.subheader("📊 Impact Bar Chart")
    bar_fig = make_impact_bar(coef_df, final_vars)
    st.plotly_chart(bar_fig, use_container_width=True)
    
    st.subheader("🚀 Waterfall – 'What-if' uplift scenarios")
    wf_fig = make_waterfall_chart(coef_df, final_vars, y)
    st.plotly_chart(wf_fig, use_container_width=True)
    
    st.caption(
        "Bar = % contribution when all selected drivers improve by 10 p.p.; "
        "Waterfall = projected Top-5 and Top-10 uplift vs. current level."
    )

def check_prerequisites():
    """Check for required data from Step 12."""
    model = None
    selected_features = None
    X = None
    y = None
    
    if hasattr(st.session_state, 'last_trained_model'):
        model = st.session_state.last_trained_model
    elif hasattr(st.session_state, 'regression_model'):
        model = st.session_state.regression_model
    
    if hasattr(st.session_state, 'model_results') and st.session_state.model_results:
        results = st.session_state.model_results
        if 'regression_model' in results:
            model = results['regression_model']
        if 'selected_features' in results:
            selected_features = results['selected_features']
        if 'X' in results:
            X = results['X']
        if 'y' in results:
            y = results['y']
    
    if model is None:
        st.error("⚠️ Missing trained model from Step 12.")
        return None
    
    if selected_features is None:
        if hasattr(st.session_state, 'sel_factored') and hasattr(st.session_state, 'sel_raw'):
            selected_features = st.session_state.sel_factored + st.session_state.sel_raw
        else:
            st.error("⚠️ Missing selected features information.")
            return None
    
    if y is None:
        if hasattr(st.session_state, 'y_target'):
            y = st.session_state.y_target
        else:
            st.error("⚠️ Missing target variable data.")
            return None
    
    return model, X, y, selected_features

def build_impact_df_with_raw_features(model, selected_features: list) -> pd.DataFrame:
    """Build impact dataframe mapping factors back to raw features using Step 10 loadings."""
    try:
        # Get coefficients
        if hasattr(model, 'coef_'):
            betas = model.coef_[0]
        else:
            st.error("Invalid model object - no coefficients found.")
            return pd.DataFrame()

        # Try to get p-values from statsmodels results if available
        pvalues = [np.nan] * len(selected_features)
        if 'sm_model_results' in getattr(st.session_state, 'model_results', {}):
            sm_res = st.session_state.model_results['sm_model_results']
            if hasattr(sm_res, 'pvalues'):
                pval_series = sm_res.pvalues
                pvalues = [pval_series.get(var, np.nan) for var in selected_features]

        # Ensure alignment
        if len(selected_features) != len(betas):
            min_len = min(len(selected_features), len(betas))
            selected_features = selected_features[:min_len]
            betas = betas[:min_len]
            pvalues = pvalues[:min_len]

        base_df = pd.DataFrame({
            "Variable": selected_features,
            "Beta": betas,
            "p_value": pvalues,
            "Type": ["Factored" if "Factor" in var else "Raw" for var in selected_features]
        })

        # --- User selection ---
        st.subheader("📌 Logistic Regression Coefficients & p-values")
        if "var_inclusion" not in st.session_state:
            st.session_state.var_inclusion = {var: True for var in base_df["Variable"]}

        for idx, row in base_df.iterrows():
            label = f"{row['Variable']} | Beta={row['Beta']:.4f}"
            if not np.isnan(row['p_value']):
                label += f", p={row['p_value']:.4f}"
            else:
                label += ", p=N/A"

            current_state = st.session_state.var_inclusion[row["Variable"]]
            st.session_state.var_inclusion[row["Variable"]] = st.checkbox(label, value=current_state)

        # Apply exclusion: Beta=0 for unchecked vars
        base_df["Included"] = base_df["Variable"].map(st.session_state.var_inclusion)
        base_df.loc[~base_df["Included"], "Beta"] = 0

        # Calculate impacts
        base_df["Abs_Beta"] = base_df["Beta"].abs()
        total_abs_impact = base_df["Abs_Beta"].sum()
        if total_abs_impact == 0:
            base_df["Impact_%"] = 0
        else:
            base_df["Impact_%"] = (base_df["Abs_Beta"] / total_abs_impact) * 100

        # Map to raw features if possible
        raw_impacts = map_factors_to_raw_features_from_step10(base_df)
        if raw_impacts.empty:
            return base_df.sort_values("Impact_%", ascending=False).reset_index(drop=True)
        return raw_impacts

    except Exception as e:
        st.error(f"Error building impact dataframe: {str(e)}")
        return pd.DataFrame()

def map_factors_to_raw_features_from_step10(coef_df: pd.DataFrame) -> pd.DataFrame:
    """Map factor impacts to raw features using Step 10's loading data."""
    try:
        if not hasattr(st.session_state, 'fa_results') or st.session_state.fa_results is None:
            return pd.DataFrame()
        
        fa_results = st.session_state.fa_results
        raw_impacts = {}
        
        factored_vars = coef_df[coef_df['Type'] == 'Factored']
        
        for _, row in factored_vars.iterrows():
            factor_var = row['Variable']
            factor_impact = row['Impact_%']
            factor_mapped = False
            
            for category, results in fa_results.items():
                if not results or not results.get('success', False):
                    continue
                    
                loadings_df = results.get('loadings')
                if loadings_df is None or not isinstance(loadings_df, pd.DataFrame):
                    continue
                
                if factor_var.startswith(category + '_Factor_'):
                    try:
                        factor_num = int(factor_var.replace(category + '_Factor_', ''))
                    except ValueError:
                        continue
                    
                    possible_factor_cols = [
                        f"Factor {factor_num}",
                        f"Factor_{factor_num}",
                        f"Factor{factor_num}"
                    ]
                    
                    factor_col = next((c for c in possible_factor_cols if c in loadings_df.columns), None)
                    if factor_col is None:
                        continue
                    
                    feature_names = loadings_df.index.tolist()
                    factor_loadings = loadings_df[factor_col].abs()
                    total_abs_loadings = factor_loadings.sum()
                    
                    if total_abs_loadings > 0:
                        for raw_feature in feature_names:
                            loading_value = abs(loadings_df.loc[raw_feature, factor_col])
                            if loading_value > 0:
                                weighted_impact = factor_impact * (loading_value / total_abs_loadings)
                                raw_impacts[raw_feature] = raw_impacts.get(raw_feature, 0) + weighted_impact
                    
                    factor_mapped = True
                    break
            
            if not factor_mapped:
                raw_impacts[factor_var] = raw_impacts.get(factor_var, 0) + factor_impact
        
        raw_vars = coef_df[coef_df['Type'] == 'Raw']
        for _, row in raw_vars.iterrows():
            raw_impacts[row['Variable']] = raw_impacts.get(row['Variable'], 0) + row['Impact_%']
        
        if raw_impacts:
            raw_impact_df = pd.DataFrame(list(raw_impacts.items()), columns=['Variable', 'Impact_%'])
            raw_impact_df = raw_impact_df.sort_values('Impact_%', ascending=False).reset_index(drop=True)
            total_impact = raw_impact_df['Impact_%'].sum()
            if total_impact > 0:
                raw_impact_df['Impact_%'] = (raw_impact_df['Impact_%'] / total_impact) * 100
            return raw_impact_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error mapping factors to raw features: {str(e)}")
        return pd.DataFrame()

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
            others_row = pd.DataFrame({"Variable": [f"Others ({len(others)})"],"Impact_%": [others["Impact_%"].sum()]})
            plot_df = pd.concat([main, others_row], ignore_index=True)
        else:
            plot_df = main.copy()
        
        colors = ["#2E86AB" if not v.startswith("Others") else "#A23B72" for v in plot_df["Variable"]]
        
        fig = px.bar(plot_df, x="Impact_%", y="Variable", orientation="h", text="Impact_%", height=max(400, 40 * len(plot_df)))
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
            top_n = df.head(n)
            uplift = top_n["Impact_%"].sum() / 100
            scaled_uplift = uplift * 0.1
            impacts.append(scaled_uplift)
        
        levels = [current, current + impacts[0], current + impacts[1]]
        labels = ["Current Level", f"10% ↑ in Top {TOP_N_LIST[0]}", f"10% ↑ in Top {TOP_N_LIST[1]}"]
        
        fig = go.Figure()
        colors = ["#3E4B8B", "#28A745", "#17A2B8"]
        
        for i, (lab, val) in enumerate(zip(labels, levels)):
            fig.add_trace(go.Bar(x=[lab], y=[val * 100], marker_color=colors[i], text=[f"{val*100:.1f}%"], textposition="outside", width=0.5))
        
        fig.update_layout(title="Projected Lift with Driver Optimization", yaxis_title="Top-2-Box Level (%)", xaxis_title="Scenario", showlegend=False, bargap=0.6, height=500)
        max_val = max([v*100 for v in levels])
        fig.update_yaxes(range=[0, max_val * 1.15])
        return fig
        
    except Exception as e:
        st.error(f"Error creating waterfall chart: {str(e)}")
        return go.Figure()

if __name__ == "__main__":
    show_page()
