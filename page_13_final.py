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
    
    # Check prerequisites
    required_data = check_prerequisites()
    if not required_data:
        return
        
    model, X_test, y_test, selected_features = required_data
    
    # Build impact dataframe - mapping to raw features using Step 9 loadings
    coef_df = build_raw_feature_impact_df(model, selected_features)
    
    if coef_df.empty:
        st.error("❌ No impact data available. Please check that factor analysis was completed successfully.")
        return
    
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
        help="These are the original survey questions that will be reflected in the charts."
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
    """Check for required data from Step 12."""
    model = None
    selected_features = None
    X_test = None
    y_test = None
    
    # Get model from various possible locations
    if hasattr(st.session_state, 'last_trained_model'):
        model = st.session_state.last_trained_model
    elif hasattr(st.session_state, 'regression_model'):
        model = st.session_state.regression_model
    
    # Get from model_results dict
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
    
    if model is None:
        st.error("⚠️ Missing trained model from Step 12.")
        st.info("Please complete Step 12 (Logistic Regression Analysis) first.")
        return None
    
    if selected_features is None:
        if hasattr(st.session_state, 'sel_factored') and hasattr(st.session_state, 'sel_raw'):
            selected_features = st.session_state.sel_factored + st.session_state.sel_raw
        else:
            st.error("⚠️ Missing selected features information.")
            return None
    
    if y_test is None:
        if hasattr(st.session_state, 'y_target'):
            y_test = st.session_state.y_target
        else:
            st.error("⚠️ Missing target variable data.")
            return None
    
    return model, X_test, y_test, selected_features

def build_raw_feature_impact_df(model, selected_features: list) -> pd.DataFrame:
    """Build impact dataframe showing impacts at RAW FEATURE level using Step 9 loadings."""
    try:
        # Get coefficients from model
        if hasattr(model, 'coef_'):
            betas = model.coef_[0]
        else:
            st.error("Invalid model object - no coefficients found.")
            return pd.DataFrame()
        
        # Ensure feature-coefficient alignment
        if len(selected_features) != len(betas):
            min_len = min(len(selected_features), len(betas))
            selected_features = selected_features[:min_len]
            betas = betas[:min_len]
        
        # Create model-level impact dataframe
        model_df = pd.DataFrame({
            "Variable": selected_features, 
            "Beta": betas,
            "Type": ["Factored" if "Factor" in var else "Raw" for var in selected_features]
        })
        
        # Calculate model-level impacts
        model_df["Abs_Beta"] = model_df["Beta"].abs()
        total_abs_impact = model_df["Abs_Beta"].sum()
        
        if total_abs_impact == 0:
            model_df["Impact_%"] = 0
        else:
            model_df["Impact_%"] = (model_df["Abs_Beta"] / total_abs_impact) * 100
        
        # Map to raw features using Step 9 fa_results
        raw_impacts = {}
        
        # Get FA results from Step 9
        fa_results = st.session_state.get('fa_results', {})
        
        # Process each model variable
        for _, row in model_df.iterrows():
            var_name = row['Variable']
            var_impact = row['Impact_%']
            var_type = row['Type']
            
            if var_type == 'Raw':
                # Raw variables go directly
                raw_impacts[var_name] = raw_impacts.get(var_name, 0) + var_impact
            
            elif var_type == 'Factored':
                # Map factored variables to raw features using loadings from fa_results
                factor_mapped = False
                
                # Find which category this factor belongs to
                for category, results in fa_results.items():
                    if not results.get('success', False):
                        continue
                    
                    # Check if this factor belongs to this category
                    if var_name.startswith(category + '_Factor_'):
                        loadings_df = results.get('loadings')
                        if loadings_df is None or not isinstance(loadings_df, pd.DataFrame):
                            continue
                        
                        # Extract factor number from variable name (e.g., "Rep Attributes_Factor_1" -> 1)
                        try:
                            factor_part = var_name.replace(category + '_Factor_', '')
                            factor_num = int(factor_part)
                        except ValueError:
                            continue
                        
                        # Find the corresponding factor column in loadings
                        factor_col = f"Factor_{factor_num}"
                        if factor_col not in loadings_df.columns:
                            continue
                        
                        # Get feature names (first column in loadings)
                        feature_col = loadings_df.columns[0]
                        
                        # Calculate total absolute loadings for proper proportional allocation
                        factor_loadings = loadings_df[factor_col].abs()
                        total_abs_loadings = factor_loadings.sum()
                        
                        if total_abs_loadings > 0:
                            # Distribute factor impact to raw features based on their loadings
                            for idx, raw_feature in enumerate(loadings_df[feature_col]):
                                loading_value = abs(loadings_df.iloc[idx][factor_col])
                                
                                if loading_value > 0:
                                    # Proportional allocation based on loading strength
                                    weighted_impact = var_impact * (loading_value / total_abs_loadings)
                                    raw_impacts[raw_feature] = raw_impacts.get(raw_feature, 0) + weighted_impact
                        
                        factor_mapped = True
                        break
                
                if not factor_mapped:
                    # If mapping failed, keep the factor name as fallback
                    st.warning(f"Could not map factor '{var_name}' to raw features. Using factor name.")
                    raw_impacts[var_name] = raw_impacts.get(var_name, 0) + var_impact
        
        # Convert to DataFrame
        if raw_impacts:
            raw_impact_df = pd.DataFrame(
                list(raw_impacts.items()), 
                columns=['Variable', 'Impact_%']
            )
            
            # Sort by impact
            raw_impact_df = raw_impact_df.sort_values('Impact_%', ascending=False).reset_index(drop=True)
            
            # Renormalize to ensure 100% total
            total_impact = raw_impact_df['Impact_%'].sum()
            if total_impact > 0:
                raw_impact_df['Impact_%'] = (raw_impact_df['Impact_%'] / total_impact) * 100
            
            return raw_impact_df
        else:
            st.warning("No impacts calculated - using model variables directly")
            return model_df[["Variable", "Impact_%"]].sort_values("Impact_%", ascending=False).reset_index(drop=True)
            
    except Exception as e:
        st.error(f"Error building raw feature impact dataframe: {str(e)}")
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
            title="Normalized Impact of Selected Drivers (Raw Features)"
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
            title="Projected Lift with Driver Optimization (Raw Features)",
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
