import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────
def render_feature_prep_page():
    """Render the feature-engineering and data-preparation page"""

    # ── Prerequisite check ────────────────────────────────────────────────
    if st.session_state.selected_target_col is None:
        st.error("❌ No target variable selected. Please complete Step 4 first.")
        return

    filtered_df          = st.session_state.filtered_df
    bin_df               = st.session_state.bin_df
    selected_target_col  = st.session_state.selected_target_col
    selected_target_name = st.session_state.selected_target_name

    st.markdown(
        f"""
        ## 🔧 Feature Engineering & Data Preparation  
        Prepare features for analysis with selected target variable: **{selected_target_name}**
        """
    )

    # ── Target variable overview ─────────────────────────────────────────
    st.subheader("🎯 Selected Target Variable")
    col1, col2, col3 = st.columns(3)

    if selected_target_col in bin_df.columns:
        vc          = bin_df[selected_target_col].value_counts().sort_index()
        total       = bin_df[selected_target_col].count()
        positive_pct = (vc.get(1, 0) / total * 100) if total else 0

        with col1:
            st.metric("Target Variable", selected_target_name)
        with col2:
            st.metric("Positive Rate", f"{positive_pct:.1f}%")
        with col3:
            st.metric("Sample Size", f"{total:,}")

    st.markdown("---")

    # ── Trigger feature preparation ──────────────────────────────────────
    if st.button("🔄 Prepare Features for Analysis", type="primary"):
        prepare_features_for_analysis()
        st.session_state.step_completed[4] = True


# ──────────────────────────────────────────────────────────────────────────
def prepare_features_for_analysis():
    """Prepare features for key-driver analysis"""

    filtered_df         = st.session_state.filtered_df
    bin_df              = st.session_state.bin_df
    selected_target_col = st.session_state.selected_target_col

    st.subheader("⚙️ Feature-Preparation Process")
    with st.spinner("Processing features..."):

        # 1. Merge target into working dataframe
        analysis_df = filtered_df.copy()
        analysis_df[selected_target_col] = bin_df[selected_target_col]

        # 2. Identify feature columns (exclude metadata & target)
        metadata_cols = [
            'Product', 'users_wave_id', 'wave_id', 'wave_number', 'user_id',
            'user_type', 'status', 'completed_date', 'completed_date_user_tz',
            'npi', 'time_period'
        ]
        feature_cols = [
            c for c in analysis_df.columns
            if c not in metadata_cols and c != selected_target_col
        ]

        # 3. Keep only numeric features & impute missing with median
        num_feats          = analysis_df[feature_cols].select_dtypes(include=[np.number])
        num_feats_filled   = num_feats.fillna(num_feats.median())

        # 4. Assemble final modelling dataframe
        final_df = num_feats_filled.copy()
        final_df[selected_target_col] = analysis_df[selected_target_col]

        # 5. Store in session state
        st.session_state.model_df     = final_df
        st.session_state.feature_list = list(num_feats_filled.columns)

    st.success("✅ Feature preparation completed successfully!")

    # ── Dataset overview ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset Shape", f"{final_df.shape[0]} × {final_df.shape[1]}")
        st.metric("Number of Features", len(st.session_state.feature_list))
    with col2:
        st.metric("Missing Values", final_df.isnull().sum().sum())
        st.metric("Target Variable", selected_target_col)

    # ── Feature category listing (unchanged) ────────────────────────────
    _display_feature_categories()

    # ── Full correlation table (ALL features) ───────────────────────────
 
# ── Correlation of all features with the three target columns ─────────

st.subheader("🎯 Correlation with Target – LTIP / Overall Quality / Overall Perception")
 
# identify the three target columns (case-insensitive)

ltip_col   = next((c for c in filtered_df.columns if 'ltip'               in c.lower()), None)

rep_col    = next((c for c in filtered_df.columns if 'overall quality'    in c.lower()), None)

percep_col = next((c for c in filtered_df.columns if 'overall perception' in c.lower()), None)

target_cols = [( "LTIP", ltip_col ), ("Overall Quality", rep_col), ("Overall Perception", percep_col)]
 
# validate presence

valid_targets = [(label, col) for label, col in target_cols if col and col in filtered_df.columns]

missing = [label for label, col in target_cols if not (col and col in filtered_df.columns)]

if missing:

    st.warning(f"Could not find these target columns in the dataframe for correlation: {', '.join(missing)}")
 
# compute & display per-target correlation tables

corr_tables = {}

for label, target in valid_targets:

    corr_rows = []

    for feat in st.session_state.feature_list:

        try:

            c = filtered_df[feat].corr(filtered_df[target])

            corr_rows.append({

                "Feature": feat,

                "Correlation": c,

                "Abs_Correlation": abs(c)

            })

        except Exception:

            continue

    if corr_rows:

        df = (

            pd.DataFrame(corr_rows)

            .sort_values("Abs_Correlation", ascending=False)

            .reset_index(drop=True)

            .loc[:, ["Feature", "Correlation"]]

        )

        st.markdown(f"**Target:** {label} (column: `{target}`)")

        st.dataframe(df, use_container_width=True)

        corr_tables[label] = df
 
# optional: aggregated view showing each feature's strongest correlation across the available targets

if corr_tables:

    aggregate = {}

    for label, df in corr_tables.items():

        for _, row in df.iterrows():

            feat = row["Feature"]

            corr = row["Correlation"]

            if feat not in aggregate or abs(corr) > aggregate[feat]["Abs"]:

                aggregate[feat] = {"Best Target": label, "Correlation": corr, "Abs": abs(corr)}

    agg_df = pd.DataFrame([

        {

            "Feature": feat,

            "Best Target": v["Best Target"],

            "Correlation": v["Correlation"]

        }

        for feat, v in aggregate.items()

    ]).sort_values("Correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
 
    st.subheader("🔥 Top Features Across All Specified Targets (by max |correlation|)")

    st.dataframe(agg_df, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
def _display_feature_categories():
    """Helper to show features by category"""
    feature_list = st.session_state.feature_list

    rep_feats   = [f for f in feature_list if "Rep Attributes"     in f]
    percep_feats= [f for f in feature_list if "Perceptions"        in f]
    deliv_feats = [f for f in feature_list if "Delivery of topic"  in f]
    misc_feats  = [f for f in feature_list if f not in rep_feats + percep_feats + deliv_feats]

    tab1, tab2, tab3, tab4 = st.tabs([
        f"📈 Rep Attributes ({len(rep_feats)})",
        f"📊 Perceptions ({len(percep_feats)})",
        f"📋 Message Delivery ({len(deliv_feats)})",
        f"📦 Miscellaneous ({len(misc_feats)})"
    ])

    for tab, feats, label in [
        (tab1, rep_feats, "Rep Attributes"),
        (tab2, percep_feats, "Perceptions"),
        (tab3, deliv_feats, "Message Delivery"),
        (tab4, misc_feats, "Miscellaneous")
    ]:
        with tab:
            if feats:
                for i, f in enumerate(feats, 1):
                    st.write(f"{i}. {f}")
            else:
                st.write(f"No {label} features found.")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    render_feature_prep_page()
