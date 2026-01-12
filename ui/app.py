"""
Batch-Centric OEE Analysis - Streamlit Application
Main UI for selecting days, batches, and viewing OEE metrics
"""
import sys
import os
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import pytz
import logging

from config import Config
from db.fetchers import (
    fetch_batches_for_day,
    fetch_job_ids_for_batch,
    fetch_print_jobreports,
    fetch_cut_jobreports,
    fetch_pick_jobreports,
    fetch_qc_data_for_batches,
    fetch_qc_data,
    fetch_components_per_job,
    fetch_batch_structure,
    fetch_fpy_data
)
from analysis.quality_overlay import apply_qc_overlay
from analysis.oee_calculator import calculate_batch_metrics, calculate_daily_metrics

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timezone
tz = pytz.timezone(Config.TIMEZONE)

# Page config
st.set_page_config(
    page_title="Batch-Centric OEE Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HEADER
# ============================================================================

st.title("🏭 Batch-Centric OEE Analysis")
st.markdown("**Manufacturing Excellence Dashboard** - Real-time batch performance tracking")
st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # Operating hours
    operating_hours = st.number_input(
        "Operating Hours per Day",
        min_value=1.0,
        max_value=24.0,
        value=Config.DEFAULT_OPERATING_HOURS,
        step=0.5,
        help="Total available hours for daily utilization calculation"
    )

    st.markdown("---")

    # Info
    st.info(f"**Timezone:** {Config.TIMEZONE}")
    st.caption("📊 Data sources: HISTORIAN + INSIDE databases")
    st.caption("🔄 Quality overlay: Auto-applies when QC data available")

# ============================================================================
# SESSION STATE
# ============================================================================

if 'batches_df' not in st.session_state:
    st.session_state.batches_df = None
if 'selected_batches' not in st.session_state:
    st.session_state.selected_batches = []
if 'print_df' not in st.session_state:
    st.session_state.print_df = None
if 'cut_df' not in st.session_state:
    st.session_state.cut_df = None
if 'pick_df' not in st.session_state:
    st.session_state.pick_df = None
if 'qc_df' not in st.session_state:
    st.session_state.qc_df = None
if 'components_per_job_df' not in st.session_state:
    st.session_state.components_per_job_df = None
if 'batch_structure_df' not in st.session_state:
    st.session_state.batch_structure_df = None
if 'fpy_df' not in st.session_state:
    st.session_state.fpy_df = None
if 'qc_applied' not in st.session_state:
    st.session_state.qc_applied = False

# ============================================================================
# STEP 1: DAY SELECTION
# ============================================================================

st.header("1️⃣ Select Production Day")

col1, col2 = st.columns([2, 1])

with col1:
    selected_date = st.date_input(
        "Production Date",
        value=date.today() - timedelta(days=1),
        help="Select the day to analyze"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("🔍 Load Batches", type="primary", use_container_width=True):
        # Calculate day boundaries
        day_start = tz.localize(datetime.combine(selected_date, datetime.min.time()))
        day_end = day_start + timedelta(days=1)

        with st.spinner("Discovering batches..."):
            batches_df = fetch_batches_for_day(
                day_start.isoformat(),
                day_end.isoformat()
            )
            st.session_state.batches_df = batches_df

            if not batches_df.empty:
                st.success(f"✅ Found {len(batches_df)} batches on {selected_date}")
            else:
                st.warning(f"⚠️ No batches found on {selected_date}")

# ============================================================================
# STEP 2: BATCH SELECTION
# ============================================================================

if st.session_state.batches_df is not None and not st.session_state.batches_df.empty:
    st.markdown("---")
    st.header("2️⃣ Select Batches to Analyze")

    batches_df = st.session_state.batches_df

    # Display batch overview
    st.dataframe(
        batches_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "batch_id": st.column_config.TextColumn("Batch ID", width="medium"),
            "job_count": st.column_config.NumberColumn("Jobs", width="small"),
            "first_job_time": st.column_config.DatetimeColumn("Start Time", width="medium"),
            "last_job_time": st.column_config.DatetimeColumn("End Time", width="medium")
        }
    )

    # Batch multiselect
    selected_batches = st.multiselect(
        "Select batches",
        options=batches_df['batch_id'].tolist(),
        default=batches_df['batch_id'].tolist(),
        help="Choose one or more batches for analysis"
    )

    st.session_state.selected_batches = selected_batches

    # ============================================================================
    # STEP 3: FETCH & ANALYZE
    # ============================================================================

    if selected_batches:
        st.markdown("---")
        st.header("3️⃣ Fetch & Analyze Data")

        if st.button("📊 Calculate OEE", type="primary", use_container_width=True):
            with st.spinner("Fetching JobReports and calculating OEE..."):

                # Get job IDs
                job_ids = fetch_job_ids_for_batch(selected_batches)

                if not job_ids:
                    st.error("❌ No job IDs found for selected batches")
                else:
                    st.info(f"📋 Found {len(job_ids)} jobs across {len(selected_batches)} batches")

                    # Fetch JobReports (Print first, then Cut needs Print data for batch_id/sheet_index)
                    # Note: Cut may happen on different day than Print, so we match by job_id + temporal proximity
                    print_df = fetch_print_jobreports(job_ids)
                    cut_df = fetch_cut_jobreports(job_ids, print_df=print_df)
                    pick_df = fetch_pick_jobreports(job_ids)

                    # Try to fetch QC data (aggregated for overlay)
                    qc_df_aggregated = fetch_qc_data_for_batches(selected_batches)

                    if qc_df_aggregated is not None and not qc_df_aggregated.empty:
                        st.info("🔬 QC inspection data found - applying quality overlay...")
                        print_df, cut_df, pick_df = apply_qc_overlay(print_df, cut_df, pick_df, qc_df_aggregated)
                        st.session_state.qc_applied = True
                    else:
                        st.info("ℹ️ No QC data available - using 100% quality assumption")
                        st.session_state.qc_applied = False

                    # Fetch detailed QC data (component-level extract)
                    qc_df_detailed = fetch_qc_data(selected_batches)
                    st.session_state.qc_df = qc_df_detailed

                    # Fetch components per job
                    components_per_job_df = fetch_components_per_job(job_ids)
                    st.session_state.components_per_job_df = components_per_job_df

                    # Fetch batch structure (styles and units per style)
                    batch_structure_df = fetch_batch_structure(selected_batches)
                    st.session_state.batch_structure_df = batch_structure_df

                    # Fetch FPY (First Pass Yield) data
                    fpy_df = fetch_fpy_data(selected_batches)
                    st.session_state.fpy_df = fpy_df

                    # Store in session state
                    st.session_state.print_df = print_df
                    st.session_state.cut_df = cut_df
                    st.session_state.pick_df = pick_df

                    st.success("✅ OEE calculation complete!")

# ============================================================================
# STEP 4: DISPLAY RESULTS
# ============================================================================

if st.session_state.print_df is not None:
    st.markdown("---")
    st.header("4️⃣ OEE Analysis Results")

    # QC Status Banner
    if st.session_state.qc_applied:
        st.success("✅ **QC Inspection Data Applied** - Quality metrics reflect actual inspection results")
    else:
        st.info("ℹ️ **No QC Data Yet** - Using 100% quality assumption (Day 0 optimistic view)")

    # Get data
    print_df = st.session_state.print_df
    cut_df = st.session_state.cut_df
    pick_df = st.session_state.pick_df

    # Calculate metrics
    batch_metrics_df = calculate_batch_metrics(
        print_df, cut_df, pick_df,
        st.session_state.selected_batches
    )

    daily_metrics = calculate_daily_metrics(batch_metrics_df, operating_hours)

    # ========================================================================
    # 4.1: BATCH-LEVEL METRICS
    # ========================================================================

    st.subheader("📦 Batch-Level OEE Metrics")

    st.dataframe(
        batch_metrics_df[[
            'batch_id',
            'print_oee', 'print_availability', 'print_performance', 'print_quality',
            'cut_oee', 'cut_availability', 'cut_performance', 'cut_quality',
            'pick_oee', 'pick_availability', 'pick_performance', 'pick_quality'
        ]].style.format({
            'print_oee': '{:.2f}%',
            'print_availability': '{:.2f}%',
            'print_performance': '{:.2f}%',
            'print_quality': '{:.2f}%',
            'cut_oee': '{:.2f}%',
            'cut_availability': '{:.2f}%',
            'cut_performance': '{:.2f}%',
            'cut_quality': '{:.2f}%',
            'pick_oee': '{:.2f}%',
            'pick_availability': '{:.2f}%',
            'pick_performance': '{:.2f}%',
            'pick_quality': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================================
    # 4.2: DAILY-LEVEL METRICS
    # ========================================================================

    st.subheader("📅 Daily-Level OEE Metrics (with Utilization)")

    daily_df = pd.DataFrame([
        {
            'Cell': '🖨️ Print',
            'Batch OEE': daily_metrics['print']['batch_oee'],
            'Production Hours': daily_metrics['print']['production_hours'],
            'Idle Hours': daily_metrics['print']['idle_hours'],
            'Utilization': daily_metrics['print']['utilization'],
            'Daily OEE': daily_metrics['print']['daily_oee']
        },
        {
            'Cell': '✂️ Cut',
            'Batch OEE': daily_metrics['cut']['batch_oee'],
            'Production Hours': daily_metrics['cut']['production_hours'],
            'Idle Hours': daily_metrics['cut']['idle_hours'],
            'Utilization': daily_metrics['cut']['utilization'],
            'Daily OEE': daily_metrics['cut']['daily_oee']
        },
        {
            'Cell': '🤖 Pick',
            'Batch OEE': daily_metrics['pick']['batch_oee'],
            'Production Hours': daily_metrics['pick']['production_hours'],
            'Idle Hours': daily_metrics['pick']['idle_hours'],
            'Utilization': daily_metrics['pick']['utilization'],
            'Daily OEE': daily_metrics['pick']['daily_oee']
        }
    ])

    st.dataframe(
        daily_df.style.format({
            'Batch OEE': '{:.2f}%',
            'Production Hours': '{:.2f}h',
            'Idle Hours': '{:.2f}h',
            'Utilization': '{:.2f}%',
            'Daily OEE': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================================
    # 4.3: JOB-LEVEL DETAILS
    # ========================================================================

    with st.expander("🔍 View Job-Level Details"):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Print Jobs", "Cut Jobs", "Pick Jobs", "QC Data", "Components/Job", "Batch Structure"])

        with tab1:
            if not print_df.empty:
                st.dataframe(
                    print_df[[
                        'job_id', 'batch_id', 'sheet_index',
                        'availability', 'performance', 'quality', 'oee',
                        'uptime_sec', 'downtime_sec', 'total_time_sec'
                    ]].style.format({
                        'availability': '{:.2f}%',
                        'performance': '{:.2f}%',
                        'quality': '{:.2f}%',
                        'oee': '{:.2f}%',
                        'uptime_sec': '{:.0f}s',
                        'downtime_sec': '{:.0f}s',
                        'total_time_sec': '{:.0f}s'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No Print data available")

        with tab2:
            if not cut_df.empty:
                st.dataframe(
                    cut_df[[
                        'job_id', 'batch_id', 'sheet_index',
                        'availability', 'performance', 'quality', 'oee',
                        'uptime_sec', 'downtime_sec', 'total_time_sec'
                    ]].style.format({
                        'availability': '{:.2f}%',
                        'performance': '{:.2f}%',
                        'quality': '{:.2f}%',
                        'oee': '{:.2f}%',
                        'uptime_sec': '{:.0f}s',
                        'downtime_sec': '{:.0f}s',
                        'total_time_sec': '{:.0f}s'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No Cut data available")

        with tab3:
            if not pick_df.empty:
                st.dataframe(
                    pick_df[[
                        'job_id', 'batch_id', 'sheet_index',
                        'availability', 'performance', 'quality', 'oee',
                        'uptime_sec', 'downtime_sec', 'total_time_sec',
                        'components_completed', 'components_failed', 'total_attempts'
                    ]].style.format({
                        'availability': '{:.2f}%',
                        'performance': '{:.2f}%',
                        'quality': '{:.2f}%',
                        'oee': '{:.2f}%',
                        'uptime_sec': '{:.0f}s',
                        'downtime_sec': '{:.0f}s',
                        'total_time_sec': '{:.0f}s'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No Pick data available")

        with tab4:
            qc_df = st.session_state.qc_df
            if qc_df is not None and not qc_df.empty:
                # Format qc_reasons for display (convert list to string)
                display_df = qc_df.copy()
                if 'qc_reasons' in display_df.columns:
                    display_df['qc_reasons'] = display_df['qc_reasons'].apply(
                        lambda x: ', '.join(x) if isinstance(x, list) and x else (str(x) if x else '')
                    )
                
                st.dataframe(
                    display_df[[
                        'component_id', 'batch_id', 'job_id', 'production_date',
                        'qc_state', 'qc_source', 'qc_reasons', 'qc_timestamp'
                    ]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "component_id": st.column_config.NumberColumn("Component ID", width="small"),
                        "batch_id": st.column_config.NumberColumn("Batch ID", width="small"),
                        "job_id": st.column_config.TextColumn("Job ID", width="medium"),
                        "production_date": st.column_config.DateColumn("Production Date", width="medium"),
                        "qc_state": st.column_config.TextColumn("QC State", width="small"),
                        "qc_source": st.column_config.TextColumn("QC Source", width="medium"),
                        "qc_reasons": st.column_config.TextColumn("QC Reasons", width="large"),
                        "qc_timestamp": st.column_config.DatetimeColumn("QC Timestamp", width="medium")
                    }
                )
                
                # QC Statistics
                st.markdown("### 📊 QC Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_components = len(qc_df)
                    st.metric("Total Components", total_components)
                
                with col2:
                    passed = len(qc_df[qc_df['qc_state'] == 'passed'])
                    st.metric("Passed", passed)
                
                with col3:
                    failed = len(qc_df[qc_df['qc_state'] == 'failed'])
                    st.metric("Failed", failed)
                
                with col4:
                    not_scanned = len(qc_df[qc_df['qc_state'] == 'not_scanned'])
                    st.metric("Not Scanned", not_scanned)
                
                # QC Source breakdown
                if 'qc_source' in qc_df.columns:
                    st.markdown("### 📱 QC Source Breakdown")
                    source_counts = qc_df[qc_df['qc_source'] != '']['qc_source'].value_counts()
                    if not source_counts.empty:
                        source_df = pd.DataFrame({
                            'QC Source': source_counts.index,
                            'Count': source_counts.values
                        })
                        source_df = source_df.sort_values('Count', ascending=False)
                        st.dataframe(
                            source_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "QC Source": st.column_config.TextColumn("QC Source", width="medium"),
                                "Count": st.column_config.NumberColumn("Count", width="small")
                            }
                        )
                    else:
                        st.info("No QC source data available")
                
                # Defect reasons breakdown (for failed components) - Top-level categories only
                failed_df = qc_df[qc_df['qc_state'] == 'failed']
                if not failed_df.empty and 'qc_reasons' in failed_df.columns:
                    st.markdown("### ⚠️ Defect Reasons (Failed Components) - Top Level")
                    # Extract unique top-level categories per component to avoid double-counting
                    component_categories = []
                    for reasons_list in failed_df['qc_reasons']:
                        # Get unique top-level categories for this component
                        unique_categories = set()
                        if isinstance(reasons_list, list):
                            for reason in reasons_list:
                                # Extract top-level category (e.g., "cut-outside-bleed" -> "cut")
                                top_level = str(reason).split('-')[0].strip() if reason else None
                                if top_level:
                                    unique_categories.add(top_level)
                        elif reasons_list:
                            top_level = str(reasons_list).split('-')[0].strip()
                            if top_level:
                                unique_categories.add(top_level)
                        
                        # Add each unique category once per component
                        component_categories.extend(list(unique_categories))
                    
                    if component_categories:
                        reason_counts = pd.Series(component_categories).value_counts()
                        reason_df = pd.DataFrame({
                            'Defect Category': reason_counts.index,
                            'Component Count': reason_counts.values
                        })
                        reason_df = reason_df.sort_values('Component Count', ascending=False)
                        st.dataframe(
                            reason_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Defect Category": st.column_config.TextColumn("Defect Category", width="medium"),
                                "Component Count": st.column_config.NumberColumn("Component Count", width="small")
                            }
                        )
                    else:
                        st.info("No defect reasons recorded")
                
                # FPY (First Pass Yield) Data
                fpy_df = st.session_state.fpy_df
                if fpy_df is not None and not fpy_df.empty:
                    st.markdown("### 📈 First Pass Yield (FPY)")
                    st.markdown("**FPY** = Garments that passed QC / Target garments × 100%")
                    
                    st.dataframe(
                        fpy_df[[
                            'item_order_id', 'garmentsGoal', 'garments_not_failed', 
                            'garments_QC_passed', 'fpy_rate'
                        ]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "item_order_id": st.column_config.NumberColumn("Style # (item_order_id)", width="small"),
                            "garmentsGoal": st.column_config.NumberColumn("Target Garments", width="small"),
                            "garments_not_failed": st.column_config.NumberColumn("Garments (No Defects)", width="small"),
                            "garments_QC_passed": st.column_config.NumberColumn("Garments (QC Passed)", width="small"),
                            "fpy_rate": st.column_config.NumberColumn("FPY Rate (%)", width="small", format="%.2f")
                        }
                    )
                    
                    # FPY Summary
                    st.markdown("### 📊 FPY Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_target = fpy_df['garmentsGoal'].sum()
                        st.metric("Total Target", total_target)
                    with col2:
                        total_passed = fpy_df['garments_QC_passed'].sum()
                        st.metric("Total (QC Passed)", total_passed)
                    with col3:
                        total_failed = total_target - total_passed
                        st.metric("Garments Failed", total_failed)
                    with col4:
                        overall_fpy = (total_passed / total_target * 100) if total_target > 0 else 0
                        st.metric("Overall FPY", f"{overall_fpy:.2f}%")
                else:
                    st.info("No FPY data available")
            else:
                st.info("No QC data available")

        with tab5:
            components_per_job_df = st.session_state.components_per_job_df
            if components_per_job_df is not None and not components_per_job_df.empty:
                st.dataframe(
                    components_per_job_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "job_id": st.column_config.TextColumn("Job ID", width="medium"),
                        "batch_id": st.column_config.NumberColumn("Batch ID", width="small"),
                        "component_count": st.column_config.NumberColumn("Component Count", width="small")
                    }
                )
                
                # Summary statistics
                st.markdown("### 📊 Summary")
                col1, col2 = st.columns(2)
                with col1:
                    total_jobs = len(components_per_job_df)
                    st.metric("Total Jobs", total_jobs)
                with col2:
                    total_components = components_per_job_df['component_count'].sum()
                    st.metric("Total Components", total_components)
            else:
                st.info("No component per job data available")

        with tab6:
            batch_structure_df = st.session_state.batch_structure_df
            if batch_structure_df is not None and not batch_structure_df.empty:
                st.markdown("### 📐 Batch Structure: Styles and Units")
                st.markdown("**Styles** = unique designs in the batch | **Units** = garments per style")
                
                # Display batch structure table
                st.dataframe(
                    batch_structure_df[[
                        'batch_id', 'style_number', 'components_per_style', 'garments_per_style', 
                        'total_components_per_style'
                    ]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "batch_id": st.column_config.NumberColumn("Batch ID", width="small"),
                        "style_number": st.column_config.NumberColumn("Style # (item_order_id)", width="small"),
                        "components_per_style": st.column_config.NumberColumn("Components/Style", width="small"),
                        "garments_per_style": st.column_config.NumberColumn("Garments/Style", width="small"),
                        "total_components_per_style": st.column_config.NumberColumn("Total Components (Style)", width="small")
                    }
                )
                
                # Summary by batch
                st.markdown("### 📊 Batch Summary")
                batch_summary = batch_structure_df.groupby('batch_id').agg({
                    'style_number': 'nunique',
                    'total_components_per_style': 'sum'
                }).reset_index()
                batch_summary.columns = ['batch_id', 'styles', 'total_components']
                batch_summary = batch_summary.sort_values('batch_id')
                
                st.dataframe(
                    batch_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "batch_id": st.column_config.NumberColumn("Batch ID", width="small"),
                        "styles": st.column_config.NumberColumn("Styles", width="small"),
                        "total_components": st.column_config.NumberColumn("Total Components", width="small")
                    }
                )
            else:
                st.info("No batch structure data available")

    # ========================================================================
    # 4.4: EXPORT
    # ========================================================================

    st.subheader("💾 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        batch_csv = batch_metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Batch Metrics CSV",
            data=batch_csv,
            file_name=f"batch_metrics_{selected_date}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        daily_csv = daily_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Daily Metrics CSV",
            data=daily_csv,
            file_name=f"daily_metrics_{selected_date}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(f"🏭 Batch-Centric OEE Analysis | Timezone: {Config.TIMEZONE} | Data: HISTORIAN + INSIDE")
