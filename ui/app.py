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
from dateutil import parser as dateutil_parser

from config import Config
from core.db.fetchers import (
    fetch_batches_for_day,
    fetch_job_ids_for_batch,
    fetch_print_jobreports,
    fetch_cut_jobreports,
    fetch_pick_jobreports,
    fetch_qc_data_for_batches,
    fetch_qc_data,
    fetch_components_per_job,
    fetch_batch_structure,
    fetch_fpy_data,
    fetch_pick_jobreports_by_jobs,
    validate_job_count_consistency,
    fetch_batch_quality_breakdown_v2,
    fetch_equipment_states_with_durations
)
from analysis.quality_overlay import apply_qc_overlay
from core.analysis.oee import apply_qc_quality_to_cells
from analysis.oee_calculator import calculate_batch_metrics, calculate_daily_metrics
from core.calculations.throughput import (
    calculate_hourly_state_breakdown,
    calculate_state_summary,
    calculate_system_throughput
)
from ui.metrics_display import display_all_cell_timelines, display_throughput_metrics

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

    batches_df = st.session_state.batches_df.copy()

    # Format timestamp columns to show timezone offset (+0100)
    timestamp_columns = ['print_start', 'print_end', 'cut_start', 'cut_end', 'pick_start', 'pick_end']
    for col in timestamp_columns:
        if col in batches_df.columns:
            # Convert timezone-aware datetime to string with +0100 format
            def format_timestamp_with_tz(ts):
                if pd.isna(ts) or ts is None:
                    return None
                # Convert pandas Timestamp to datetime if needed
                if isinstance(ts, pd.Timestamp):
                    ts = ts.to_pydatetime()
                # If timezone-aware, format with offset
                if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                    # Format as YYYY-MM-DD HH:mm:ss +0100
                    offset = ts.strftime('%z')  # Gets +0100 or +0200 format
                    return ts.strftime('%Y-%m-%d %H:%M:%S') + f' {offset}'
                else:
                    # If naive, assume it's already in UTC+1 and format accordingly
                    return ts.strftime('%Y-%m-%d %H:%M:%S') + ' +0100'
            
            batches_df[col] = batches_df[col].apply(format_timestamp_with_tz)

    # Display batch overview with Print/Cut/Pick counts
    # Select columns to display
    display_columns = ['batch_id', 'print_jobs', 'cut_jobs', 'pick_jobs']
    if 'print_start' in batches_df.columns:
        display_columns.extend(['print_start', 'print_end'])
    if 'cut_start' in batches_df.columns:
        display_columns.extend(['cut_start', 'cut_end'])
    if 'pick_start' in batches_df.columns:
        display_columns.extend(['pick_start', 'pick_end'])
    
    # Filter to only existing columns
    display_columns = [col for col in display_columns if col in batches_df.columns]
    
    st.dataframe(
        batches_df[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "batch_id": st.column_config.TextColumn("Batch ID", width="small"),
            "print_jobs": st.column_config.NumberColumn("Print Jobs", width="small"),
            "cut_jobs": st.column_config.NumberColumn("Cut Jobs", width="small"),
            "pick_jobs": st.column_config.NumberColumn("Pick Jobs", width="small"),
            "print_start": st.column_config.TextColumn("Print Start", width="medium"),
            "print_end": st.column_config.TextColumn("Print End", width="medium"),
            "cut_start": st.column_config.TextColumn("Cut Start", width="medium"),
            "cut_end": st.column_config.TextColumn("Cut End", width="medium"),
            "pick_start": st.column_config.TextColumn("Pick Start", width="medium"),
            "pick_end": st.column_config.TextColumn("Pick End", width="medium")
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

        # Lunch break input
        st.subheader("⏰ Break Time Configuration")
        st.caption("Specify when planned breaks (e.g., lunch) occurred to exclude from productive time")

        col1, col2, col3 = st.columns(3)
        with col1:
            include_break = st.checkbox("Include Break Period", value=False, help="Check if this batch included a scheduled break")

        break_start = None
        break_end = None
        break_duration_minutes = 0.0

        if include_break:
            with col2:
                break_start = st.time_input(
                    "Break Start Time",
                    value=None,
                    help="Time when the break started (e.g., 12:12)"
                )
            with col3:
                break_end = st.time_input(
                    "Break End Time",
                    value=None,
                    help="Time when the break ended (e.g., 12:55)"
                )

            # Calculate duration if both times provided
            if break_start and break_end:
                # Convert to datetime for calculation
                from datetime import datetime, timedelta
                start_dt = datetime.combine(datetime.today(), break_start)
                end_dt = datetime.combine(datetime.today(), break_end)

                # Handle case where break crosses midnight
                if end_dt < start_dt:
                    end_dt += timedelta(days=1)

                break_duration = end_dt - start_dt
                break_duration_minutes = break_duration.total_seconds() / 60.0

                st.info(f"⏱️ Break Duration: {break_duration_minutes:.0f} minutes ({break_duration_minutes/60:.1f} hours)")

        # Store in session state
        st.session_state.break_start = break_start
        st.session_state.break_end = break_end
        st.session_state.break_duration_minutes = break_duration_minutes

        st.markdown("---")

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
                    
                    # Fetch pick data using new function
                    pick_df = fetch_pick_jobreports_by_jobs(job_ids, time_window=None)
                    
                    # Merge sheet_index from Print to Pick (Pick jobs don't have sheet_index in their JobReport)
                    if not pick_df.empty and not print_df.empty and 'sheet_index' in print_df.columns:
                        # Create a lookup map from print_df: job_id -> sheet_index
                        print_sheet_map = print_df.set_index('job_id')['sheet_index'].to_dict()
                        # Map sheet_index to pick_df by job_id
                        pick_df['sheet_index'] = pick_df['job_id'].map(print_sheet_map)
                        logger.info(f"Merged sheet_index from Print to Pick: {pick_df['sheet_index'].notna().sum()}/{len(pick_df)} jobs matched")
                    
                    # Debug output
                    if pick_df.empty:
                        st.warning(f"⚠️ No Pick data found for {len(job_ids)} jobs")
                        st.caption(f"Sample job IDs searched: {job_ids[:5] if len(job_ids) > 0 else 'none'}")
                    else:
                        st.success(f"✅ Found {len(pick_df)} Pick JobReports")
                        st.caption(f"Sample job IDs found: {pick_df['job_id'].head(3).tolist() if 'job_id' in pick_df.columns else 'N/A'}")
                    
                    # Validate job count consistency
                    validation = validate_job_count_consistency(print_df, cut_df, pick_df)
                    if not validation['is_valid']:
                        st.warning("⚠️ Job Count Mismatch")
                        for msg in validation['messages']:
                            st.text(msg)

                    # Try to fetch QC data (aggregated for overlay)
                    qc_df_aggregated = fetch_qc_data_for_batches(selected_batches)
                    
                    # Fetch quality breakdown for two-layer Pick quality and batch-level quality calculation
                    quality_breakdown = fetch_batch_quality_breakdown_v2(selected_batches)
                    st.session_state.quality_breakdown = quality_breakdown

                    if qc_df_aggregated is not None and not qc_df_aggregated.empty:
                        st.info("🔬 QC inspection data found - applying quality overlay...")
                        # Use apply_qc_quality_to_cells for two-layer Pick quality support
                        if not quality_breakdown.empty:
                            print_df, cut_df, pick_df = apply_qc_quality_to_cells(
                                print_df, cut_df, pick_df, quality_breakdown
                            )
                        else:
                            # Fallback to simple overlay if quality_breakdown not available
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

                    # Step 4: Extract timestamps from batch data
                    # IMPORTANT: Use the ORIGINAL batches_df from session state (with datetime objects),
                    # NOT the formatted batches_df copy (with string timestamps for display)
                    if st.session_state.batches_df is not None and not st.session_state.batches_df.empty and len(selected_batches) > 0:
                        # Get the first selected batch row (assuming single batch analysis for now)
                        batch_row = st.session_state.batches_df[st.session_state.batches_df['batch_id'].astype(str).isin([str(b) for b in selected_batches])].iloc[0]

                        # Extract timestamps (these are datetime objects, NOT strings)
                        cell_timestamps = {
                            'print_start': batch_row.get('print_start'),
                            'print_end': batch_row.get('print_end'),
                            'cut_start': batch_row.get('cut_start'),
                            'cut_end': batch_row.get('cut_end'),
                            'pick_start': batch_row.get('pick_start'),
                            'pick_end': batch_row.get('pick_end')
                        }
                        
                        # Calculate system start (earliest) and end (latest)
                        starts = [cell_timestamps['print_start'], cell_timestamps['cut_start'], cell_timestamps['pick_start']]
                        ends = [cell_timestamps['print_end'], cell_timestamps['cut_end'], cell_timestamps['pick_end']]
                        
                        starts = [s for s in starts if pd.notna(s)]
                        ends = [e for e in ends if pd.notna(e)]
                        
                        if starts:
                            cell_timestamps['system_start'] = min(starts)
                        if ends:
                            cell_timestamps['system_end'] = max(ends)
                        
                        st.session_state.cell_timestamps = cell_timestamps
                        logger.info(f"Extracted cell timestamps: {cell_timestamps}")

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
    
    # Ensure pick_df is a DataFrame (not None)
    if pick_df is None:
        pick_df = pd.DataFrame()

    # Calculate metrics
    # Get quality_breakdown from session state if available
    quality_breakdown = st.session_state.get('quality_breakdown', pd.DataFrame())
    batch_metrics_df = calculate_batch_metrics(
        print_df, cut_df, pick_df,
        st.session_state.selected_batches,
        quality_breakdown=quality_breakdown
    )

    daily_metrics = calculate_daily_metrics(batch_metrics_df, operating_hours)

    # ========================================================================
    # 4.1: BATCH-LEVEL METRICS
    # ========================================================================

    st.subheader("📦 Batch-Level OEE Metrics")

    # Add custom CSS to increase font size in the dataframe
    st.markdown("""
        <style>
        /* Target the batch metrics dataframe specifically */
        div[data-testid="stDataFrame"] table {
            font-size: 16px !important;
        }
        div[data-testid="stDataFrame"] table td {
            font-size: 16px !important;
            font-weight: 500 !important;
        }
        div[data-testid="stDataFrame"] table th {
            font-size: 14px !important;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Select columns for display, including actual performance
    display_columns = [
        'batch_id',
        'print_oee', 'print_availability', 'print_performance', 'print_performance_actual', 'print_quality',
        'cut_oee', 'cut_availability', 'cut_performance', 'cut_quality',
        'pick_oee', 'pick_availability', 'pick_performance', 'pick_quality'
    ]
    
    # Filter to only include columns that exist in the DataFrame
    available_columns = [col for col in display_columns if col in batch_metrics_df.columns]
    
    # Create a formatted copy for display with larger, more readable numbers
    display_df = batch_metrics_df[available_columns].copy()
    
    # Format percentage columns to show 1 decimal place with % symbol
    percentage_columns = [
        'print_oee', 'print_availability', 'print_performance', 'print_performance_actual', 'print_quality',
        'cut_oee', 'cut_availability', 'cut_performance', 'cut_quality',
        'pick_oee', 'pick_availability', 'pick_performance', 'pick_quality'
    ]
    
    for col in percentage_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "batch_id": st.column_config.NumberColumn("Batch ID", width="small"),
            "print_oee": st.column_config.TextColumn("Print OEE", width="medium"),
            "print_availability": st.column_config.TextColumn("Print Avail.", width="medium"),
            "print_performance": st.column_config.TextColumn("Print Perf. (Capped)", width="medium"),
            "print_performance_actual": st.column_config.TextColumn("Print Perf. (Actual)", width="medium"),
            "print_quality": st.column_config.TextColumn("Print Quality", width="medium"),
            "cut_oee": st.column_config.TextColumn("Cut OEE", width="medium"),
            "cut_availability": st.column_config.TextColumn("Cut Avail.", width="medium"),
            "cut_performance": st.column_config.TextColumn("Cut Perf.", width="medium"),
            "cut_quality": st.column_config.TextColumn("Cut Quality", width="medium"),
            "pick_oee": st.column_config.TextColumn("Pick OEE", width="medium"),
            "pick_availability": st.column_config.TextColumn("Pick Avail.", width="medium"),
            "pick_performance": st.column_config.TextColumn("Pick Perf.", width="medium"),
            "pick_quality": st.column_config.TextColumn("Pick Quality", width="medium"),
        }
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
            if pick_df is not None and not pick_df.empty:
                # Select columns for display
                display_columns = [
                    'job_id', 'batch_id', 'sheet_index',
                    'job_active_window_duration', 'successful_picks_time',
                    'order_count', 'components_completed', 'components_failed', 'components_ignored', 
                    'avg_component_pick_time_s'
                ]
                
                # Filter to only columns that exist
                available_columns = [col for col in display_columns if col in pick_df.columns]
                display_df = pick_df[available_columns].copy()
                
                st.dataframe(
                    display_df.style.format({
                        'job_active_window_duration': '{:.2f}s',
                        'successful_picks_time': '{:.2f}s',
                        'avg_component_pick_time_s': '{:.2f}s'
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

    # ========================================================================
    # 4.5: EQUIPMENT STATE ANALYSIS
    # ========================================================================
    
    if 'cell_timestamps' in st.session_state and st.session_state.cell_timestamps:
        st.markdown("---")
        st.header("5️⃣ Equipment State Analysis")
        
        cell_timestamps = st.session_state.cell_timestamps
        cell_data = {}
        
        with st.spinner("Loading equipment state data..."):
            cells_config = [
                ('Print1', 'print_start', 'print_end'),
                ('Cut1', 'cut_start', 'cut_end'),
                ('Pick1', 'pick_start', 'pick_end')
            ]
            
            for cell_name, start_key, end_key in cells_config:
                start_ts = cell_timestamps.get(start_key)
                end_ts = cell_timestamps.get(end_key)
                
                if start_ts and end_ts:
                    try:
                        # Convert to datetime if string
                        if isinstance(start_ts, str):
                            start_ts = dateutil_parser.parse(start_ts)
                        if isinstance(end_ts, str):
                            end_ts = dateutil_parser.parse(end_ts)
                        
                        # Normalize timestamps to UTC+1 (Europe/Copenhagen) for consistent querying
                        from dateutil.tz import gettz
                        copenhagen_tz = gettz('Europe/Copenhagen')
                        
                        # Ensure timestamps are timezone-aware in UTC+1
                        if start_ts.tzinfo is None:
                            # Naive datetime - assume it's already in UTC+1 and localize
                            start_ts = start_ts.replace(tzinfo=copenhagen_tz)
                        else:
                            # Timezone-aware - convert to UTC+1
                            start_ts = start_ts.astimezone(copenhagen_tz)
                        
                        if end_ts.tzinfo is None:
                            end_ts = end_ts.replace(tzinfo=copenhagen_tz)
                        else:
                            end_ts = end_ts.astimezone(copenhagen_tz)
                        
                        # Pass timestamps to PostgreSQL with timezone info (as ISO format strings)
                        # PostgreSQL will handle the timezone conversion internally
                        states_df = fetch_equipment_states_with_durations(
                            cell=cell_name,
                            start_ts=start_ts.isoformat(),
                            end_ts=end_ts.isoformat()
                        )
                        
                        if not states_df.empty:
                            # Calculate hourly breakdown
                            hourly_df = calculate_hourly_state_breakdown(
                                states_df, start_ts, end_ts
                            )
                            
                            # Calculate summary (pass start_ts, end_ts, and break window)
                            break_start_time = st.session_state.get('break_start')
                            break_end_time = st.session_state.get('break_end')

                            # Convert break times to full datetime objects if provided
                            break_start_dt = None
                            break_end_dt = None
                            if break_start_time and break_end_time:
                                from datetime import datetime, timedelta
                                # Use the date from start_ts to create break datetime
                                break_date = start_ts.date()
                                break_start_dt = datetime.combine(break_date, break_start_time)
                                break_end_dt = datetime.combine(break_date, break_end_time)

                                # Ensure timezone-aware (match the cell timestamps)
                                from dateutil.tz import gettz
                                copenhagen_tz = gettz('Europe/Copenhagen')
                                if break_start_dt.tzinfo is None:
                                    break_start_dt = break_start_dt.replace(tzinfo=copenhagen_tz)
                                if break_end_dt.tzinfo is None:
                                    break_end_dt = break_end_dt.replace(tzinfo=copenhagen_tz)

                            summary = calculate_state_summary(
                                states_df,
                                start_ts,
                                end_ts,
                                break_start=break_start_dt,
                                break_end=break_end_dt
                            )
                            
                            cell_data[cell_name] = {
                                'hourly_df': hourly_df,
                                'summary': summary,
                                'start_ts': start_ts,
                                'end_ts': end_ts,
                                'states_df': states_df
                            }
                        else:
                            logger.warning(f"No equipment state data found for {cell_name}")
                    except Exception as e:
                        logger.error(f"Error processing {cell_name} equipment states: {e}", exc_info=True)
                        st.error(f"Error loading {cell_name} equipment states: {e}")
        
        # Store in session state for throughput calculation
        st.session_state.equipment_state_data = cell_data
        
        # Display timelines
        if cell_data:
            # Get break times from session state and convert to datetime
            break_start_time = st.session_state.get('break_start')
            break_end_time = st.session_state.get('break_end')

            break_start_dt = None
            break_end_dt = None
            if break_start_time and break_end_time:
                from datetime import datetime
                from dateutil.tz import gettz
                copenhagen_tz = gettz('Europe/Copenhagen')

                # Use system start date for break datetime
                if 'cell_timestamps' in st.session_state and st.session_state.cell_timestamps:
                    system_start = st.session_state.cell_timestamps.get('system_start')
                    if system_start:
                        break_date = system_start.date()
                        break_start_dt = datetime.combine(break_date, break_start_time)
                        break_end_dt = datetime.combine(break_date, break_end_time)

                        # Ensure timezone-aware
                        if break_start_dt.tzinfo is None:
                            break_start_dt = break_start_dt.replace(tzinfo=copenhagen_tz)
                        if break_end_dt.tzinfo is None:
                            break_end_dt = break_end_dt.replace(tzinfo=copenhagen_tz)

            display_all_cell_timelines(cell_data, break_start=break_start_dt, break_end=break_end_dt)
        else:
            st.warning("No equipment state data available for selected batches")
    
    # ========================================================================
    # 4.6: THROUGHPUT ANALYSIS
    # ========================================================================
    
    if ('equipment_state_data' in st.session_state and 
        st.session_state.equipment_state_data and
        st.session_state.print_df is not None):
        
        st.markdown("---")
        st.header("6️⃣ Throughput Analysis")
        
        with st.spinner("Calculating throughput metrics..."):
            # Prepare data for throughput calculation
            print_data = None
            cut_data = None
            pick_data = None
            
            equipment_data = st.session_state.equipment_state_data
            
            # Prepare cell data (all cells process the same total components)
            # Component count comes from batch_structure_df, not individual cell data
            if 'Print1' in equipment_data:
                print_data = {
                    'state_summary': equipment_data['Print1']['summary'],
                    'start_ts': equipment_data['Print1']['start_ts'],
                    'end_ts': equipment_data['Print1']['end_ts']
                }
            
            if 'Cut1' in equipment_data:
                cut_data = {
                    'state_summary': equipment_data['Cut1']['summary'],
                    'start_ts': equipment_data['Cut1']['start_ts'],
                    'end_ts': equipment_data['Cut1']['end_ts']
                }
            
            if 'Pick1' in equipment_data:
                pick_data = {
                    'state_summary': equipment_data['Pick1']['summary'],
                    'start_ts': equipment_data['Pick1']['start_ts'],
                    'end_ts': equipment_data['Pick1']['end_ts']
                }
            
            # Calculate system throughput
            throughput_results = calculate_system_throughput(
                print_data=print_data,
                cut_data=cut_data,
                pick_data=pick_data,
                batch_structure_df=st.session_state.get('batch_structure_df'),
                fpy_df=st.session_state.get('fpy_df')
            )
            
            # Display throughput metrics
            display_throughput_metrics(throughput_results)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(f"🏭 Batch-Centric OEE Analysis | Timezone: {Config.TIMEZONE} | Data: HISTORIAN + INSIDE")
