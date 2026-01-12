"""
Manufacturing OEE Analytics v2 - Main Application

Simplified OEE calculation and analysis tool for three manufacturing cells:
- Printer: OEE = Availability × Performance × 100%
- Cut: OEE = Availability × 100% × 100%
- Pick: OEE = Availability × 100% × Quality_accuracy
"""

import streamlit as st
import pandas as pd
import logging
import io
from typing import Optional
from datetime import date

logger = logging.getLogger(__name__)

from utils.config import load_config, validate_config
from core.db.fetchers import (
    fetch_print_data_by_batch,
    fetch_print_data_by_time_window,
    fetch_cut_data_by_jobs,
    fetch_pick_data_by_batch,
    fetch_quality_metrics_by_batch,
    fetch_batch_quality_breakdown,
    fetch_batch_quality_breakdown_v2
)
from core.calculations.oee import (
    calculate_cell_oee,
    aggregate_batch_metrics,
    calculate_performance_ratio,
    calculate_quality_accuracy,
    calculate_reconciled_cell_oee,
    OEEMetrics,
    BatchQualityBreakdown,
    CellOEEReconciled
)
from ui.time_window_selector import render_time_window_editor
from ui.metrics_display import (
    render_cell_metrics_card,
    render_printer_metrics_detailed,
    render_cut_metrics_detailed,
    render_pick_metrics_detailed,
    render_reconciled_oee_comparison,
    render_batch_quality_breakdown
)
from ui.log_display import LogCollector, render_compact_log_area

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
load_config()

# Validate configuration
config_errors = validate_config()
if config_errors:
    st.error("❌ Configuration errors detected:")
    for error in config_errors:
        st.error(error)
    st.stop()

# Streamlit page config
st.set_page_config(
    page_title="Manufacturing OEE Analytics v2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app title
st.title("🏭 Manufacturing OEE Analytics v2")
st.markdown("**Simplified OEE calculations for Printer, Cut, and Pick cells**")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool calculates OEE (Overall Equipment Effectiveness) for three manufacturing cells:
    
    - **Printer**: A × P × 100%
    - **Cut**: A × 100% × 100%
    - **Pick**: A × 100% × Q_acc%
    
    **Workflow:**
    1. Configure time windows for each cell
    2. The app automatically detects which batches were active
    3. OEE calculations are based on the detected batches
    """)

# Main content
st.divider()

# 1. Time Window Configuration (per cell) - NOW FIRST
st.header("⏰ Time Window Configuration")
st.markdown("Enable cells and configure time windows. The app will automatically detect which batches were active during these time periods.")

col1, col2, col3 = st.columns(3)

with col1:
    printer_window = render_time_window_editor('printer', key_prefix='printer_window')
    printer_enabled = st.session_state.get('printer_window_enabled', True)

with col2:
    cut_window = render_time_window_editor('cut', key_prefix='cut_window')
    cut_enabled = st.session_state.get('cut_window_enabled', False)

with col3:
    pick_window = render_time_window_editor('pick', key_prefix='pick_window')
    pick_enabled = st.session_state.get('pick_window_enabled', False)

st.divider()

# Initialize log collector
log_collector = LogCollector("app_logs")

# 2. Load Data button - fetches data based on time windows
if st.button("📊 Load Data & Calculate OEE", type="primary", use_container_width=True):
    # Clear previous logs
    log_collector.clear()
    
    # Check which cells are enabled
    enabled_cells = []
    if printer_enabled and printer_window and printer_window.segments:
        enabled_cells.append('printer')
    if cut_enabled and cut_window and cut_window.segments:
        enabled_cells.append('cut')
    if pick_enabled and pick_window and pick_window.segments:
        enabled_cells.append('pick')
    
    if not enabled_cells:
        log_collector.add_error("Please enable and configure at least one cell's time window before loading data.")
        st.error("❌ Please enable and configure at least one cell's time window before loading data.")
        st.stop()
    
    log_collector.add_info(f"Analyzing {len(enabled_cells)} enabled cell(s): {', '.join([c.capitalize() for c in enabled_cells])}")
    
    with st.spinner("Loading data and calculating OEE metrics..."):
        try:
            print_df = pd.DataFrame()
            cut_df = pd.DataFrame()
            pick_df = pd.DataFrame()
            job_ids = []
            detected_batches = []
            
            # Fetch print data if printer is enabled
            if 'printer' in enabled_cells:
                # Step 1: Use time window to discover which batches were active
                log_collector.add_info("Step 1: Discovering batches active during the time window...")
                discovery_df = fetch_print_data_by_time_window(printer_window)
                
                if discovery_df.empty:
                    log_collector.add_error("No print data found for the configured time window.")
                    st.error("❌ No print data found for the configured time window.")
                    st.stop()
                
                log_collector.add_info(f"Found {len(discovery_df)} jobs that ran during the time window")
                
                # Extract batch IDs from jobs that ran in the time window
                if 'batchId' in discovery_df.columns:
                    detected_batches = discovery_df['batchId'].dropna().unique().tolist()
                    log_collector.add_success(f"Detected {len(detected_batches)} batch(es) with activity: {', '.join(map(str, detected_batches[:10]))}")
                    if len(detected_batches) > 10:
                        log_collector.add_info(f"... and {len(detected_batches) - 10} more batches")
                else:
                    log_collector.add_warning("No batch IDs found in print data")
                    detected_batches = []
                
                # Step 2: Fetch ALL jobs for these batches (complete batch analysis)
                if detected_batches:
                    log_collector.add_info(f"Step 2: Fetching all jobs for {len(detected_batches)} batch(es)...")
                    print_df = fetch_print_data_by_batch(detected_batches, time_window=None)
                    
                    if print_df.empty:
                        log_collector.add_warning("No jobs found for detected batches")
                        print_df = discovery_df  # Fallback
                    else:
                        log_collector.add_success(f"Loaded {len(print_df)} print records (all jobs for detected batches)")
                        
                        # Calculate actual batch time span per batch
                        if 'timing_start' in print_df.columns and 'timing_end' in print_df.columns and 'batchId' in print_df.columns:
                            batch_time_spans = {}
                            for batch_id in detected_batches:
                                batch_data = print_df[print_df['batchId'] == batch_id]
                                if not batch_data.empty:
                                    batch_start = batch_data['timing_start'].min()
                                    batch_end = batch_data['timing_end'].max()
                                    if pd.notna(batch_start) and pd.notna(batch_end):
                                        batch_duration = (batch_end - batch_start).total_seconds() / 60.0
                                        batch_duration_hours = batch_duration / 60.0
                                        batch_time_spans[batch_id] = {
                                            'start': batch_start,
                                            'end': batch_end,
                                            'duration_minutes': batch_duration,
                                            'duration_hours': batch_duration_hours,
                                            'job_count': len(batch_data)
                                        }
                                        log_collector.add_info(
                                            f"Batch {batch_id}: {batch_start.strftime('%Y-%m-%d %H:%M:%S')} → "
                                            f"{batch_end.strftime('%Y-%m-%d %H:%M:%S')} ({batch_duration_hours:.1f} hours, {len(batch_data)} jobs)"
                                        )
                            
                            # Store batch time spans
                            st.session_state['batch_time_spans'] = batch_time_spans
                else:
                    print_df = discovery_df  # Fallback
                
                # Extract job IDs for cut and pick queries
                job_ids = print_df['jobId'].dropna().unique().tolist()
                log_collector.add_info(f"Found {len(job_ids)} unique job IDs")
            
            # Fetch cut data if cut is enabled
            if 'cut' in enabled_cells:
                if not job_ids and 'printer' not in enabled_cells:
                    log_collector.add_warning("Cut cell requires job IDs. Enable Printer cell or provide job IDs.")
                    cut_df = pd.DataFrame()
                else:
                    log_collector.add_info("Fetching cut data...")
                    if job_ids:
                        cut_df = fetch_cut_data_by_jobs(job_ids, cut_window)
                        log_collector.add_success(f"Loaded {len(cut_df)} cut records")
                    else:
                        log_collector.add_warning("No job IDs available for cut data")
                        cut_df = pd.DataFrame()
            
            # Fetch pick data if pick is enabled
            if 'pick' in enabled_cells:
                if not detected_batches:
                    log_collector.add_warning("Pick cell requires batch IDs. Enable Printer cell to detect batches.")
                    pick_df = pd.DataFrame()
                else:
                    log_collector.add_info("Fetching pick data...")
                    if detected_batches:
                        pick_df = fetch_pick_data_by_batch(detected_batches, pick_window)
                        log_collector.add_success(f"Loaded {len(pick_df)} pick records")
                    else:
                        log_collector.add_warning("No batch IDs available for pick data")
                        pick_df = pd.DataFrame()
            
            # Fetch quality metrics for detected batches
            if detected_batches:
                log_collector.add_info("Fetching quality metrics...")
                quality_df = fetch_quality_metrics_by_batch(detected_batches)
                log_collector.add_success(f"Loaded quality metrics")
            else:
                quality_df = pd.DataFrame()
            
            # Store data in session state
            st.session_state['print_df'] = print_df
            st.session_state['cut_df'] = cut_df
            st.session_state['pick_df'] = pick_df
            st.session_state['quality_df'] = quality_df
            st.session_state['job_ids'] = job_ids
            st.session_state['detected_batches'] = detected_batches
            st.session_state['enabled_cells'] = enabled_cells
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            log_collector.add_error(f"Error loading data: {e}")
            st.error(f"❌ Error loading data: {e}")
            st.stop()

# Show log area if there are logs
if 'app_logs' in st.session_state and st.session_state['app_logs']:
    st.divider()
    render_compact_log_area(log_collector)

# 3. Display Detected Batches and Batch Time Spans
if 'detected_batches' in st.session_state and st.session_state['detected_batches']:
    st.divider()
    st.header("📦 Detected Batches")
    detected_batches = st.session_state['detected_batches']
    batch_time_spans = st.session_state.get('batch_time_spans', {})
    
    if len(detected_batches) == 1:
        st.info(f"**1 batch** was active during the configured time window:")
        batch_id = detected_batches[0]
        st.code(batch_id, language=None)
        
        if batch_id in batch_time_spans:
            span = batch_time_spans[batch_id]
            st.success(
                f"⏰ **Batch Execution Time:** "
                f"{span['start'].strftime('%Y-%m-%d %H:%M:%S')} → "
                f"{span['end'].strftime('%Y-%m-%d %H:%M:%S')} "
                f"({span['duration_hours']:.1f} hours, {span['job_count']} jobs)"
            )
            st.caption("ℹ️ OEE calculations are based on the actual batch execution time, not the configured time window.")
    else:
        st.info(f"**{len(detected_batches)} batches** were active during the configured time windows:")
        
        # Display batch details in a table
        batch_data = []
        for batch_id in detected_batches:
            if batch_id in batch_time_spans:
                span = batch_time_spans[batch_id]
                batch_data.append({
                    'Batch ID': batch_id,
                    'Start Time': span['start'].strftime('%Y-%m-%d %H:%M:%S'),
                    'End Time': span['end'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Duration (hours)': f"{span['duration_hours']:.1f}",
                    'Job Count': span['job_count']
                })
            else:
                batch_data.append({
                    'Batch ID': batch_id,
                    'Start Time': 'N/A',
                    'End Time': 'N/A',
                    'Duration (hours)': 'N/A',
                    'Job Count': 'N/A'
                })
        
        if batch_data:
            batch_df = pd.DataFrame(batch_data)
            st.dataframe(batch_df, use_container_width=True, hide_index=True)
        
        st.caption("ℹ️ OEE calculations will be aggregated across all batches. Each batch's execution time is calculated from its first job start to last job end.")

# 4. Calculate and Display OEE Metrics
enabled_cells = st.session_state.get('enabled_cells', [])
has_data = False

if enabled_cells:
    # Check if we have data for any enabled cell
    if 'printer' in enabled_cells and 'print_df' in st.session_state and not st.session_state['print_df'].empty:
        has_data = True
    elif 'cut' in enabled_cells and 'cut_df' in st.session_state and not st.session_state['cut_df'].empty:
        has_data = True
    elif 'pick' in enabled_cells and 'pick_df' in st.session_state and not st.session_state['pick_df'].empty:
        has_data = True

if has_data:
    st.divider()
    st.header("📊 OEE Metrics")
    
    print_df = st.session_state.get('print_df', pd.DataFrame())
    cut_df = st.session_state.get('cut_df', pd.DataFrame())
    pick_df = st.session_state.get('pick_df', pd.DataFrame())
    quality_df = st.session_state.get('quality_df', pd.DataFrame())
    detected_batches = st.session_state.get('detected_batches', [])
    
    # Check if we have multiple batches
    has_multiple_batches = len(detected_batches) > 1 and 'batchId' in print_df.columns if not print_df.empty else False
    
    printer_oee = None
    cut_oee = None
    pick_oee = None
    
    # If multiple batches, show per-batch breakdown first
    if has_multiple_batches and 'printer' in enabled_cells:
        st.subheader("📋 Per-Batch OEE Breakdown")
        st.info(f"Showing OEE metrics for each of the {len(detected_batches)} detected batches:")
        
        batch_oee_data = []
        for batch_id in detected_batches:
            batch_print_df = print_df[print_df['batchId'] == batch_id]
            if not batch_print_df.empty:
                batch_uptime = float(batch_print_df['uptime (min)'].sum()) if 'uptime (min)' in batch_print_df.columns else 0.0
                batch_downtime = float(batch_print_df['downtime (min)'].sum()) if 'downtime (min)' in batch_print_df.columns else 0.0
                
                batch_performance = 1.0
                if 'nominalSpeed' in batch_print_df.columns and 'speed' in batch_print_df.columns:
                    total_nominal = float(batch_print_df['nominalSpeed'].sum())
                    total_actual = float(batch_print_df['speed'].sum())
                    if total_nominal > 0:
                        batch_performance = min(total_actual / total_nominal, 1.0)
                
                batch_oee_metrics = calculate_cell_oee('printer', batch_uptime, batch_downtime, performance_ratio=batch_performance)
                
                batch_oee_data.append({
                    'Batch ID': batch_id,
                    'Jobs': len(batch_print_df),
                    'Availability (%)': f"{batch_oee_metrics.availability * 100:.1f}",
                    'Performance (%)': f"{batch_oee_metrics.performance * 100:.1f}",
                    'Quality (%)': f"{batch_oee_metrics.quality * 100:.1f}",
                    'OEE (%)': f"{batch_oee_metrics.oee * 100:.1f}"
                })
        
        if batch_oee_data:
            batch_oee_df = pd.DataFrame(batch_oee_data)
            st.dataframe(batch_oee_df, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("📊 Aggregated OEE Metrics (All Batches Combined)")
    
    # Calculate aggregated Printer OEE (all batches combined)
    if 'printer' in enabled_cells and not print_df.empty:
        printer_uptime = float(print_df['uptime (min)'].sum()) if 'uptime (min)' in print_df.columns else 0.0
        printer_downtime = float(print_df['downtime (min)'].sum()) if 'downtime (min)' in print_df.columns else 0.0
        
        # Calculate performance ratio for printer (aggregated across all batches)
        printer_performance = 1.0
        if 'nominalSpeed' in print_df.columns and 'speed' in print_df.columns:
            total_nominal = float(print_df['nominalSpeed'].sum())
            total_actual = float(print_df['speed'].sum())
            if total_nominal > 0:
                printer_performance = min(total_actual / total_nominal, 1.0)
        
        # For printer: Quality is fixed at 100% (per specifications: Printer OEE = A × P × 100%)
        # Note: The quality_df from fetch_quality_metrics_by_batch() contains batch-level quality metrics
        # (quality_rate_inclusive_percent), but this is NOT used for printer OEE calculation.
        # Printer quality is always 100% by design. The quality data is used for Pick cell OEE instead.
        printer_oee = calculate_cell_oee(
            'printer',
            printer_uptime,
            printer_downtime,
            performance_ratio=printer_performance
        )
        # Store in session state for reconciliation
        st.session_state['printer_oee'] = printer_oee
    
    # Calculate aggregated Cut OEE (all batches combined)
    if 'cut' in enabled_cells and not cut_df.empty:
        cut_uptime = float(cut_df['uptime (min)'].sum()) if 'uptime (min)' in cut_df.columns else 0.0
        cut_downtime = float(cut_df['downtime (min)'].sum()) if 'downtime (min)' in cut_df.columns else 0.0
        cut_oee = calculate_cell_oee('cut', cut_uptime, cut_downtime)
        # Store in session state for reconciliation
        st.session_state['cut_oee'] = cut_oee
    
    # Calculate aggregated Pick OEE (all batches combined)
    if 'pick' in enabled_cells and not pick_df.empty:
        # Uptime/downtime are calculated from equipment state, not per job
        # All job rows have the same values, so take the first (or max) instead of summing
        if 'uptime (min)' in pick_df.columns and 'downtime (min)' in pick_df.columns:
            pick_uptime = float(pick_df['uptime (min)'].iloc[0]) if len(pick_df) > 0 else 0.0
            pick_downtime = float(pick_df['downtime (min)'].iloc[0]) if len(pick_df) > 0 else 0.0
        else:
            pick_uptime = 0.0
            pick_downtime = 0.0
        
        # Pick OEE = Availability × Pick Quality Accuracy × Quality (100%)
        # Quality is always 100% for individual cell OEE calculation
        # Pick Quality Accuracy = successful_picks / total_picks
        # Now we have component-level data with pick_status, so we count successful vs total
        pick_quality_accuracy = 1.0  # Default to 100% if data not available
        if 'pick_status' in pick_df.columns:
            # Count successful picks (pick_status = 'successful') vs total picks (successful + failed)
            total_successful = int((pick_df['pick_status'] == 'successful').sum())
            total_picks = int((pick_df['pick_status'].isin(['successful', 'failed'])).sum())
            
            log_collector.add_info(f"Pick quality calculation: {len(pick_df)} components, {total_successful} successful, {total_picks} total picks")
            
            if total_picks > 0:
                pick_quality_accuracy = calculate_quality_accuracy(total_successful, total_picks)
                log_collector.add_info(f"Pick Quality: {total_successful}/{total_picks} successful picks ({pick_quality_accuracy * 100:.1f}%)")
            else:
                log_collector.add_warning("No pick attempts found (all components are 'not_picked'), using 100%")
        elif 'successful_picks' in pick_df.columns and 'total_picks' in pick_df.columns:
            # Fallback: use pre-calculated columns if available
            total_successful = int(pick_df['successful_picks'].sum()) if pick_df['successful_picks'].notna().any() else 0
            total_picks = int(pick_df['total_picks'].sum()) if pick_df['total_picks'].notna().any() else 0
            if total_picks > 0:
                pick_quality_accuracy = calculate_quality_accuracy(total_successful, total_picks)
                log_collector.add_info(f"Pick Quality (fallback): {total_successful}/{total_picks} successful picks ({pick_quality_accuracy * 100:.1f}%)")
            else:
                log_collector.add_warning("No pick data available for quality calculation, using 100%")
        else:
            log_collector.add_warning("Pick quality metrics (pick_status) not found in data, using 100%")
        
        # Note: quality_ratio parameter in calculate_cell_oee is used as "Pick Quality Accuracy" for pick cell
        pick_oee = calculate_cell_oee('pick', pick_uptime, pick_downtime, quality_ratio=pick_quality_accuracy)
        # Store in session state for reconciliation
        st.session_state['pick_oee'] = pick_oee
    
    # Check if reconciliation is complete and get reconciled OEEs
    reconciliation_complete = st.session_state.get('reconciliation_complete', False)
    reconciled_oees_all = st.session_state.get('reconciled_oees', {})
    
    # For single batch, use that batch's reconciled OEEs; for multiple batches, aggregate
    reconciled_oees = {}
    if reconciliation_complete and reconciled_oees_all and len(reconciled_oees_all) > 0:
        # Try to match batch ID first
        matched = False
        for batch_id_str in detected_batches:
            # Try as integer (most likely - this is how it's stored)
            try:
                batch_id_int = int(str(batch_id_str).replace('B', '').replace('b', '').strip())
                if batch_id_int in reconciled_oees_all:
                    matched_value = reconciled_oees_all[batch_id_int]
                    if matched_value and isinstance(matched_value, dict) and len(matched_value) > 0:
                        reconciled_oees = matched_value
                        matched = True
                        logger.info(f"Matched batch ID {batch_id_str} as integer {batch_id_int}, got {len(reconciled_oees)} reconciled OEEs")
                        break
            except (ValueError, AttributeError):
                pass
            
            # Try as string
            if batch_id_str in reconciled_oees_all:
                matched_value = reconciled_oees_all[batch_id_str]
                if matched_value and isinstance(matched_value, dict) and len(matched_value) > 0:
                    reconciled_oees = matched_value
                    matched = True
                    logger.info(f"Matched batch ID {batch_id_str} as string, got {len(reconciled_oees)} reconciled OEEs")
                    break
            
            # Try with 'B' prefix
            batch_with_b = f"B{batch_id_str}" if not str(batch_id_str).startswith('B') else batch_id_str
            if batch_with_b in reconciled_oees_all:
                matched_value = reconciled_oees_all[batch_with_b]
                if matched_value and isinstance(matched_value, dict) and len(matched_value) > 0:
                    reconciled_oees = matched_value
                    matched = True
                    logger.info(f"Matched batch ID {batch_id_str} as {batch_with_b}, got {len(reconciled_oees)} reconciled OEEs")
                    break
        
        # Fallback: Always use the first reconciled OEE if we have any and haven't matched
        # This is safe for single batch scenarios and handles format mismatches
        if not matched:
            first_key = list(reconciled_oees_all.keys())[0]
            first_value = reconciled_oees_all[first_key]
            if first_value and isinstance(first_value, dict) and len(first_value) > 0:
                reconciled_oees = first_value
                logger.info(f"Fallback: Using first key {first_key} (type: {type(first_key)}), value type: {type(first_value)}, contains: {list(reconciled_oees.keys())}")
            else:
                logger.warning(f"Fallback: First value is not a valid dict. Type: {type(first_value)}, Value: {first_value}")
    
    # Debug: Show what we found (can be removed later)
    if reconciliation_complete:
        with st.expander("🔍 Debug: Reconciliation Status", expanded=False):
            st.write(f"Reconciliation complete: {reconciliation_complete}")
            st.write(f"Detected batches: {detected_batches}")
            st.write(f"Reconciled OEEs all exists: {reconciled_oees_all is not None}")
            st.write(f"Reconciled OEEs all keys: {list(reconciled_oees_all.keys()) if reconciled_oees_all else 'None'}")
            st.write(f"Reconciled OEEs all type: {type(reconciled_oees_all)}")
            st.write(f"Reconciled OEEs all length: {len(reconciled_oees_all) if reconciled_oees_all else 0}")
            st.write(f"Condition met: {reconciliation_complete and reconciled_oees_all and len(reconciled_oees_all) > 0}")
            
            if reconciled_oees_all and len(reconciled_oees_all) > 0:
                first_key = list(reconciled_oees_all.keys())[0]
                st.write(f"First key: {first_key}, type: {type(first_key)}")
                first_value = reconciled_oees_all[first_key]
                st.write(f"First key value type: {type(first_value)}")
                st.write(f"First key value is dict: {isinstance(first_value, dict)}")
                if isinstance(first_value, dict):
                    st.write(f"First key value keys: {list(first_value.keys())}")
                    st.write(f"First key value length: {len(first_value)}")
                    if 'printer' in first_value:
                        st.write(f"Printer in first value: {type(first_value['printer'])}")
                        if hasattr(first_value['printer'], 'quality_attributed'):
                            st.write(f"Printer quality: {first_value['printer'].quality_attributed * 100:.1f}%")
                else:
                    st.write(f"First key value (not dict): {first_value}")
            
            st.write(f"---")
            st.write(f"Found reconciled OEEs: {list(reconciled_oees.keys()) if reconciled_oees else 'None'}")
            st.write(f"Reconciled OEEs type: {type(reconciled_oees)}")
            st.write(f"Reconciled OEEs length: {len(reconciled_oees) if reconciled_oees else 0}")
            if reconciled_oees and 'printer' in reconciled_oees:
                st.write(f"Printer reconciled quality: {reconciled_oees['printer'].quality_attributed * 100:.1f}%")
                st.write(f"Printer reconciled OEE: {reconciled_oees['printer'].oee_reconciled * 100:.1f}%")
            elif reconciled_oees:
                st.write(f"Reconciled OEEs keys found: {list(reconciled_oees.keys())}")
                st.write(f"Reconciled OEEs content: {reconciled_oees}")
    
    # Final check: If we have reconciled_oees_all but reconciled_oees is still empty, force use first key
    if reconciliation_complete and reconciled_oees_all and len(reconciled_oees_all) > 0 and len(reconciled_oees) == 0:
        first_key = list(reconciled_oees_all.keys())[0]
        reconciled_oees = reconciled_oees_all[first_key]
        logger.warning(f"Final fallback: Force using first key {first_key}, got {len(reconciled_oees)} reconciled OEEs")
    
    # Display metrics per cell (only for enabled cells)
    # Use reconciled OEE if available, otherwise use real-time OEE
    
    # Printer metrics (detailed format)
    if printer_oee and not print_df.empty:
        # Use reconciled OEE if available, otherwise real-time
        if reconciliation_complete and reconciled_oees and 'printer' in reconciled_oees:
            reconciled_printer = reconciled_oees['printer']
            # Create OEEMetrics object from reconciled data for display
            reconciled_display = OEEMetrics(
                availability=reconciled_printer.availability,
                performance=reconciled_printer.performance,
                quality=reconciled_printer.quality_attributed,
                oee=reconciled_printer.oee_reconciled
            )
            render_printer_metrics_detailed(reconciled_display, print_df)
        else:
            render_printer_metrics_detailed(printer_oee, print_df)
        
        st.divider()
    
    # Display other cells (cut, pick) using the detailed format
    if cut_oee:
        # Use reconciled OEE if available, otherwise real-time
        if reconciliation_complete and reconciled_oees and 'cut' in reconciled_oees:
            reconciled_cut = reconciled_oees['cut']
            reconciled_display = OEEMetrics(
                availability=reconciled_cut.availability,
                performance=reconciled_cut.performance,
                quality=reconciled_cut.quality_attributed,
                oee=reconciled_cut.oee_reconciled
            )
            render_cut_metrics_detailed(reconciled_display, cut_df)
        else:
            render_cut_metrics_detailed(cut_oee, cut_df)
        st.divider()
    
    if pick_oee:
        # Use reconciled OEE if available, otherwise real-time
        if reconciliation_complete and reconciled_oees and 'pick' in reconciled_oees:
            reconciled_pick = reconciled_oees['pick']
            reconciled_display = OEEMetrics(
                availability=reconciled_pick.availability,
                performance=reconciled_pick.performance,
                quality=reconciled_pick.quality_attributed,
                oee=reconciled_pick.oee_reconciled
            )
            render_pick_metrics_detailed(reconciled_display, pick_df)
        else:
            render_pick_metrics_detailed(pick_oee, pick_df)
        st.divider()
    
    st.divider()
    
    # 5. Phase 2: Post-QC Reconciliation (Optional)
    if detected_batches and len(detected_batches) > 0:
        st.header("🔄 Phase 2: Post-QC Reconciliation")
        st.info(
            "**Post-QC Reconciliation** attributes defects to specific cells based on QC event descriptions. "
            "This provides more accurate OEE calculations after batch QC is complete."
        )
        
        # Show quality breakdown if already calculated
        reconciliation_complete = st.session_state.get('reconciliation_complete', False)
        quality_breakdown_df = st.session_state.get('quality_breakdown_df', pd.DataFrame())
        
        if reconciliation_complete and not quality_breakdown_df.empty:
            st.success("✅ Post-QC Reconciliation has been completed. Quality breakdown and reconciled OEE are shown below.")
            # Display quality breakdown for each batch
            for _, row in quality_breakdown_df.iterrows():
                batch_id = int(row['production_batch_id'])
                prod_date = row['production_date']
                if isinstance(prod_date, pd.Timestamp):
                    production_date = prod_date.date()
                elif isinstance(prod_date, str):
                    from datetime import datetime
                    production_date = datetime.strptime(prod_date, '%Y-%m-%d').date()
                else:
                    production_date = prod_date
                
                breakdown = BatchQualityBreakdown(
                    production_batch_id=batch_id,
                    production_date=production_date,
                    total_components=int(row['total_components']),
                    total_passed=int(row['total_passed']),
                    total_failed=int(row['total_failed']),
                    not_scanned=int(row['not_scanned']),
                    print_defects=int(row['print_defects']),
                    cut_defects=int(row['cut_defects']),
                    pick_defects=int(row['pick_defects']),
                    fabric_defects=int(row.get('fabric_defects', 0)),
                    file_defects=int(row.get('file_defects', 0)),
                    other_defects=int(row.get('other_defects', 0)),
                    scanned_handheld=int(row.get('scanned_handheld', 0)),
                    scanned_manual=int(row.get('scanned_manual', 0)),
                    scanned_inside=int(row.get('scanned_inside', 0)),
                    quality_rate_checked_percent=float(row['quality_rate_checked_percent']) if pd.notna(row.get('quality_rate_checked_percent')) else None,
                    scan_coverage_percent=float(row.get('scan_coverage_percent', 0.0)),
                    unchecked_rate_percent=float(row.get('unchecked_rate_percent', 0.0))
                )
                render_batch_quality_breakdown(breakdown)
        
        if st.button("🔍 Run Post-QC Reconciliation", type="secondary"):
            with st.spinner("Running post-QC reconciliation with defect attribution..."):
                try:
                    # Fetch quality breakdown with defect attribution using two-database approach
                    breakdown_df = fetch_batch_quality_breakdown_v2(detected_batches)
                    
                    if breakdown_df.empty:
                        st.warning("⚠️ No quality breakdown data found. QC may not be complete for these batches.")
                    else:
                        log_collector.add_success(f"Loaded quality breakdown for {len(breakdown_df)} batch(es)")
                        
                        # Process each batch
                        for _, row in breakdown_df.iterrows():
                            batch_id = int(row['production_batch_id'])
                            
                            # Convert production_date to date object if needed
                            prod_date = row['production_date']
                            if isinstance(prod_date, pd.Timestamp):
                                production_date = prod_date.date()
                            elif isinstance(prod_date, str):
                                from datetime import datetime
                                production_date = datetime.strptime(prod_date, '%Y-%m-%d').date()
                            else:
                                production_date = prod_date
                            
                            # Create BatchQualityBreakdown object
                            breakdown = BatchQualityBreakdown(
                                production_batch_id=batch_id,
                                production_date=production_date,
                                total_components=int(row['total_components']),
                                total_passed=int(row['total_passed']),
                                total_failed=int(row['total_failed']),
                                not_scanned=int(row['not_scanned']),
                                print_defects=int(row['print_defects']),
                                cut_defects=int(row['cut_defects']),
                                pick_defects=int(row['pick_defects']),
                                fabric_defects=int(row.get('fabric_defects', 0)),
                                file_defects=int(row.get('file_defects', 0)),
                                other_defects=int(row.get('other_defects', 0)),
                                scanned_handheld=int(row.get('scanned_handheld', 0)),
                                scanned_manual=int(row.get('scanned_manual', 0)),
                                scanned_inside=int(row.get('scanned_inside', 0)),
                                quality_rate_checked_percent=float(row['quality_rate_checked_percent']) if pd.notna(row.get('quality_rate_checked_percent')) else None,
                                scan_coverage_percent=float(row.get('scan_coverage_percent', 0.0)),
                                unchecked_rate_percent=float(row.get('unchecked_rate_percent', 0.0))
                            )
                            
                            # Display quality breakdown
                            render_batch_quality_breakdown(breakdown)
                            
                            # Calculate reconciled OEE for each enabled cell using existing OEE metrics
                            # We already have availability and performance calculated, just need to apply quality from breakdown
                            reconciled_oees = {}
                            
                            # Get existing OEE metrics from session state (calculated earlier)
                            printer_oee_existing = st.session_state.get('printer_oee')
                            cut_oee_existing = st.session_state.get('cut_oee')
                            pick_oee_existing = st.session_state.get('pick_oee')
                            
                            # Printer: Use existing availability and performance, apply quality from breakdown
                            if 'printer' in enabled_cells and printer_oee_existing:
                                reconciled_printer = calculate_reconciled_cell_oee(
                                    'Printer',
                                    batch_id,
                                    production_date,
                                    printer_oee_existing.availability,  # Use existing availability
                                    printer_oee_existing.performance,  # Use existing performance
                                    breakdown  # Quality comes from breakdown
                                )
                                reconciled_oees['printer'] = reconciled_printer
                                logger.info(f"Batch {batch_id}: Calculated reconciled Printer OEE - Quality: {reconciled_printer.quality_attributed * 100:.1f}%, OEE: {reconciled_printer.oee_reconciled * 100:.1f}%")
                            
                            # Cut: Use existing availability, performance is always 100%, apply quality from breakdown
                            if 'cut' in enabled_cells and cut_oee_existing:
                                reconciled_cut = calculate_reconciled_cell_oee(
                                    'Cut',
                                    batch_id,
                                    production_date,
                                    cut_oee_existing.availability,  # Use existing availability
                                    1.0,  # Cut performance is always 100%
                                    breakdown  # Quality comes from breakdown
                                )
                                reconciled_oees['cut'] = reconciled_cut
                                logger.info(f"Batch {batch_id}: Calculated reconciled Cut OEE - Quality: {reconciled_cut.quality_attributed * 100:.1f}%, OEE: {reconciled_cut.oee_reconciled * 100:.1f}%")
                            
                            # Pick: Use existing availability, performance is always 100%, apply quality from breakdown
                            if 'pick' in enabled_cells and pick_oee_existing:
                                reconciled_pick = calculate_reconciled_cell_oee(
                                    'Pick',
                                    batch_id,
                                    production_date,
                                    pick_oee_existing.availability,  # Use existing availability
                                    1.0,  # Pick performance is always 100%
                                    breakdown  # Quality comes from breakdown
                                )
                                reconciled_oees['pick'] = reconciled_pick
                                logger.info(f"Batch {batch_id}: Calculated reconciled Pick OEE - Quality: {reconciled_pick.quality_attributed * 100:.1f}%, OEE: {reconciled_pick.oee_reconciled * 100:.1f}%")
                            
                            # Store reconciled OEEs in session state (keyed by batch_id)
                            if 'reconciled_oees' not in st.session_state:
                                st.session_state['reconciled_oees'] = {}
                            st.session_state['reconciled_oees'][batch_id] = reconciled_oees
                            logger.info(f"Stored reconciled OEEs for batch {batch_id}: {list(reconciled_oees.keys())}, length: {len(reconciled_oees)}")
                        
                        # Store breakdown data in session state
                        st.session_state['quality_breakdown_df'] = breakdown_df
                        st.session_state['reconciliation_complete'] = True
                        
                        # Rerun to update the display with reconciled OEE values
                        st.rerun()
                        
                except Exception as e:
                    logger.error(f"Error running post-QC reconciliation: {e}", exc_info=True)
                    log_collector.add_error(f"Error running post-QC reconciliation: {e}")
                    st.error(f"❌ Error running post-QC reconciliation: {e}")
    
    st.divider()
    
    # 7. Data Export
    st.header("💾 Export Data")
    
    # Prepare CSV data (only for enabled cells)
    export_data = {
        'Cell': [],
        'Availability (%)': [],
        'Performance (%)': [],
        'Quality (%)': [],
        'OEE (%)': []
    }
    
    if printer_oee:
        export_data['Cell'].append('Printer')
        export_data['Availability (%)'].append(printer_oee.availability * 100)
        export_data['Performance (%)'].append(printer_oee.performance * 100)
        export_data['Quality (%)'].append(printer_oee.quality * 100)
        export_data['OEE (%)'].append(printer_oee.oee * 100)
    
    if cut_oee:
        export_data['Cell'].append('Cut')
        export_data['Availability (%)'].append(cut_oee.availability * 100)
        export_data['Performance (%)'].append(cut_oee.performance * 100)
        export_data['Quality (%)'].append(cut_oee.quality * 100)
        export_data['OEE (%)'].append(cut_oee.oee * 100)
    
    if pick_oee:
        export_data['Cell'].append('Pick')
        export_data['Availability (%)'].append(pick_oee.availability * 100)
        export_data['Performance (%)'].append(pick_oee.performance * 100)
        export_data['Quality (%)'].append(pick_oee.quality * 100)
        export_data['OEE (%)'].append(pick_oee.oee * 100)
    
    export_df = pd.DataFrame(export_data)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download OEE Metrics (CSV)",
        data=csv_data,
        file_name="oee_metrics.csv",
        mime="text/csv"
    )
    
    # Show data preview (only for enabled cells) - full data for each cell
    with st.expander("📋 Data Preview"):
        if 'printer' in enabled_cells and not print_df.empty:
            st.subheader(f"Print Data ({len(print_df)} records)")
            st.dataframe(print_df, use_container_width=True)
        
        if 'cut' in enabled_cells and not cut_df.empty:
            st.subheader(f"Cut Data ({len(cut_df)} records)")
            st.dataframe(cut_df, use_container_width=True)
        
        if 'pick' in enabled_cells and not pick_df.empty:
            st.subheader(f"Pick Data ({len(pick_df)} records)")
            st.dataframe(pick_df, use_container_width=True)
        
        if detected_batches and not quality_df.empty:
            st.subheader(f"Quality Metrics ({len(quality_df)} records)")
            st.dataframe(quality_df, use_container_width=True)

