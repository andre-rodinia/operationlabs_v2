"""
Throughput Calculation Functions

Calculates equipment state categorizations, hourly breakdowns, and throughput metrics
for manufacturing cells (Print, Cut, Pick).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def categorize_state(state_name: str) -> str:
    """
    Categorize a state name into one of: "running", "idle", "fault", or "other".
    
    Uses keyword matching to determine category:
    - "running": states containing "run", "active", "operat", "work", "print" (for printing states)
    - "idle": states containing "idle", "standby", "wait", "ready"
    - "fault": states containing "fault", "error", "fail", "down", "stop", "alarm", "maintenance"
    - "other": everything else
    
    Args:
        state_name: State name from equipment table (case-insensitive matching)
        
    Returns:
        Category string: "running", "idle", "fault", or "other"
    """
    if not state_name or not isinstance(state_name, str):
        return "other"
    
    state_lower = state_name.lower()
    
    # Running states - expanded to include printing-related states
    if any(keyword in state_lower for keyword in ["run", "active", "operat", "work", "print", "printing", "produc"]):
        return "running"
    
    # Idle states
    if any(keyword in state_lower for keyword in ["idle", "standby", "wait", "ready"]):
        return "idle"
    
    # Fault/Error states
    if any(keyword in state_lower for keyword in ["fault", "error", "fail", "down", "stop", "alarm", "maintenance"]):
        return "fault"
    
    # Default to other
    return "other"


def calculate_hourly_state_breakdown(
    states_df: pd.DataFrame,
    start_ts: datetime,
    end_ts: datetime
) -> pd.DataFrame:
    """
    Split equipment states into hourly buckets and calculate time spent in each category.
    
    For each hour between start_ts and end_ts:
    - Calculate seconds spent in "running", "idle", "fault", "other"
    - Calculate percentages for each category
    - Return DataFrame with one row per hour
    
    Args:
        states_df: DataFrame from fetch_equipment_states_with_durations with columns:
                   ts, state, description, duration_seconds, next_ts
        start_ts: Start timestamp for the analysis window
        end_ts: End timestamp for the analysis window
        
    Returns:
        DataFrame with columns:
        - hour_start: Start of the hour
        - hour_end: End of the hour
        - running_seconds: Seconds in running state
        - idle_seconds: Seconds in idle state
        - fault_seconds: Seconds in fault state
        - other_seconds: Seconds in other state
        - total_seconds: Total seconds in hour
        - running_percent: Percentage running
        - idle_percent: Percentage idle
        - fault_percent: Percentage fault
        - other_percent: Percentage other
    """
    if states_df.empty:
        logger.warning("Empty states DataFrame provided")
        return pd.DataFrame()
    
    # Categorize all states
    states_df = states_df.copy()
    states_df['category'] = states_df['state'].apply(categorize_state)
    
    # Generate hourly buckets - only for hours that overlap with the time window
    hourly_data = []
    # Start from the hour containing start_ts, but only include hours that overlap with the window
    current_hour = start_ts.replace(minute=0, second=0, microsecond=0)
    
    # Only generate buckets for hours that actually overlap with the time window
    while current_hour < end_ts:
        hour_end = current_hour + timedelta(hours=1)
        hour_start_actual = max(current_hour, start_ts)
        hour_end_actual = min(hour_end, end_ts)
        
        # Filter states that overlap with this hour
        hour_states = states_df[
            (states_df['ts'] < hour_end_actual) & 
            (states_df['next_ts'] > hour_start_actual)
        ].copy()
        
        total_sec = (hour_end_actual - hour_start_actual).total_seconds()
        
        if hour_states.empty:
            # No states in this hour - assign all time to "other"
            running_sec = 0.0
            idle_sec = 0.0
            fault_sec = 0.0
            other_sec = total_sec  # All unrecorded time goes to "other"
        else:
            # Calculate duration in this hour for each state
            # Ensure ts and next_ts are datetime Series
            ts_series = pd.to_datetime(hour_states['ts'], errors='coerce')
            next_ts_series = pd.to_datetime(hour_states['next_ts'], errors='coerce')
            
            # Calculate overlap using vectorized operations
            # Convert to numeric (timestamps) for comparison, then back to datetime
            ts_numeric = ts_series.astype('int64')
            next_ts_numeric = next_ts_series.astype('int64')
            hour_start_numeric = pd.Timestamp(hour_start_actual).value
            hour_end_numeric = pd.Timestamp(hour_end_actual).value
            
            # Calculate overlap start/end as numeric, then convert back to datetime
            overlap_start_numeric = np.maximum(ts_numeric, hour_start_numeric)
            overlap_end_numeric = np.minimum(next_ts_numeric, hour_end_numeric)
            
            hour_states['overlap_start'] = pd.to_datetime(overlap_start_numeric)
            hour_states['overlap_end'] = pd.to_datetime(overlap_end_numeric)
            
            hour_states['overlap_duration'] = (
                hour_states['overlap_end'] - hour_states['overlap_start']
            ).dt.total_seconds()
            
            # Sum by category
            category_sums = hour_states.groupby('category')['overlap_duration'].sum()
            running_sec = float(category_sums.get('running', 0.0))
            idle_sec = float(category_sums.get('idle', 0.0))
            fault_sec = float(category_sums.get('fault', 0.0))
            other_sec = float(category_sums.get('other', 0.0))
            
            # If there's a gap (unrecorded time) in this hour, assign it to "other"
            recorded_sec = running_sec + idle_sec + fault_sec + other_sec
            gap_sec = max(0.0, total_sec - recorded_sec)
            if gap_sec > 0:
                other_sec += gap_sec
        
        # Calculate percentages
        if total_sec > 0:
            running_pct = (running_sec / total_sec) * 100
            idle_pct = (idle_sec / total_sec) * 100
            fault_pct = (fault_sec / total_sec) * 100
            other_pct = (other_sec / total_sec) * 100
        else:
            running_pct = idle_pct = fault_pct = other_pct = 0.0
        
        hourly_data.append({
            'hour_start': current_hour,
            'hour_end': hour_end,
            'running_seconds': running_sec,
            'idle_seconds': idle_sec,
            'fault_seconds': fault_sec,
            'other_seconds': other_sec,
            'total_seconds': total_sec,
            'running_percent': running_pct,
            'idle_percent': idle_pct,
            'fault_percent': fault_pct,
            'other_percent': other_pct
        })
        
        current_hour = hour_end
    
    result_df = pd.DataFrame(hourly_data)
    logger.info(f"Calculated hourly breakdown: {len(result_df)} hours")
    
    return result_df


def calculate_state_summary(
    states_df: pd.DataFrame,
    start_ts: datetime,
    end_ts: datetime
) -> Dict[str, float]:
    """
    Sum up total time in each state category and calculate utilization.
    
    Uses the full time window (start_ts to end_ts) as the total time, not just
    the sum of recorded state durations. This ensures consistency with hourly breakdown.
    
    Args:
        states_df: DataFrame from fetch_equipment_states_with_durations with columns:
                   ts, state, description, duration_seconds, next_ts
        start_ts: Start timestamp for the analysis window
        end_ts: End timestamp for the analysis window
                   
    Returns:
        Dictionary with:
        - running_seconds: Total seconds in running state
        - idle_seconds: Total seconds in idle state
        - fault_seconds: Total seconds in fault state
        - other_seconds: Total seconds in other state
        - total_seconds: Total time analyzed (end_ts - start_ts)
        - running_percent: Percentage running
        - idle_percent: Percentage idle
        - fault_percent: Percentage fault
        - other_percent: Percentage other
        - utilization: Running time / Total time (0-1)
    """
    if states_df.empty:
        logger.warning("Empty states DataFrame provided")
        total_sec = (end_ts - start_ts).total_seconds() if start_ts and end_ts else 0.0
        return {
            'running_seconds': 0.0,
            'idle_seconds': 0.0,
            'fault_seconds': 0.0,
            'other_seconds': 0.0,
            'total_seconds': total_sec,
            'running_percent': 0.0,
            'idle_percent': 0.0,
            'fault_percent': 0.0,
            'other_percent': 0.0,
            'utilization': 0.0
        }
    
    # Categorize states
    states_df = states_df.copy()
    states_df['category'] = states_df['state'].apply(categorize_state)
    
    # Log raw state to category mapping for debugging
    state_to_category = states_df.groupby('state')['category'].first().to_dict()
    logger.info(f"State categorization mapping: {state_to_category}")
    
    # Use duration_seconds directly from query (already calculated for the time window)
    # The SQL query already handles overlap with the window correctly
    if 'duration_seconds' not in states_df.columns:
        logger.warning("duration_seconds column not found, calculating overlap manually")
        # Fallback: calculate overlap manually if column missing
        ts_series = pd.to_datetime(states_df['ts'], errors='coerce')
        next_ts_series = pd.to_datetime(states_df['next_ts'], errors='coerce')
        ts_numeric = ts_series.astype('int64')
        next_ts_numeric = next_ts_series.astype('int64')
        start_numeric = pd.Timestamp(start_ts).value
        end_numeric = pd.Timestamp(end_ts).value
        overlap_start_numeric = np.maximum(ts_numeric, start_numeric)
        overlap_end_numeric = np.minimum(next_ts_numeric, end_numeric)
        overlap_start = pd.to_datetime(overlap_start_numeric)
        overlap_end = pd.to_datetime(overlap_end_numeric)
        states_df['duration_seconds'] = (overlap_end - overlap_start).dt.total_seconds()
        states_df['duration_seconds'] = states_df['duration_seconds'].clip(lower=0)
    else:
        # Ensure duration_seconds is numeric
        states_df['duration_seconds'] = pd.to_numeric(states_df['duration_seconds'], errors='coerce').fillna(0.0)
    
    # Sum durations by category (using duration_seconds from query)
    category_sums = states_df.groupby('category')['duration_seconds'].sum()
    
    # Also log raw state totals for comparison
    raw_state_totals = states_df.groupby('state')['duration_seconds'].sum().to_dict()
    logger.info(f"Raw state totals (seconds): {raw_state_totals}")
    
    # Log category totals to verify categorization
    category_totals = states_df.groupby('category')['duration_seconds'].sum().to_dict()
    total_categorized = sum(category_totals.values())
    logger.info(f"Category totals (seconds): {category_totals}")
    if total_categorized > 0:
        for cat, sec in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
            pct = (sec / total_categorized) * 100
            logger.info(f"  {cat}: {sec:.1f}s ({sec/60:.1f} min, {pct:.1f}%)")
    
    running_sec = float(category_sums.get('running', 0.0))
    idle_sec = float(category_sums.get('idle', 0.0))
    fault_sec = float(category_sums.get('fault', 0.0))
    other_sec = float(category_sums.get('other', 0.0))
    
    # Use the full time window as total time (not just sum of recorded states)
    total_sec = (end_ts - start_ts).total_seconds()
    
    # If there are gaps (unrecorded time), assign them to "other" or "idle"
    recorded_sec = running_sec + idle_sec + fault_sec + other_sec
    gap_sec = max(0.0, total_sec - recorded_sec)
    
    if gap_sec > 0:
        logger.debug(f"Time gap detected: {gap_sec:.1f}s ({gap_sec/60:.1f} min) not covered by state records")
        # Assign gaps to "other" category
        other_sec += gap_sec
    
    # Calculate percentages and utilization
    if total_sec > 0:
        running_pct = (running_sec / total_sec) * 100
        idle_pct = (idle_sec / total_sec) * 100
        fault_pct = (fault_sec / total_sec) * 100
        other_pct = (other_sec / total_sec) * 100
        utilization = running_sec / total_sec
    else:
        running_pct = idle_pct = fault_pct = other_pct = 0.0
        utilization = 0.0
    
    logger.info(f"State summary: Running={running_pct:.1f}%, Idle={idle_pct:.1f}%, Fault={fault_pct:.1f}%, Utilization={utilization:.2%}")
    
    return {
        'running_seconds': running_sec,
        'idle_seconds': idle_sec,
        'fault_seconds': fault_sec,
        'other_seconds': other_sec,
        'total_seconds': total_sec,
        'running_percent': running_pct,
        'idle_percent': idle_pct,
        'fault_percent': fault_pct,
        'other_percent': other_pct,
        'utilization': utilization
    }


def calculate_cell_throughput(
    components_count: int,
    state_summary: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate throughput metrics for one cell.
    
    Args:
        components_count: Total number of components produced
        state_summary: Dictionary from calculate_state_summary
        
    Returns:
        Dictionary with:
        - active_throughput_components_per_hour: Components / running hours
        - calendar_throughput_components_per_hour: Components / total hours
        - utilization: Running time / Total time (0-1)
        - running_hours: Total running time in hours
        - total_hours: Total time in hours
    """
    running_hours = state_summary['running_seconds'] / 3600.0
    total_hours = state_summary['total_seconds'] / 3600.0
    
    if running_hours > 0:
        active_throughput = components_count / running_hours
    else:
        active_throughput = 0.0
    
    if total_hours > 0:
        calendar_throughput = components_count / total_hours
    else:
        calendar_throughput = 0.0
    
    utilization = state_summary['utilization']
    
    return {
        'active_throughput_components_per_hour': active_throughput,
        'calendar_throughput_components_per_hour': calendar_throughput,
        'utilization': utilization,
        'running_hours': running_hours,
        'total_hours': total_hours
    }


def calculate_system_throughput(
    print_data: Optional[Dict],
    cut_data: Optional[Dict],
    pick_data: Optional[Dict],
    batch_structure_df: Optional[pd.DataFrame],
    fpy_df: Optional[pd.DataFrame]
) -> Dict:
    """
    Calculate system-level throughput metrics for all cells.
    
    Key principle: All cells (Print, Cut, Pick) process the SAME total components.
    Each cell processes all 1,537 components (or whatever the batch total is).
    
    Args:
        print_data: Dictionary with 'state_summary', 'start_ts', 'end_ts' for Print
        cut_data: Dictionary with 'state_summary', 'start_ts', 'end_ts' for Cut
        pick_data: Dictionary with 'state_summary', 'start_ts', 'end_ts' for Pick
        batch_structure_df: DataFrame with batch structure (garments, components)
        fpy_df: DataFrame with FPY data for sellable calculations
        
    Returns:
        Dictionary with:
        - cell_throughputs: Dict with throughput data per cell
        - system_metrics: System-level raw and sellable metrics
        - bottleneck: Cell with lowest active throughput
        - improvement_opportunity: Cell with most improvement potential
    """
    # Step 1: Get total components from batch structure (same for all cells)
    total_components = 0
    total_garments = 0
    
    if batch_structure_df is not None and not batch_structure_df.empty:
        if 'garments_per_style' in batch_structure_df.columns:
            total_garments = int(batch_structure_df['garments_per_style'].sum())
        if 'total_components_per_style' in batch_structure_df.columns:
            total_components = int(batch_structure_df['total_components_per_style'].sum())
        elif 'garments_per_style' in batch_structure_df.columns and 'components_per_style' in batch_structure_df.columns:
            # Calculate total components from garments × components per style
            total_components = int((batch_structure_df['garments_per_style'] * batch_structure_df['components_per_style']).sum())
    
    if total_components == 0:
        logger.error("No components found in batch structure - cannot calculate throughput")
        return {
            'cell_throughputs': {},
            'system_metrics': {},
            'bottleneck': None,
            'improvement_opportunity': None
        }
    
    logger.info(f"Total components for batch: {total_components} (same for all cells)")
    
    # Step 2: Calculate cell-level throughput (using SAME component count for all)
    cell_throughputs = {}
    cells_data = [
        ('Print1', print_data),
        ('Cut1', cut_data),
        ('Pick1', pick_data)
    ]
    
    for cell_name, cell_info in cells_data:
        if cell_info and 'state_summary' in cell_info:
            throughput = calculate_cell_throughput(
                total_components,  # ✅ Same for all cells
                cell_info['state_summary']
            )
            cell_throughputs[cell_name] = throughput
    
    # Step 3: Calculate system time window (wall-clock)
    system_start = None
    system_end = None
    
    for cell_name, cell_info in cells_data:
        if cell_info and 'start_ts' in cell_info and 'end_ts' in cell_info:
            if system_start is None or cell_info['start_ts'] < system_start:
                system_start = cell_info['start_ts']
            if system_end is None or cell_info['end_ts'] > system_end:
                system_end = cell_info['end_ts']
    
    # Calculate system duration (wall-clock time from first start to last end)
    if system_start and system_end:
        system_total_hours = (system_end - system_start).total_seconds() / 3600.0
    else:
        system_total_hours = 0.0
    
    # System-level throughputs (components and garments per hour)
    if system_total_hours > 0:
        system_components_per_hour = total_components / system_total_hours
        system_garments_per_hour = total_garments / system_total_hours if total_garments > 0 else 0.0
    else:
        system_components_per_hour = 0.0
        system_garments_per_hour = 0.0
    
    # Calculate sellable metrics if FPY data exists
    sellable_components = None
    sellable_garments = None
    sellable_components_per_hour = None
    sellable_garments_per_hour = None
    
    if fpy_df is not None and not fpy_df.empty and 'garments_QC_passed' in fpy_df.columns:
        sellable_garments = int(fpy_df['garments_QC_passed'].sum())
        
        # Estimate sellable components from sellable garments
        if total_garments > 0 and total_components > 0:
            avg_components_per_garment = total_components / total_garments
            sellable_components = int(sellable_garments * avg_components_per_garment)
        else:
            sellable_components = sellable_garments  # Fallback (assume 1 component per garment)
        
        if system_total_hours > 0:
            sellable_components_per_hour = sellable_components / system_total_hours
            sellable_garments_per_hour = sellable_garments / system_total_hours
    
    # Find bottleneck (lowest active throughput)
    bottleneck = None
    lowest_throughput = float('inf')
    
    for cell_name, throughput in cell_throughputs.items():
        if throughput['active_throughput_components_per_hour'] > 0:
            if throughput['active_throughput_components_per_hour'] < lowest_throughput:
                lowest_throughput = throughput['active_throughput_components_per_hour']
                bottleneck = cell_name
    
    # Find improvement opportunity (lowest utilization)
    improvement_opportunity = None
    lowest_utilization = float('inf')
    
    for cell_name, throughput in cell_throughputs.items():
        if throughput['utilization'] < lowest_utilization:
            lowest_utilization = throughput['utilization']
            improvement_opportunity = cell_name
    
    return {
        'cell_throughputs': cell_throughputs,
        'system_metrics': {
            'total_components': total_components,
            'total_garments': total_garments,
            'system_total_hours': system_total_hours,  # Wall-clock time from first start to last end
            'system_components_per_hour': system_components_per_hour,
            'system_garments_per_hour': system_garments_per_hour,
            'sellable_components': sellable_components,
            'sellable_garments': sellable_garments,
            'sellable_components_per_hour': sellable_components_per_hour,
            'sellable_garments_per_hour': sellable_garments_per_hour
        },
        'bottleneck': bottleneck,
        'improvement_opportunity': improvement_opportunity
    }
