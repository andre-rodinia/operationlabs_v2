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
    Categorize a state name into one of: "running", "idle", "fault", "blocked", or "other".

    Uses keyword matching to determine category:
    - "running": states containing "run", "active", "operat", "work", "print" (for printing states)
    - "blocked": states containing "wait", "waiting", "blocked" (tandem interference)
    - "idle": states containing "idle", "standby", "ready"
    - "fault": states containing "fault", "error", "fail", "down", "stop", "alarm", "maintenance"
    - "other": everything else

    Note: "blocked" is for future use when Pick tracks "waiting_for_cut" states.
    Currently, waiting states may still be categorized as "running".

    Args:
        state_name: State name from equipment table (case-insensitive matching)

    Returns:
        Category string: "running", "idle", "fault", "blocked", or "other"
    """
    if not state_name or not isinstance(state_name, str):
        return "other"

    state_lower = state_name.lower()

    # Running states - expanded to include printing-related states
    # Note: Check this BEFORE "wait" to avoid false positives (e.g., "wait" in "await")
    if any(keyword in state_lower for keyword in ["run", "active", "operat", "work", "print", "printing", "produc"]):
        return "running"

    # Blocked states (tandem interference) - check before general "idle"
    # These are waiting states due to equipment dependencies, not true idle
    if any(keyword in state_lower for keyword in ["waiting_for", "blocked", "waiting"]):
        return "blocked"

    # Idle states
    if any(keyword in state_lower for keyword in ["idle", "standby", "ready"]):
        return "idle"

    # Fault/Error states
    if any(keyword in state_lower for keyword in ["fault", "error", "fail", "down", "stop", "alarm", "maintenance"]):
        return "fault"

    # Default to other
    return "other"


def calculate_hourly_state_breakdown(
    states_df: pd.DataFrame,
    start_ts: datetime,
    end_ts: datetime,
    break_start: Optional[datetime] = None,
    break_end: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Split equipment states into hourly buckets and calculate time spent in each category.

    For each hour between start_ts and end_ts:
    - Calculate seconds spent in "running", "idle", "fault", "other", "break"
    - Calculate percentages for each category (based on productive time, excluding breaks)
    - Return DataFrame with one row per hour

    Args:
        states_df: DataFrame from fetch_equipment_states_with_durations with columns:
                   ts, state, description, duration_seconds, next_ts
        start_ts: Start timestamp for the analysis window
        end_ts: End timestamp for the analysis window
        break_start: Optional start timestamp of break period
        break_end: Optional end timestamp of break period

    Returns:
        DataFrame with columns:
        - hour_start: Start of the hour
        - hour_end: End of the hour
        - running_seconds: Seconds in running state
        - idle_seconds: Seconds in idle state
        - fault_seconds: Seconds in fault state
        - blocked_seconds: Seconds in blocked state
        - other_seconds: Seconds in other state
        - break_seconds: Seconds in break period
        - total_seconds: Total seconds in hour
        - productive_seconds: Total seconds excluding break
        - running_percent: Percentage running (of productive time)
        - idle_percent: Percentage idle (of productive time)
        - fault_percent: Percentage fault (of productive time)
        - blocked_percent: Percentage blocked (of productive time)
        - other_percent: Percentage other (of productive time)
        - break_percent: Percentage break (of total time)
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

        # Calculate break overlap with this hour
        break_sec = 0.0
        if break_start and break_end:
            # Calculate overlap between this hour and the break period
            overlap_start = max(hour_start_actual, break_start)
            overlap_end = min(hour_end_actual, break_end)
            if overlap_end > overlap_start:
                break_sec = (overlap_end - overlap_start).total_seconds()

        # Productive time is total time minus break time
        productive_sec = total_sec - break_sec

        if hour_states.empty:
            # No states in this hour - assign all time to "other"
            running_sec = 0.0
            idle_sec = 0.0
            fault_sec = 0.0
            blocked_sec = 0.0
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
            blocked_sec = float(category_sums.get('blocked', 0.0))
            other_sec = float(category_sums.get('other', 0.0))

            # If there's a gap (unrecorded time) in this hour, assign it to "other"
            recorded_sec = running_sec + idle_sec + fault_sec + blocked_sec + other_sec
            gap_sec = max(0.0, total_sec - recorded_sec)
            if gap_sec > 0:
                other_sec += gap_sec
        
        # Calculate percentages based on productive time (excluding breaks)
        # State percentages are relative to productive time
        if productive_sec > 0:
            running_pct = (running_sec / productive_sec) * 100
            idle_pct = (idle_sec / productive_sec) * 100
            fault_pct = (fault_sec / productive_sec) * 100
            blocked_pct = (blocked_sec / productive_sec) * 100
            other_pct = (other_sec / productive_sec) * 100
        else:
            running_pct = idle_pct = fault_pct = blocked_pct = other_pct = 0.0

        # Break percentage is relative to total time (for stacked bar visualization)
        if total_sec > 0:
            break_pct = (break_sec / total_sec) * 100
        else:
            break_pct = 0.0

        hourly_data.append({
            'hour_start': current_hour,
            'hour_end': hour_end,
            'running_seconds': running_sec,
            'idle_seconds': idle_sec,
            'fault_seconds': fault_sec,
            'blocked_seconds': blocked_sec,
            'other_seconds': other_sec,
            'break_seconds': break_sec,
            'total_seconds': total_sec,
            'productive_seconds': productive_sec,
            'running_percent': running_pct,
            'idle_percent': idle_pct,
            'fault_percent': fault_pct,
            'blocked_percent': blocked_pct,
            'other_percent': other_pct,
            'break_percent': break_pct
        })
        
        current_hour = hour_end
    
    result_df = pd.DataFrame(hourly_data)
    logger.info(f"Calculated hourly breakdown: {len(result_df)} hours")
    
    return result_df


def calculate_state_summary(
    states_df: pd.DataFrame,
    start_ts: datetime,
    end_ts: datetime,
    break_start: Optional[datetime] = None,
    break_end: Optional[datetime] = None
) -> Dict[str, float]:
    """
    Sum up total time in each state category and calculate utilization.

    Uses the full time window (start_ts to end_ts) as the total time, not just
    the sum of recorded state durations. Only subtracts break time if the break
    period overlaps with this cell's time window.

    Args:
        states_df: DataFrame from fetch_equipment_states_with_durations with columns:
                   ts, state, description, duration_seconds, next_ts
        start_ts: Start timestamp for the analysis window
        end_ts: End timestamp for the analysis window
        break_start: Break start timestamp (optional, only if break occurred)
        break_end: Break end timestamp (optional, only if break occurred)

    Returns:
        Dictionary with:
        - running_seconds: Total seconds in running state
        - idle_seconds: Total seconds in idle state
        - fault_seconds: Total seconds in fault state
        - blocked_seconds: Total seconds in blocked/waiting state (tandem interference)
        - other_seconds: Total seconds in other state
        - total_seconds: Total time analyzed (end_ts - start_ts)
        - productive_seconds: Total time minus break overlap (total_seconds - break_overlap_seconds)
        - break_overlap_seconds: Break time that overlaps with this cell's window
        - running_percent: Percentage running (of total time)
        - idle_percent: Percentage idle (of total time)
        - fault_percent: Percentage fault (of total time)
        - blocked_percent: Percentage blocked (of total time)
        - other_percent: Percentage other (of total time)
        - utilization: Running time / Total time (0-1)
        - productive_utilization: Running time / Productive time (0-1, excluding break overlap)
    """
    # Calculate break overlap with this cell's time window
    break_overlap_sec = 0.0
    if break_start and break_end and start_ts and end_ts:
        # Only subtract break time if it overlaps with this cell's window
        # Overlap calculation: max(0, min(end1, end2) - max(start1, start2))
        overlap_start = max(start_ts, break_start)
        overlap_end = min(end_ts, break_end)

        if overlap_end > overlap_start:
            break_overlap_sec = (overlap_end - overlap_start).total_seconds()
            logger.info(f"Break overlap with cell window: {break_overlap_sec/60:.1f} minutes")
        else:
            logger.info("Break does not overlap with this cell's time window")

    if states_df.empty:
        logger.warning("Empty states DataFrame provided")
        total_sec = (end_ts - start_ts).total_seconds() if start_ts and end_ts else 0.0
        productive_sec = max(0.0, total_sec - break_overlap_sec)
        return {
            'running_seconds': 0.0,
            'idle_seconds': 0.0,
            'fault_seconds': 0.0,
            'blocked_seconds': 0.0,
            'other_seconds': 0.0,
            'total_seconds': total_sec,
            'productive_seconds': productive_sec,
            'break_overlap_seconds': break_overlap_sec,
            'running_percent': 0.0,
            'idle_percent': 0.0,
            'fault_percent': 0.0,
            'blocked_percent': 0.0,
            'other_percent': 0.0,
            'utilization': 0.0,
            'productive_utilization': 0.0
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
    blocked_sec = float(category_sums.get('blocked', 0.0))
    other_sec = float(category_sums.get('other', 0.0))

    # Use the full time window as total time (not just sum of recorded states)
    total_sec = (end_ts - start_ts).total_seconds()
    productive_sec = max(0.0, total_sec - break_overlap_sec)

    # If there are gaps (unrecorded time), assign them to "other" or "idle"
    recorded_sec = running_sec + idle_sec + fault_sec + blocked_sec + other_sec
    gap_sec = max(0.0, total_sec - recorded_sec)

    if gap_sec > 0:
        logger.debug(f"Time gap detected: {gap_sec:.1f}s ({gap_sec/60:.1f} min) not covered by state records")
        # Assign gaps to "other" category
        other_sec += gap_sec

    # Calculate percentages (of total time including breaks)
    if total_sec > 0:
        running_pct = (running_sec / total_sec) * 100
        idle_pct = (idle_sec / total_sec) * 100
        fault_pct = (fault_sec / total_sec) * 100
        blocked_pct = (blocked_sec / total_sec) * 100
        other_pct = (other_sec / total_sec) * 100
        utilization = running_sec / total_sec
    else:
        running_pct = idle_pct = fault_pct = blocked_pct = other_pct = 0.0
        utilization = 0.0

    # Calculate productive utilization (excluding breaks)
    if productive_sec > 0:
        productive_utilization = running_sec / productive_sec
    else:
        productive_utilization = 0.0

    logger.info(f"State summary: Running={running_pct:.1f}%, Idle={idle_pct:.1f}%, Fault={fault_pct:.1f}%, Blocked={blocked_pct:.1f}%, Utilization={utilization:.2%}, Productive Utilization={productive_utilization:.2%}")
    if break_overlap_sec > 0:
        logger.info(f"Break overlap: {break_overlap_sec/60:.1f} min, Productive time: {productive_sec/60:.1f} min")

    return {
        'running_seconds': running_sec,
        'idle_seconds': idle_sec,
        'fault_seconds': fault_sec,
        'blocked_seconds': blocked_sec,
        'other_seconds': other_sec,
        'total_seconds': total_sec,
        'productive_seconds': productive_sec,
        'break_overlap_seconds': break_overlap_sec,
        'running_percent': running_pct,
        'idle_percent': idle_pct,
        'fault_percent': fault_pct,
        'blocked_percent': blocked_pct,
        'other_percent': other_pct,
        'utilization': utilization,
        'productive_utilization': productive_utilization
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
        - productive_throughput_components_per_hour: Components / productive hours (excluding breaks)
        - calendar_throughput_components_per_hour: Components / total hours (including breaks)
        - utilization: Running time / Total time (0-1)
        - productive_utilization: Running time / Productive time (0-1, excluding breaks)
        - running_hours: Total running time in hours
        - productive_hours: Productive time in hours (total - breaks)
        - total_hours: Total time in hours
        - break_hours: Break time in hours
        - blocked_hours: Blocked/waiting time in hours (tandem interference)
    """
    running_hours = state_summary['running_seconds'] / 3600.0
    total_hours = state_summary['total_seconds'] / 3600.0
    productive_hours = state_summary['productive_seconds'] / 3600.0
    break_overlap_hours = state_summary.get('break_overlap_seconds', 0.0) / 3600.0
    blocked_hours = state_summary.get('blocked_seconds', 0.0) / 3600.0

    # Active throughput: when machine is actually running
    if running_hours > 0:
        active_throughput = components_count / running_hours
    else:
        active_throughput = 0.0

    # Productive throughput: during scheduled work hours (excluding breaks)
    if productive_hours > 0:
        productive_throughput = components_count / productive_hours
    else:
        productive_throughput = 0.0

    # Calendar throughput: wall-clock (including breaks)
    if total_hours > 0:
        calendar_throughput = components_count / total_hours
    else:
        calendar_throughput = 0.0

    utilization = state_summary['utilization']
    productive_utilization = state_summary.get('productive_utilization', utilization)

    return {
        'active_throughput_components_per_hour': active_throughput,
        'productive_throughput_components_per_hour': productive_throughput,
        'calendar_throughput_components_per_hour': calendar_throughput,
        'utilization': utilization,
        'productive_utilization': productive_utilization,
        'running_hours': running_hours,
        'productive_hours': productive_hours,
        'total_hours': total_hours,
        'break_overlap_hours': break_overlap_hours,
        'blocked_hours': blocked_hours
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

    # Calculate productive hours (sum of productive time across all cells, not wall-clock)
    # This represents actual scheduled work time excluding break overlaps
    system_productive_hours = 0.0
    system_break_overlap_hours = 0.0
    for cell_name, throughput in cell_throughputs.items():
        if 'productive_hours' in throughput:
            # Use the longest cell productive time as system productive time
            # (since all cells must finish for batch to complete)
            system_productive_hours = max(system_productive_hours, throughput['productive_hours'])
            system_break_overlap_hours = max(system_break_overlap_hours, throughput.get('break_overlap_hours', 0.0))

    # System-level throughputs based on productive time (excluding breaks)
    # This is the primary metric for system performance
    if system_productive_hours > 0:
        system_components_per_hour = total_components / system_productive_hours
        system_garments_per_hour = total_garments / system_productive_hours if total_garments > 0 else 0.0
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

        if system_productive_hours > 0:
            sellable_components_per_hour = sellable_components / system_productive_hours
            sellable_garments_per_hour = sellable_garments / system_productive_hours
    
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
            'system_productive_hours': system_productive_hours,  # Productive time (excluding break overlaps)
            'system_break_overlap_hours': system_break_overlap_hours,  # Break overlap time
            'system_components_per_hour': system_components_per_hour,  # Based on productive time
            'system_garments_per_hour': system_garments_per_hour,  # Based on productive time
            'bottleneck_active_throughput': lowest_throughput if bottleneck else None,  # Theoretical max
            'sellable_components': sellable_components,
            'sellable_garments': sellable_garments,
            'sellable_components_per_hour': sellable_components_per_hour,
            'sellable_garments_per_hour': sellable_garments_per_hour
        },
        'bottleneck': bottleneck,
        'improvement_opportunity': improvement_opportunity
    }
