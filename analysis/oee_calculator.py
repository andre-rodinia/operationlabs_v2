"""
OEE Calculator
Aggregates job-level OEE into batch-level and daily-level metrics
"""
import pandas as pd
import logging
from typing import Dict, Optional
from datetime import datetime

from core.db.fetchers import get_batch_pick_time_window, fetch_robot_equipment_states

logger = logging.getLogger(__name__)


def calculate_break_overlap_per_cell(
    cell_start: Optional[pd.Timestamp],
    cell_end: Optional[pd.Timestamp],
    break_start: Optional[datetime],
    break_end: Optional[datetime]
) -> float:
    """
    Calculate overlap between break window and cell production window.

    Args:
        cell_start: Start timestamp of cell production
        cell_end: End timestamp of cell production
        break_start: Start of break period
        break_end: End of break period

    Returns:
        Overlap duration in hours (0.0 if no overlap)

    Edge Cases:
    - Break outside production window: Returns 0.0
    - Partial overlap: Returns only overlapping portion
    - None values: Returns 0.0
    """
    if not all([cell_start, cell_end, break_start, break_end]):
        return 0.0

    # Convert pd.Timestamp to datetime if needed
    if isinstance(cell_start, pd.Timestamp):
        cell_start = cell_start.to_pydatetime()
    if isinstance(cell_end, pd.Timestamp):
        cell_end = cell_end.to_pydatetime()

    # Calculate overlap
    overlap_start = max(cell_start, break_start)
    overlap_end = min(cell_end, break_end)

    if overlap_end > overlap_start:
        overlap_seconds = (overlap_end - overlap_start).total_seconds()
        overlap_hours = overlap_seconds / 3600.0
        logger.debug(f"Break overlap: {overlap_hours:.2f}h ({overlap_start} to {overlap_end})")
        return overlap_hours

    logger.debug("No break overlap with cell production window")
    return 0.0

def calculate_batch_metrics(
    print_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    pick_df: pd.DataFrame,
    batch_ids: list,
    quality_breakdown: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Calculate batch-level OEE metrics by aggregating job-level data.

    Args:
        print_df: Print jobs DataFrame
        cut_df: Cut jobs DataFrame
        pick_df: Pick jobs DataFrame
        batch_ids: List of batch IDs to calculate metrics for

    Returns:
        DataFrame with batch-level metrics (one row per batch)

    Edge Cases:
    - Batch has no jobs in a cell: Metrics = 0 for that cell
    - Empty DataFrames: Returns zeros for all metrics
    """
    batch_metrics = []

    # Helper function to normalize batch_id for comparison
    def normalize_batch_id(bid):
        if pd.isna(bid):
            return None
        bid_str = str(bid)
        # Remove B10- prefix and B prefix, strip whitespace, then strip leading zeros
        normalized = bid_str.replace('B10-', '').replace('B', '').strip()
        # Strip leading zeros to handle formats like "0000000955" vs "955"
        try:
            # If it's a number, convert to int then back to string to remove leading zeros
            normalized = str(int(normalized))
        except (ValueError, TypeError):
            # If not a number, keep as is
            pass
        return normalized

    for batch_id in batch_ids:
        # Filter jobs for this batch
        # Handle different batch_id formats (e.g., "B10-0000000955", "955", "B955")
        batch_id_str = str(batch_id)
        batch_id_clean = normalize_batch_id(batch_id_str)
        
        logger.info(f"Filtering for batch_id: '{batch_id}' (normalized: '{batch_id_clean}')")
        
        # Try multiple formats for matching
        batch_print = pd.DataFrame()
        batch_cut = pd.DataFrame()
        batch_pick = pd.DataFrame()
        
        if not print_df.empty and 'batch_id' in print_df.columns:
            # Try exact match first
            batch_print = print_df[print_df['batch_id'] == batch_id]
            if batch_print.empty:
                # Try matching cleaned version
                print_df_normalized = print_df['batch_id'].apply(normalize_batch_id)
                batch_print = print_df[print_df_normalized == batch_id_clean]
            logger.debug(f"  Print: found {len(batch_print)} jobs (exact match: {len(print_df[print_df['batch_id'] == batch_id])}, normalized match: {len(batch_print)})")
        
        if not cut_df.empty and 'batch_id' in cut_df.columns:
            batch_cut = cut_df[cut_df['batch_id'] == batch_id]
            if batch_cut.empty:
                cut_df_normalized = cut_df['batch_id'].apply(normalize_batch_id)
                batch_cut = cut_df[cut_df_normalized == batch_id_clean]
            logger.debug(f"  Cut: found {len(batch_cut)} jobs")
        
        if not pick_df.empty and 'batch_id' in pick_df.columns:
            logger.debug(f"  Pick: pick_df has {len(pick_df)} rows, batch_id column exists")
            logger.debug(f"  Pick: Unique batch_ids in pick_df: {pick_df['batch_id'].unique().tolist()}")
            logger.debug(f"  Pick: Looking for batch_id='{batch_id}' or normalized='{batch_id_clean}'")
            
            # Try exact match first
            batch_pick = pick_df[pick_df['batch_id'] == batch_id]
            logger.debug(f"  Pick: Exact match found {len(batch_pick)} jobs")
            
            if batch_pick.empty:
                # Try matching cleaned version
                pick_df_normalized = pick_df['batch_id'].apply(normalize_batch_id)
                logger.debug(f"  Pick: Normalized batch_ids: {pick_df_normalized.unique().tolist()}")
                logger.debug(f"  Pick: Looking for normalized='{batch_id_clean}'")
                batch_pick = pick_df[pick_df_normalized == batch_id_clean]
                logger.debug(f"  Pick: Normalized match found {len(batch_pick)} jobs")
                if len(batch_pick) > 0:
                    logger.debug(f"  Pick: batch_ids in filtered data: {batch_pick['batch_id'].unique().tolist()}")
            else:
                logger.debug(f"  Pick: Using exact match, found {len(batch_pick)} jobs")
        
        logger.info(f"Batch {batch_id}: Print={len(batch_print)} jobs, Cut={len(batch_cut)} jobs, Pick={len(batch_pick)} jobs")

        # Helper function to calculate weighted average
        def weighted_avg(df, value_col, weight_col):
            if df.empty or value_col not in df.columns or weight_col not in df.columns:
                return 0.0
            total_weight = df[weight_col].sum()
            if total_weight == 0:
                return 0.0
            return (df[value_col] * df[weight_col]).sum() / total_weight
        
        # Calculate batch-level metrics with weighted averages for availability and performance
        # Quality is calculated from batch-level QC data (stored in batch_quality_percent or quality_qc_percent)
        
        # Print metrics
        if not batch_print.empty:
            logger.debug(f"Batch {batch_id} Print: {len(batch_print)} jobs, columns: {batch_print.columns.tolist()}")
            print_total_time = batch_print['total_time_sec'].sum() if 'total_time_sec' in batch_print.columns else 0
            print_uptime = batch_print['uptime_sec'].sum() if 'uptime_sec' in batch_print.columns else 0
            print_downtime = batch_print['downtime_sec'].sum() if 'downtime_sec' in batch_print.columns else 0
            logger.debug(f"Batch {batch_id} Print: total_time={print_total_time}, uptime={print_uptime}, downtime={print_downtime}")
            
            # Weighted availability: Σ(availability × total_time) / Σ(total_time)
            if 'availability' in batch_print.columns and 'total_time_sec' in batch_print.columns:
                print_availability = weighted_avg(batch_print, 'availability', 'total_time_sec')
            else:
                print_availability = batch_print['availability'].mean() if 'availability' in batch_print.columns else 0
            
            # Weighted performance (capped): Σ(performance_capped × uptime) / Σ(uptime)
            # Use performance_capped column (capped at 100%) for batch-level OEE calculations
            if 'performance_capped' in batch_print.columns and 'uptime_sec' in batch_print.columns and print_uptime > 0:
                print_performance = weighted_avg(batch_print, 'performance_capped', 'uptime_sec')
            elif 'performance' in batch_print.columns and 'uptime_sec' in batch_print.columns and print_uptime > 0:
                # Fallback: cap the original performance at 100% if capped column doesn't exist
                batch_print_temp = batch_print.copy()
                batch_print_temp['performance_capped'] = batch_print_temp['performance'].clip(upper=100.0)
                print_performance = weighted_avg(batch_print_temp, 'performance_capped', 'uptime_sec')
            else:
                print_performance = batch_print['performance'].mean() if 'performance' in batch_print.columns else 0
                # Cap at 100% if using mean fallback
                print_performance = min(print_performance, 100.0)
            
            # Weighted performance (actual/uncapped): Σ(performance × uptime) / Σ(uptime)
            # This shows the actual measured performance for assessment purposes
            if 'performance' in batch_print.columns and 'uptime_sec' in batch_print.columns and print_uptime > 0:
                print_performance_actual = weighted_avg(batch_print, 'performance', 'uptime_sec')
            else:
                print_performance_actual = batch_print['performance'].mean() if 'performance' in batch_print.columns else 0
        else:
            print_total_time = 0
            print_uptime = 0
            print_downtime = 0
            print_availability = 0
            print_performance = 0
            print_performance_actual = 0
        
        # Quality from batch-level QC data (calculate from quality_breakdown)
        # Extract numeric batch ID for lookup
        batch_id_numeric = int(str(batch_id).replace('B10-', '').replace('B', '').strip()) if isinstance(batch_id, str) else batch_id
        
        # Get batch quality from quality_breakdown if available
        print_quality = 100.0  # Default
        if quality_breakdown is not None and not quality_breakdown.empty:
            batch_qc = quality_breakdown[quality_breakdown['production_batch_id'] == batch_id_numeric]
            if not batch_qc.empty:
                # Calculate Print quality: (total_components - print_defects) / total_components
                if 'total_components' in batch_qc.columns and 'print_defects' in batch_qc.columns:
                    total = batch_qc['total_components'].iloc[0]
                    defects = batch_qc['print_defects'].iloc[0]
                    if total > 0:
                        print_quality = ((total - defects) / total) * 100
                        logger.debug(f"Batch {batch_id}: Print quality = ({total} - {defects}) / {total} = {print_quality:.2f}%")
        
        # Cut metrics
        if not batch_cut.empty:
            logger.debug(f"Batch {batch_id} Cut: {len(batch_cut)} jobs, columns: {batch_cut.columns.tolist()}")
            cut_total_time = batch_cut['total_time_sec'].sum() if 'total_time_sec' in batch_cut.columns else 0
            cut_uptime = batch_cut['uptime_sec'].sum() if 'uptime_sec' in batch_cut.columns else 0
            cut_downtime = batch_cut['downtime_sec'].sum() if 'downtime_sec' in batch_cut.columns else 0
            logger.debug(f"Batch {batch_id} Cut: total_time={cut_total_time}, uptime={cut_uptime}, downtime={cut_downtime}")
            
            # Weighted availability: Σ(availability × total_time) / Σ(total_time)
            if 'availability' in batch_cut.columns and 'total_time_sec' in batch_cut.columns:
                cut_availability = weighted_avg(batch_cut, 'availability', 'total_time_sec')
            else:
                cut_availability = batch_cut['availability'].mean() if 'availability' in batch_cut.columns else 0
            
            # Weighted performance: Σ(performance × uptime) / Σ(uptime)
            if 'performance' in batch_cut.columns and 'uptime_sec' in batch_cut.columns and cut_uptime > 0:
                cut_performance = weighted_avg(batch_cut, 'performance', 'uptime_sec')
            else:
                cut_performance = batch_cut['performance'].mean() if 'performance' in batch_cut.columns else 0
        else:
            cut_total_time = 0
            cut_uptime = 0
            cut_downtime = 0
            cut_availability = 0
            cut_performance = 0
        
        # Quality from batch-level QC data (calculate from quality_breakdown)
        cut_quality = 100.0  # Default
        if quality_breakdown is not None and not quality_breakdown.empty:
            batch_qc = quality_breakdown[quality_breakdown['production_batch_id'] == batch_id_numeric]
            if not batch_qc.empty:
                # Calculate Cut quality: (total_components - cut_defects) / total_components
                if 'total_components' in batch_qc.columns and 'cut_defects' in batch_qc.columns:
                    total = batch_qc['total_components'].iloc[0]
                    defects = batch_qc['cut_defects'].iloc[0]
                    if total > 0:
                        cut_quality = ((total - defects) / total) * 100
                        logger.debug(f"Batch {batch_id}: Cut quality = ({total} - {defects}) / {total} = {cut_quality:.2f}%")
        
        # Pick metrics - use equipment state data for availability
        logger.info(f"🤖 Calculating Pick metrics for batch {batch_id}")
        logger.info(f"  batch_pick empty: {batch_pick.empty}")
        if not batch_pick.empty:
            logger.info(f"  batch_pick columns: {batch_pick.columns.tolist()}")
            logger.info(f"  batch_pick shape: {batch_pick.shape}")
            logger.info(f"  batch_pick batch_ids: {batch_pick['batch_id'].unique().tolist() if 'batch_id' in batch_pick.columns else 'N/A'}")
        
        # Get batch time window from Pick JobReports (sheetStart_ts to sheetEnd_ts)
        # Use the already-filtered batch_pick if available (it's already filtered by normalized batch_id)
        if not batch_pick.empty:
            # batch_pick is already filtered, so we can use it directly without filtering again
            # But we still need to pass batch_id for logging
            pick_start_ts, pick_end_ts = get_batch_pick_time_window(batch_pick, batch_id)
        else:
            # Fallback: try with full pick_df (will filter inside function with normalized matching)
            pick_start_ts, pick_end_ts = get_batch_pick_time_window(pick_df, batch_id)
        
        if pick_start_ts and pick_end_ts:
            logger.info(f"  ✅ Got time window: {pick_start_ts} to {pick_end_ts}")

            # Convert timestamps to datetime if they're strings
            if isinstance(pick_start_ts, str):
                pick_start_ts = pd.to_datetime(pick_start_ts)
            if isinstance(pick_end_ts, str):
                pick_end_ts = pd.to_datetime(pick_end_ts)

            # Calculate total time from time window for daily metrics (like Print/Cut use job runtime)
            pick_total_time = (pick_end_ts - pick_start_ts).total_seconds()
            logger.info(f"  Batch time window duration: {pick_total_time:.1f}s ({pick_total_time/3600:.2f}h)")

            # Query equipment states for the batch time window
            equipment_states = fetch_robot_equipment_states('Pick1', pick_start_ts, pick_end_ts)
            running_time_sec = equipment_states['running_time_sec']
            downtime_sec = equipment_states['downtime_sec']

            logger.info(f"  Equipment states returned: {equipment_states}")
            logger.info(f"  Running time: {running_time_sec:.1f}s, Downtime from states: {downtime_sec:.1f}s")

            # Calculate availability using equipment states only (for batch OEE)
            # This keeps batch OEE calculation consistent with the original method
            total_time_for_availability = running_time_sec + downtime_sec
            if total_time_for_availability > 0:
                pick_availability = (running_time_sec / total_time_for_availability) * 100
                logger.info(f"  ✅ Calculated availability: {pick_availability:.2f}% = ({running_time_sec:.1f}s / {total_time_for_availability:.1f}s) × 100")
            else:
                pick_availability = 0
                logger.warning(f"  ⚠️ Total time for availability is 0, setting availability to 0%")

            logger.info(f"  Final: availability={pick_availability:.2f}%, total_time_for_daily={pick_total_time:.1f}s ({pick_total_time/3600:.2f}h)")
        else:
            # Fallback to job-level data if time window not available
            logger.warning(f"  ❌ Could not get time window (start={pick_start_ts}, end={pick_end_ts}), using job-level data fallback")
            pick_total_time = batch_pick['total_time_sec'].sum() if not batch_pick.empty and 'total_time_sec' in batch_pick.columns else 0
            pick_uptime = batch_pick['uptime_sec'].sum() if not batch_pick.empty and 'uptime_sec' in batch_pick.columns else 0
            pick_availability = weighted_avg(batch_pick, 'availability', 'total_time_sec') if not batch_pick.empty and 'total_time_sec' in batch_pick.columns else 0
            logger.info(f"  Fallback: availability={pick_availability:.2f}%, total_time={pick_total_time:.1f}s")
        
        # Performance is always 100% for Pick
        pick_performance = 100.0
        
        # Quality from batch-level QC data or job-level average (Pick uses job-level quality)
        pick_quality = batch_pick['quality'].mean() if not batch_pick.empty else 0
        logger.info(f"  Pick quality: {pick_quality:.2f}%")
        
        # Calculate OEE
        pick_oee = (pick_availability / 100) * (pick_performance / 100) * (pick_quality / 100) * 100
        logger.info(f"  ✅ Final Pick OEE: {pick_oee:.2f}% = ({pick_availability:.2f}% × {pick_performance:.2f}% × {pick_quality:.2f}%) / 10000")
        
        # Calculate OEE from weighted components
        print_oee = (print_availability / 100) * (print_performance / 100) * (print_quality / 100) * 100
        cut_oee = (cut_availability / 100) * (cut_performance / 100) * (cut_quality / 100) * 100
        pick_oee = (pick_availability / 100) * (pick_performance / 100) * (pick_quality / 100) * 100
        
        metrics = {
            'batch_id': batch_id,

            # Print metrics
            'print_job_count': len(batch_print),
            'print_oee': print_oee,
            'print_availability': print_availability,
            'print_performance': print_performance,  # Capped at 100% for OEE calculation
            'print_performance_actual': print_performance_actual,  # Actual measured performance (uncapped)
            'print_quality': print_quality,
            'print_total_time_sec': print_total_time,

            # Cut metrics
            'cut_job_count': len(batch_cut),
            'cut_oee': cut_oee,
            'cut_availability': cut_availability,
            'cut_performance': cut_performance,
            'cut_quality': cut_quality,
            'cut_total_time_sec': cut_total_time,

            # Pick metrics
            'pick_job_count': len(batch_pick),
            'pick_oee': pick_oee,
            'pick_availability': pick_availability,
            'pick_performance': pick_performance,
            'pick_quality': pick_quality,
            'pick_total_time_sec': pick_total_time,
        }

        batch_metrics.append(metrics)

    return pd.DataFrame(batch_metrics)

def calculate_daily_metrics(
    batch_metrics_df: pd.DataFrame,
    scheduled_hours: float,
    break_start: Optional[datetime] = None,
    break_end: Optional[datetime] = None,
    batches_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Calculate daily-level OEE metrics with break-aware utilization.

    Formulas:
    - Active Production Hours = Production Hours - Break Overlap
    - Utilization = Active Production Hours / Scheduled Hours × 100%
    - Weighted Batch OEE = Σ(OEE × duration) / Σ(duration)
    - Daily OEE = Weighted Batch OEE × Utilization / 100

    Args:
        batch_metrics_df: Batch-level metrics DataFrame
        scheduled_hours: Scheduled shift hours (shift duration - break duration)
        break_start: Optional start timestamp of break period
        break_end: Optional end timestamp of break period
        batches_df: Optional DataFrame with batch timestamps for break overlap calculation

    Returns:
        Dictionary with daily metrics for each cell

    Edge Cases:
    - scheduled_hours = 0: Utilization = 0%
    - Production time > scheduled_hours: Utilization > 100% (overtime/multi-shift)
    - No breaks configured: Break overlap = 0, Active = Production hours
    - Break outside production window: Break overlap = 0
    """
    # Helper function for weighted batch OEE
    def weighted_batch_oee(df: pd.DataFrame, oee_col: str, time_col: str) -> float:
        if df.empty or oee_col not in df.columns or time_col not in df.columns:
            return 0.0
        total_time = df[time_col].sum()
        if total_time == 0:
            return 0.0
        weighted = (df[oee_col] * df[time_col]).sum() / total_time
        logger.debug(f"Weighted {oee_col}: {weighted:.2f}% (vs simple mean: {df[oee_col].mean():.2f}%)")
        return weighted

    # Calculate total production time (sum across batches)
    print_production_hours = batch_metrics_df['print_total_time_sec'].sum() / 3600
    cut_production_hours = batch_metrics_df['cut_total_time_sec'].sum() / 3600
    pick_production_hours = batch_metrics_df['pick_total_time_sec'].sum() / 3600

    # Calculate break overlap for each cell
    print_break_overlap = 0.0
    cut_break_overlap = 0.0
    pick_break_overlap = 0.0

    if break_start and break_end and batches_df is not None and not batches_df.empty:
        logger.info(f"Calculating break overlap for break window: {break_start} to {break_end}")

        # Extract cell timestamps from batches_df
        if 'print_start' in batches_df.columns and 'print_end' in batches_df.columns:
            print_start = batches_df['print_start'].min()
            print_end = batches_df['print_end'].max()
            print_break_overlap = calculate_break_overlap_per_cell(print_start, print_end, break_start, break_end)
            logger.info(f"Print break overlap: {print_break_overlap:.2f}h")

        if 'cut_start' in batches_df.columns and 'cut_end' in batches_df.columns:
            cut_start = batches_df['cut_start'].min()
            cut_end = batches_df['cut_end'].max()
            cut_break_overlap = calculate_break_overlap_per_cell(cut_start, cut_end, break_start, break_end)
            logger.info(f"Cut break overlap: {cut_break_overlap:.2f}h")

        if 'pick_start' in batches_df.columns and 'pick_end' in batches_df.columns:
            pick_start = batches_df['pick_start'].min()
            pick_end = batches_df['pick_end'].max()
            pick_break_overlap = calculate_break_overlap_per_cell(pick_start, pick_end, break_start, break_end)
            logger.info(f"Pick break overlap: {pick_break_overlap:.2f}h")

    # Calculate active production hours (production minus break overlap)
    print_active_hours = max(0, print_production_hours - print_break_overlap)
    cut_active_hours = max(0, cut_production_hours - cut_break_overlap)
    pick_active_hours = max(0, pick_production_hours - pick_break_overlap)

    logger.info(f"Active production hours: Print={print_active_hours:.2f}h, Cut={cut_active_hours:.2f}h, Pick={pick_active_hours:.2f}h")

    # Calculate utilization (active hours vs scheduled hours)
    print_utilization = (print_active_hours / scheduled_hours * 100) if scheduled_hours > 0 else 0
    cut_utilization = (cut_active_hours / scheduled_hours * 100) if scheduled_hours > 0 else 0
    pick_utilization = (pick_active_hours / scheduled_hours * 100) if scheduled_hours > 0 else 0

    # Calculate scheduled idle time (unused shift time)
    print_scheduled_idle = max(0, scheduled_hours - print_active_hours)
    cut_scheduled_idle = max(0, scheduled_hours - cut_active_hours)
    pick_scheduled_idle = max(0, scheduled_hours - pick_active_hours)

    # Calculate weighted batch OEE (duration-weighted average)
    print_batch_oee = weighted_batch_oee(batch_metrics_df, 'print_oee', 'print_total_time_sec')
    cut_batch_oee = weighted_batch_oee(batch_metrics_df, 'cut_oee', 'cut_total_time_sec')
    pick_batch_oee = weighted_batch_oee(batch_metrics_df, 'pick_oee', 'pick_total_time_sec')

    # Calculate daily OEE
    print_daily_oee = (print_batch_oee * print_utilization) / 100
    cut_daily_oee = (cut_batch_oee * cut_utilization) / 100
    pick_daily_oee = (pick_batch_oee * pick_utilization) / 100

    daily_metrics = {
        'print': {
            'batch_oee': print_batch_oee,
            'production_hours': print_production_hours,
            'active_production_hours': print_active_hours,
            'break_overlap_hours': print_break_overlap,
            'scheduled_idle_hours': print_scheduled_idle,
            'utilization': print_utilization,
            'daily_oee': print_daily_oee
        },
        'cut': {
            'batch_oee': cut_batch_oee,
            'production_hours': cut_production_hours,
            'active_production_hours': cut_active_hours,
            'break_overlap_hours': cut_break_overlap,
            'scheduled_idle_hours': cut_scheduled_idle,
            'utilization': cut_utilization,
            'daily_oee': cut_daily_oee
        },
        'pick': {
            'batch_oee': pick_batch_oee,
            'production_hours': pick_production_hours,
            'active_production_hours': pick_active_hours,
            'break_overlap_hours': pick_break_overlap,
            'scheduled_idle_hours': pick_scheduled_idle,
            'utilization': pick_utilization,
            'daily_oee': pick_daily_oee
        }
    }

    logger.info(f"Calculated daily OEE: Print={print_daily_oee:.2f}%, Cut={cut_daily_oee:.2f}%, Pick={pick_daily_oee:.2f}%")
    logger.info(f"Utilization: Print={print_utilization:.2f}%, Cut={cut_utilization:.2f}%, Pick={pick_utilization:.2f}%")

    return daily_metrics
