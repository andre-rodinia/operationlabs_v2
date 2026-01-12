"""
OEE Calculator
Aggregates job-level OEE into batch-level and daily-level metrics
"""
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def calculate_batch_metrics(
    print_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    pick_df: pd.DataFrame,
    batch_ids: list
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

    for batch_id in batch_ids:
        # Filter jobs for this batch
        batch_print = print_df[print_df['batch_id'] == batch_id] if not print_df.empty else pd.DataFrame()
        batch_cut = cut_df[cut_df['batch_id'] == batch_id] if not cut_df.empty else pd.DataFrame()
        batch_pick = pick_df[pick_df['batch_id'] == batch_id] if not pick_df.empty else pd.DataFrame()

        # Aggregate metrics (mean for percentages, sum for times)
        metrics = {
            'batch_id': batch_id,

            # Print metrics
            'print_job_count': len(batch_print),
            'print_oee': batch_print['oee'].mean() if not batch_print.empty else 0,
            'print_availability': batch_print['availability'].mean() if not batch_print.empty else 0,
            'print_performance': batch_print['performance'].mean() if not batch_print.empty else 0,
            'print_quality': batch_print['quality'].mean() if not batch_print.empty else 0,
            'print_total_time_sec': batch_print['total_time_sec'].sum() if not batch_print.empty else 0,

            # Cut metrics
            'cut_job_count': len(batch_cut),
            'cut_oee': batch_cut['oee'].mean() if not batch_cut.empty else 0,
            'cut_availability': batch_cut['availability'].mean() if not batch_cut.empty else 0,
            'cut_performance': batch_cut['performance'].mean() if not batch_cut.empty else 0,
            'cut_quality': batch_cut['quality'].mean() if not batch_cut.empty else 0,
            'cut_total_time_sec': batch_cut['total_time_sec'].sum() if not batch_cut.empty else 0,

            # Pick metrics
            'pick_job_count': len(batch_pick),
            'pick_oee': batch_pick['oee'].mean() if not batch_pick.empty else 0,
            'pick_availability': batch_pick['availability'].mean() if not batch_pick.empty else 0,
            'pick_performance': batch_pick['performance'].mean() if not batch_pick.empty else 0,
            'pick_quality': batch_pick['quality'].mean() if not batch_pick.empty else 0,
            'pick_total_time_sec': batch_pick['total_time_sec'].sum() if not batch_pick.empty else 0,
        }

        batch_metrics.append(metrics)

    return pd.DataFrame(batch_metrics)

def calculate_daily_metrics(
    batch_metrics_df: pd.DataFrame,
    available_hours: float
) -> Dict:
    """
    Calculate daily-level OEE metrics with utilization.

    Formula:
    - Utilization = Production Time / Available Time × 100%
    - Daily OEE = Batch OEE × Utilization / 100

    Args:
        batch_metrics_df: Batch-level metrics DataFrame
        available_hours: Total available hours in the day (e.g., 24)

    Returns:
        Dictionary with daily metrics for each cell

    Edge Cases:
    - available_hours = 0: Utilization = 0%
    - Production time > available hours: Utilization > 100% (possible in multi-shift)
    """
    # Calculate total production time (sum across batches)
    print_production_hours = batch_metrics_df['print_total_time_sec'].sum() / 3600
    cut_production_hours = batch_metrics_df['cut_total_time_sec'].sum() / 3600
    pick_production_hours = batch_metrics_df['pick_total_time_sec'].sum() / 3600

    # Calculate utilization
    print_utilization = (print_production_hours / available_hours * 100) if available_hours > 0 else 0
    cut_utilization = (cut_production_hours / available_hours * 100) if available_hours > 0 else 0
    pick_utilization = (pick_production_hours / available_hours * 100) if available_hours > 0 else 0

    # Calculate idle time
    print_idle_hours = max(0, available_hours - print_production_hours)
    cut_idle_hours = max(0, available_hours - cut_production_hours)
    pick_idle_hours = max(0, available_hours - pick_production_hours)

    # Calculate batch OEE (average across batches)
    print_batch_oee = batch_metrics_df['print_oee'].mean()
    cut_batch_oee = batch_metrics_df['cut_oee'].mean()
    pick_batch_oee = batch_metrics_df['pick_oee'].mean()

    # Calculate daily OEE
    print_daily_oee = (print_batch_oee * print_utilization) / 100
    cut_daily_oee = (cut_batch_oee * cut_utilization) / 100
    pick_daily_oee = (pick_batch_oee * pick_utilization) / 100

    daily_metrics = {
        'print': {
            'batch_oee': print_batch_oee,
            'production_hours': print_production_hours,
            'idle_hours': print_idle_hours,
            'utilization': print_utilization,
            'daily_oee': print_daily_oee
        },
        'cut': {
            'batch_oee': cut_batch_oee,
            'production_hours': cut_production_hours,
            'idle_hours': cut_idle_hours,
            'utilization': cut_utilization,
            'daily_oee': cut_daily_oee
        },
        'pick': {
            'batch_oee': pick_batch_oee,
            'production_hours': pick_production_hours,
            'idle_hours': pick_idle_hours,
            'utilization': pick_utilization,
            'daily_oee': pick_daily_oee
        }
    }

    logger.info(f"Calculated daily metrics: Print={print_daily_oee:.2f}%, Cut={cut_daily_oee:.2f}%, Pick={pick_daily_oee:.2f}%")

    return daily_metrics
