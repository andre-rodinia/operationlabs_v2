"""
OEE Calculation Functions
Calculates OEE metrics from JobReport data with custom performance calculations.

Three-level framework:
1. Job-level OEE: Individual job performance
2. Batch-level OEE: Aggregated batch performance (when running)
3. Daily OEE: Overall effectiveness including utilization

Quality Strategy:
- Day 0 (Production): Assume 100% quality (or pick success rate for Pick cell)
- Day 3+ (Post-QC): Apply actual QC inspection results as overlay
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ============================================================
# PERFORMANCE CALCULATIONS (CORRECTED)
# ============================================================

def calculate_print_performance(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Print performance based on actual vs nominal speed.
    Performance = (Actual Speed / Nominal Speed) * 100
    
    Args:
        df: DataFrame with 'speed_actual' and 'speed_nominal' columns
        
    Returns:
        Series with performance percentages
    """
    if 'speed_actual' not in df.columns or 'speed_nominal' not in df.columns:
        logger.warning("Missing speed columns for performance calculation, defaulting to 100%")
        return pd.Series([100.0] * len(df), index=df.index)
    
    # Calculate performance
    performance = np.where(
        (df['speed_nominal'] > 0) & (df['speed_actual'].notna()),
        (df['speed_actual'] / df['speed_nominal']) * 100,
        100.0
    )
    
    # Cap at reasonable limits (equipment can exceed nominal)
    performance = np.clip(performance, 0, 150)
    
    
    return pd.Series(performance, index=df.index)


def calculate_print_performance_capped(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Print performance capped at 100% for batch-level weighted averages.
    Performance = min((Actual Speed / Nominal Speed) * 100, 100)
    
    Args:
        df: DataFrame with 'speed_actual' and 'speed_nominal' columns
        
    Returns:
        Series with performance percentages capped at 100%
    """
    if 'speed_actual' not in df.columns or 'speed_nominal' not in df.columns:
        logger.warning("Missing speed columns for capped performance calculation, defaulting to 100%")
        return pd.Series([100.0] * len(df), index=df.index)
    
    # Calculate performance
    performance = np.where(
        (df['speed_nominal'] > 0) & (df['speed_actual'].notna()),
        (df['speed_actual'] / df['speed_nominal']) * 100,
        100.0
    )
    
    # Cap at 100% for batch-level calculations
    performance_capped = np.clip(performance, 0, 100)
    
    
    return pd.Series(performance_capped, index=df.index)


def calculate_cut_performance(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Cut performance.
    Currently assumes 100% (nominal performance).
    
    RATIONALE: Cut equipment operates at consistent speed.
    Future enhancement: Calculate based on expected vs actual cut time.
    
    Args:
        df: DataFrame with Cut job data
        
    Returns:
        Series with performance percentages (always 100%)
    """
    return pd.Series([100.0] * len(df), index=df.index)


def calculate_pick_performance(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Pick performance.
    
    CORRECTED APPROACH:
    Pick robots operate at consistent speed - they don't speed up or slow down.
    Performance is always 100% (nominal).
    
    The success/failure of picks is a QUALITY metric, not performance.
    
    Args:
        df: DataFrame with Pick job data
        
    Returns:
        Series with performance percentages (always 100%)
    """
    return pd.Series([100.0] * len(df), index=df.index)


# ============================================================
# QUALITY CALCULATIONS (CORRECTED)
# ============================================================

def calculate_pick_quality_from_jobreport(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Pick quality from JobReport data.
    
    Quality = Successful Picks / Total Attempts
    
    This is calculated from the closingReport in Pick1/JobReport which tracks
    componentsCompleted vs componentsFailed in real-time during production.
    
    This is "Day 0" quality - immediate pick success rate.
    Later QC inspection quality can be overlaid using apply_qc_quality_to_cells().
    
    Args:
        df: DataFrame with 'components_completed' and 'components_failed' columns
        
    Returns:
        Series with quality percentages
    """
    if 'components_completed' not in df.columns or 'components_failed' not in df.columns:
        logger.warning("Missing component count columns for Pick quality, defaulting to 100%")
        return pd.Series([100.0] * len(df), index=df.index)
    
    # Calculate quality per job
    total_attempts = df['components_completed'] + df['components_failed']
    
    quality = np.where(
        total_attempts > 0,
        (df['components_completed'] / total_attempts) * 100,
        100.0  # If no attempts recorded, assume 100%
    )
    
    
    return pd.Series(quality, index=df.index)


def apply_qc_quality_to_cells(
    print_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    pick_df: pd.DataFrame,
    quality_breakdown: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply QC inspection quality data to cell OEE calculations.
    
    This function overlays actual QC results onto the OEE calculations,
    replacing the default 100% quality assumption with real data.
    
    QC quality applies to:
    - Print cell: QC inspection of printed components (replaces 100% assumption)
    - Cut cell: QC inspection of cut components (replaces 100% assumption)
    - Pick cell: QC inspection MULTIPLIED by pick success rate
      (component must be successfully picked AND pass QC inspection)
    
    Args:
        print_df: Print OEE DataFrame (with initial quality = 100%)
        cut_df: Cut OEE DataFrame (with initial quality = 100%)
        pick_df: Pick OEE DataFrame (with initial quality = pick success rate)
        quality_breakdown: Output from fetch_batch_quality_breakdown_v2()
            Expected columns: production_batch_id, quality_rate_checked_percent
        
    Returns:
        Tuple of (print_df, cut_df, pick_df) with updated quality metrics
    """
    if quality_breakdown is None or quality_breakdown.empty:
        logger.info("No QC data available, using default quality assumptions")
        return print_df, cut_df, pick_df
    
    logger.info("Applying QC quality data to OEE calculations")
    
    # Make copies to avoid modifying originals
    print_df = print_df.copy() if not print_df.empty else print_df
    cut_df = cut_df.copy() if not cut_df.empty else cut_df
    pick_df = pick_df.copy() if not pick_df.empty else pick_df
    
    # Create quality lookup from breakdown
    if 'production_batch_id' not in quality_breakdown.columns or 'quality_rate_checked_percent' not in quality_breakdown.columns:
        logger.error("Quality breakdown missing required columns")
        return print_df, cut_df, pick_df
    
    quality_map = quality_breakdown.set_index('production_batch_id')['quality_rate_checked_percent'].to_dict()
    
    # Apply to each cell
    for df, cell_name in [(print_df, 'Print'), (cut_df, 'Cut'), (pick_df, 'Pick')]:
        if df.empty:
            continue
            
        # Extract numeric batch ID from format like "B10-0000000955" -> 955
        if 'batch_id' in df.columns:
            df['batch_id_numeric'] = df['batch_id'].apply(
                lambda x: int(str(x).replace('B10-', '').replace('B', '').strip()) if pd.notna(x) else None
            )
            
            # Map QC quality (as percentage)
            df['quality_qc_percent'] = df['batch_id_numeric'].map(quality_map)
            
            # Store original quality for comparison
            df['quality_before_qc'] = df['quality'].copy()
            
            # Apply QC quality based on cell type
            if cell_name == 'Pick':
                # Pick: Two-layer quality = Robot Accuracy × QC Pick Quality
                # Store robot accuracy (original quality)
                df['robot_accuracy'] = df['quality_before_qc'].copy()
                
                # Calculate QC pick quality: (total_components - pick_defects) / total_components
                # Check if quality_breakdown has pick_defects column
                if 'pick_defects' in quality_breakdown.columns and 'total_components' in quality_breakdown.columns:
                    qc_pick_quality_map = {}
                    for _, qc_row in quality_breakdown.iterrows():
                        batch_id = qc_row.get('production_batch_id')
                        total = qc_row.get('total_components', 0)
                        defects = qc_row.get('pick_defects', 0)
                        if total > 0:
                            qc_pick_quality_map[batch_id] = ((total - defects) / total) * 100
                        else:
                            qc_pick_quality_map[batch_id] = 100.0
                    
                    df['qc_pick_quality_percent'] = df['batch_id_numeric'].map(qc_pick_quality_map)
                    
                    # Combined quality = Robot Accuracy × QC Pick Quality
                    df['quality'] = np.where(
                        df['qc_pick_quality_percent'].notna(),
                        (df['robot_accuracy'] / 100) * (df['qc_pick_quality_percent'] / 100) * 100,
                        df['robot_accuracy']  # Keep robot accuracy if QC not available
                    )
                    logger.info(f"Pick: Two-layer quality applied (Robot Accuracy × QC Pick Quality)")
                else:
                    # Fallback to original method if pick_defects not available
                    df['quality'] = np.where(
                        df['quality_qc_percent'].notna(),
                        (df['quality_before_qc'] / 100) * (df['quality_qc_percent'] / 100) * 100,
                        df['quality_before_qc']
                    )
                    logger.info(f"Pick: Combined pick success × QC pass rate (fallback)")
            else:
                # Print/Cut: Keep job-level quality at 100% (quality calculated at batch level only)
                # Store batch-level quality for reference but don't apply to jobs
                df['batch_quality_percent'] = df['quality_qc_percent']
                # Keep quality at 100% for job-level display
                df['quality'] = df['quality_before_qc']  # Keep at 100%
                logger.info(f"{cell_name}: Job-level quality kept at 100%, batch-level quality stored separately")
            
            # Recalculate OEE with new quality
            # For Print/Cut, quality stays at 100% so OEE doesn't change
            # For Pick, quality may have changed so recalculate
            if cell_name != 'Pick':
                # Print/Cut: OEE stays the same since quality is still 100%
                df['oee_before_qc'] = df['oee'].copy()
                # OEE remains unchanged (quality still 100%)
            else:
                # Pick: Recalculate OEE with new quality
                df['oee_before_qc'] = df['oee'].copy()
                df['oee'] = (
                    (df['availability'] / 100) *
                    (df['performance'] / 100) *
                    (df['quality'] / 100) *
                    100
                )
            
            qc_applied_count = df['quality_qc_percent'].notna().sum()
            logger.info(f"{cell_name}: Applied QC to {qc_applied_count}/{len(df)} jobs")
            
            # Clean up temporary columns
            df.drop(['batch_id_numeric', 'quality_qc_percent'], axis=1, inplace=True, errors='ignore')
    
    return print_df, cut_df, pick_df


# ============================================================
# LEVEL 1: JOB-LEVEL OEE
# ============================================================

def calculate_cell_oee(df: pd.DataFrame, cell_type: str) -> pd.DataFrame:
    """
    Calculate OEE for a cell's JobReport data.
    OEE = Availability × Performance × Quality
    
    Quality defaults to 100% for Print/Cut (until QC overlay applied).
    Quality calculated from pick success rate for Pick cell.
    
    Args:
        df: DataFrame with JobReport data (from fetch_*_jobreports functions)
        cell_type: 'print', 'cut', or 'pick'
        
    Returns:
        DataFrame with added OEE columns:
        - availability: %
        - performance: %
        - quality: %
        - oee: %
    """
    if df.empty:
        return df
    
    result_df = df.copy()
    
    # Availability (already calculated in fetchers)
    result_df['availability'] = result_df['availability_calculated']
    
    # Performance (calculated based on cell type)
    if cell_type == 'print':
        result_df['performance'] = calculate_print_performance(result_df)
        # Add capped performance column for batch-level weighted averages
        result_df['performance_capped'] = calculate_print_performance_capped(result_df)
    elif cell_type == 'cut':
        result_df['performance'] = calculate_cut_performance(result_df)
    elif cell_type == 'pick':
        result_df['performance'] = calculate_pick_performance(result_df)
    else:
        logger.warning(f"Unknown cell type: {cell_type}, defaulting to 100%")
        result_df['performance'] = 100.0
    
    # Quality (different for each cell type)
    if cell_type == 'pick':
        # Pick: Calculate from pick success rate
        result_df['quality'] = calculate_pick_quality_from_jobreport(result_df)
    else:
        # Print/Cut: Default to 100% (will be replaced by QC overlay later)
        result_df['quality'] = 100.0
        logger.info(f"{cell_type.title()}: Quality set to 100% (awaiting QC inspection data)")
    
    # Calculate OEE
    result_df['oee'] = (
        (result_df['availability'] / 100) *
        (result_df['performance'] / 100) *
        (result_df['quality'] / 100) *
        100
    )
    
    logger.info(f"Calculated OEE for {len(result_df)} {cell_type} jobs")
    logger.info(f"  Avg Quality: {result_df['quality'].mean():.1f}%")
    
    return result_df


# ============================================================
# LEVEL 2: BATCH-LEVEL OEE
# ============================================================

def aggregate_batch_oee(df: pd.DataFrame, batch_id: str = None) -> Dict:
    """
    Aggregate job-level OEE metrics to batch level.
    
    Args:
        df: DataFrame with job-level OEE data
        batch_id: Optional batch identifier
        
    Returns:
        Dictionary with batch-level metrics
    """
    if df.empty:
        return {}
    
    total_uptime_min = df['uptime_min'].sum()
    total_downtime_min = df['downtime_min'].sum()
    total_time_min = total_uptime_min + total_downtime_min
    
    # Weighted OEE (weighted by job uptime)
    if total_uptime_min > 0:
        weighted_oee = (df['oee'] * df['uptime_min']).sum() / total_uptime_min
    else:
        weighted_oee = df['oee'].mean()
    
    metrics = {
        'batch_id': batch_id,
        'cell': df['cell'].iloc[0] if 'cell' in df.columns else 'Unknown',
        'job_count': len(df),
        'total_uptime_min': total_uptime_min,
        'total_downtime_min': total_downtime_min,
        'total_time_min': total_time_min,
        'avg_availability': df['availability'].mean(),
        'avg_performance': df['performance'].mean(),
        'avg_quality': df['quality'].mean(),
        'avg_oee': df['oee'].mean(),
        'weighted_oee': weighted_oee,
    }
    
    # Add cell-specific metrics
    if 'area_sqm' in df.columns:
        metrics['total_area_sqm'] = df['area_sqm'].sum()
    if 'component_count' in df.columns:
        metrics['total_components'] = int(df['component_count'].sum())
    if 'components_completed' in df.columns:
        metrics['components_completed'] = int(df['components_completed'].sum())
        metrics['components_failed'] = int(df['components_failed'].sum())
    
    # Include QC comparison if available
    if 'oee_before_qc' in df.columns:
        metrics['avg_oee_before_qc'] = df['oee_before_qc'].mean()
        metrics['avg_quality_before_qc'] = df['quality_before_qc'].mean()
        metrics['qc_impact'] = metrics['avg_oee'] - metrics['avg_oee_before_qc']
    
    return metrics


# ============================================================
# LEVEL 3: DAILY UTILIZATION & OEE
# ============================================================

def calculate_daily_utilization(
    print_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    pick_df: pd.DataFrame,
    day_start: datetime,
    day_end: datetime,
    operating_hours_per_day: float = 24.0,
    debug_mode: bool = False
) -> Dict:
    """
    Calculate daily utilization and overall daily OEE.
    
    Utilization = Production Time / Available Time
    Daily OEE = Batch OEE × Utilization
    
    Args:
        print_df, cut_df, pick_df: DataFrames with OEE calculations
        day_start, day_end: Day boundaries (datetime with timezone)
        operating_hours_per_day: Available operating hours (24, 16, or 8)
        debug_mode: Enable debug logging
        
    Returns:
        Dictionary with daily metrics for each cell and summary
    """
    available_time_minutes = operating_hours_per_day * 60
    
    results = {
        'day_start': day_start.isoformat(),
        'day_end': day_end.isoformat(),
        'available_time_hours': operating_hours_per_day,
        'available_time_minutes': available_time_minutes
    }
    
    # Calculate for each cell
    for cell_name, df in [('print', print_df), ('cut', cut_df), ('pick', pick_df)]:
        if df.empty:
            results[cell_name] = {
                'production_time_min': 0,
                'uptime_min': 0,
                'downtime_min': 0,
                'idle_time_min': available_time_minutes,
                'utilization': 0,
                'batch_oee': 0,
                'daily_oee': 0
            }
            continue
        
        # Total production time (uptime + downtime)
        total_uptime = df['uptime_min'].sum()
        total_downtime = df['downtime_min'].sum()
        production_time = total_uptime + total_downtime
        
        # Idle time = Available - Production
        idle_time = available_time_minutes - production_time
        
        # Utilization
        utilization = (production_time / available_time_minutes * 100) if available_time_minutes > 0 else 0
        
        # Batch OEE (while running)
        batch_availability = (total_uptime / production_time * 100) if production_time > 0 else 0
        batch_performance = df['performance'].mean()
        batch_quality = df['quality'].mean()
        batch_oee = (batch_availability / 100) * (batch_performance / 100) * (batch_quality / 100) * 100
        
        # Daily OEE = Batch OEE × Utilization
        daily_oee = (batch_oee / 100) * (utilization / 100) * 100
        
        results[cell_name] = {
            'job_count': len(df),
            'production_time_min': production_time,
            'uptime_min': total_uptime,
            'downtime_min': total_downtime,
            'idle_time_min': idle_time,
            'utilization': utilization,
            'batch_availability': batch_availability,
            'batch_performance': batch_performance,
            'batch_quality': batch_quality,
            'batch_oee': batch_oee,
            'daily_oee': daily_oee
        }

    # Calculate summary
    cells_with_data = [c for c in ['print', 'cut', 'pick'] if results[c]['production_time_min'] > 0]
    
    if cells_with_data:
        results['summary'] = {
            'avg_utilization': np.mean([results[c]['utilization'] for c in cells_with_data]),
            'avg_batch_oee': np.mean([results[c]['batch_oee'] for c in cells_with_data]),
            'avg_daily_oee': np.mean([results[c]['daily_oee'] for c in cells_with_data]),
            'cells_active': len(cells_with_data)
        }
    else:
        results['summary'] = {
            'avg_utilization': 0,
            'avg_batch_oee': 0,
            'avg_daily_oee': 0,
            'cells_active': 0
        }
    
    return results
