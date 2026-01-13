"""
Quality Overlay Logic
Applies QC inspection data to job-level OEE calculations
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_qc_overlay(
    print_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    pick_df: pd.DataFrame,
    qc_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply QC inspection data overlay to cell DataFrames.

    Strategy:
    - Print/Cut: QC pass rate replaces 100% quality assumption
    - Pick: QC pass rate × pick success rate (compound quality)

    Args:
        print_df: Print job DataFrame (with quality=100%)
        cut_df: Cut job DataFrame (with quality=100%)
        pick_df: Pick job DataFrame (with quality=pick_success_rate)
        qc_df: QC data DataFrame with batch_id and pass_rate columns

    Returns:
        Tuple of (print_df, cut_df, pick_df) with updated quality and OEE

    Edge Cases:
    - Empty qc_df: Returns original DataFrames unchanged
    - Batch not in QC data: Retains 100% assumption for that batch
    - QC pass rate = 0%: Quality becomes 0%, OEE becomes 0%
    """
    if qc_df is None or qc_df.empty:
        logger.info("No QC data to apply - retaining 100% quality assumption")
        return print_df, cut_df, pick_df

    # Group QC data by batch (in case of multiple entries)
    qc_by_batch = qc_df.groupby('batch_id').agg({
        'pass_rate': 'mean'  # Average pass rate
    }).reset_index()
    
    # Normalize batch_id format: QC returns integers (955), JobReports may have "B955" or "955"
    # Convert QC batch_id to string and ensure it matches JobReport format
    qc_by_batch['batch_id'] = qc_by_batch['batch_id'].astype(str)
    
    # Also create a version without 'B' prefix for matching
    # JobReports might have "B955" or "955", QC has "955"
    # We'll try to match both formats
    qc_by_batch['batch_id_clean'] = qc_by_batch['batch_id'].str.replace('B', '').str.replace('b', '')

    # Apply to Print
    if not print_df.empty:
        # Ensure batch_id is string type and normalize format
        if 'batch_id' in print_df.columns:
            print_df['batch_id'] = print_df['batch_id'].astype(str)
            # Create clean version for matching (remove 'B' prefix)
            print_df['batch_id_clean'] = print_df['batch_id'].str.replace('B', '').str.replace('b', '')
        # Merge on clean batch_id (handles both "B955" and "955" formats)
        print_df = print_df.merge(qc_by_batch, left_on='batch_id_clean', right_on='batch_id_clean', how='left', suffixes=('', '_qc'))
        # Drop temporary columns
        print_df = print_df.drop(columns=['batch_id_clean'], errors='ignore')
        # Replace quality with QC pass rate (keep 100% if no QC data for that batch)
        print_df['quality'] = print_df['pass_rate'].fillna(print_df['quality'])
        # Recalculate OEE
        print_df['oee'] = (print_df['availability'] * print_df['performance'] * print_df['quality']) / 10000
        # Drop temporary pass_rate column
        print_df = print_df.drop(columns=['pass_rate'])
        logger.info(f"Applied QC overlay to {len(print_df)} Print jobs")

    # Apply to Cut
    if not cut_df.empty:
        # Ensure batch_id is string type and normalize format
        if 'batch_id' in cut_df.columns:
            cut_df['batch_id'] = cut_df['batch_id'].astype(str)
            # Create clean version for matching (remove 'B' prefix)
            cut_df['batch_id_clean'] = cut_df['batch_id'].str.replace('B', '').str.replace('b', '')
        # Merge on clean batch_id
        cut_df = cut_df.merge(qc_by_batch, left_on='batch_id_clean', right_on='batch_id_clean', how='left', suffixes=('', '_qc'))
        # Drop temporary columns
        cut_df = cut_df.drop(columns=['batch_id_clean'], errors='ignore')
        cut_df['quality'] = cut_df['pass_rate'].fillna(cut_df['quality'])
        cut_df['oee'] = (cut_df['availability'] * cut_df['performance'] * cut_df['quality']) / 10000
        cut_df = cut_df.drop(columns=['pass_rate'])
        logger.info(f"Applied QC overlay to {len(cut_df)} Cut jobs")

    # Apply to Pick (two-layer quality: Robot Accuracy × QC Pick Quality)
    if not pick_df.empty:
        # Ensure batch_id is string type and normalize format
        if 'batch_id' in pick_df.columns:
            pick_df['batch_id'] = pick_df['batch_id'].astype(str)
            # Create clean version for matching (remove 'B' prefix)
            pick_df['batch_id_clean'] = pick_df['batch_id'].str.replace('B', '').str.replace('b', '')
        # Merge on clean batch_id
        pick_df = pick_df.merge(qc_by_batch, left_on='batch_id_clean', right_on='batch_id_clean', how='left', suffixes=('', '_qc'))
        # Drop temporary columns
        pick_df = pick_df.drop(columns=['batch_id_clean'], errors='ignore')
        
        # Store robot accuracy (original quality = success_rate)
        pick_df['robot_accuracy'] = pick_df['quality'].copy()
        
        # For two-layer quality, we need pick_defects from quality breakdown
        # For now, use pass_rate as QC pick quality (will be updated when quality_breakdown is available)
        # Compound quality: robot_accuracy × qc_pass_rate / 100
        pick_df['quality'] = np.where(
            pick_df['pass_rate'].notna(),
            (pick_df['robot_accuracy'] * pick_df['pass_rate']) / 100,
            pick_df['robot_accuracy']  # Keep robot accuracy if no QC data
        )
        pick_df['oee'] = (pick_df['availability'] * pick_df['performance'] * pick_df['quality']) / 10000
        pick_df = pick_df.drop(columns=['pass_rate'])
        logger.info(f"Applied QC overlay to {len(pick_df)} Pick jobs (Robot Accuracy × QC Pass Rate)")

    return print_df, cut_df, pick_df
