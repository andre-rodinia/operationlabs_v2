"""
Data Fetching Module

This module contains all database data fetching functions for the manufacturing data import tool.
Handles connections to both HISTORIAN and INSIDE databases with time window filtering support.
"""

import logging
import json
import pandas as pd
import numpy as np
import psycopg2
from typing import List, Optional, Dict, Tuple
from datetime import timedelta, timezone
from dateutil import parser as dateutil_parser
from dateutil.tz import UTC, gettz

from .pool import get_historian_connection, get_inside_connection
from .queries import secure_query_builder
from core.time_windows.filters import filter_dataframe_by_time_window
from core.time_windows.models import CellTimeWindow

# Import Config for TOPICS
import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
from config import Config

logger = logging.getLogger(__name__)


def safe_sum(series):
    """Convert pandas sum result to float to avoid Decimal arithmetic issues."""
    result = series.sum()
    return float(result) if result is not None else 0.0


def fetch_production_printjob_data(selected_batches: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch production printjob data from the INSIDE database."""
    logger.info(f"Fetching production printjob data for batches: {selected_batches}")
    
    try:
        with get_inside_connection() as conn:
            cursor = conn.cursor()

            query, parameters = secure_query_builder.build_production_query(selected_batches)
            cursor.execute(query, parameters)
            data = cursor.fetchall()

            columns = [
                "rg_id",
                "status",
                "production_batch_id",
                "component_order_count",
                "index",
            ]
            df = pd.DataFrame(data, columns=columns)
            logger.info(f"Successfully fetched {len(df)} production printjob records")
            return df

    except Exception as e:
        logger.error(f"Error fetching production_printjob data: {e}", exc_info=True)
        return pd.DataFrame()


def fetch_print_data_by_batch(
    batch_ids: List[str],
    time_window: Optional[CellTimeWindow] = None
) -> pd.DataFrame:
    """
    Fetch print data for specific batch IDs regardless of time range.
    Enables analysis of batches that span multiple days/shifts.
    
    Optionally filters by time window after fetching.

    Args:
        batch_ids: List of batch IDs to query
        time_window: Optional CellTimeWindow to filter results

    Returns:
        DataFrame with print data for specified batches
    """
    logger.info(f"Fetching print data for batches: {batch_ids}")

    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()

            # Use secure batch-aware query
            query, parameters = secure_query_builder.build_print_data_by_batch_query(batch_ids)
            cursor.execute(query, parameters)
            data = cursor.fetchall()

            logger.info(f"PostgreSQL returned {len(data)} rows for batch print data")

            if not data:
                logger.warning(f"No print data found for batches: {batch_ids}")
                return pd.DataFrame()

            # Create DataFrame using column names from cursor description
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(data, columns=columns)

            # Process timing columns (convert ISO strings to datetime and rename)
            if 'timing_start_iso' in df.columns:
                df['timing_start'] = pd.to_datetime(df['timing_start_iso'], errors='coerce')
                df = df.drop('timing_start_iso', axis=1)

            if 'timing_end_iso' in df.columns:
                df['timing_end'] = pd.to_datetime(df['timing_end_iso'], errors='coerce')
                df = df.drop('timing_end_iso', axis=1)

            # Merge with production_printjob data to get component counts
            try:
                production_df = fetch_production_printjob_data(batch_ids)
                if not production_df.empty:
                    df = pd.merge(
                        df,
                        production_df,
                        left_on="jobId",
                        right_on="rg_id",
                        how="left"
                    )
                    df = df.drop("rg_id", axis=1)
                    logger.info("Successfully merged production_printjob data")
            except Exception as e:
                logger.warning(f"Could not merge production data: {e}")

            # Apply time window filter if provided
            if time_window is not None:
                # Use ts column for filtering (timestamp of the job report)
                if 'ts' in df.columns:
                    original_count = len(df)
                    df = filter_dataframe_by_time_window(df, time_window, timestamp_column='ts')
                    logger.info(
                        f"Time window filter: {original_count} → {len(df)} rows "
                        f"({len(df)/original_count*100:.1f}% coverage)"
                    )
                else:
                    logger.warning("Cannot apply time window filter: 'ts' column not found")

            logger.info(f"Successfully processed {len(df)} batch print records")
            return df

    except Exception as e:
        logger.error(f"Error fetching batch print data: {e}", exc_info=True)
        return pd.DataFrame()


def fetch_cut_data_by_jobs(
    job_ids: List[str],
    time_window: Optional[CellTimeWindow] = None
) -> pd.DataFrame:
    """
    Fetch cut data for specific job IDs.
    This is the correct way to match cut data with print data since cut records don't have batch IDs.
    
    Optionally filters by time window after fetching.

    Args:
        job_ids: List of job IDs from print data
        time_window: Optional CellTimeWindow to filter results

    Returns:
        DataFrame with cut data matched to the job IDs
    """
    try:
        if not job_ids or len(job_ids) == 0:
            logger.warning("No job IDs provided for cut data query")
            return pd.DataFrame()

        with get_historian_connection() as conn:
            cursor = conn.cursor()

            # Build secure parameterized query
            query, parameters = secure_query_builder.build_cut_data_by_job_query(job_ids)

            logger.info(f"Fetching cut data for {len(job_ids)} jobs")

            # Execute query
            cursor.execute(query, parameters)
            results = cursor.fetchall()

            if not results:
                logger.info(f"No cut data found for {len(job_ids)} jobs")
                return pd.DataFrame()

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Create DataFrame
            df = pd.DataFrame(results, columns=columns)

            # Convert timing columns to datetime if present
            if 'timing_start' in df.columns:
                df['timing_start'] = pd.to_datetime(df['timing_start'], errors='coerce')
            if 'timing_end' in df.columns:
                df['timing_end'] = pd.to_datetime(df['timing_end'], errors='coerce')

            # Apply time window filter if provided
            if time_window is not None:
                # Use ts column for filtering (timestamp of the job report)
                if 'ts' in df.columns:
                    original_count = len(df)
                    df = filter_dataframe_by_time_window(df, time_window, timestamp_column='ts')
                    logger.info(
                        f"Time window filter: {original_count} → {len(df)} rows "
                        f"({len(df)/original_count*100:.1f}% coverage)"
                    )
                else:
                    logger.warning("Cannot apply time window filter: 'ts' column not found")

            logger.info(f"Retrieved {len(df)} cut records for {len(job_ids)} jobs")
            return df

    except Exception as e:
        logger.error(f"Error fetching cut data by jobs: {e}", exc_info=True)
        return pd.DataFrame()


def fetch_pick_data_by_batch(
    batch_ids: List[str],
    time_window: CellTimeWindow
) -> pd.DataFrame:
    """
    Fetch Pick/Robot data by batch using simplified approach.
    
    This function:
    1. Gets components and pick status from INSIDE database (production_componentorder + production_componentorderevent)
    2. Gets uptime/downtime from equipment table (historian) filtered by time window
    3. Combines them into a single DataFrame

    Args:
        batch_ids: List of batch IDs (can be 'B955' or '955' format)
        time_window: CellTimeWindow with segments (required) - used to filter equipment state changes

    Returns:
        DataFrame with Pick data including component-level pick status and availability

    Raises:
        ValueError: If time_window is None or has no segments
    """
    try:
        if not batch_ids or len(batch_ids) == 0:
            logger.warning("No batch IDs provided for Pick data query")
            return pd.DataFrame()

        if not time_window or not time_window.segments:
            raise ValueError(
                "Time window with segments is required for Pick data query. "
                "Time segments are used to filter equipment state changes for accurate uptime/downtime calculation."
            )

        # Step 1: Get components and pick status from INSIDE database
        with get_inside_connection() as conn:
            cursor = conn.cursor()
            query, parameters = secure_query_builder.build_pick_data_by_batch_query(batch_ids)
            logger.info(f"Fetching Pick components and status for {len(batch_ids)} batches")
            cursor.execute(query, parameters)
            component_results = cursor.fetchall()
            
            if not component_results:
                logger.warning(f"No components found for batches: {batch_ids}")
                return pd.DataFrame()
            
            columns = [desc[0] for desc in cursor.description]
            component_df = pd.DataFrame(component_results, columns=columns)
            logger.info(f"Found {len(component_df)} components with pick status")

        # Step 2: Get uptime/downtime from equipment table (historian)
        with get_historian_connection() as conn:
            cursor = conn.cursor()
            
            # Build time window conditions for equipment state filtering
            segment_conditions = []
            time_window_params = []
            for segment in time_window.segments:
                segment_conditions.append("(e.ts > %s AND e.ts < %s)")
                time_window_params.append(segment.start.isoformat())
                time_window_params.append(segment.end.isoformat())
            
            time_window_conditions = " AND (" + " OR ".join(segment_conditions) + ")"
            
            equipment_query = f"""
                WITH state_changes AS (
                    SELECT
                        e.ts,
                        e.cell,
                        e.state,
                        LEAD(e.state) OVER (PARTITION BY e.cell ORDER BY e.ts) AS next_state,
                        LEAD(e.ts) OVER (PARTITION BY e.cell ORDER BY e.ts) AS next_ts
                    FROM equipment e
                    WHERE e.cell = 'Pick1'
                    {time_window_conditions}
                ),
                running_duration AS (
                    SELECT
                        SUM(EXTRACT(EPOCH FROM (next_ts - ts))) AS running_duration_seconds
                    FROM state_changes
                    WHERE state = 'running' 
                    AND next_state IN ('idle', 'down')
                    AND ts IS NOT NULL 
                    AND next_ts IS NOT NULL
                ),
                downtime_duration AS (
                    SELECT
                        SUM(EXTRACT(EPOCH FROM (next_ts - ts))) AS downtime_duration_seconds
                    FROM state_changes
                    WHERE state = 'down'
                    AND next_state IN ('idle', 'running')
                    AND ts IS NOT NULL 
                    AND next_ts IS NOT NULL
                )
                SELECT
                    COALESCE((SELECT running_duration_seconds FROM running_duration), 0.0) / 60.0 AS "uptime (min)",
                    COALESCE((SELECT downtime_duration_seconds FROM downtime_duration), 0.0) / 60.0 AS "downtime (min)"
            """
            
            cursor.execute(equipment_query, time_window_params)
            equipment_results = cursor.fetchone()
            
            if equipment_results:
                uptime_min = float(equipment_results[0]) if equipment_results[0] is not None else 0.0
                downtime_min = float(equipment_results[1]) if equipment_results[1] is not None else 0.0
                logger.info(f"Pick uptime/downtime: {uptime_min:.2f} min / {downtime_min:.2f} min")
            else:
                uptime_min = 0.0
                downtime_min = 0.0
                logger.warning("No equipment state data found for Pick cell")

        # Step 3: Combine component data with uptime/downtime
        component_df['uptime (min)'] = uptime_min
        component_df['downtime (min)'] = downtime_min
        component_df['cell'] = 'Pick1'
        
        # Calculate successful_picks and total_picks for aggregation
        component_df['successful_picks'] = (component_df['pick_status'] == 'successful').astype(int)
        component_df['total_picks'] = (component_df['pick_status'].isin(['successful', 'failed'])).astype(int)
        
        logger.info(f"Retrieved {len(component_df)} Pick records for {len(batch_ids)} batches")
        return component_df

    except Exception as e:
        logger.error(f"Error fetching Pick data by batch: {e}", exc_info=True)
        return pd.DataFrame()


def fetch_print_data_by_time_window(time_window: CellTimeWindow) -> pd.DataFrame:
    """
    Fetch print data for a time window and return data with batch IDs.
    This function queries by time range and extracts batch IDs from the results.

    Args:
        time_window: CellTimeWindow configuration

    Returns:
        DataFrame with print data for the time window
    """
    if not time_window.segments:
        logger.warning("No time segments in time window")
        return pd.DataFrame()

    # Get earliest start and latest end across all segments
    start_time = time_window.get_earliest_start()
    end_time = time_window.get_latest_end()

    if not start_time or not end_time:
        logger.warning("Invalid time window: missing start or end time")
        return pd.DataFrame()

    # Convert to ISO format strings
    start_iso = start_time.isoformat()
    end_iso = end_time.isoformat()

    logger.info(f"Fetching print data for time window: {start_iso} to {end_iso}")

    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()

            # Use secure time-based query
            query, parameters = secure_query_builder.build_print_data_by_time_query(start_iso, end_iso)
            cursor.execute(query, parameters)
            data = cursor.fetchall()

            logger.info(f"PostgreSQL returned {len(data)} rows for time-based print data")

            if not data:
                logger.warning(f"No print data found for time window")
                return pd.DataFrame()

            # Create DataFrame using column names from cursor description
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(data, columns=columns)

            # Process timing columns
            if 'timing_start_iso' in df.columns:
                df['timing_start'] = pd.to_datetime(df['timing_start_iso'], errors='coerce')
                df = df.drop('timing_start_iso', axis=1)

            if 'timing_end_iso' in df.columns:
                df['timing_end'] = pd.to_datetime(df['timing_end_iso'], errors='coerce')
                df = df.drop('timing_end_iso', axis=1)

            # Apply time window filter to match segments exactly (using actual job timing)
            if 'timing_start' in df.columns:
                original_count = len(df)
                df = filter_dataframe_by_time_window(df, time_window, timestamp_column='timing_start')
                logger.info(
                    f"Time window filter (by job timing): {original_count} → {len(df)} rows "
                    f"({len(df)/original_count*100:.1f}% coverage)" if original_count > 0 else "0 rows"
                )

            # Merge with production_printjob data if batch IDs are available
            if 'batchId' in df.columns and not df['batchId'].isna().all():
                batch_ids = df['batchId'].dropna().unique().tolist()
                try:
                    production_df = fetch_production_printjob_data(batch_ids)
                    if not production_df.empty:
                        df = pd.merge(
                            df,
                            production_df,
                            left_on="jobId",
                            right_on="rg_id",
                            how="left"
                        )
                        df = df.drop("rg_id", axis=1)
                        logger.info("Successfully merged production_printjob data")
                except Exception as e:
                    logger.warning(f"Could not merge production data: {e}")

            logger.info(f"Successfully processed {len(df)} time-based print records")
            return df

    except Exception as e:
        logger.error(f"Error fetching time-based print data: {e}", exc_info=True)
        return pd.DataFrame()


def fetch_quality_metrics_by_batch(batch_ids: List[str]) -> pd.DataFrame:
    """
    Fetch quality metrics (QC data) for specific batches from production_componentorder.
    Provides the Quality component of OEE calculation.

    Args:
        batch_ids: List of batch IDs to query

    Returns:
        DataFrame with quality metrics for specified batches
    """
    logger.info(f"Fetching quality metrics for batches: {batch_ids}")

    try:
        with get_inside_connection() as conn:
            cursor = conn.cursor()

            # Use secure quality metrics query
            query, parameters = secure_query_builder.build_quality_metrics_by_batch_query(batch_ids)
            cursor.execute(query, parameters)
            data = cursor.fetchall()

            logger.info(f"PostgreSQL returned {len(data)} rows for quality metrics")

            if not data:
                logger.warning(f"No quality data found for batches: {batch_ids}")
                return pd.DataFrame()

            # Create DataFrame
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(data, columns=columns)

            logger.info(f"Successfully processed {len(df)} quality metric records")
            return df

    except Exception as e:
        logger.error(f"Error fetching quality metrics: {e}", exc_info=True)
        return pd.DataFrame()


def fetch_batch_quality_breakdown(batch_ids: List[str]) -> pd.DataFrame:
    """
    Fetch Phase 2: Post-QC Reconciliation data with defect attribution.
    
    This function retrieves quality breakdown for batches with defects attributed
    to specific cells (print, cut, pick, fabric) based on QC event descriptions.
    
    Args:
        batch_ids: List of batch IDs to query
        
    Returns:
        DataFrame with quality breakdown and defect attribution
    """
    logger.info(f"Fetching batch quality breakdown for batches: {batch_ids}")

    try:
        with get_inside_connection() as conn:
            cursor = conn.cursor()

            # Use secure batch quality breakdown query
            query, parameters = secure_query_builder.build_batch_quality_breakdown_query(batch_ids)
            
            # Log the query and parameters for debugging
            logger.debug(f"Executing batch quality breakdown query")
            logger.debug(f"Parameters type: {type(parameters)}, length: {len(parameters) if parameters else 0}")
            logger.debug(f"Parameters content: {parameters}")
            if parameters and len(parameters) > 0:
                logger.debug(f"First parameter type: {type(parameters[0])}, content: {parameters[0]}")
            logger.debug(f"Query: {query[:200]}...")  # Log first 200 chars
            
            # Ensure parameters is a list/tuple for cursor.execute
            if not isinstance(parameters, (list, tuple)):
                parameters = [parameters]
            
            # For ANY(%s::bigint[]), psycopg2 expects the array as the parameter value
            # Verify parameters structure matches what psycopg2 expects
            if not parameters or len(parameters) == 0:
                raise ValueError("No parameters provided for query execution")
            
            if len(parameters) != 1:
                raise ValueError(f"Expected 1 parameter for query, got {len(parameters)}")
            
            # For ANY(%s::bigint[]), parameters should be [validated_batches] where validated_batches is a list
            # This matches the exact format used in the working quality_metrics query
            if not isinstance(parameters, list):
                raise ValueError(f"Expected list parameter for ANY(%s::bigint[]), got {type(parameters)}")
            
            if len(parameters) != 1:
                raise ValueError(f"Expected 1 parameter (list of batch IDs), got {len(parameters)}")
            
            if not isinstance(parameters[0], list):
                raise ValueError(f"Expected list of batch IDs as first parameter, got {type(parameters[0])}")
            
            if len(parameters[0]) == 0:
                raise ValueError("Parameter list is empty - no batch IDs to query")
            
            logger.debug(f"Executing query with {len(parameters[0])} batch ID(s): {parameters[0]}")
            logger.debug(f"Parameters type: {type(parameters)}, structure: {parameters}")
            logger.debug(f"Query placeholders count: {query.count('%s')}")
            
            # Use the exact same approach as the working quality_metrics query (line 391)
            # It just passes parameters directly: cursor.execute(query, parameters)
            try:
                cursor.execute(query, parameters)
                data = cursor.fetchall()
                logger.info(f"PostgreSQL returned {len(data)} rows for batch quality breakdown")
            except Exception as exec_error:
                logger.error(f"Error executing batch quality breakdown query: {exec_error}")
                logger.error(f"Query: {query[:500]}...")
                logger.error(f"Parameters: {parameters}")
                raise
            
            if not data:
                logger.warning(
                    f"No quality breakdown data found for batches: {batch_ids}. "
                    f"This could mean: (1) No components exist for these batches, "
                    f"(2) All print jobs are cancelled, or (3) Batch IDs don't match."
                )
                # Try a simpler test query to see if components exist at all
                test_query = """
                    SELECT COUNT(*) as component_count
                    FROM production_componentorder pc
                    JOIN production_printjob pj ON pc.print_job_id = pj.id
                    WHERE pc.production_batch_id = ANY(%s::bigint[])
                      AND pj.status != 'cancelled'
                """
                try:
                    cursor.execute(test_query, parameters)
                    test_result = cursor.fetchone()
                    if test_result:
                        logger.info(f"Test query found {test_result[0]} components for batch(es) {batch_ids}")
                except Exception as test_error:
                    logger.debug(f"Test query failed: {test_error}")
                
                return pd.DataFrame()

            # Create DataFrame
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(data, columns=columns)

            logger.info(f"Successfully processed {len(df)} batch quality breakdown records")
            return df

    except Exception as e:
        logger.error(f"Error fetching batch quality breakdown: {e}", exc_info=True)
        return pd.DataFrame()


def aggregate_qc_by_batch(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate QC data by batch with defect attribution.
    
    Args:
        merged_df: DataFrame with component_id, production_batch_id, 
                   production_date, qc_state, qc_source, qc_reasons
                   
    Returns:
        DataFrame with aggregated metrics per batch
    """
    
    def count_reason_exact(reasons_list, patterns: List[str]) -> int:
        """Count components with any of the exact reason patterns."""
        count = 0
        for reasons in reasons_list:
            if isinstance(reasons, list):
                for reason in reasons:
                    reason_lower = str(reason).lower()
                    if any(p.lower() == reason_lower or reason_lower.startswith(p.lower()) 
                           for p in patterns):
                        count += 1
                        break
        return count
    
    results = []
    
    for (batch_id, prod_date), group in merged_df.groupby(['production_batch_id', 'production_date']):
        total = len(group)
        passed = len(group[group['qc_state'] == 'passed'])
        failed = len(group[group['qc_state'] == 'failed'])
        not_scanned = len(group[group['qc_state'] == 'not_scanned'])
        
        # Get failed reasons for defect attribution
        failed_reasons = group[group['qc_state'] == 'failed']['qc_reasons'].tolist()
        
        # Defect attribution by cell
        # Print defects: print, print-water-mark
        print_defects = count_reason_exact(failed_reasons, ['print', 'print-water-mark'])
        
        # Cut defects: cut, cut-outside-bleed, cut-measurement*, cut-fraying
        cut_defects = count_reason_exact(failed_reasons, [
            'cut', 'cut-outside-bleed', 'cut-measurement', 
            'cut-measurement-small', 'cut-measurement-big', 'cut-fraying'
        ])
        
        # Fabric defects: fabric
        fabric_defects = count_reason_exact(failed_reasons, ['fabric'])
        
        # File/Pre-production defects: file, file-product
        file_defects = count_reason_exact(failed_reasons, ['file', 'file-product'])
        
        # Pick defects: pick
        pick_defects = count_reason_exact(failed_reasons, ['pick'])
        
        # Other defects: other
        other_defects = count_reason_exact(failed_reasons, ['other'])
        
        # QC Source breakdown
        scanned_handheld = len(group[group['qc_source'] == 'handheld-scan'])
        scanned_manual = len(group[group['qc_source'] == 'handheld-manual'])
        scanned_inside = len(group[group['qc_source'] == 'inside'])
        
        # Calculate rates
        scanned_total = passed + failed
        quality_rate = (passed / scanned_total * 100) if scanned_total > 0 else None
        scan_coverage = (scanned_total / total * 100) if total > 0 else 0
        unchecked_rate = (not_scanned / total * 100) if total > 0 else 0
        
        results.append({
            'production_batch_id': batch_id,
            'production_date': prod_date,
            'total_components': total,
            'total_passed': passed,
            'total_failed': failed,
            'not_scanned': not_scanned,
            'print_defects': print_defects,
            'cut_defects': cut_defects,
            'fabric_defects': fabric_defects,
            'file_defects': file_defects,
            'pick_defects': pick_defects,
            'other_defects': other_defects,
            'scanned_handheld': scanned_handheld,
            'scanned_manual': scanned_manual,
            'scanned_inside': scanned_inside,
            'quality_rate_checked_percent': quality_rate,
            'scan_coverage_percent': scan_coverage,
            'unchecked_rate_percent': unchecked_rate
        })
    
    return pd.DataFrame(results)


def fetch_batch_quality_breakdown_v2(batch_ids: List[str]) -> pd.DataFrame:
    """
    Fetch comprehensive QC breakdown using INSIDE database only.
    
    Uses production_componentorderqc table which contains all QC information.
    This is simpler and faster than cross-database queries.
    
    Args:
        batch_ids: List of batch IDs to analyze
        
    Returns:
        DataFrame with quality breakdown per batch
    """
    logger.info(f"Fetching batch quality breakdown (v2) for batches: {batch_ids}")
    
    # Validate and convert batch IDs
    validated_batches = []
    for bid in batch_ids:
        try:
            clean_id = str(bid).replace('B', '').replace('b', '').strip()
            validated_batches.append(int(clean_id))
        except (ValueError, AttributeError):
            logger.warning(f"Invalid batch ID format: {bid}")
            continue
    
    if not validated_batches:
        logger.error("No valid batch IDs provided")
        return pd.DataFrame()
    
    try:
        # Single query to get all components with QC data from INSIDE database
        with get_inside_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    pc.id as component_id,
                    pc.production_batch_id,
                    DATE(pc.created_at) as production_date,
                    qc.state as qc_state,
                    qc.source as qc_source,
                    qc.reasons as qc_reasons
                FROM production_componentorder pc
                JOIN production_printjob pj ON pc.print_job_id = pj.id
                LEFT JOIN production_componentorderqc qc ON qc.component_order_id = pc.id
                WHERE pc.production_batch_id = ANY(%s::bigint[])
                  AND pj.status != 'cancelled'
            """
            
            cursor.execute(query, [validated_batches])
            results = cursor.fetchall()
            
            if not results:
                logger.warning(f"No components found for batches: {batch_ids}")
                return pd.DataFrame()
            
            # Create DataFrame with all component and QC data
            columns = [desc[0] for desc in cursor.description]
            merged_df = pd.DataFrame(results, columns=columns)
            
            logger.info(f"Found {len(merged_df)} components across {len(validated_batches)} batches")
            
            # Fill missing QC state as 'not_scanned' (components without QC records)
            merged_df['qc_state'] = merged_df['qc_state'].fillna('not_scanned')
            
            # Handle reasons array - it might come as array or JSON string
            if 'qc_reasons' in merged_df.columns:
                merged_df['qc_reasons'] = merged_df['qc_reasons'].apply(
                    lambda x: (
                        json.loads(x) if isinstance(x, str) 
                        else (x if isinstance(x, list) 
                              else (list(x) if hasattr(x, '__iter__') and not isinstance(x, str) 
                                    else []))
                    )
                )
            else:
                merged_df['qc_reasons'] = [[]] * len(merged_df)
            
            # Log QC state distribution
            state_counts = merged_df['qc_state'].value_counts().to_dict()
            logger.info(f"QC state distribution: {state_counts}")
            
            logger.debug(f"Components with QC data: {(merged_df['qc_state'] != 'not_scanned').sum()}")
            logger.debug(f"Components without QC data (not_scanned): {(merged_df['qc_state'] == 'not_scanned').sum()}")
            
            # Aggregate by batch
            aggregated = aggregate_qc_by_batch(merged_df)
            
            logger.info(f"Successfully processed QC breakdown for {len(aggregated)} batches")
            if not aggregated.empty:
                logger.debug(f"Aggregated results:\n{aggregated.to_string()}")
            return aggregated

    except Exception as e:
        logger.error(f"Error in batch quality breakdown v2: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_pick_jobreports_by_jobs(job_ids: List[str], time_window: Optional[CellTimeWindow] = None) -> pd.DataFrame:
    """
    Fetch Pick JobReport JSON from historian and parse into job-level dataframe.
    
    Matches Print/Cut structure with two-layer quality: Robot Accuracy × QC Pick Quality.
    
    Args:
        job_ids: List of job IDs to fetch Pick reports for
        time_window: Optional time window for filtering
        
    Returns:
        DataFrame with columns:
        - job_id, batch_id, cell, ts
        - job_start, job_end, job_duration_s
        - uptime (min), downtime (min)
        - components_per_job, components_completed, components_failed, components_queued, components_ignored
        - success_rate, failure_rate
        - avg_component_pick_time_s
        - availability, performance, quality, oee
    """
    if not job_ids:
        return pd.DataFrame()
    
    try:
        # Use secure query builder
        query, parameters = secure_query_builder.build_pick_data_by_job_query(job_ids)
        
        logger.info(f"Executing Pick query for {len(job_ids)} jobs")
        logger.debug(f"Sample job IDs: {job_ids[:3] if len(job_ids) > 0 else 'none'}")
        
        with get_historian_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, parameters)
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
        
        logger.info(f"Query returned {len(results)} rows")
        logger.info(f"Query columns: {columns}")
        
        # Debug: Check sheet_start_ts and sheet_end_ts in raw results
        if results and len(results) > 0:
            sheet_start_idx = columns.index('sheet_start_ts') if 'sheet_start_ts' in columns else -1
            sheet_end_idx = columns.index('sheet_end_ts') if 'sheet_end_ts' in columns else -1
            if sheet_start_idx >= 0 and sheet_end_idx >= 0:
                sample_record = results[0]
                logger.info(f"🔍 Debug: Sample record sheet_start_ts: {sample_record[sheet_start_idx]} (type: {type(sample_record[sheet_start_idx])})")
                logger.info(f"🔍 Debug: Sample record sheet_end_ts: {sample_record[sheet_end_idx]} (type: {type(sample_record[sheet_end_idx])})")
                # Count non-null values
                non_null_start = sum(1 for r in results if r[sheet_start_idx] is not None and str(r[sheet_start_idx]).strip() != '')
                non_null_end = sum(1 for r in results if r[sheet_end_idx] is not None and str(r[sheet_end_idx]).strip() != '')
                logger.info(f"🔍 Debug: Non-null sheet_start_ts in raw results: {non_null_start}/{len(results)}")
                logger.info(f"🔍 Debug: Non-null sheet_end_ts in raw results: {non_null_end}/{len(results)}")
        
        # Parse results
        if not results:
            logger.warning(f"No Pick JobReports found for {len(job_ids)} jobs")
            logger.debug(f"Query executed with {len(job_ids)} job IDs: {job_ids[:5]}...")  # Log first 5
            logger.debug(f"Query topic: {parameters[0] if len(parameters) > 0 else 'N/A'}")
            return pd.DataFrame()
        
        df = pd.DataFrame(results, columns=columns)
        
        if df.empty:
            logger.warning(f"DataFrame is empty after parsing {len(results)} results")
            return pd.DataFrame()
        
        # Log sample of what we got
        if len(df) > 0:
            sample_row = df.iloc[0].to_dict()
            logger.info(f"Sample row keys: {list(sample_row.keys())}")
            # Check if our expected columns exist
            expected_cols = ['components_completed', 'components_failed', 'components_state_unknown', 'sheetIndex']
            for col in expected_cols:
                if col in sample_row:
                    val = sample_row[col]
                    val_type = type(val).__name__
                    if val is None:
                        val_preview = 'None'
                    elif isinstance(val, (list, dict)):
                        val_preview = f"{val_type} with {len(val)} items" if hasattr(val, '__len__') else val_type
                    else:
                        val_preview = str(val)[:100]
                    logger.info(f"  {col}: type={val_type}, preview={val_preview}")
                else:
                    logger.warning(f"  {col}: NOT FOUND in columns!")
        
        # Check if we have valid job_id values (not all NULL)
        if 'job_id' in df.columns:
            valid_jobs = df['job_id'].notna().sum()
            logger.info(f"Found {valid_jobs} rows with valid job_id out of {len(df)} total rows")
            if valid_jobs == 0:
                logger.error("All job_id values are NULL - query may not be matching correctly")
                logger.error(f"Sample row data: {df.iloc[0].to_dict() if len(df) > 0 else 'No rows'}")
                logger.error(f"Columns in result: {df.columns.tolist()}")
                return pd.DataFrame()
            # Filter out rows with NULL job_id
            df = df[df['job_id'].notna()].copy()
            logger.info(f"Filtered to {len(df)} rows with valid job_id")
        else:
            logger.error(f"job_id column not found in results! Columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        if df.empty:
            logger.warning(f"No Pick JobReports found for {len(job_ids)} jobs")
            return pd.DataFrame()
        
        # Deduplicate: If multiple reports for same job_id, keep the latest (by ts)
        if 'job_id' in df.columns and 'ts' in df.columns:
            initial_count = len(df)
            df = df.sort_values('ts', ascending=False).drop_duplicates(subset=['job_id'], keep='first')
            df = df.sort_values('ts', ascending=True)  # Re-sort chronologically
            if len(df) < initial_count:
                logger.info(f"Deduplicated Pick reports: {initial_count} -> {len(df)} (kept latest per job_id)")
        
        logger.info(f"Found {len(df)} Pick JobReports for {df['job_id'].nunique()} unique jobs")
        
        # Verify expected columns exist
        expected_cols = ['components_completed', 'components_failed', 'components_state_unknown', 'successful_picks', 'total_picks']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing expected columns in DataFrame: {missing_cols}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            # Don't return empty - try to continue with what we have
        
        # Parse component arrays from closingReport for each row
        component_metrics = []
        logger.info(f"Starting to parse component metrics for {len(df)} rows")
        for idx, row in df.iterrows():
            try:
                components_completed = row.get('components_completed') if 'components_completed' in df.columns else None
                components_failed = row.get('components_failed') if 'components_failed' in df.columns else None
                components_state_unknown = row.get('components_state_unknown') if 'components_state_unknown' in df.columns else None
                successful_picks = row.get('successful_picks', 0) or 0 if 'successful_picks' in df.columns else 0
                total_picks = row.get('total_picks', 0) or 0 if 'total_picks' in df.columns else 0
                
                # Initialize counts
                completed = 0
                failed = 0
                ignored = 0
                pick_times = []
                
                # Parse componentsCompleted array (most reliable source)
                if components_completed is not None and not (isinstance(components_completed, float) and pd.isna(components_completed)):
                    try:
                        if isinstance(components_completed, str):
                            completed_list = json.loads(components_completed)
                        else:
                            completed_list = components_completed
                        
                        if isinstance(completed_list, list):
                            completed = len(completed_list)
                            # Extract pickJobDuration from each completed component
                            for comp in completed_list:
                                if isinstance(comp, dict):
                                    info = comp.get('info', {})
                                    if isinstance(info, dict) and 'pickJobDuration' in info:
                                        try:
                                            # pickJobDuration is in milliseconds, convert to seconds
                                            pick_times.append(float(info['pickJobDuration']) / 1000)
                                        except (ValueError, TypeError):
                                            pass
                    except (json.JSONDecodeError, TypeError, AttributeError) as e:
                        logger.warning(f"Error parsing components_completed for job {row.get('job_id')}: {e}")
                        logger.debug(f"  components_completed type: {type(components_completed)}, value: {str(components_completed)[:200]}")
                
                # Parse componentsFailed array
                if components_failed is not None and not (isinstance(components_failed, float) and pd.isna(components_failed)):
                    try:
                        if isinstance(components_failed, str):
                            failed_list = json.loads(components_failed)
                        else:
                            failed_list = components_failed
                        
                        if isinstance(failed_list, list):
                            failed = len(failed_list)
                    except (json.JSONDecodeError, TypeError, AttributeError) as e:
                        logger.debug(f"Error parsing components_failed for job {row.get('job_id')}: {e}")
                
                # Don't parse componentsStateUnknown array (may be corrupted)
                # Instead, calculate ignored as: order_count - completed - failed
                
                # Fallback: Use successful_picks/total_picks from metrics if arrays not available
                if completed == 0 and failed == 0:
                    if successful_picks > 0:
                        completed = int(successful_picks)
                        failed = int(total_picks - successful_picks) if total_picks >= successful_picks else 0
                        logger.debug(f"Using metrics fallback for job {row.get('job_id')}: completed={completed}, failed={failed}")
                
                # Calculate ignored components as: order_count - completed - failed
                # Get order_count from component_count or total_picks
                order_count = row.get('component_count', 0) or row.get('total_picks', 0) or 0
                if isinstance(order_count, (int, float)):
                    order_count = int(order_count)
                else:
                    order_count = 0
                
                # Calculate ignored: order_count - completed - failed (ensure non-negative)
                ignored = max(0, order_count - completed - failed)
                
                # Calculate average pick time
                avg_pick_time = np.mean(pick_times) if pick_times else 0.0
                
                component_metrics.append({
                    'components_completed': completed,
                    'components_failed': failed,
                    'components_queued': 0,  # Not used in current structure
                    'components_ignored': ignored,
                    'avg_component_pick_time_s': avg_pick_time
                })
            except Exception as row_error:
                logger.error(f"Error processing row {idx} (job_id: {row.get('job_id', 'unknown')}): {row_error}", exc_info=True)
                # Add default values for this row to maintain DataFrame shape
                component_metrics.append({
                    'components_completed': 0,
                    'components_failed': 0,
                    'components_queued': 0,
                    'components_ignored': 0,
                    'avg_component_pick_time_s': 0.0
                })
        
        logger.info(f"Parsed component metrics for {len(component_metrics)} rows")
        if len(component_metrics) > 0:
            logger.info(f"Sample metrics - completed: {component_metrics[0].get('components_completed')}, failed: {component_metrics[0].get('components_failed')}, avg_time: {component_metrics[0].get('avg_component_pick_time_s')}")
        
        # Add component metrics to dataframe
        if len(component_metrics) != len(df):
            logger.error(f"Mismatch: component_metrics has {len(component_metrics)} items but df has {len(df)} rows")
            logger.error(f"This will cause concat to fail. Returning empty DataFrame.")
            return pd.DataFrame()
        
        try:
            # Drop the raw JSON array columns from SQL query before concatenating
            # We'll use the parsed integer counts from component_metrics instead
            columns_to_drop = ['components_completed', 'components_failed', 'components_state_unknown', 'components_events']
            existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
            if existing_cols_to_drop:
                logger.info(f"Dropping raw JSON array columns before concat: {existing_cols_to_drop}")
                df = df.drop(columns=existing_cols_to_drop)
            
            metrics_df = pd.DataFrame(component_metrics)
            df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)
            logger.info(f"Successfully merged component metrics: df now has {len(df)} rows and {len(df.columns)} columns")
            
            # Check for duplicate column names (shouldn't happen now, but just in case)
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicate_cols:
                logger.warning(f"Found duplicate column names after concat: {duplicate_cols}")
                logger.warning(f"All columns: {df.columns.tolist()}")
                # Keep only the first occurrence of each duplicate column
                df = df.loc[:, ~df.columns.duplicated()]
                logger.info(f"Removed duplicate columns. Now has {len(df.columns)} columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error concatenating component metrics: {e}", exc_info=True)
            logger.error(f"df shape: {df.shape if not df.empty else 'empty'}, component_metrics length: {len(component_metrics)}")
            if len(component_metrics) > 0:
                logger.error(f"Sample component_metrics item: {component_metrics[0]}")
            return pd.DataFrame()
        
        # Rename and calculate derived metrics
        # Log before rename to see what columns we have
        logger.info(f"Columns before rename: {df.columns.tolist()}")
        if 'sheetIndex' in df.columns:
            logger.info(f"sheetIndex column exists before rename. Sample values: {df['sheetIndex'].head(3).tolist()}")
            logger.info(f"sheetIndex non-null count: {df['sheetIndex'].notna().sum()}/{len(df)}")
        
        # Check for sheet timing columns before rename
        if 'sheet_start_ts' in df.columns:
            non_null_count = df['sheet_start_ts'].notna().sum()
            logger.info(f"sheet_start_ts found! Non-null count: {non_null_count}/{len(df)}")
            if non_null_count > 0:
                logger.info(f"  Sample sheet_start_ts values: {df[df['sheet_start_ts'].notna()]['sheet_start_ts'].head(3).tolist()}")
            else:
                logger.warning(f"  ⚠️ All sheet_start_ts values are NULL - JSON path may be incorrect")
                # Try to inspect the actual JSON structure for debugging
                if len(df) > 0:
                    sample_job_id = df.iloc[0]['job_id'] if 'job_id' in df.columns else 'unknown'
                    logger.warning(f"  Checking JSON structure for job {sample_job_id}...")
        else:
            logger.warning(f"sheet_start_ts NOT found in columns: {df.columns.tolist()}")
        
        if 'sheet_end_ts' in df.columns:
            non_null_count = df['sheet_end_ts'].notna().sum()
            logger.info(f"sheet_end_ts found! Non-null count: {non_null_count}/{len(df)}")
            if non_null_count > 0:
                logger.info(f"  Sample sheet_end_ts values: {df[df['sheet_end_ts'].notna()]['sheet_end_ts'].head(3).tolist()}")
            else:
                logger.warning(f"  ⚠️ All sheet_end_ts values are NULL - JSON path may be incorrect")
        else:
            logger.warning(f"sheet_end_ts NOT found in columns: {df.columns.tolist()}")
        
        df = df.rename(columns={
            'batchId': 'batch_id',
            'sheetIndex': 'sheet_index',
            'uptime (min)': 'uptime_min',
            'downtime (min)': 'downtime_min',
            'component_count': 'components_per_job'
            # Note: sheet_start_ts and sheet_end_ts keep their names (no rename needed)
        })
        
        logger.info(f"Columns after rename: {df.columns.tolist()}")
        if 'sheet_start_ts' in df.columns:
            logger.info(f"✅ sheet_start_ts preserved after rename")
        if 'sheet_end_ts' in df.columns:
            logger.info(f"✅ sheet_end_ts preserved after rename")
        
        # Convert Decimal types to float to avoid arithmetic errors
        # PostgreSQL returns numeric columns as Decimal, which can't be multiplied with float
        numeric_columns = ['uptime_min', 'downtime_min', 'job_duration_s', 'successful_picks', 'total_picks', 'components_per_job']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        # Calculate total_attempts (using parsed integer counts, not raw JSON arrays)
        if 'components_completed' in df.columns and 'components_failed' in df.columns:
            # Ensure these are numeric (they should be integers from component_metrics)
            df['components_completed'] = pd.to_numeric(df['components_completed'], errors='coerce').fillna(0).astype(int)
            df['components_failed'] = pd.to_numeric(df['components_failed'], errors='coerce').fillna(0).astype(int)
            df['total_attempts'] = (df['components_completed'] + df['components_failed']).astype(int)
        else:
            logger.warning("components_completed or components_failed columns missing, setting total_attempts to 0")
            df['total_attempts'] = 0
        
        # Add order_count column (from component_count or total_picks)
        if 'components_per_job' in df.columns:
            df['order_count'] = pd.to_numeric(df['components_per_job'], errors='coerce').fillna(0).astype(int)
        elif 'total_picks' in df.columns:
            df['order_count'] = pd.to_numeric(df['total_picks'], errors='coerce').fillna(0).astype(int)
        else:
            df['order_count'] = 0
        
        # Add new columns: job_active_window_duration and successful_picks_time
        # job_active_window_duration is already in job_duration_s (from jobActiveWindow.duration_s)
        if 'job_duration_s' in df.columns:
            df['job_active_window_duration'] = pd.to_numeric(df['job_duration_s'], errors='coerce').fillna(0.0).astype(float)
        else:
            df['job_active_window_duration'] = 0.0
        
        # successful_picks_time = components_completed × avg_component_pick_time_s
        if 'components_completed' in df.columns and 'avg_component_pick_time_s' in df.columns:
            df['successful_picks_time'] = (
                pd.to_numeric(df['components_completed'], errors='coerce').fillna(0) *
                pd.to_numeric(df['avg_component_pick_time_s'], errors='coerce').fillna(0.0)
            ).astype(float)
        else:
            df['successful_picks_time'] = 0.0
        
        # Ensure sheet_index column exists (will be populated from Print JobReports by job_id in ui/app.py)
        # Pick JobReports don't contain sheetIndex, so we initialize it as None
        if 'sheet_index' not in df.columns:
            df['sheet_index'] = None
        else:
            # Convert to numeric if it exists (should be NULL from SQL query)
            df['sheet_index'] = pd.to_numeric(df['sheet_index'], errors='coerce')
            # Convert NaN to None for consistency with other cells
            df['sheet_index'] = df['sheet_index'].where(pd.notna(df['sheet_index']), None)
            logger.info(f"sheet_index initialized (will be merged from Print): {df['sheet_index'].notna().sum()}/{len(df)} rows have values")
        
        # Calculate success_rate and failure_rate (for internal use, not displayed)
        total_attempts = df['total_attempts']
        df['success_rate'] = np.where(
            total_attempts > 0,
            (df['components_completed'] / total_attempts) * 100,
            100.0  # If no attempts, assume 100% (benefit of doubt)
        )
        df['failure_rate'] = np.where(
            total_attempts > 0,
            (df['components_failed'] / total_attempts) * 100,
            0.0
        )
        
        # Calculate availability, performance, quality, and oee for batch-level aggregation
        # (These are not displayed in job-level details but needed for batch metrics)
        total_time = df['uptime_min'] + df['downtime_min']
        df['availability'] = np.where(
            total_time > 0,
            (df['uptime_min'] / total_time) * 100,
            0.0
        )
        
        # Performance is always 100% for Pick
        df['performance'] = 100.0
        
        # Quality = success_rate (robot accuracy)
        df['quality'] = df['success_rate']
        
        # OEE = availability × performance × quality / 10000
        df['oee'] = (df['availability'] * df['performance'] * df['quality']) / 10000
        
        # Handle NULL values and ensure proper types
        df['components_per_job'] = pd.to_numeric(df['components_per_job'], errors='coerce').fillna(0).astype(int)
        df['components_completed'] = pd.to_numeric(df['components_completed'], errors='coerce').fillna(0).astype(int)
        df['components_failed'] = pd.to_numeric(df['components_failed'], errors='coerce').fillna(0).astype(int)
        df['components_queued'] = pd.to_numeric(df['components_queued'], errors='coerce').fillna(0).astype(int)
        df['components_ignored'] = pd.to_numeric(df['components_ignored'], errors='coerce').fillna(0).astype(int)
        df['uptime_min'] = pd.to_numeric(df['uptime_min'], errors='coerce').fillna(0.0).astype(float)
        df['downtime_min'] = pd.to_numeric(df['downtime_min'], errors='coerce').fillna(0.0).astype(float)
        df['avg_component_pick_time_s'] = pd.to_numeric(df['avg_component_pick_time_s'], errors='coerce').fillna(0.0).astype(float)
        
        # Ensure calculated columns are also float
        # Note: uptime_sec, downtime_sec, total_time_sec are removed from Pick data extract
        # availability, performance, quality, oee are still calculated for batch-level aggregation
        df['availability'] = pd.to_numeric(df['availability'], errors='coerce').fillna(0.0).astype(float)
        df['performance'] = pd.to_numeric(df['performance'], errors='coerce').fillna(100.0).astype(float)
        df['quality'] = pd.to_numeric(df['quality'], errors='coerce').fillna(0.0).astype(float)
        df['oee'] = pd.to_numeric(df['oee'], errors='coerce').fillna(0.0).astype(float)
        
        # Apply time window filter if provided
        if time_window and not df.empty:
            if 'ts' in df.columns:
                df = filter_dataframe_by_time_window(df, time_window, 'ts')
            else:
                logger.warning("Cannot apply time window filter: 'ts' column not found")
        
        logger.info(f"Fetched {len(df)} Pick JobReports for {len(job_ids)} requested jobs")
        if not df.empty:
            logger.info(f"  Unique job IDs found: {df['job_id'].nunique()}")
            logger.info(f"  Sample job IDs: {df['job_id'].head(3).tolist()}")
            logger.info(f"  Columns: {df.columns.tolist()}")
            if len(df) > 0:
                logger.info(f"  Sample data - first row job_id: {df.iloc[0]['job_id']}")
                if 'components_completed' in df.columns:
                    logger.info(f"  Components completed (sum): {df['components_completed'].sum()}")
                    logger.info(f"  Components failed (sum): {df['components_failed'].sum()}")
                    logger.info(f"  Components ignored (sum): {df['components_ignored'].sum()}")
                    logger.info(f"  Avg pick time (mean): {df['avg_component_pick_time_s'].mean():.2f}s")
                else:
                    logger.error(f"  components_completed column missing! Available: {df.columns.tolist()}")
        else:
            logger.warning(f"DataFrame is empty - no Pick data to return")
        
        # Final validation before returning
        if df.empty:
            logger.error("RETURNING EMPTY DATAFRAME - Pick data fetch failed")
        else:
            logger.info(f"RETURNING DATAFRAME with {len(df)} rows and columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in fetch_pick_jobreports_by_jobs: {e}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def get_batch_pick_time_window(pick_df: pd.DataFrame, batch_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Calculate batch time window from Pick JobReports using sheetStart_ts and sheetEnd_ts.
    
    Args:
        pick_df: Pick jobs DataFrame with sheet_start_ts and sheet_end_ts columns
        batch_id: Batch ID to filter by (can be in various formats: "955", "B955", "B10-0000000955")
        
    Returns:
        Tuple of (start_timestamp, end_timestamp) or (None, None) if no data
    """
    logger.info(f"🔍 Getting batch time window for batch {batch_id}")
    logger.info(f"  pick_df empty: {pick_df.empty}, columns: {pick_df.columns.tolist() if not pick_df.empty else 'N/A'}")
    
    if pick_df.empty:
        logger.warning(f"  ❌ pick_df is empty for batch {batch_id}")
        return None, None
    
    if 'batch_id' not in pick_df.columns:
        logger.warning(f"  ❌ 'batch_id' column not found in pick_df. Available columns: {pick_df.columns.tolist()}")
        return None, None
    
    # Check if DataFrame is already filtered (all rows have same batch_id)
    # If so, skip filtering and use directly
    if 'batch_id' in pick_df.columns and len(pick_df) > 0:
        unique_batch_ids = pick_df['batch_id'].unique()
        if len(unique_batch_ids) == 1:
            logger.info(f"  ✅ DataFrame already filtered to single batch_id: {unique_batch_ids[0]}")
            batch_pick = pick_df  # Use directly, no need to filter
        else:
            # Need to filter - use normalized matching
            def normalize_batch_id(bid):
                if pd.isna(bid):
                    return None
                bid_str = str(bid)
                normalized = bid_str.replace('B10-', '').replace('B', '').strip()
                try:
                    normalized = str(int(normalized))  # Strip leading zeros
                except (ValueError, TypeError):
                    pass
                return normalized
            
            batch_id_normalized = normalize_batch_id(batch_id)
            logger.info(f"  Normalized batch_id '{batch_id}' to '{batch_id_normalized}'")
            
            # Filter by batch_id (try exact match first, then normalized)
            batch_pick = pick_df[pick_df['batch_id'] == batch_id] if not pick_df.empty else pd.DataFrame()
            if batch_pick.empty:
                # Try normalized matching
                pick_df_normalized = pick_df['batch_id'].apply(normalize_batch_id)
                batch_pick = pick_df[pick_df_normalized == batch_id_normalized]
                logger.info(f"  Exact match: 0 jobs, Normalized match: {len(batch_pick)} jobs")
            else:
                logger.info(f"  Exact match: {len(batch_pick)} jobs")
            
            if batch_pick.empty:
                logger.warning(f"  ❌ No Pick jobs found for batch {batch_id} (normalized: {batch_id_normalized})")
                logger.info(f"  Available batch_ids in pick_df: {pick_df['batch_id'].unique().tolist()}")
                if 'batch_id' in pick_df.columns:
                    normalized_available = pick_df['batch_id'].apply(normalize_batch_id).unique().tolist()
                    logger.info(f"  Normalized batch_ids in pick_df: {normalized_available}")
                return None, None
    else:
        if 'batch_id' not in pick_df.columns:
            logger.warning(f"  ❌ 'batch_id' column not found in pick_df. Available columns: {pick_df.columns.tolist()}")
            return None, None
        batch_pick = pick_df
    
    # Check for sheet timing columns
    logger.info(f"  Checking for sheet timing columns...")
    logger.info(f"  Available columns: {batch_pick.columns.tolist()}")
    
    if 'sheet_start_ts' not in batch_pick.columns:
        logger.error(f"  ❌ 'sheet_start_ts' column not found! Available: {batch_pick.columns.tolist()}")
        return None, None
    
    if 'sheet_end_ts' not in batch_pick.columns:
        logger.error(f"  ❌ 'sheet_end_ts' column not found! Available: {batch_pick.columns.tolist()}")
        return None, None
    
    logger.info(f"  ✅ Found sheet_start_ts and sheet_end_ts columns")
    
    # Get MIN(sheetStart_ts) and MAX(sheetEnd_ts)
    # Log raw values to debug
    logger.info(f"  Checking sheet_start_ts and sheet_end_ts values...")
    logger.info(f"  Total rows: {len(batch_pick)}")
    if len(batch_pick) > 0:
        sample_row = batch_pick.iloc[0]
        logger.info(f"  Sample row sheet_start_ts: {sample_row.get('sheet_start_ts')} (type: {type(sample_row.get('sheet_start_ts'))})")
        logger.info(f"  Sample row sheet_end_ts: {sample_row.get('sheet_end_ts')} (type: {type(sample_row.get('sheet_end_ts'))})")
        # Check if they're empty strings vs NULL
        if 'sheet_start_ts' in batch_pick.columns:
            non_null_count = batch_pick['sheet_start_ts'].notna().sum()
            empty_string_count = (batch_pick['sheet_start_ts'].astype(str).str.strip() == '').sum()
            logger.info(f"  sheet_start_ts: {non_null_count} non-null, {empty_string_count} empty strings, {len(batch_pick) - non_null_count - empty_string_count} NULL")
        if 'sheet_end_ts' in batch_pick.columns:
            non_null_count = batch_pick['sheet_end_ts'].notna().sum()
            empty_string_count = (batch_pick['sheet_end_ts'].astype(str).str.strip() == '').sum()
            logger.info(f"  sheet_end_ts: {non_null_count} non-null, {empty_string_count} empty strings, {len(batch_pick) - non_null_count - empty_string_count} NULL")
    
    start_times = batch_pick['sheet_start_ts'].dropna()
    end_times = batch_pick['sheet_end_ts'].dropna()
    
    # Also filter out empty strings
    if 'sheet_start_ts' in batch_pick.columns:
        start_times = batch_pick[batch_pick['sheet_start_ts'].notna() & (batch_pick['sheet_start_ts'].astype(str).str.strip() != '')]['sheet_start_ts']
    if 'sheet_end_ts' in batch_pick.columns:
        end_times = batch_pick[batch_pick['sheet_end_ts'].notna() & (batch_pick['sheet_end_ts'].astype(str).str.strip() != '')]['sheet_end_ts']
    
    logger.info(f"  Non-null start_times: {len(start_times)}/{len(batch_pick)}")
    logger.info(f"  Non-null end_times: {len(end_times)}/{len(batch_pick)}")
    
    if not start_times.empty:
        logger.info(f"  Sample start_times: {start_times.head(3).tolist()}")
    if not end_times.empty:
        logger.info(f"  Sample end_times: {end_times.head(3).tolist()}")
    
    if start_times.empty or end_times.empty:
        logger.warning(f"  ❌ No valid sheet timestamps found for batch {batch_id}")
        logger.warning(f"    start_times empty: {start_times.empty}, end_times empty: {end_times.empty}")
        # Check if the JSON path in SQL query is correct
        logger.warning(f"    This suggests sheetStart_ts/sheetEnd_ts are not being extracted from JSON correctly")
        return None, None
    
    batch_start = start_times.min()
    batch_end = end_times.max()
    
    logger.info(f"  ✅ Batch {batch_id} time window: {batch_start} to {batch_end}")
    return batch_start, batch_end


def fetch_robot_equipment_states(
    cell: str,
    start_ts: str,
    end_ts: str
) -> Dict[str, float]:
    """
    Query equipment state changes for robot cell within time window.
    
    Calculates total running time and downtime from equipment state transitions.
    
    Args:
        cell: Cell name (e.g., 'Pick1')
        start_ts: Start timestamp (ISO format string)
        end_ts: End timestamp (ISO format string)
        
    Returns:
        Dictionary with 'running_time_sec' and 'downtime_sec'
    """
    logger.info(f"🔍 Fetching robot equipment states for {cell}")
    logger.info(f"  Time window: {start_ts} to {end_ts}")
    
    if not start_ts or not end_ts:
        logger.warning(f"  ❌ Missing start or end timestamp: start_ts={start_ts}, end_ts={end_ts}")
        return {'running_time_sec': 0.0, 'downtime_sec': 0.0}
    
    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                WITH state_changes AS (
                    SELECT 
                        ts, 
                        cell, 
                        state,
                        LEAD(state) OVER (PARTITION BY cell ORDER BY ts ASC) AS next_state,
                        LEAD(ts) OVER (PARTITION BY cell ORDER BY ts ASC) AS next_ts
                    FROM equipment e
                    WHERE e.cell = %s
                      AND e.ts >= %s::timestamp
                      AND e.ts <= %s::timestamp
                )
                SELECT 
                    state,
                    SUM(EXTRACT(EPOCH FROM (
                        LEAST(COALESCE(next_ts, %s::timestamp), %s::timestamp) - 
                        GREATEST(ts, %s::timestamp)
                    ))) AS duration_seconds
                FROM state_changes
                WHERE ts < %s::timestamp
                  AND (next_ts IS NULL OR next_ts > %s::timestamp)
                GROUP BY state;
            """
            
            logger.info(f"  Executing query with parameters: cell={cell}, start={start_ts}, end={end_ts}")
            cursor.execute(query, [cell, start_ts, end_ts, end_ts, end_ts, start_ts, end_ts, start_ts])
            results = cursor.fetchall()
            cursor.close()
            
            logger.info(f"  Query returned {len(results)} state groups")
            
            running_time_sec = 0.0
            downtime_sec = 0.0
            
            for state, duration in results:
                duration_float = float(duration) if duration else 0.0
                logger.info(f"  State '{state}': {duration_float:.1f}s")
                if state == 'running':
                    running_time_sec += duration_float
                elif state == 'down':
                    downtime_sec += duration_float
                # Other states (idle, maintenance, etc.) can be added if needed
            
            total_time = running_time_sec + downtime_sec
            logger.info(f"  ✅ Robot {cell} states summary:")
            logger.info(f"    Running: {running_time_sec:.1f}s")
            logger.info(f"    Down: {downtime_sec:.1f}s")
            logger.info(f"    Total: {total_time:.1f}s")
            if total_time > 0:
                availability = (running_time_sec / total_time) * 100
                logger.info(f"    Availability: {availability:.2f}%")
            
            return {
                'running_time_sec': running_time_sec,
                'downtime_sec': downtime_sec
            }
            
    except Exception as e:
        logger.error(f"  ❌ Error querying equipment states for {cell}: {e}", exc_info=True)
        import traceback
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return {'running_time_sec': 0.0, 'downtime_sec': 0.0}


def validate_job_count_consistency(
    print_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    pick_df: pd.DataFrame
) -> Dict:
    """
    Validate that print, cut, and pick have the same job counts.
    
    Args:
        print_df: Print job DataFrame
        cut_df: Cut job DataFrame
        pick_df: Pick job DataFrame
        
    Returns:
        Dict with validation results:
        {
            'is_valid': bool,
            'print_job_count': int,
            'cut_job_count': int,
            'pick_job_count': int,
            'missing_in_cut': List[str],
            'missing_in_pick': List[str],
            'messages': List[str]
        }
    """
    result = {
        'is_valid': True,
        'print_job_count': 0,
        'cut_job_count': 0,
        'pick_job_count': 0,
        'missing_in_cut': [],
        'missing_in_pick': [],
        'messages': []
    }
    
    try:
        # Extract unique job_ids from each dataframe
        print_jobs = set()
        cut_jobs = set()
        pick_jobs = set()
        
        if not print_df.empty and 'job_id' in print_df.columns:
            print_jobs = set(print_df['job_id'].dropna().unique())
        if not cut_df.empty and 'job_id' in cut_df.columns:
            cut_jobs = set(cut_df['job_id'].dropna().unique())
        if not pick_df.empty and 'job_id' in pick_df.columns:
            pick_jobs = set(pick_df['job_id'].dropna().unique())
        
        result['print_job_count'] = len(print_jobs)
        result['cut_job_count'] = len(cut_jobs)
        result['pick_job_count'] = len(pick_jobs)
        
        # Find mismatches
        missing_in_cut = print_jobs - cut_jobs
        missing_in_pick = print_jobs - pick_jobs
        
        result['missing_in_cut'] = sorted(list(missing_in_cut))
        result['missing_in_pick'] = sorted(list(missing_in_pick))
        
        # Check if valid
        if missing_in_cut or missing_in_pick:
            result['is_valid'] = False
            result['messages'].append("✗ Job count mismatch detected:")
            
            if missing_in_cut:
                result['messages'].append(f"  - {len(missing_in_cut)} job(s) in print but missing in cut: {', '.join(list(missing_in_cut)[:5])}")
            if missing_in_pick:
                result['messages'].append(f"  - {len(missing_in_pick)} job(s) in print but missing in pick: {', '.join(list(missing_in_pick)[:5])}")
        else:
            result['messages'].append("✓ All job counts match")
        
        logger.info(f"Job count validation: Print={len(print_jobs)}, Cut={len(cut_jobs)}, Pick={len(pick_jobs)}")
        if not result['is_valid']:
            logger.warning(f"Job count mismatch: {result['messages']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in validate_job_count_consistency: {e}", exc_info=True)
        result['is_valid'] = False
        result['messages'].append(f"Error during validation: {str(e)}")
        return result

def fetch_batches_for_day(day_start: str, day_end: str) -> pd.DataFrame:
    """
    Discover batches produced on a specific day with Print, Cut, and Pick job counts.

    Strategy:
    1. Query HISTORIAN for Print JobReports in time window
    2. Extract job IDs from JobReports and map to batch IDs via INSIDE database
    3. Query HISTORIAN for Cut JobReports in time window (extended to next day for Cut)
    4. Query HISTORIAN for Pick JobReports in time window (extended to next day for Pick)
    5. Match Cut/Pick job_ids to batch_ids via Print job mapping
    6. Group by batch_id with aggregated metadata for all three cells

    Args:
        day_start: ISO format timestamp (start of day)
        day_end: ISO format timestamp (end of day)

    Returns:
        DataFrame with columns:
        - batch_id: Batch identifier
        - print_jobs: Number of Print jobs in batch
        - cut_jobs: Number of Cut jobs in batch
        - pick_jobs: Number of Pick jobs in batch
        - print_start: Earliest Print job timestamp
        - print_end: Latest Print job timestamp
        - cut_start: Earliest Cut job start time (from events)
        - cut_end: Latest Cut job end time (from events)
        - pick_start: Earliest Pick job start time (from sheetStart_ts)
        - pick_end: Latest Pick job end time (from sheetEnd_ts)
        - first_job_time: Overall earliest job timestamp (for sorting)
        - last_job_time: Overall latest job timestamp (for sorting)

    Edge Cases:
    - No JobReports found: Returns empty DataFrame
    - Jobs without batch mapping: Excluded from results
    - Multiple batches in same day: All returned
    """
    try:
        # Extend time window for Cut and Pick (they may happen next day)
        day_end_extended = (dateutil_parser.parse(day_end) + timedelta(days=1)).isoformat()
        
        # Step 1: Query HISTORIAN for Print JobReports
        with get_historian_connection() as historian_conn:
            cursor = historian_conn.cursor()

            query = """
                SELECT
                    (CAST(payload AS jsonb))->>'jobId' as job_id,
                    ts as job_time
                FROM jobs
                WHERE topic = %s
                AND ts >= %s::timestamp
                AND ts < %s::timestamp
                AND (CAST(payload AS jsonb))->>'jobId' IS NOT NULL
                ORDER BY ts ASC
            """

            cursor.execute(query, (Config.TOPICS['print'], day_start, day_end))
            print_records = cursor.fetchall()
            
            # Step 2: Query HISTORIAN for Cut JobReports (extended window)
            # Extract first event (start) and last event (end) from events array
            query_cut = """
                SELECT
                    COALESCE(
                        (CAST(payload AS jsonb))->>'jobId',
                        (CAST(payload AS jsonb)->'data'->'ids'->>'jobId')
                    ) as job_id,
                    -- First event timestamp (start)
                    (CAST(payload AS jsonb)->'data'->'events'->0->>'ts') as job_start,
                    -- Last event timestamp (end) - use array length to get last index
                    (
                        CAST(payload AS jsonb)->'data'->'events'->(jsonb_array_length((CAST(payload AS jsonb)->'data'->'events')) - 1)->>'ts'
                    ) as job_end,
                    ts as report_time
                FROM jobs
                WHERE topic = %s
                AND ts >= %s::timestamp
                AND ts < %s::timestamp
                AND (CAST(payload AS jsonb))->>'state' = 'completed'
                AND (
                    (CAST(payload AS jsonb))->>'jobId' IS NOT NULL
                    OR (CAST(payload AS jsonb)->'data'->'ids'->>'jobId') IS NOT NULL
                )
                ORDER BY ts ASC
            """
            
            cursor.execute(query_cut, (Config.TOPICS['cut'], day_start, day_end_extended))
            cut_records = cursor.fetchall()
            
            # Step 3: Query HISTORIAN for Pick JobReports (extended window)
            query_pick = """
                SELECT
                    (CAST(payload AS jsonb))->>'jobId' as job_id,
                    (CAST(payload AS jsonb)->'data'->'insideSummary'->>'sheetStart_ts') as job_start,
                    (CAST(payload AS jsonb)->'data'->'insideSummary'->>'sheetEnd_ts') as job_end,
                    ts as report_time
                FROM jobs
                WHERE topic = %s
                AND ts >= %s::timestamp
                AND ts < %s::timestamp
                AND (CAST(payload AS jsonb))->>'jobId' IS NOT NULL
                ORDER BY ts ASC
            """
            
            cursor.execute(query_pick, (Config.TOPICS['pick'], day_start, day_end_extended))
            pick_records = cursor.fetchall()
            
            cursor.close()

        if not print_records:
            logger.debug(f"No Print JobReports found for {day_start} to {day_end}")
            return pd.DataFrame(columns=[
                'batch_id', 'print_jobs', 'cut_jobs', 'pick_jobs',
                'print_start', 'print_end', 'cut_start', 'cut_end', 'pick_start', 'pick_end',
                'first_job_time', 'last_job_time'
            ])

        # Extract Print job IDs and timestamps
        print_job_ids = [row[0] for row in print_records]
        print_job_times = {row[0]: row[1] for row in print_records}

        logger.info(f"Found {len(print_job_ids)} Print JobReports, {len(cut_records)} Cut JobReports, {len(pick_records)} Pick JobReports")

        # Step 4: Map Print job IDs to batch IDs via INSIDE database
        with get_inside_connection() as inside_conn:
            cursor = inside_conn.cursor()

            placeholders = ','.join(['%s'] * len(print_job_ids))
            query = f"""
                SELECT
                    rg_id as job_id,
                    production_batch_id as batch_id
                FROM production_printjob
                WHERE rg_id IN ({placeholders})
                AND production_batch_id IS NOT NULL
            """

            cursor.execute(query, print_job_ids)
            mapping_records = cursor.fetchall()
            cursor.close()

        if not mapping_records:
            logger.warning("No batch mappings found for job IDs")
            return pd.DataFrame(columns=[
                'batch_id', 'print_jobs', 'cut_jobs', 'pick_jobs',
                'print_start', 'print_end', 'cut_start', 'cut_end', 'pick_start', 'pick_end',
                'first_job_time', 'last_job_time'
            ])

        # Create job_id -> batch_id mapping
        job_to_batch = {job_id: batch_id for job_id, batch_id in mapping_records}

        # Step 5: Build batch metadata for Print
        batch_metadata = {}
        for job_id, batch_id in mapping_records:
            if batch_id not in batch_metadata:
                batch_metadata[batch_id] = {
                    'print_job_ids': [],
                    'print_times': [],
                    'cut_job_ids': [],
                    'cut_starts': [],
                    'cut_ends': [],
                    'pick_job_ids': [],
                    'pick_starts': [],
                    'pick_ends': []
                }
            batch_metadata[batch_id]['print_job_ids'].append(job_id)
            if job_id in print_job_times:
                batch_metadata[batch_id]['print_times'].append(print_job_times[job_id])

        # Step 6: Match Cut jobs to batches via job_id
        for row in cut_records:
            job_id = row[0]
            job_start = row[1]
            job_end = row[2]
            if job_id in job_to_batch:
                batch_id = job_to_batch[job_id]
                if batch_id in batch_metadata:
                    batch_metadata[batch_id]['cut_job_ids'].append(job_id)
                    if job_start:
                        batch_metadata[batch_id]['cut_starts'].append(job_start)
                    if job_end:
                        batch_metadata[batch_id]['cut_ends'].append(job_end)

        # Step 7: Match Pick jobs to batches via job_id
        for row in pick_records:
            job_id = row[0]
            job_start = row[1]
            job_end = row[2]
            if job_id in job_to_batch:
                batch_id = job_to_batch[job_id]
                if batch_id in batch_metadata:
                    batch_metadata[batch_id]['pick_job_ids'].append(job_id)
                    if job_start:
                        batch_metadata[batch_id]['pick_starts'].append(job_start)
                    if job_end:
                        batch_metadata[batch_id]['pick_ends'].append(job_end)

        # Helper function to convert timestamp strings to datetime with timezone
        def parse_timestamp(ts):
            """Convert timestamp string or datetime to timezone-aware datetime in UTC+1 (Europe/Copenhagen)."""
            if ts is None:
                return None
            if isinstance(ts, pd.Timestamp):
                dt = ts.to_pydatetime()
            elif isinstance(ts, str):
                try:
                    dt = dateutil_parser.parse(ts)
                except (ValueError, TypeError):
                    return None
            else:
                dt = ts
            
            # Normalize to UTC+1 (Europe/Copenhagen) and keep timezone info
            copenhagen_tz = gettz('Europe/Copenhagen')
            if dt.tzinfo is not None:
                # Convert timezone-aware datetime to UTC+1
                dt = dt.astimezone(copenhagen_tz)
            else:
                # If naive, assume it's UTC and convert to UTC+1
                # Database timestamps are typically stored in UTC, but we want to display in local time
                dt_utc = dt.replace(tzinfo=UTC)
                dt = dt_utc.astimezone(copenhagen_tz)
            
            return dt

        # Step 8: Aggregate into DataFrame
        batch_rows = []
        for batch_id, metadata in batch_metadata.items():
            # Convert all timestamps to datetime objects for comparison
            print_times_dt = [parse_timestamp(t) for t in metadata['print_times'] if t is not None]
            cut_starts_dt = [parse_timestamp(t) for t in metadata['cut_starts'] if t is not None]
            cut_ends_dt = [parse_timestamp(t) for t in metadata['cut_ends'] if t is not None]
            pick_starts_dt = [parse_timestamp(t) for t in metadata['pick_starts'] if t is not None]
            pick_ends_dt = [parse_timestamp(t) for t in metadata['pick_ends'] if t is not None]
            
            # Collect all timestamps for overall first/last
            all_times = []
            all_times.extend(print_times_dt)
            all_times.extend(cut_starts_dt)
            all_times.extend(cut_ends_dt)
            all_times.extend(pick_starts_dt)
            all_times.extend(pick_ends_dt)
            # Filter out None values
            all_times = [t for t in all_times if t is not None]
            
            batch_rows.append({
                'batch_id': batch_id,
                'print_jobs': len(metadata['print_job_ids']),
                'cut_jobs': len(metadata['cut_job_ids']),
                'pick_jobs': len(metadata['pick_job_ids']),
                'print_start': min(print_times_dt) if print_times_dt else None,
                'print_end': max(print_times_dt) if print_times_dt else None,
                'cut_start': min(cut_starts_dt) if cut_starts_dt else None,
                'cut_end': max(cut_ends_dt) if cut_ends_dt else None,
                'pick_start': min(pick_starts_dt) if pick_starts_dt else None,
                'pick_end': max(pick_ends_dt) if pick_ends_dt else None,
                'first_job_time': min(all_times) if all_times else None,
                'last_job_time': max(all_times) if all_times else None
            })

        batches_df = pd.DataFrame(batch_rows)
        batches_df = batches_df.sort_values('first_job_time')

        logger.info(f"Discovered {len(batches_df)} batches with Print/Cut/Pick counts")
        return batches_df

    except Exception as e:
        logger.error(f"Error in fetch_batches_for_day: {e}", exc_info=True)
        return pd.DataFrame(columns=[
            'batch_id', 'print_jobs', 'cut_jobs', 'pick_jobs',
            'print_start', 'print_end', 'cut_start', 'cut_end', 'pick_start', 'pick_end',
            'first_job_time', 'last_job_time'
        ])

def fetch_job_ids_for_batch(batch_ids: List[str]) -> List[str]:
    """
    Map batch IDs to job IDs via INSIDE database.

    Args:
        batch_ids: List of batch identifiers

    Returns:
        List of job IDs (rg_ids) associated with the batches

    Edge Cases:
    - Empty batch_ids list: Returns empty list
    - Batch not found: Excluded from results
    - Multiple jobs per batch: All returned
    """
    if not batch_ids:
        return []

    try:
        with get_inside_connection() as inside_conn:
            cursor = inside_conn.cursor()

            placeholders = ','.join(['%s'] * len(batch_ids))
            query = f"""
                SELECT DISTINCT rg_id
                FROM production_printjob
                WHERE production_batch_id IN ({placeholders})
                AND rg_id IS NOT NULL
                ORDER BY rg_id
            """

            cursor.execute(query, batch_ids)
            job_records = cursor.fetchall()
            cursor.close()

        job_ids = [row[0] for row in job_records]
        logger.info(f"Mapped {len(batch_ids)} batches to {len(job_ids)} job IDs")

        return job_ids

    except Exception as e:
        logger.error(f"Error in fetch_job_ids_for_batch: {e}", exc_info=True)
        return []

# ============================================================================
# JOBREPORT FETCHERS
# ============================================================================

def fetch_print_jobreports(job_ids: List[str]) -> pd.DataFrame:
    """
    Fetch Print1 JobReports from HISTORIAN.

    Topic: rg_v2/RG/CPH/Prod/ComponentLine/Print1/JobReport

    Extracts:
    - job_id, batch_id, sheet_index
    - uptime_sec, downtime_sec (from reportData)
    - speed_actual, speed_nominal (from reportData)
    - area_sqm (from closingReport)
    - job_start, job_end timestamps

    Calculates:
    - total_time_sec = uptime + downtime
    - availability = uptime / total_time × 100%
    - performance = actual_speed / nominal_speed × 100%
    - quality = 100% (initial assumption, QC overlay applied later)
    - oee = availability × performance × quality / 10000

    Returns:
        DataFrame with Print job metrics

    Edge Cases:
    - Empty job_ids: Returns empty DataFrame
    - Missing JSON fields: Returns NULL for that field
    - Division by zero: Returns 0% for that metric
    - No records found: Returns empty DataFrame
    """
    if not job_ids:
        return pd.DataFrame()

    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()

            placeholders = ','.join(['%s'] * len(job_ids))
            query = f"""
                SELECT
                    CAST(payload AS jsonb)->>'jobId' as job_id,
                    CAST(payload AS jsonb)->'data'->'ids'->>'batchId' as batch_id,
                    CAST((CAST(payload AS jsonb)->'data'->'ids'->>'sheetIndex') AS FLOAT) as sheet_index,
                    CAST(payload AS jsonb)->'data'->'reportData'->'timing'->>'start' as job_start,
                    CAST(payload AS jsonb)->'data'->'reportData'->'timing'->>'end' as job_end,
                    CAST((CAST(payload AS jsonb)->'data'->'time'->>'uptime') AS FLOAT) as uptime_sec,
                    CAST((CAST(payload AS jsonb)->'data'->'time'->>'downtime') AS FLOAT) as downtime_sec,
                    CAST((CAST(payload AS jsonb)->'data'->'reportData'->'printSettings'->>'speed') AS FLOAT) as speed_actual,
                    CAST((CAST(payload AS jsonb)->'data'->'reportData'->'printSettings'->>'nominalSpeed') AS FLOAT) as speed_nominal,
                    CAST((CAST(payload AS jsonb)->'data'->'reportData'->'jobInfo'->>'area') AS FLOAT) as area_sqm
                FROM jobs
                WHERE topic = %s
                AND CAST(payload AS jsonb)->>'jobId' IN ({placeholders})
                ORDER BY ts ASC
            """

            cursor.execute(query, [Config.TOPICS['print']] + job_ids)
            records = cursor.fetchall()
            cursor.close()

        if not records:
            logger.warning(f"No Print JobReports found for {len(job_ids)} jobs")
            return pd.DataFrame()

        # Convert to DataFrame
        columns = [
            'job_id', 'batch_id', 'sheet_index', 'job_start', 'job_end',
            'uptime_sec', 'downtime_sec', 'speed_actual', 'speed_nominal', 'area_sqm'
        ]
        df = pd.DataFrame(records, columns=columns)

        # Handle NULL values
        df['uptime_sec'] = df['uptime_sec'].fillna(0)
        df['downtime_sec'] = df['downtime_sec'].fillna(0)
        df['speed_actual'] = df['speed_actual'].fillna(0)
        df['speed_nominal'] = df['speed_nominal'].fillna(0)
        df['area_sqm'] = df['area_sqm'].fillna(0)

        # Calculate derived metrics
        df['total_time_sec'] = df['uptime_sec'] + df['downtime_sec']

        # Availability (handle division by zero)
        df['availability'] = np.where(
            df['total_time_sec'] > 0,
            (df['uptime_sec'] / df['total_time_sec']) * 100,
            0.0
        )

        # Performance (handle division by zero)
        df['performance'] = np.where(
            df['speed_nominal'] > 0,
            (df['speed_actual'] / df['speed_nominal']) * 100,
            0.0
        )

        # Quality (initial 100% assumption)
        df['quality'] = 100.0

        # OEE
        df['oee'] = (df['availability'] * df['performance'] * df['quality']) / 10000

        logger.info(f"Fetched {len(df)} Print JobReports")
        return df

    except Exception as e:
        logger.error(f"Error in fetch_print_jobreports: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_cut_jobreports(job_ids: List[str], print_df: pd.DataFrame = None, time_window: tuple = None) -> pd.DataFrame:
    # Updated: Now accepts print_df parameter for batch_id/sheet_index join
    """
    Fetch Cut1 JobReports from HISTORIAN.

    Topic: rg_v2/RG/CPH/Prod/ComponentLine/Cut1/JobReport

    Cut JobReport structure:
    - jobId in data.ids.jobId
    - uptime/downtime in data.time
    - componentCount in data.componentCount
    - events array with timestamps (first = start, last = end)
    - NO batchId, sheetIndex, or speed data in Cut payload

    Extracts:
    - job_id (from data.ids.jobId)
    - uptime_sec, downtime_sec (from data.time)
    - component_count (from data.componentCount)
    - job_start, job_end (from data.events array)
    - ts (database timestamp for deduplication)

    Note: batch_id and sheet_index come from Print data via join

    Args:
        job_ids: List of job IDs to fetch Cut reports for
        print_df: Print DataFrame for batch_id/sheet_index join
        time_window: Optional tuple of (start_time, end_time) to filter Cut reports by date

    Returns:
        DataFrame with Cut job metrics (1:1 with Print jobs after deduplication)

    Edge Cases:
    - Multiple Cut reports per job_id: Filters by time window, then takes latest by timestamp
    - Missing Print data: batch_id/sheet_index will be NULL
    - Job IDs reused across days: Time window filter ensures we get the correct day's report
    """
    if not job_ids:
        return pd.DataFrame()

    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()

            placeholders = ','.join(['%s'] * len(job_ids))
            
            # Don't filter by time window - Cut might happen on different day than Print
            # We'll match by job_id + temporal proximity to Print jobs instead
            time_filter = ""
            query_params = [Config.TOPICS['cut']] + list(job_ids) + list(job_ids)
            
            # Extract from Cut payload structure
            # jobId is in data.ids.jobId (also check top level as fallback)
            query = f"""
                SELECT
                    COALESCE(
                        CAST(payload AS jsonb)->>'jobId',
                        CAST(payload AS jsonb)->'data'->'ids'->>'jobId'
                    ) as job_id,
                    CAST((CAST(payload AS jsonb)->'data'->'time'->>'uptime') AS FLOAT) as uptime_sec,
                    CAST((CAST(payload AS jsonb)->'data'->'time'->>'downtime') AS FLOAT) as downtime_sec,
                    CAST((CAST(payload AS jsonb)->'data'->>'componentCount') AS INT) as component_count,
                    -- Extract events array as JSON for processing in Python
                    CAST(payload AS jsonb)->'data'->'events' as events_array,
                    CAST(payload AS jsonb)->>'state' as job_state,
                    ts as report_timestamp
                FROM jobs
                WHERE topic = %s
                AND (
                    CAST(payload AS jsonb)->>'jobId' IN ({placeholders})
                    OR CAST(payload AS jsonb)->'data'->'ids'->>'jobId' IN ({placeholders})
                )
                AND CAST(payload AS jsonb)->>'state' = 'completed'
                {time_filter}
                ORDER BY ts DESC
            """

            # Execute query with parameters
            cursor.execute(query, query_params)
            records = cursor.fetchall()
            cursor.close()

        if not records:
            logger.warning(f"No Cut JobReports found for {len(job_ids)} jobs")
            return pd.DataFrame()

        columns = [
            'job_id', 'uptime_sec', 'downtime_sec', 'component_count',
            'events_array', 'job_state', 'report_timestamp'
        ]
        df = pd.DataFrame(records, columns=columns)
        
        # Extract job_start and job_end from events array
        def extract_timestamps(events_json):
            """Extract first and last event timestamps from events array"""
            if events_json is None:
                return None, None
            try:
                import json
                if isinstance(events_json, str):
                    events = json.loads(events_json)
                else:
                    events = events_json
                if isinstance(events, list) and len(events) > 0:
                    first_ts = events[0].get('ts') if events[0] else None
                    last_ts = events[-1].get('ts') if events[-1] else None
                    return first_ts, last_ts
                return None, None
            except Exception as e:
                logger.warning(f"Error parsing events array: {e}")
                return None, None
        
        # Apply extraction
        df[['job_start', 'job_end']] = df['events_array'].apply(
            lambda x: pd.Series(extract_timestamps(x))
        )
        df = df.drop(columns=['events_array'], errors='ignore')

        # Debug: Log all records for specific job before deduplication
        if 'S10-0000008031' in df['job_id'].values:
            job_8031_before = df[df['job_id'] == 'S10-0000008031']
            logger.warning(f"Job S10-0000008031 BEFORE deduplication: {len(job_8031_before)} records")
            for idx, row in job_8031_before.iterrows():
                logger.warning(f"  Record {idx}: uptime={row['uptime_sec']}, downtime={row['downtime_sec']}, state={row.get('job_state', 'N/A')}, timestamp={row['report_timestamp']}")
        
        # Match Cut reports to Print jobs by job_id and temporal proximity
        # (Print might be on one day, Cut on another - match by job_id + closest time to Print job)
        if print_df is not None and not print_df.empty and 'job_id' in print_df.columns:
            # Get Print job end timestamps for temporal matching (Cut happens after Print)
            print_times = None
            if 'job_end' in print_df.columns:
                print_times = print_df.set_index('job_id')['job_end']
            elif 'job_start' in print_df.columns:
                print_times = print_df.set_index('job_id')['job_start']
            
            # If we have multiple Cut reports per job_id, match to Print by temporal proximity
            if len(df) > len(df['job_id'].unique()):
                logger.warning(f"Found {len(df)} completed Cut reports for {len(df['job_id'].unique())} unique jobs - matching to Print jobs by temporal proximity")
                
                matched_cut_reports = []
                for job_id in df['job_id'].unique():
                    cut_reports_for_job = df[df['job_id'] == job_id].copy()
                    
                    if print_times is not None and job_id in print_times.index:
                        # Find Cut report closest to Print job time
                        # Prefer Cut reports that happen after Print (within reasonable window, e.g., 7 days)
                        print_time = pd.to_datetime(print_times[job_id])
                        cut_times = pd.to_datetime(cut_reports_for_job['report_timestamp'])
                        
                        # Calculate time differences
                        cut_reports_for_job['time_diff'] = (cut_times - print_time).abs()
                        cut_reports_for_job['is_after_print'] = cut_times > print_time
                        cut_reports_for_job['days_after_print'] = (cut_times - print_time).dt.total_seconds() / 86400
                        
                        # Scoring system for temporal matching (Cut typically happens same day or next day):
                        # 1) After Print (required for logical flow)
                        # 2) Same day (highest priority - within 24 hours)
                        # 3) Next day (high priority - within 2 days)
                        # 4) Beyond 2 days (very low priority - only if no better option)
                        # 5) Closest time (tie-breaker)
                        cut_reports_for_job['score'] = (
                            cut_reports_for_job['is_after_print'].astype(int) * 10000 +  # Must be after Print
                            (cut_reports_for_job['days_after_print'] <= 1).astype(int) * 1000 +  # Same day (within 24 hours)
                            (cut_reports_for_job['days_after_print'] <= 2).astype(int) * 500 +  # Next day (within 2 days)
                            (cut_reports_for_job['days_after_print'] > 2).astype(int) * (-1000) +  # Penalize beyond 2 days
                            (1 / (cut_reports_for_job['time_diff'].dt.total_seconds() / 3600 + 1)) * 10  # Closer is better (inverse of hours)
                        )
                        
                        selected = cut_reports_for_job.sort_values('score', ascending=False).iloc[0]
                        matched_cut_reports.append(selected)
                        logger.info(f"Job {job_id}: Print at {print_time}, Selected Cut at {selected['report_timestamp']} ({selected['days_after_print']:.1f} days after, score={selected['score']:.2f})")
                        
                        # Debug for specific job
                        if job_id == 'S10-0000008031':
                            logger.warning(f"  Job S10-0000008031 matching details:")
                            for idx, row in cut_reports_for_job.iterrows():
                                logger.warning(f"    Cut report {idx}: timestamp={row['report_timestamp']}, uptime={row['uptime_sec']}, downtime={row['downtime_sec']}, score={row['score']:.2f}, days_after={row['days_after_print']:.1f}")
                    else:
                        # No Print time available, just use latest
                        selected = cut_reports_for_job.sort_values('report_timestamp', ascending=False).iloc[0]
                        matched_cut_reports.append(selected)
                        logger.warning(f"Job {job_id}: No Print time found, using latest Cut report")
                
                df = pd.DataFrame(matched_cut_reports).reset_index(drop=True)
                # Drop temporary columns
                df = df.drop(columns=['time_diff', 'is_after_print', 'days_after_print', 'score'], errors='ignore')
                logger.info(f"After temporal matching: {len(df)} Cut reports (1:1 with Print)")
            else:
                logger.info(f"Found {len(df)} Cut reports (1:1 with Print jobs)")
        else:
            # No Print data, just deduplicate by latest timestamp
            if len(df) > len(df['job_id'].unique()):
                logger.warning(f"Found {len(df)} completed Cut reports for {len(df['job_id'].unique())} unique jobs - deduplicating")
                df = df.sort_values('report_timestamp', ascending=False).drop_duplicates(subset=['job_id'], keep='first')
                logger.info(f"After deduplication: {len(df)} Cut reports")
        
        # Debug: Check for specific job after deduplication
        if 'S10-0000008031' in df['job_id'].values:
            job_8031_after = df[df['job_id'] == 'S10-0000008031']
            logger.warning(f"Job S10-0000008031 AFTER deduplication: {len(job_8031_after)} records")
            if not job_8031_after.empty:
                row = job_8031_after.iloc[0]
                logger.warning(f"  Final: uptime={row['uptime_sec']}, downtime={row['downtime_sec']}, state={row.get('job_state', 'N/A')}, timestamp={row['report_timestamp']}")
                # Expected values from CSV: uptime=68.066, downtime=0
                if abs(row['uptime_sec'] - 68.066) > 1 or row['downtime_sec'] != 0:
                    logger.error(f"  ⚠️ MISMATCH! Expected uptime=68.066, downtime=0, but got uptime={row['uptime_sec']}, downtime={row['downtime_sec']}")

        # Debug: Log raw values before processing
        if not df.empty:
            sample_job = df.iloc[0]['job_id']
            sample_row = df[df['job_id'] == sample_job].iloc[0]
            logger.info(f"Sample Cut job {sample_job} - Raw uptime: {sample_row['uptime_sec']}, downtime: {sample_row['downtime_sec']}")
        
        # Handle NULL values
        df['uptime_sec'] = df['uptime_sec'].fillna(0)
        df['downtime_sec'] = df['downtime_sec'].fillna(0)
        df['component_count'] = df['component_count'].fillna(0)
        
        # Debug: Log after fillna
        if not df.empty:
            sample_row = df[df['job_id'] == sample_job].iloc[0]
            logger.info(f"Sample Cut job {sample_job} - After fillna: uptime: {sample_row['uptime_sec']}, downtime: {sample_row['downtime_sec']}")

        # Join with Print data to get batch_id and sheet_index
        if print_df is not None and not print_df.empty and 'job_id' in print_df.columns:
            # Select only the columns we need from Print
            print_lookup = print_df[['job_id', 'batch_id', 'sheet_index']].copy()
            
            # Debug: Check for duplicate job_ids in Print data
            print_duplicates = print_lookup[print_lookup.duplicated(subset=['job_id'], keep=False)]
            if not print_duplicates.empty:
                logger.warning(f"Found {len(print_duplicates)} duplicate job_ids in Print data - this may cause join issues")
            
            # Debug: Check job_id overlap
            cut_job_ids = set(df['job_id'].unique())
            print_job_ids = set(print_lookup['job_id'].unique())
            overlap = cut_job_ids & print_job_ids
            logger.info(f"Cut job_ids: {len(cut_job_ids)}, Print job_ids: {len(print_job_ids)}, Overlap: {len(overlap)}")
            if len(overlap) < len(cut_job_ids):
                missing = cut_job_ids - print_job_ids
                logger.warning(f"Cut job_ids not found in Print: {list(missing)[:5]}... (showing first 5)")
            
            # Merge to get batch_id and sheet_index
            df = df.merge(print_lookup, on='job_id', how='left')
            
            # Debug: Check batch_id distribution after join
            batch_counts = df['batch_id'].value_counts()
            logger.info(f"Batch distribution after join: {dict(batch_counts.head(5))}")
            logger.info(f"Joined Cut data with Print data: {df['batch_id'].notna().sum()}/{len(df)} jobs have batch_id")
        else:
            # If no Print data, add empty columns
            df['batch_id'] = None
            df['sheet_index'] = None
            logger.warning("No Print data provided for Cut join - batch_id and sheet_index will be NULL")

        # Calculate derived metrics
        df['total_time_sec'] = df['uptime_sec'] + df['downtime_sec']
        df['availability'] = np.where(
            df['total_time_sec'] > 0,
            (df['uptime_sec'] / df['total_time_sec']) * 100,
            0.0
        )
        # Performance is always 100% for Cut (no speed tracking)
        df['performance'] = 100.0
        df['quality'] = 100.0
        df['oee'] = (df['availability'] * df['performance'] * df['quality']) / 10000

        # Drop temporary column
        df = df.drop(columns=['report_timestamp'], errors='ignore')

        logger.info(f"Fetched {len(df)} Cut JobReports (deduplicated, 1:1 with Print)")
        return df

    except Exception as e:
        logger.error(f"Error in fetch_cut_jobreports: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_pick_jobreports(job_ids: List[str]) -> pd.DataFrame:
    """
    Fetch Pick1 JobReports from HISTORIAN.

    Topic: rg_v2/RG/CPH/Prod/ComponentLine/Pick1/JobReport

    KEY LOGIC:
    - Performance = 100% (robots operate at consistent speed)
    - Quality = successful picks / total attempts

    Extracts:
    - job_id, batch_id, sheet_index
    - uptime_sec, downtime_sec
    - components_completed, components_failed (from closingReport)
    - average_pick_time_sec (from closingReport)
    - job_start, job_end timestamps

    Calculates:
    - availability = uptime / total_time × 100%
    - performance = 100% (FIXED - robots don't vary speed)
    - quality = completed / (completed + failed) × 100%
    - oee = availability × performance × quality / 10000

    Returns:
        DataFrame with Pick job metrics

    Edge Cases:
    - No components picked: quality = 100% (benefit of doubt)
    - All components failed: quality = 0%
    - Missing failed count: Assumes 0 failures
    """
    if not job_ids:
        return pd.DataFrame()

    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()

            placeholders = ','.join(['%s'] * len(job_ids))
            query = f"""
                SELECT
                    CAST(payload AS jsonb)->>'jobId' as job_id,
                    CAST(payload AS jsonb)->'data'->'ids'->>'batchId' as batch_id,
                    CAST((CAST(payload AS jsonb)->'data'->'ids'->>'sheetIndex') AS FLOAT) as sheet_index,
                    CAST(payload AS jsonb)->'data'->'reportData'->'closingReport'->'jobActiveWindow'->>'startTs' as job_start,
                    CAST(payload AS jsonb)->'data'->'reportData'->'closingReport'->'jobActiveWindow'->>'endTs' as job_end,
                    CAST((CAST(payload AS jsonb)->'data'->'time'->>'uptime') AS FLOAT) as uptime_sec,
                    CAST((CAST(payload AS jsonb)->'data'->'time'->>'downtime') AS FLOAT) as downtime_sec,
                    CAST((CAST(payload AS jsonb)->'data'->'reportData'->'metrics'->>'successfulPicks') AS INT) as components_completed,
                    CAST((CAST(payload AS jsonb)->'data'->'reportData'->'metrics'->>'totalPicks') AS INT) as total_picks,
                    CAST(((CAST(payload AS jsonb)->'data'->'reportData'->'metrics'->>'totalPicks')::numeric - 
                         (CAST(payload AS jsonb)->'data'->'reportData'->'metrics'->>'successfulPicks')::numeric) AS INT) as components_failed
                FROM jobs
                WHERE topic = %s
                AND CAST(payload AS jsonb)->>'jobId' IN ({placeholders})
                ORDER BY ts ASC
            """

            cursor.execute(query, [Config.TOPICS['pick']] + job_ids)
            records = cursor.fetchall()
            cursor.close()

        if not records:
            logger.warning(f"No Pick JobReports found for {len(job_ids)} jobs")
            return pd.DataFrame()

        columns = [
            'job_id', 'batch_id', 'sheet_index', 'job_start', 'job_end',
            'uptime_sec', 'downtime_sec', 'components_completed', 'total_picks', 'components_failed'
        ]
        df = pd.DataFrame(records, columns=columns)

        # Handle NULL values
        df['uptime_sec'] = df['uptime_sec'].fillna(0)
        df['downtime_sec'] = df['downtime_sec'].fillna(0)
        df['components_completed'] = df['components_completed'].fillna(0)
        df['total_picks'] = df['total_picks'].fillna(0)
        df['components_failed'] = df['components_failed'].fillna(0)
        
        # Calculate components_failed if not provided (total_picks - components_completed)
        df['components_failed'] = df['total_picks'] - df['components_completed']

        # Calculate derived metrics
        df['total_time_sec'] = df['uptime_sec'] + df['downtime_sec']
        df['availability'] = np.where(
            df['total_time_sec'] > 0,
            (df['uptime_sec'] / df['total_time_sec']) * 100,
            0.0
        )

        # CORRECTED: Pick performance is always 100%
        df['performance'] = 100.0

        # CORRECTED: Quality = successful picks / total attempts
        df['total_attempts'] = df['components_completed'] + df['components_failed']
        df['quality'] = np.where(
            df['total_attempts'] > 0,
            (df['components_completed'] / df['total_attempts']) * 100,
            100.0  # If no attempts, assume 100% (benefit of doubt)
        )

        # OEE
        df['oee'] = (df['availability'] * df['performance'] * df['quality']) / 10000

        logger.info(f"Fetched {len(df)} Pick JobReports")
        return df

    except Exception as e:
        logger.error(f"Error in fetch_pick_jobreports: {e}", exc_info=True)
        return pd.DataFrame()

# ============================================================================
# QUALITY / QC DATA FETCHERS
# ============================================================================

def fetch_qc_data_for_batches(batch_ids: List[str]) -> Optional[pd.DataFrame]:
    """
    Fetch QC inspection data for batches from INSIDE database.

    Queries production_componentorderqc table to get inspection results.

    Args:
        batch_ids: List of batch identifiers

    Returns:
        DataFrame with columns:
        - batch_id: Batch identifier
        - total_inspected: Total components inspected
        - passed: Number of components that passed
        - failed: Number of components that failed
        - pass_rate: Percentage of components that passed (0-100)

    Returns None if no QC data available (Day 0 scenario).

    Edge Cases:
    - No QC data: Returns None (not empty DataFrame)
    - Partial QC data: Returns only batches with QC results
    - Multiple inspections per batch: Aggregated into single pass rate
    """
    if not batch_ids:
        return None

    try:
        with get_inside_connection() as conn:
            cursor = conn.cursor()

            # Convert batch IDs to integers (strip 'B' prefix if present)
            validated_batches = []
            for bid in batch_ids:
                try:
                    clean_id = str(bid).replace('B', '').replace('b', '').strip()
                    validated_batches.append(int(clean_id))
                except (ValueError, AttributeError, TypeError):
                    logger.warning(f"Invalid batch ID format: {bid}")
                    continue
            
            if not validated_batches:
                logger.warning("No valid batch IDs for QC query")
                return None
            
            query = """
                SELECT
                    pc.production_batch_id as batch_id,
                    COUNT(DISTINCT pc.id) as total_inspected,
                    COUNT(DISTINCT CASE 
                        WHEN EXISTS (
                            SELECT 1 FROM production_componentorderevent pce 
                            WHERE pce.component_order_id = pc.id 
                            AND pce.event_type = 'qc' 
                            AND pce.state = 'passed'
                        ) THEN pc.id 
                    END) as passed,
                    COUNT(DISTINCT CASE 
                        WHEN EXISTS (
                            SELECT 1 FROM production_componentorderevent pce 
                            WHERE pce.component_order_id = pc.id 
                            AND pce.event_type = 'qc' 
                            AND pce.state = 'failed'
                        ) THEN pc.id 
                    END) as failed,
                    CASE
                        WHEN COUNT(DISTINCT CASE 
                            WHEN EXISTS (
                                SELECT 1 FROM production_componentorderevent pce 
                                WHERE pce.component_order_id = pc.id 
                                AND pce.event_type = 'qc' 
                                AND pce.state IN ('passed', 'failed')
                            ) THEN pc.id 
                        END) > 0
                        THEN
                            (COUNT(DISTINCT CASE 
                                WHEN EXISTS (
                                    SELECT 1 FROM production_componentorderevent pce 
                                    WHERE pce.component_order_id = pc.id 
                                    AND pce.event_type = 'qc' 
                                    AND pce.state = 'passed'
                                ) THEN pc.id 
                            END) * 100.0) /
                            COUNT(DISTINCT CASE 
                                WHEN EXISTS (
                                    SELECT 1 FROM production_componentorderevent pce 
                                    WHERE pce.component_order_id = pc.id 
                                    AND pce.event_type = 'qc' 
                                    AND pce.state IN ('passed', 'failed')
                                ) THEN pc.id 
                            END)
                        ELSE NULL
                    END as pass_rate
                FROM production_componentorder pc
                JOIN production_printjob pj ON pc.print_job_id = pj.id
                WHERE pc.production_batch_id = ANY(%s::bigint[])
                AND pj.status != 'cancelled'
                AND EXISTS (
                    SELECT 1 FROM production_componentorderevent pce 
                    WHERE pce.component_order_id = pc.id 
                    AND pce.event_type = 'qc'
                )
                GROUP BY pc.production_batch_id
            """

            cursor.execute(query, [validated_batches])
            results = cursor.fetchall()
            cursor.close()

        if not results:
            logger.info(f"No QC data found for {len(batch_ids)} batches (Day 0 scenario)")
            return None

        df = pd.DataFrame(results, columns=['batch_id', 'total_inspected', 'passed', 'failed', 'pass_rate'])
        logger.info(f"Fetched QC data for {len(df)} batches")

        return df

    except Exception as e:
        logger.error(f"Error in fetch_qc_data_for_batches: {e}", exc_info=True)
        return None

def fetch_qc_data(batch_ids: List[str]) -> pd.DataFrame:
    """
    Fetch component-level QC inspection data for batches from INSIDE database.
    
    Similar to fetch_print_jobreports/fetch_cut_jobreports/fetch_pick_jobreports,
    this returns one row per component with all QC details.
    
    Queries production_componentorderqc table to get inspection results.
    
    Args:
        batch_ids: List of batch identifiers
        
    Returns:
        DataFrame with columns:
        - component_id: Component order ID
        - batch_id: Production batch ID
        - job_id: Print job rg_id (for matching with Print/Cut/Pick data)
        - production_date: Date component was created
        - qc_state: QC state ('passed', 'failed', or 'not_scanned')
        - qc_source: QC source ('handheld-scan', 'handheld-manual', 'inside', or NULL)
        - qc_reasons: Array of defect reasons (e.g., ['print', 'cut-outside-bleed'])
        - qc_timestamp: When QC inspection occurred (if available)
        
    Returns empty DataFrame if no QC data available.
    
    Edge Cases:
    - No QC data: Returns empty DataFrame
    - Components without QC: qc_state = 'not_scanned', qc_source = NULL, qc_reasons = []
    - Multiple QC records per component: Returns latest (by timestamp if available)
    """
    if not batch_ids:
        return pd.DataFrame()
    
    try:
        with get_inside_connection() as conn:
            cursor = conn.cursor()
            
            # Convert batch IDs to integers (strip 'B' prefix if present)
            validated_batches = []
            for bid in batch_ids:
                try:
                    clean_id = str(bid).replace('B', '').replace('b', '').strip()
                    validated_batches.append(int(clean_id))
                except (ValueError, AttributeError, TypeError):
                    logger.warning(f"Invalid batch ID format: {bid}")
                    continue
            
            if not validated_batches:
                logger.warning("No valid batch IDs for QC query")
                return pd.DataFrame()
            
            # Query to get component-level QC data
            # Join with print_job to get rg_id (job_id) for matching with JobReports
            query = """
                SELECT 
                    pc.id as component_id,
                    pc.production_batch_id as batch_id,
                    pj.rg_id as job_id,
                    DATE(pc.created_at) as production_date,
                    qc.state as qc_state,
                    qc.source as qc_source,
                    qc.reasons as qc_reasons,
                    qc.created_at as qc_timestamp
                FROM production_componentorder pc
                JOIN production_printjob pj ON pc.print_job_id = pj.id
                LEFT JOIN production_componentorderqc qc ON qc.component_order_id = pc.id
                WHERE pc.production_batch_id = ANY(%s::bigint[])
                  AND pj.status != 'cancelled'
                ORDER BY pc.id, qc.created_at DESC NULLS LAST
            """
            
            cursor.execute(query, [validated_batches])
            results = cursor.fetchall()
            cursor.close()
            
        if not results:
            logger.warning(f"No QC data found for batches: {batch_ids}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        columns = [
            'component_id', 'batch_id', 'job_id', 'production_date',
            'qc_state', 'qc_source', 'qc_reasons', 'qc_timestamp'
        ]
        df = pd.DataFrame(results, columns=columns)
        
        # Handle NULL values
        df['qc_state'] = df['qc_state'].fillna('not_scanned')
        df['qc_source'] = df['qc_source'].fillna('')
        df['qc_timestamp'] = pd.to_datetime(df['qc_timestamp'], errors='coerce')
        
        # Handle reasons array - it might come as array, JSON string, or NULL
        if 'qc_reasons' in df.columns:
            df['qc_reasons'] = df['qc_reasons'].apply(
                lambda x: (
                    json.loads(x) if isinstance(x, str) 
                    else (x if isinstance(x, list) 
                          else (list(x) if hasattr(x, '__iter__') and not isinstance(x, str) 
                                else []))
                )
            )
        else:
            df['qc_reasons'] = [[]] * len(df)
        
        # If multiple QC records per component, keep the latest one (already sorted by qc.created_at DESC)
        if len(df) > len(df['component_id'].unique()):
            logger.info(f"Found multiple QC records for some components - keeping latest per component")
            df = df.drop_duplicates(subset=['component_id'], keep='first')
            logger.info(f"After deduplication: {len(df)} component QC records")
        
        # Log QC state distribution
        state_counts = df['qc_state'].value_counts().to_dict()
        logger.info(f"QC extract - Components: {len(df)}, State distribution: {state_counts}")
        logger.info(f"QC extract - Components with QC data: {(df['qc_state'] != 'not_scanned').sum()}")
        logger.info(f"QC extract - Components without QC data: {(df['qc_state'] == 'not_scanned').sum()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in fetch_qc_data: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_components_per_job(job_ids: List[str]) -> pd.DataFrame:
    """
    Fetch component counts per job from INSIDE database.
    
    Similar to other extract functions, returns one row per job with component count.
    
    Args:
        job_ids: List of job IDs (rg_ids) to fetch component counts for
        
    Returns:
        DataFrame with columns:
        - job_id: Print job rg_id
        - batch_id: Production batch ID
        - component_count: Number of components in this job
        
    Returns empty DataFrame if no data available.
    """
    if not job_ids:
        return pd.DataFrame()
    
    try:
        with get_inside_connection() as conn:
            cursor = conn.cursor()
            
            placeholders = ','.join(['%s'] * len(job_ids))
            query = f"""
                SELECT 
                    pj.rg_id as job_id,
                    pj.production_batch_id as batch_id,
                    COUNT(pc.id) as component_count
                FROM production_printjob pj
                LEFT JOIN production_componentorder pc ON pc.print_job_id = pj.id
                WHERE pj.rg_id IN ({placeholders})
                  AND pj.status != 'cancelled'
                GROUP BY pj.rg_id, pj.production_batch_id
                ORDER BY pj.rg_id
            """
            
            cursor.execute(query, job_ids)
            results = cursor.fetchall()
            cursor.close()
            
        if not results:
            logger.warning(f"No component counts found for {len(job_ids)} jobs")
            return pd.DataFrame()
        
        columns = ['job_id', 'batch_id', 'component_count']
        df = pd.DataFrame(results, columns=columns)
        
        # Handle NULL values
        df['component_count'] = df['component_count'].fillna(0).astype(int)
        
        logger.info(f"Fetched component counts for {len(df)} jobs")
        return df
        
    except Exception as e:
        logger.error(f"Error in fetch_components_per_job: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_batch_structure(batch_ids: List[str]) -> pd.DataFrame:
    """
    Fetch batch structure: styles, components per style, and garments per style.
    
    Batch Structure:
    - item_order_id: Represents the style number within the batch
    - component_id: Unique identifier for each component within a style (multiple per style)
    - rg_id: Unique identifier for each garment (multiple per component_id)
    
    Logic:
    - Number of styles = number of unique item_order_id per batch
    - Components per style = count of unique component_id per item_order_id
    - Garments per style = count of unique rg_id per item_order_id
    - Total garments = sum of garments across all styles
    
    Args:
        batch_ids: List of batch identifiers
        
    Returns:
        DataFrame with columns:
        - batch_id: Production batch ID
        - style_number: Style number within batch (item_order_id)
        - components_per_style: Number of components for this style (count of unique component_id)
        - unique_rg_ids_per_style: Total unique rg_ids for this style
        - garments_per_style: Number of garments for this style (unique_rg_ids / components_per_style)
        - total_styles: Total number of styles in the batch
        - total_components: Total unique rg_ids across batch (total components to produce)
        
    Returns empty DataFrame if no data available.
    """
    if not batch_ids:
        return pd.DataFrame()
    
    try:
        with get_inside_connection() as conn:
            cursor = conn.cursor()
            
            # Convert batch IDs to integers (strip 'B' prefix if present)
            validated_batches = []
            for bid in batch_ids:
                try:
                    clean_id = str(bid).replace('B', '').replace('b', '').strip()
                    validated_batches.append(int(clean_id))
                except (ValueError, AttributeError, TypeError):
                    logger.warning(f"Invalid batch ID format: {bid}")
                    continue
            
            if not validated_batches:
                logger.warning("No valid batch IDs for batch structure query")
                return pd.DataFrame()
            
            # Query to get batch structure
            # For each style (item_order_id):
            # 1. Count distinct component_id = number of components per style (fixed)
            # 2. Count distinct rg_id = total unique rg_ids per style
            # 3. Garments per style = unique rg_ids / components per style
            query = """
                SELECT 
                    pc.production_batch_id as batch_id,
                    pc.item_order_id as style_number,
                    COUNT(DISTINCT pc.component_id) as components_per_style,
                    COUNT(DISTINCT pc.rg_id) as unique_rg_ids_per_style,
                    COUNT(DISTINCT pc.rg_id)::float / NULLIF(COUNT(DISTINCT pc.component_id), 0) as garments_per_style
                FROM production_componentorder pc
                JOIN production_printjob pj ON pc.print_job_id = pj.id
                WHERE pc.production_batch_id = ANY(%s::bigint[])
                  AND pj.status != 'cancelled'
                  AND pc.item_order_id IS NOT NULL
                GROUP BY pc.production_batch_id, pc.item_order_id
                ORDER BY pc.production_batch_id, pc.item_order_id ASC
            """
            
            cursor.execute(query, [validated_batches])
            results = cursor.fetchall()
            cursor.close()
            
        if not results:
            logger.warning(f"No batch structure data found for batches: {batch_ids}")
            return pd.DataFrame()
        
        columns = ['batch_id', 'style_number', 'components_per_style', 'unique_rg_ids_per_style', 'garments_per_style']
        df = pd.DataFrame(results, columns=columns)
        
        # Round garments_per_style to integer (should be whole number)
        df['garments_per_style'] = df['garments_per_style'].fillna(0).round().astype(int)
        
        # Calculate total components per style = garments_per_style × components_per_style
        df['total_components_per_style'] = df['garments_per_style'] * df['components_per_style']
        
        # Calculate batch-level totals
        # Total styles = count of unique item_order_id per batch
        # Total components (batch) = SUM of total_components_per_style (sum across all styles in batch)
        batch_totals = df.groupby('batch_id').agg({
            'style_number': 'nunique',  # Total number of unique styles (item_order_id)
            'total_components_per_style': 'sum'  # Sum of components per style = total components for batch
        }).reset_index()
        batch_totals.columns = ['batch_id', 'total_styles', 'total_components']
        
        # Merge totals back into main dataframe
        df = df.merge(batch_totals, on='batch_id', how='left')
        
        # Handle NULL values
        df['components_per_style'] = df['components_per_style'].fillna(0).astype(int)
        df['unique_rg_ids_per_style'] = df['unique_rg_ids_per_style'].fillna(0).astype(int)
        df['total_components_per_style'] = df['total_components_per_style'].fillna(0).astype(int)
        df['total_styles'] = df['total_styles'].fillna(0).astype(int)
        df['total_components'] = df['total_components'].fillna(0).astype(int)
        
        logger.info(f"Fetched batch structure for {len(df)} styles across {df['batch_id'].nunique()} batches")
        if not df.empty:
            logger.info(f"Sample data - Batch {df.iloc[0]['batch_id']}: {df.iloc[0]['total_styles']} styles, {df.iloc[0]['total_components']} total components")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in fetch_batch_structure: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_fpy_data(batch_ids: List[str]) -> pd.DataFrame:
    """
    Fetch First Pass Yield (FPY) data for batches.
    
    FPY calculates how many garments could be produced without a defect round.
    
    Logic:
    - For each style (item_order_id), find max defects and min passed across all components
    - garments_not_failed = garmentsGoal - max_defects (garments without any defect)
    - garments_QC_passed = min_passed (garments that passed QC)
    - FPY rate = garments_QC_passed / garmentsGoal × 100%
    
    Args:
        batch_ids: List of batch identifiers
        
    Returns:
        DataFrame with columns:
        - item_order_id: Style identifier
        - garmentsGoal: Target number of garments for this style
        - garments_not_failed: Garments without any defect (quantity - max_defects)
        - garments_QC_passed: Garments that passed QC (min_passed across components)
        - fpy_rate: First Pass Yield rate (garments_QC_passed / garmentsGoal × 100)
        
    Returns empty DataFrame if no data available.
    """
    if not batch_ids:
        return pd.DataFrame()
    
    try:
        with get_inside_connection() as conn:
            cursor = conn.cursor()
            
            # Convert batch IDs to integers (strip 'B' prefix if present)
            validated_batches = []
            for bid in batch_ids:
                try:
                    clean_id = str(bid).replace('B', '').replace('b', '').strip()
                    validated_batches.append(int(clean_id))
                except (ValueError, AttributeError, TypeError):
                    logger.warning(f"Invalid batch ID format: {bid}")
                    continue
            
            if not validated_batches:
                logger.warning("No valid batch IDs for FPY query")
                return pd.DataFrame()
            
            # Query to get FPY data
            query = """
                WITH component_qc AS (
                  SELECT
                    pco.item_order_id,
                    pco.component_id,
                    COUNT(DISTINCT CASE WHEN qc.state = 'failed' THEN pco.id END) AS failed_count,
                    COUNT(DISTINCT CASE WHEN qc.state = 'passed' THEN pco.id END) AS passed_count
                  FROM production_componentorder pco
                  LEFT JOIN production_componentorderqc qc
                    ON qc.component_order_id = pco.id
                  WHERE pco.production_batch_id = ANY(%s::bigint[])
                  GROUP BY
                    pco.item_order_id,
                    pco.component_id
                ),
                itemorder_rollup AS (
                  SELECT
                    item_order_id,
                    MAX(failed_count) AS max_defects,
                    MIN(passed_count) AS min_passed
                  FROM component_qc
                  GROUP BY item_order_id
                )
                SELECT
                  pi.id AS item_order_id,
                  pi.quantity AS garmentsGoal,
                  pi.quantity - COALESCE(r.max_defects, 0) AS garments_not_failed,
                  COALESCE(r.min_passed, 0) AS garments_QC_passed
                FROM production_itemorder pi
                LEFT JOIN itemorder_rollup r
                  ON r.item_order_id = pi.id
                WHERE pi.id IN (
                  SELECT DISTINCT item_order_id
                  FROM production_componentorder
                  WHERE production_batch_id = ANY(%s::bigint[])
                )
                ORDER BY pi.id
            """
            
            cursor.execute(query, [validated_batches, validated_batches])
            results = cursor.fetchall()
            cursor.close()
            
        if not results:
            logger.warning(f"No FPY data found for batches: {batch_ids}")
            return pd.DataFrame()
        
        columns = ['item_order_id', 'garmentsGoal', 'garments_not_failed', 'garments_QC_passed']
        df = pd.DataFrame(results, columns=columns)
        
        # Handle NULL values
        df['garmentsGoal'] = df['garmentsGoal'].fillna(0).astype(int)
        df['garments_not_failed'] = df['garments_not_failed'].fillna(0).astype(int)
        df['garments_QC_passed'] = df['garments_QC_passed'].fillna(0).astype(int)
        
        # Calculate FPY rate
        df['fpy_rate'] = np.where(
            df['garmentsGoal'] > 0,
            (df['garments_QC_passed'] / df['garmentsGoal']) * 100,
            0.0
        )
        
        logger.info(f"Fetched FPY data for {len(df)} styles across {len(validated_batches)} batches")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in fetch_fpy_data: {e}", exc_info=True)
        return pd.DataFrame()


def get_distinct_equipment_states(start_date: str = '2025-12-01') -> List[str]:
    """
    Query distinct state names from the equipment table.
    
    Useful for understanding what state values exist in the database before
    implementing state-based logic.
    
    Args:
        start_date: Start date to filter states (default: '2025-12-01')
        
    Returns:
        List of distinct state names found in the equipment table
    """
    logger.info(f"🔍 Querying distinct equipment states since {start_date}")
    
    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT DISTINCT state 
                FROM public.equipment 
                WHERE ts >= %s::timestamp
                ORDER BY state;
            """
            
            cursor.execute(query, [start_date])
            results = cursor.fetchall()
            cursor.close()
            
            states = [row[0] for row in results if row[0] is not None]
            logger.info(f"  Found {len(states)} distinct states: {', '.join(states)}")
            
            return states
            
    except Exception as e:
        logger.error(f"  ❌ Error querying distinct states: {e}", exc_info=True)
        return []


def fetch_equipment_states_with_durations(
    cell: str,
    start_ts: str,
    end_ts: str
) -> pd.DataFrame:
    """
    Fetch equipment states with durations for a specific cell within a time window.
    
    Queries the equipment table for state changes and calculates how long each state lasted
    by looking at the time to the next state change.
    
    Args:
        cell: Cell name (e.g., 'Print1', 'Cut1', 'Pick1')
        start_ts: Start timestamp (ISO format string or datetime)
        end_ts: End timestamp (ISO format string or datetime)
        
    Returns:
        DataFrame with columns:
        - ts: Timestamp when state started
        - state: State name (e.g., 'running', 'idle', 'down', 'fault')
        - description: State description (if available)
        - duration_seconds: How long this state lasted
        - next_ts: Timestamp when state ended (next state change)
    """
    logger.info(f"🔍 Fetching equipment states with durations for {cell}")
    logger.info(f"  Time window: {start_ts} to {end_ts}")
    
    if not start_ts or not end_ts:
        logger.warning("  Missing start or end timestamp")
        return pd.DataFrame(columns=['ts', 'state', 'description', 'duration_seconds', 'next_ts'])
    
    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()
            
            # Query equipment states - match user's query pattern but include state before window
            # User's query: ts > start AND ts < end (exclusive), but we need state before start too
            # Get state before window + states in window, calculate durations, then filter to window
            query = """
                WITH state_before_window AS (
                    -- Get the state that was active BEFORE the window started
                    SELECT 
                        ts,
                        state,
                        COALESCE(description, '') AS description
                    FROM public.equipment
                    WHERE cell = %s
                      AND ts <= %s::timestamp  -- States up to window start
                    ORDER BY ts DESC
                    LIMIT 1
                ),
                states_in_window AS (
                    -- States that start within the window (matches user's query: ts > start AND ts < end)
                    SELECT 
                        ts,
                        state,
                        COALESCE(description, '') AS description
                    FROM public.equipment
                    WHERE cell = %s
                      AND ts > %s::timestamp   -- Exclusive start (matches user's query)
                      AND ts < %s::timestamp   -- Exclusive end (matches user's query)
                    ORDER BY ts ASC
                ),
                all_relevant_states AS (
                    -- Combine: state before window + states in window
                    SELECT ts, state, description FROM state_before_window
                    UNION ALL
                    SELECT ts, state, description FROM states_in_window
                    ORDER BY ts ASC
                ),
                state_durations AS (
                    SELECT 
                        ts,
                        state,
                        description,
                        LEAD(ts) OVER (ORDER BY ts ASC) AS next_ts,
                        LEAD(ts) OVER (ORDER BY ts ASC) - ts AS duration,
                        EXTRACT(EPOCH FROM (LEAD(ts) OVER (ORDER BY ts ASC) - ts)) AS duration_seconds
                    FROM all_relevant_states
                )
                SELECT 
                    -- Adjust ts to window start if state started before window
                    CASE 
                        WHEN ts <= %s::timestamp THEN %s::timestamp
                        ELSE ts
                    END AS ts,
                    state,
                    description,
                    -- Calculate duration within window only
                    CASE 
                        WHEN ts <= %s::timestamp THEN
                            -- State started before/at window start: duration from window start to next_ts or window end
                            CASE 
                                WHEN next_ts IS NOT NULL AND next_ts < %s::timestamp THEN
                                    EXTRACT(EPOCH FROM (next_ts - %s::timestamp))
                                WHEN next_ts IS NULL OR next_ts >= %s::timestamp THEN
                                    EXTRACT(EPOCH FROM (%s::timestamp - %s::timestamp))
                                ELSE 0
                            END
                        ELSE
                            -- State started in window: use calculated duration, but cap at window end
                            CASE 
                                WHEN next_ts IS NOT NULL AND next_ts < %s::timestamp THEN
                                    duration_seconds
                                WHEN next_ts IS NULL OR next_ts >= %s::timestamp THEN
                                    EXTRACT(EPOCH FROM (%s::timestamp - ts))
                                ELSE 0
                            END
                    END AS duration_seconds,
                    CASE
                        WHEN next_ts IS NOT NULL AND next_ts < %s::timestamp THEN
                            next_ts
                        ELSE
                            %s::timestamp
                    END AS next_ts
                FROM state_durations
                WHERE (ts > %s::timestamp AND ts < %s::timestamp)  -- States starting in window
                   OR (ts <= %s::timestamp AND (next_ts IS NULL OR next_ts > %s::timestamp))  -- State active at window start
                  AND duration_seconds IS NOT NULL
                ORDER BY ts ASC;
            """
            
            logger.info(f"  Executing query with parameters: cell={cell}, start={start_ts}, end={end_ts}")
            cursor.execute(query, [
                cell,           # 1: cell = %s (state_before_window)
                start_ts,       # 2: ts <= %s::timestamp (state_before_window)
                cell,           # 3: cell = %s (states_in_window)
                start_ts,       # 4: ts > %s::timestamp (states_in_window)
                end_ts,         # 5: ts < %s::timestamp (states_in_window)
                start_ts,       # 6: WHEN ts <= %s::timestamp (ts adjustment)
                start_ts,       # 7: THEN %s::timestamp (ts adjustment)
                start_ts,       # 8: WHEN ts <= %s::timestamp (duration calc)
                end_ts,         # 9: next_ts < %s::timestamp
                start_ts,       # 10: next_ts - %s::timestamp
                end_ts,         # 11: next_ts >= %s::timestamp
                start_ts,       # 12: %s::timestamp - %s::timestamp (first)
                start_ts,       # 13: %s::timestamp - %s::timestamp (second)
                end_ts,         # 14: next_ts < %s::timestamp
                end_ts,         # 15: next_ts >= %s::timestamp
                end_ts,         # 16: %s::timestamp - ts
                end_ts,         # 17: next_ts < %s::timestamp (next_ts)
                end_ts,         # 18: %s::timestamp (next_ts)
                start_ts,       # 19: ts > %s::timestamp (WHERE)
                end_ts,         # 20: ts < %s::timestamp (WHERE)
                start_ts,       # 21: ts <= %s::timestamp (WHERE OR)
                start_ts        # 22: next_ts > %s::timestamp (WHERE OR)
            ])
            results = cursor.fetchall()
            cursor.close()
            
            logger.info(f"  Query returned {len(results)} state records")
            
            if not results:
                logger.warning(f"  No equipment state data found for {cell}")
                return pd.DataFrame(columns=['ts', 'state', 'description', 'duration_seconds', 'next_ts'])
            
            # Convert to DataFrame
            columns = ['ts', 'state', 'description', 'duration_seconds', 'next_ts']
            df = pd.DataFrame(results, columns=columns)
            
            # Convert timestamp columns to datetime and normalize to UTC+1 (Europe/Copenhagen)
            # Database timestamps are typically in UTC, but we want to work in UTC+1
            copenhagen_tz = gettz('Europe/Copenhagen')
            
            df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
            df['next_ts'] = pd.to_datetime(df['next_ts'], errors='coerce')
            
            # Normalize timestamps to UTC+1
            # If timestamps are naive (no timezone), assume they're UTC and convert to UTC+1
            # If they're timezone-aware, convert to UTC+1
            for col in ['ts', 'next_ts']:
                if df[col].dt.tz is None:
                    # Naive datetime - assume UTC and convert to UTC+1
                    df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert(copenhagen_tz)
                else:
                    # Timezone-aware - convert to UTC+1
                    df[col] = df[col].dt.tz_convert(copenhagen_tz)
            
            # Ensure duration_seconds is numeric
            df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce').fillna(0.0)
            
            # Log raw state summary (before categorization) - this should match user's direct SQL query
            raw_state_summary = df.groupby('state')['duration_seconds'].sum().to_dict()
            total_raw_seconds = sum(raw_state_summary.values())
            window_duration = (pd.to_datetime(end_ts) - pd.to_datetime(start_ts)).total_seconds()
            logger.info(f"  📊 Raw equipment states summary for {cell} (before categorization):")
            logger.info(f"  ⏱️  Time window duration: {window_duration:.1f}s ({window_duration/60:.1f} min)")
            logger.info(f"  📈 Total recorded state time: {total_raw_seconds:.1f}s ({total_raw_seconds/60:.1f} min)")
            logger.info(f"  📉 Gap (unrecorded time): {max(0, window_duration - total_raw_seconds):.1f}s ({max(0, window_duration - total_raw_seconds)/60:.1f} min)")
            for state, total_duration in sorted(raw_state_summary.items(), key=lambda x: x[1], reverse=True):
                pct = (total_duration / total_raw_seconds * 100) if total_raw_seconds > 0 else 0
                pct_of_window = (total_duration / window_duration * 100) if window_duration > 0 else 0
                logger.info(f"    {state}: {total_duration:.1f}s ({total_duration/60:.1f} min, {pct:.1f}% of recorded, {pct_of_window:.1f}% of window)")
            
            # Also log first few rows for debugging
            if len(df) > 0:
                logger.info(f"  🔍 First 5 state records:")
                for idx, row in df.head(5).iterrows():
                    logger.info(f"    Row {idx}: ts={row['ts']}, state={row['state']}, duration={row['duration_seconds']:.1f}s, next_ts={row['next_ts']}")
            
            # Log summary
            state_summary = df.groupby('state')['duration_seconds'].sum().to_dict()
            logger.info(f"  ✅ Equipment states summary for {cell}:")
            for state, total_duration in state_summary.items():
                logger.info(f"    {state}: {total_duration:.1f}s ({total_duration/60:.1f} min)")
            
            return df
            
    except Exception as e:
        logger.error(f"  ❌ Error fetching equipment states for {cell}: {e}", exc_info=True)
        return pd.DataFrame(columns=['ts', 'state', 'description', 'duration_seconds', 'next_ts'])

