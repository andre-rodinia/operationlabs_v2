"""
Data Fetching Module

This module contains all database data fetching functions for the manufacturing data import tool.
Handles connections to both HISTORIAN and INSIDE databases with time window filtering support.
"""

import logging
import json
import pandas as pd
import psycopg2
from typing import List, Optional

from .pool import get_historian_connection, get_inside_connection
from .queries import secure_query_builder
from core.time_windows.filters import filter_dataframe_by_time_window
from core.time_windows.models import CellTimeWindow

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


def fetch_pick_data_by_jobs(
    job_ids: List[str],
    time_window: Optional[CellTimeWindow] = None
) -> pd.DataFrame:
    """
    Fetch Pick/Robot data for specific job IDs with component-level details.
    
    Optionally filters by time window after fetching.

    Args:
        job_ids: List of job IDs from print data
        time_window: Optional CellTimeWindow to filter results

    Returns:
        DataFrame with Pick data including component counts and availability
    """
    try:
        if not job_ids or len(job_ids) == 0:
            logger.warning("No job IDs provided for Pick data query")
            return pd.DataFrame()

        with get_historian_connection() as conn:
            cursor = conn.cursor()

            # Build secure parameterized query
            query, parameters = secure_query_builder.build_pick_data_by_job_query(job_ids)

            logger.info(f"Fetching Pick data for {len(job_ids)} jobs")

            # Execute query
            cursor.execute(query, parameters)
            results = cursor.fetchall()

            if not results:
                logger.info(f"No Pick data found for {len(job_ids)} jobs")
                return pd.DataFrame()

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Create DataFrame
            df = pd.DataFrame(results, columns=columns)

            # Convert timing columns to datetime if present
            if 'job_start' in df.columns:
                df['job_start'] = pd.to_datetime(df['job_start'], errors='coerce')
            if 'job_end' in df.columns:
                df['job_end'] = pd.to_datetime(df['job_end'], errors='coerce')

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

            logger.info(f"Retrieved {len(df)} Pick records for {len(job_ids)} jobs")
            return df

    except Exception as e:
        logger.error(f"Error fetching Pick data by jobs: {e}", exc_info=True)
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

