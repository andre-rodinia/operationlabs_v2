"""
Data Fetchers
All database query functions for batch discovery and JobReport fetching
"""
import pandas as pd
import numpy as np
import logging
import json
from typing import List, Optional
from db.connections import get_historian_connection, get_inside_connection
from config import Config

logger = logging.getLogger(__name__)

# ============================================================================
# BATCH DISCOVERY
# ============================================================================

def fetch_batches_for_day(day_start: str, day_end: str) -> pd.DataFrame:
    """
    Discover batches produced on a specific day.

    Strategy:
    1. Query HISTORIAN for Print1 JobReports in time window
    2. Extract job IDs from JobReports
    3. Map job IDs to batch IDs via INSIDE database
    4. Group by batch_id with aggregated metadata

    Args:
        day_start: ISO format timestamp (start of day)
        day_end: ISO format timestamp (end of day)

    Returns:
        DataFrame with columns:
        - batch_id: Batch identifier
        - job_count: Number of jobs in batch
        - first_job_time: Earliest job timestamp
        - last_job_time: Latest job timestamp

    Edge Cases:
    - No JobReports found: Returns empty DataFrame
    - Jobs without batch mapping: Excluded from results
    - Multiple batches in same day: All returned
    """
    try:
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
            job_records = cursor.fetchall()
            cursor.close()

        if not job_records:
            logger.debug(f"No JobReports found for {day_start} to {day_end}")
            return pd.DataFrame(columns=['batch_id', 'job_count', 'first_job_time', 'last_job_time'])

        # Extract job IDs and timestamps
        job_ids = [row[0] for row in job_records]
        job_times = {row[0]: row[1] for row in job_records}

        logger.info(f"Found {len(job_ids)} Print JobReports")

        # Step 2: Map job IDs to batch IDs via INSIDE database
        with get_inside_connection() as inside_conn:
            cursor = inside_conn.cursor()

            placeholders = ','.join(['%s'] * len(job_ids))
            query = f"""
                SELECT
                    rg_id as job_id,
                    production_batch_id as batch_id
                FROM production_printjob
                WHERE rg_id IN ({placeholders})
                AND production_batch_id IS NOT NULL
            """

            cursor.execute(query, job_ids)
            mapping_records = cursor.fetchall()
            cursor.close()

        if not mapping_records:
            logger.warning("No batch mappings found for job IDs")
            return pd.DataFrame(columns=['batch_id', 'job_count', 'first_job_time', 'last_job_time'])

        # Step 3: Build batch metadata
        batch_metadata = {}
        for job_id, batch_id in mapping_records:
            if batch_id not in batch_metadata:
                batch_metadata[batch_id] = {
                    'job_ids': [],
                    'job_times': []
                }
            batch_metadata[batch_id]['job_ids'].append(job_id)
            if job_id in job_times:
                batch_metadata[batch_id]['job_times'].append(job_times[job_id])

        # Step 4: Aggregate into DataFrame
        batch_rows = []
        for batch_id, metadata in batch_metadata.items():
            batch_rows.append({
                'batch_id': batch_id,
                'job_count': len(metadata['job_ids']),
                'first_job_time': min(metadata['job_times']) if metadata['job_times'] else None,
                'last_job_time': max(metadata['job_times']) if metadata['job_times'] else None
            })

        batches_df = pd.DataFrame(batch_rows)
        batches_df = batches_df.sort_values('first_job_time')

        logger.info(f"Discovered {len(batches_df)} batches")
        return batches_df

    except Exception as e:
        logger.error(f"Error in fetch_batches_for_day: {e}", exc_info=True)
        return pd.DataFrame(columns=['batch_id', 'job_count', 'first_job_time', 'last_job_time'])

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

