"""
Time Window Filtering Utilities

Functions to filter DataFrames and data based on time window configurations.
"""

from typing import List, Optional
import pandas as pd
from datetime import datetime

from .models import CellTimeWindow, TimeSegment


def filter_dataframe_by_time_window(
    df: pd.DataFrame,
    time_window: CellTimeWindow,
    timestamp_column: str = 'start_time'
) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows within time window segments.

    Args:
        df: DataFrame with timestamp column
        time_window: CellTimeWindow configuration
        timestamp_column: Name of the datetime column to filter on

    Returns:
        Filtered DataFrame containing only rows within time window

    Example:
        >>> window = CellTimeWindow('printer', [segment1, segment2])
        >>> filtered_df = filter_dataframe_by_time_window(print_df, window)
        >>> print(f"Filtered from {len(print_df)} to {len(filtered_df)} rows")
    """
    if df.empty:
        return df

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Column '{timestamp_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Create boolean mask for rows within any segment
    mask = df[timestamp_column].apply(time_window.is_within_window)

    return df[mask].copy()


def filter_jobs_by_time_window(
    jobs: List[dict],
    time_window: CellTimeWindow,
    timestamp_key: str = 'start_time'
) -> List[dict]:
    """
    Filter list of job dictionaries by time window.

    Args:
        jobs: List of dictionaries containing job data
        time_window: CellTimeWindow configuration
        timestamp_key: Key name for timestamp in each dictionary

    Returns:
        Filtered list of jobs within time window

    Example:
        >>> jobs = [{'job_id': 'RG-123', 'start_time': dt1}, ...]
        >>> window = CellTimeWindow('cut', [segment])
        >>> filtered = filter_jobs_by_time_window(jobs, window)
    """
    filtered_jobs = []

    for job in jobs:
        if timestamp_key not in job:
            continue

        timestamp = job[timestamp_key]

        # Convert to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        if time_window.is_within_window(timestamp):
            filtered_jobs.append(job)

    return filtered_jobs


def get_time_window_summary(time_window: CellTimeWindow) -> dict:
    """
    Generate summary statistics for a time window.

    Args:
        time_window: CellTimeWindow to summarize

    Returns:
        Dictionary with summary information

    Example:
        >>> summary = get_time_window_summary(window)
        >>> print(f"Total duration: {summary['total_hours']} hours")
    """
    return {
        'cell_type': time_window.cell_type,
        'segment_count': len(time_window.segments),
        'total_minutes': time_window.total_duration_minutes(),
        'total_hours': time_window.total_duration_hours(),
        'earliest_start': time_window.get_earliest_start(),
        'latest_end': time_window.get_latest_end(),
        'segments': [
            {
                'start': seg.start,
                'end': seg.end,
                'duration_hours': seg.duration_hours,
                'description': seg.description
            }
            for seg in time_window.segments
        ]
    }


def validate_time_window_coverage(
    df: pd.DataFrame,
    time_window: CellTimeWindow,
    timestamp_column: str = 'start_time'
) -> dict:
    """
    Validate how much of the data falls within the time window.

    Useful for checking if time window configuration is reasonable.

    Args:
        df: DataFrame to check
        time_window: CellTimeWindow configuration
        timestamp_column: Name of the datetime column

    Returns:
        Dictionary with coverage statistics

    Example:
        >>> coverage = validate_time_window_coverage(df, window)
        >>> print(f"Coverage: {coverage['coverage_percentage']:.1f}%")
    """
    if df.empty:
        return {
            'total_rows': 0,
            'included_rows': 0,
            'excluded_rows': 0,
            'coverage_percentage': 0.0
        }

    total_rows = len(df)
    filtered_df = filter_dataframe_by_time_window(df, time_window, timestamp_column)
    included_rows = len(filtered_df)
    excluded_rows = total_rows - included_rows

    return {
        'total_rows': total_rows,
        'included_rows': included_rows,
        'excluded_rows': excluded_rows,
        'coverage_percentage': (included_rows / total_rows * 100) if total_rows > 0 else 0.0,
        'data_start': df[timestamp_column].min(),
        'data_end': df[timestamp_column].max(),
        'window_start': time_window.get_earliest_start(),
        'window_end': time_window.get_latest_end()
    }


def merge_adjacent_segments(segments: List[TimeSegment], gap_tolerance_minutes: float = 0) -> List[TimeSegment]:
    """
    Merge adjacent or overlapping time segments.

    Useful for simplifying time windows with many small segments.

    Args:
        segments: List of TimeSegment objects
        gap_tolerance_minutes: Maximum gap between segments to merge (default 0)

    Returns:
        List of merged TimeSegment objects

    Example:
        >>> # Merge segments with gaps up to 30 minutes
        >>> merged = merge_adjacent_segments(segments, gap_tolerance_minutes=30)
    """
    if not segments:
        return []

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda s: s.start)

    merged = [sorted_segments[0]]

    for current in sorted_segments[1:]:
        previous = merged[-1]

        # Calculate gap in minutes
        gap_minutes = (current.start - previous.end).total_seconds() / 60.0

        # Merge if segments overlap or are within tolerance
        if gap_minutes <= gap_tolerance_minutes:
            # Extend the previous segment
            merged[-1] = TimeSegment(
                start=previous.start,
                end=max(previous.end, current.end),
                description=f"{previous.description} + {current.description}"
            )
        else:
            # Add as new segment
            merged.append(current)

    return merged
