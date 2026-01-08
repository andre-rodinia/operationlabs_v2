"""
Formatting Utilities

Functions for formatting timestamps, converting data types, and validating time ranges.
"""

import json
import logging
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Tuple, List

logger = logging.getLogger(__name__)


def format_timestamp(iso_timestamp) -> str:
    """
    Convert ISO 8601 timestamp to readable format (YYYY-MM-DD HH:MM:SS), handling potential errors.
    
    Args:
        iso_timestamp: ISO timestamp string, datetime object, or None
        
    Returns:
        Formatted timestamp string or empty string if invalid
    """
    if not iso_timestamp or pd.isna(iso_timestamp):
        return ""
    try:
        # Handle potential 'Z' suffix for UTC
        if isinstance(iso_timestamp, str) and iso_timestamp.endswith("Z"):
            # Parse as UTC and remove timezone info for consistent display
            dt_obj = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        # Handle cases where it might already be a datetime object or requires basic parsing
        if isinstance(iso_timestamp, str):
            dt_obj = datetime.fromisoformat(iso_timestamp)
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(iso_timestamp, datetime):
            return iso_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Attempt conversion if possible, otherwise return original
            return datetime.fromisoformat(str(iso_timestamp)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    except (ValueError, TypeError):
        # If parsing fails, return the original string or an empty string
        return str(iso_timestamp) if iso_timestamp else ""


def format_downtime_reasons_from_json(input_data: str) -> str:
    """
    Parse downtime reasons (either JSON string or list) and format into a readable string.
    
    Args:
        input_data: JSON string or list containing downtime reasons
        
    Returns:
        Formatted string with downtime reasons
    """
    # Handle empty or NaN values
    if (
        input_data is None
        or pd.isna(input_data)
        or input_data == "nan"
        or input_data == "None"
    ):
        return ""

    # Handle empty strings
    if isinstance(input_data, str) and not input_data.strip():
        return ""

    downtime_reasons = None
    try:
        # Check if it's a string that needs parsing
        if isinstance(input_data, str):
            downtime_reasons = json.loads(input_data)
        else:
            # If it's not a string, try converting to string and return
            return str(input_data)

        # Ensure we ended up with a list after potential parsing
        if not isinstance(downtime_reasons, list):
            return str(input_data)  # Return original if parsing didn't yield a list

        reasons_formatted = []
        for reason in downtime_reasons:
            if isinstance(reason, dict):
                reason_ts_str = format_timestamp(reason.get("ts", ""))
                reason_text = reason.get("reason", "Unknown Reason")
                try:
                    duration_seconds = float(reason.get("duration", 0))
                    duration_minutes = duration_seconds / 60.0
                    reasons_formatted.append(
                        f"{reason_ts_str}: {reason_text} (Reason duration: {duration_minutes:.2f} min)"
                    )
                except (ValueError, TypeError):
                    reasons_formatted.append(
                        f"{reason_ts_str}: {reason_text} (duration error)"
                    )
            else:
                reasons_formatted.append(str(reason))

        return "; ".join(reasons_formatted) if reasons_formatted else ""

    except json.JSONDecodeError:
        # If JSON parsing fails on a string input
        return str(input_data)
    except Exception as e:
        # Catch other potential errors during processing
        logger.warning(f"Error formatting downtime reason: {e} - Input: {input_data}")
        return str(input_data)


def convert_all_datetime_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all datetime columns in a DataFrame to string format (YYYY-MM-DD HH:MM:SS).
    
    Args:
        df: DataFrame with datetime columns
        
    Returns:
        DataFrame with datetime columns converted to strings
    """
    try:
        df = df.copy()  # Avoid modifying original
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df
    except Exception as e:
        logger.error(f"Error converting datetime columns: {e}", exc_info=True)
        return df


def validate_time_range(start_dt: datetime, end_dt: datetime) -> Tuple[List[str], List[str], bool]:
    """
    Validate time range and return validation results with warnings/errors.
    
    Args:
        start_dt: Start datetime
        end_dt: End datetime
        
    Returns:
        Tuple of (validation_errors, validation_warnings, is_valid)
    """
    validation_errors = []
    validation_warnings = []
    
    # Check if end time is after start time
    if end_dt <= start_dt:
        validation_errors.append("End time must be after start time")
        return validation_errors, validation_warnings, False
    
    # Calculate time difference
    time_diff = end_dt - start_dt
    
    # Check for unreasonably short periods (less than 1 minute)
    if time_diff.total_seconds() < 60:
        validation_warnings.append("⚠️ Very short time range (< 1 minute) - may not capture meaningful data")
    
    # Check for unreasonably long periods (more than 90 days)
    if time_diff.days > 90:
        validation_errors.append("Time range too large (> 90 days) - please select a smaller range to avoid performance issues")
        return validation_errors, validation_warnings, False
    
    # Warn for long periods (more than 7 days)
    if time_diff.days > 7:
        validation_warnings.append(f"⚠️ Large time range ({time_diff.days} days) - queries may take longer to complete")
    
    # Check if dates are in the future
    now_utc = datetime.now(pytz.UTC)
    
    # Handle timezone-naive datetimes by assuming UTC
    start_dt_tz = start_dt if start_dt.tzinfo else pytz.UTC.localize(start_dt)
    end_dt_tz = end_dt if end_dt.tzinfo else pytz.UTC.localize(end_dt)
    
    if start_dt_tz > now_utc:
        validation_errors.append("Start time cannot be in the future")
        return validation_errors, validation_warnings, False
    
    if end_dt_tz > now_utc + timedelta(hours=1):  # Allow 1 hour buffer for clock differences
        validation_warnings.append("⚠️ End time is in the future - current data may be incomplete")
    
    # Check if dates are too far in the past (more than 2 years)
    two_years_ago = now_utc - timedelta(days=730)
    if start_dt_tz < two_years_ago:
        validation_warnings.append("⚠️ Data older than 2 years may not be available or archived")
    
    return validation_errors, validation_warnings, True

