"""
Time Window Selector UI Component

Provides UI components for configuring time windows per manufacturing cell.
Supports manual segment entry and preset templates.
"""

import streamlit as st
import logging
import pytz
from datetime import datetime, timedelta
from typing import Optional

from core.time_windows.models import CellTimeWindow, TimeSegment, TimeWindowPreset

logger = logging.getLogger(__name__)


def render_time_window_editor(
    cell_type: str,
    key_prefix: Optional[str] = None
) -> Optional[CellTimeWindow]:
    """
    Render UI for time window configuration for a specific cell.

    Args:
        cell_type: 'printer', 'cut', or 'pick'
        key_prefix: Optional unique key prefix for Streamlit widgets

    Returns:
        CellTimeWindow object or None if not configured
    """
    if key_prefix is None:
        key_prefix = f"{cell_type}_time_window"

    # Initialize enable state
    if f"{key_prefix}_enabled" not in st.session_state:
        # Default: printer enabled, others disabled
        st.session_state[f"{key_prefix}_enabled"] = (cell_type == 'printer')

    # Enable/Disable toggle
    enabled = st.checkbox(
        f"Enable {cell_type.capitalize()} Cell Analysis",
        value=st.session_state[f"{key_prefix}_enabled"],
        key=f"{key_prefix}_enabled_checkbox",
        help=f"Toggle to include {cell_type} cell in OEE analysis"
    )
    st.session_state[f"{key_prefix}_enabled"] = enabled

    if not enabled:
        st.info(f"⏸️ {cell_type.capitalize()} cell is disabled. Configure time window below to enable it.")
        return None

    st.subheader(f"⏰ {cell_type.capitalize()} Time Window Configuration")

    # Initialize session state for time window if not exists
    if f"{key_prefix}_window" not in st.session_state:
        st.session_state[f"{key_prefix}_window"] = CellTimeWindow(cell_type=cell_type)

    time_window: CellTimeWindow = st.session_state[f"{key_prefix}_window"]

    # Preset templates
    st.markdown("**Quick Templates:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(f"Same Day (8 AM - 4 PM)", key=f"{key_prefix}_preset_sameday"):
            today = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            preset_window = TimeWindowPreset.same_day(cell_type, today, 8, 16)
            st.session_state[f"{key_prefix}_window"] = preset_window
            st.rerun()

    with col2:
        if st.button(f"Overnight Batch", key=f"{key_prefix}_preset_overnight"):
            today = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)
            preset_window = TimeWindowPreset.overnight_batch(cell_type, today, 18, 6)
            st.session_state[f"{key_prefix}_window"] = preset_window
            st.rerun()

    with col3:
        if st.button(f"Clear All", key=f"{key_prefix}_preset_clear"):
            st.session_state[f"{key_prefix}_window"] = CellTimeWindow(cell_type=cell_type)
            st.rerun()

    st.divider()

    # Display current segments
    st.markdown("**Current Time Segments:**")
    if time_window.segments:
        for i, segment in enumerate(time_window.segments):
            with st.expander(
                f"Segment {i+1}: {segment.start.strftime('%Y-%m-%d %H:%M')} → "
                f"{segment.end.strftime('%Y-%m-%d %H:%M')} "
                f"({segment.duration_hours:.1f} hours)",
                expanded=False
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Description:** {segment.description or 'No description'}")
                with col2:
                    if st.button("Remove", key=f"{key_prefix}_remove_{i}"):
                        time_window.remove_segment(i)
                        st.rerun()
    else:
        st.info("No time segments configured. Add a segment below or use a preset template.")

    # Summary
    if time_window.segments:
        total_hours = time_window.total_duration_hours()
        st.success(f"📊 Total time window: {total_hours:.1f} hours ({len(time_window.segments)} segment(s))")

    st.divider()

    # Add new segment
    st.markdown("**Add Time Segment:**")
    
    # Timezone selection
    timezone_options = ["Europe/Copenhagen", "UTC", "America/New_York", "Asia/Tokyo"]
    default_tz = time_window.timezone if hasattr(time_window, 'timezone') else "Europe/Copenhagen"
    selected_tz = st.selectbox(
        "Timezone:",
        options=timezone_options,
        index=timezone_options.index(default_tz) if default_tz in timezone_options else 0,
        key=f"{key_prefix}_timezone",
        help="Select the timezone for the time inputs"
    )
    
    # Default values for manual input (in selected timezone)
    tz = pytz.timezone(selected_tz)
    now_tz = datetime.now(tz)
    default_start = now_tz.replace(hour=8, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
    default_end = now_tz.replace(hour=16, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
    
    col1, col2 = st.columns(2)

    with col1:
        start_time_str = st.text_input(
            f"Start Time (YYYY-MM-DD HH:MM:SS) [{selected_tz}]:",
            value=default_start,
            key=f"{key_prefix}_new_start",
            help="Enter start time in format: YYYY-MM-DD HH:MM:SS (e.g., 2024-01-15 08:30:00). Time is interpreted in the selected timezone."
        )

    with col2:
        end_time_str = st.text_input(
            f"End Time (YYYY-MM-DD HH:MM:SS) [{selected_tz}]:",
            value=default_end,
            key=f"{key_prefix}_new_end",
            help="Enter end time in format: YYYY-MM-DD HH:MM:SS (e.g., 2024-01-15 16:45:00). Time is interpreted in the selected timezone."
        )

    segment_description = st.text_input(
        "Description (optional):",
        value="",
        key=f"{key_prefix}_new_desc",
        help="Optional description for this time segment"
    )

    if st.button("Add Segment", key=f"{key_prefix}_add_segment"):
        try:
            tz = pytz.timezone(selected_tz)
            
            # Parse the datetime strings and make them timezone-aware
            try:
                segment_start_naive = datetime.strptime(start_time_str.strip(), '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # Try without seconds
                try:
                    segment_start_naive = datetime.strptime(start_time_str.strip(), '%Y-%m-%d %H:%M')
                except ValueError:
                    st.error(f"❌ Invalid start time format. Use YYYY-MM-DD HH:MM:SS or YYYY-MM-DD HH:MM")
                    segment_start = None
                else:
                    segment_start = tz.localize(segment_start_naive)
            else:
                segment_start = tz.localize(segment_start_naive)
            
            try:
                segment_end_naive = datetime.strptime(end_time_str.strip(), '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # Try without seconds
                try:
                    segment_end_naive = datetime.strptime(end_time_str.strip(), '%Y-%m-%d %H:%M')
                except ValueError:
                    st.error(f"❌ Invalid end time format. Use YYYY-MM-DD HH:MM:SS or YYYY-MM-DD HH:MM")
                    segment_end = None
                else:
                    segment_end = tz.localize(segment_end_naive)
            else:
                segment_end = tz.localize(segment_end_naive)
            
            if segment_start and segment_end:
                # Update timezone in time window
                time_window.timezone = selected_tz
                time_window.add_segment(segment_start, segment_end, segment_description)
                st.success(f"✅ Added time segment: {segment_start.strftime('%Y-%m-%d %H:%M:%S')} → "
                          f"{segment_end.strftime('%Y-%m-%d %H:%M:%S')} ({selected_tz})")
                st.rerun()
        except ValueError as e:
            st.error(f"❌ Error adding segment: {e}")

    # Return time window if it has segments, otherwise None
    if time_window.segments:
        return time_window
    else:
        return None


def apply_preset_template(
    cell_type: str,
    preset_name: str,
    **kwargs
) -> CellTimeWindow:
    """
    Apply a preset time window template.

    Args:
        cell_type: 'printer', 'cut', or 'pick'
        preset_name: 'same_day', 'overnight_batch', or 'multi_shift'
        **kwargs: Additional arguments for the preset

    Returns:
        CellTimeWindow configured with the preset
    """
    if preset_name == "same_day":
        date = kwargs.get("date", datetime.now())
        start_hour = kwargs.get("start_hour", 8)
        end_hour = kwargs.get("end_hour", 16)
        return TimeWindowPreset.same_day(cell_type, date, start_hour, end_hour)

    elif preset_name == "overnight_batch":
        start_date = kwargs.get("start_date", datetime.now())
        evening_start = kwargs.get("evening_start_hour", 18)
        morning_end = kwargs.get("morning_end_hour", 6)
        return TimeWindowPreset.overnight_batch(cell_type, start_date, evening_start, morning_end)

    elif preset_name == "multi_shift":
        date = kwargs.get("date", datetime.now())
        shift_hours = kwargs.get("shift_hours", [(8, 16)])
        return TimeWindowPreset.multi_shift(cell_type, date, shift_hours)

    else:
        raise ValueError(f"Unknown preset name: {preset_name}")

