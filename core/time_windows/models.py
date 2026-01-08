"""
Time Window Management Models

Handles per-cell time windows with multiple segments to support:
- Multi-day batch operations
- Shift-based tracking
- Custom time ranges per cell
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import pytz


@dataclass
class TimeSegment:
    """
    Represents a single time range for analysis.

    A time segment defines a specific period to include in calculations.
    Multiple segments can be combined to handle non-continuous production.
    """
    start: datetime
    end: datetime
    description: str = ""

    def __post_init__(self):
        """Validate time segment"""
        if self.end <= self.start:
            raise ValueError(
                f"End time ({self.end}) must be after start time ({self.start})"
            )

    @property
    def duration_minutes(self) -> float:
        """Calculate segment duration in minutes"""
        return (self.end - self.start).total_seconds() / 60.0

    @property
    def duration_hours(self) -> float:
        """Calculate segment duration in hours"""
        return self.duration_minutes / 60.0

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within this segment"""
        return self.start <= timestamp <= self.end

    def overlaps_with(self, other: 'TimeSegment') -> bool:
        """Check if this segment overlaps with another"""
        return not (self.end <= other.start or self.start >= other.end)

    def __repr__(self) -> str:
        desc = f" ({self.description})" if self.description else ""
        return (
            f"TimeSegment({self.start.strftime('%Y-%m-%d %H:%M')} → "
            f"{self.end.strftime('%Y-%m-%d %H:%M')}{desc})"
        )


@dataclass
class CellTimeWindow:
    """
    Time window configuration for a manufacturing cell.

    Supports multiple time segments to handle complex scenarios like:
    - Batch spanning multiple days
    - Production across different shifts
    - Excluding maintenance/downtime periods
    """
    cell_type: str  # 'printer', 'cut', or 'pick'
    segments: List[TimeSegment] = field(default_factory=list)
    timezone: str = "Europe/Copenhagen"

    def __post_init__(self):
        """Validate cell type"""
        valid_types = ['printer', 'cut', 'pick']
        if self.cell_type.lower() not in valid_types:
            raise ValueError(
                f"Invalid cell_type: '{self.cell_type}'. "
                f"Must be one of: {valid_types}"
            )
        self.cell_type = self.cell_type.lower()

        # Check for overlapping segments
        for i, seg1 in enumerate(self.segments):
            for seg2 in self.segments[i+1:]:
                if seg1.overlaps_with(seg2):
                    raise ValueError(
                        f"Overlapping segments detected: {seg1} and {seg2}"
                    )

    def add_segment(self, start: datetime, end: datetime, description: str = ""):
        """Add a new time segment to the window"""
        segment = TimeSegment(start, end, description)

        # Check for overlaps with existing segments
        for existing in self.segments:
            if segment.overlaps_with(existing):
                raise ValueError(
                    f"New segment {segment} overlaps with existing {existing}"
                )

        self.segments.append(segment)
        # Keep segments sorted by start time
        self.segments.sort(key=lambda s: s.start)

    def remove_segment(self, index: int):
        """Remove a segment by index"""
        if 0 <= index < len(self.segments):
            self.segments.pop(index)

    def total_duration_minutes(self) -> float:
        """Calculate total time across all segments in minutes"""
        return sum(seg.duration_minutes for seg in self.segments)

    def total_duration_hours(self) -> float:
        """Calculate total time across all segments in hours"""
        return self.total_duration_minutes() / 60.0

    def is_within_window(self, timestamp: datetime) -> bool:
        """
        Check if timestamp falls within any segment.

        Args:
            timestamp: Datetime to check

        Returns:
            True if timestamp is within any segment, False otherwise
        """
        return any(seg.contains(timestamp) for seg in self.segments)

    def get_earliest_start(self) -> Optional[datetime]:
        """Get the earliest start time across all segments"""
        if not self.segments:
            return None
        return min(seg.start for seg in self.segments)

    def get_latest_end(self) -> Optional[datetime]:
        """Get the latest end time across all segments"""
        if not self.segments:
            return None
        return max(seg.end for seg in self.segments)

    def __repr__(self) -> str:
        seg_count = len(self.segments)
        total_hrs = self.total_duration_hours()
        return (
            f"CellTimeWindow(cell={self.cell_type}, "
            f"segments={seg_count}, total_hours={total_hrs:.1f})"
        )


@dataclass
class TimeWindowPreset:
    """
    Preset time window templates for common scenarios.

    Makes it easy to apply standard time configurations without
    manually creating segments.
    """
    name: str
    description: str

    @staticmethod
    def same_day(
        cell_type: str,
        date: datetime,
        start_hour: int = 8,
        end_hour: int = 16,
        timezone: str = "Europe/Copenhagen"
    ) -> CellTimeWindow:
        """
        Create a single-day time window (e.g., 8 AM - 4 PM).

        Args:
            cell_type: 'printer', 'cut', or 'pick'
            date: The date for the window
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)
            timezone: Timezone name

        Returns:
            CellTimeWindow with a single segment
        """
        tz = pytz.timezone(timezone)
        start = tz.localize(datetime(
            date.year, date.month, date.day, start_hour, 0, 0
        ))
        end = tz.localize(datetime(
            date.year, date.month, date.day, end_hour, 0, 0
        ))

        segment = TimeSegment(start, end, f"{cell_type} day shift")
        return CellTimeWindow(cell_type, [segment], timezone)

    @staticmethod
    def overnight_batch(
        cell_type: str,
        start_date: datetime,
        evening_start_hour: int = 18,
        morning_end_hour: int = 6,
        timezone: str = "Europe/Copenhagen"
    ) -> CellTimeWindow:
        """
        Create overnight time window (e.g., 6 PM Day 1 → 6 AM Day 2).

        Args:
            cell_type: 'printer', 'cut', or 'pick'
            start_date: First day of the batch
            evening_start_hour: Start hour on first day
            morning_end_hour: End hour on second day
            timezone: Timezone name

        Returns:
            CellTimeWindow spanning two days
        """
        tz = pytz.timezone(timezone)

        # Evening segment (Day 1)
        evening_start = tz.localize(datetime(
            start_date.year, start_date.month, start_date.day,
            evening_start_hour, 0, 0
        ))
        evening_end = tz.localize(datetime(
            start_date.year, start_date.month, start_date.day,
            23, 59, 59
        ))

        # Calculate next day
        next_day = start_date.replace(
            day=start_date.day + 1
        )

        # Morning segment (Day 2)
        morning_start = tz.localize(datetime(
            next_day.year, next_day.month, next_day.day,
            0, 0, 0
        ))
        morning_end = tz.localize(datetime(
            next_day.year, next_day.month, next_day.day,
            morning_end_hour, 0, 0
        ))

        segments = [
            TimeSegment(evening_start, evening_end, f"{cell_type} evening shift"),
            TimeSegment(morning_start, morning_end, f"{cell_type} morning shift")
        ]

        return CellTimeWindow(cell_type, segments, timezone)

    @staticmethod
    def multi_shift(
        cell_type: str,
        date: datetime,
        shift_hours: List[tuple],
        timezone: str = "Europe/Copenhagen"
    ) -> CellTimeWindow:
        """
        Create time window with multiple shifts in a day.

        Args:
            cell_type: 'printer', 'cut', or 'pick'
            date: The date for shifts
            shift_hours: List of (start_hour, end_hour) tuples
                        e.g., [(8, 16), (18, 23)] for day and evening shifts
            timezone: Timezone name

        Returns:
            CellTimeWindow with multiple segments
        """
        tz = pytz.timezone(timezone)
        segments = []

        for i, (start_hour, end_hour) in enumerate(shift_hours):
            start = tz.localize(datetime(
                date.year, date.month, date.day, start_hour, 0, 0
            ))
            end = tz.localize(datetime(
                date.year, date.month, date.day, end_hour, 0, 0
            ))
            segments.append(
                TimeSegment(start, end, f"{cell_type} shift {i+1}")
            )

        return CellTimeWindow(cell_type, segments, timezone)


def auto_detect_time_window(
    cell_type: str,
    timestamps: List[datetime],
    timezone: str = "Europe/Copenhagen"
) -> CellTimeWindow:
    """
    Automatically detect time window from data timestamps.

    Creates a single segment from earliest to latest timestamp.

    Args:
        cell_type: 'printer', 'cut', or 'pick'
        timestamps: List of datetime objects from the data
        timezone: Timezone name

    Returns:
        CellTimeWindow covering the full data range
    """
    if not timestamps:
        raise ValueError("Cannot auto-detect time window from empty timestamp list")

    start = min(timestamps)
    end = max(timestamps)

    segment = TimeSegment(start, end, f"Auto-detected from {len(timestamps)} jobs")
    return CellTimeWindow(cell_type, [segment], timezone)
