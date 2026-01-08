"""
Simplified OEE Calculator for Manufacturing Analytics

This module provides clean, straightforward OEE calculations for three production cells:
- Printer: OEE = Availability × Performance × 100%
- Cut: OEE = Availability × 100% × 100%
- Pick: OEE = Availability × 100% × Quality_accuracy
"""

from typing import Dict, Optional, Union
from dataclasses import dataclass
from decimal import Decimal
from datetime import date


@dataclass
class OEEMetrics:
    """Container for OEE calculation results"""
    availability: float  # 0.0 to 1.0
    performance: float   # 0.0 to 1.0
    quality: float       # 0.0 to 1.0
    oee: float          # 0.0 to 1.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy display"""
        return {
            'availability': self.availability,
            'performance': self.performance,
            'quality': self.quality,
            'oee': self.oee
        }

    def to_percentage_dict(self) -> Dict[str, float]:
        """Convert to percentage values for display"""
        return {
            'availability': round(self.availability * 100, 2),
            'performance': round(self.performance * 100, 2),
            'quality': round(self.quality * 100, 2),
            'oee': round(self.oee * 100, 2)
        }


def calculate_cell_oee(
    cell_type: str,
    uptime: Union[float, Decimal],
    downtime: Union[float, Decimal],
    performance_ratio: Union[float, Decimal] = 1.0,
    quality_ratio: Union[float, Decimal] = 1.0
) -> OEEMetrics:
    """
    Calculate OEE for a specific manufacturing cell.

    Args:
        cell_type: 'printer', 'cut', or 'pick'
        uptime: Total uptime in minutes
        downtime: Total downtime in minutes
        performance_ratio: Actual speed / Nominal speed (0.0-1.0)
        quality_ratio: Success rate (0.0-1.0)

    Returns:
        OEEMetrics object with all OEE components

    Raises:
        ValueError: If cell_type is not recognized

    Examples:
        >>> # Printer with 350 min uptime, 90 min downtime, 87% performance
        >>> metrics = calculate_cell_oee('printer', 350, 90, 0.87)
        >>> print(f"Printer OEE: {metrics.oee:.1%}")

        >>> # Cut with 280 min uptime, 20 min downtime
        >>> metrics = calculate_cell_oee('cut', 280, 20)
        >>> print(f"Cut OEE: {metrics.oee:.1%}")

        >>> # Pick with 310 min uptime, 50 min downtime, 94% quality
        >>> metrics = calculate_cell_oee('pick', 310, 50, quality_ratio=0.94)
        >>> print(f"Pick OEE: {metrics.oee:.1%}")
    """
    # Convert Decimal to float if needed (PostgreSQL returns Decimal types)
    uptime = float(uptime) if isinstance(uptime, Decimal) else uptime
    downtime = float(downtime) if isinstance(downtime, Decimal) else downtime
    performance_ratio = float(performance_ratio) if isinstance(performance_ratio, Decimal) else performance_ratio
    quality_ratio = float(quality_ratio) if isinstance(quality_ratio, Decimal) else quality_ratio
    
    # Calculate availability (common to all cells)
    total_time = uptime + downtime
    availability = uptime / total_time if total_time > 0 else 0.0

    # Apply cell-specific OEE formulas
    cell_type = cell_type.lower()

    if cell_type == 'printer':
        # Printer: A × P × 100%
        performance = performance_ratio
        quality = 1.0  # Fixed at 100%

    elif cell_type == 'cut':
        # Cut: A × 100% × 100%
        performance = 1.0  # Fixed at 100%
        quality = 1.0      # Fixed at 100%

    elif cell_type == 'pick':
        # Pick: A × Pick Quality Accuracy × Quality (100%)
        # Pick Quality Accuracy is passed as quality_ratio parameter
        # Quality component is always 100% (fixed)
        performance = quality_ratio  # Pick Quality Accuracy (0.0-1.0)
        quality = 1.0                # Quality is fixed at 100%

    else:
        raise ValueError(
            f"Unknown cell type: '{cell_type}'. "
            f"Valid options: 'printer', 'cut', 'pick'"
        )

    # Calculate final OEE
    oee = availability * performance * quality

    return OEEMetrics(
        availability=availability,
        performance=performance,
        quality=quality,
        oee=oee
    )


@dataclass
class BatchQualityBreakdown:
    """Quality breakdown for a batch with defect attribution."""
    production_batch_id: int
    production_date: date
    
    # Overall Counts
    total_components: int
    total_passed: int
    total_failed: int
    not_scanned: int
    
    # Defect Attribution by Cell
    print_defects: int          # Printer cell
    cut_defects: int            # Cutter cell
    pick_defects: int           # Pick cell
    
    # Non-cell defects (upstream issues)
    fabric_defects: int         # Incoming material
    file_defects: int           # Pre-production/file issues
    other_defects: int          # Unattributed
    
    # QC Source breakdown
    scanned_handheld: int
    scanned_manual: int
    scanned_inside: int
    
    # Rates
    quality_rate_checked_percent: Optional[float]
    scan_coverage_percent: float
    unchecked_rate_percent: float
    
    @property
    def cell_attributed_defects(self) -> int:
        """Defects that can be attributed to production cells."""
        return self.print_defects + self.cut_defects + self.pick_defects
    
    @property
    def upstream_defects(self) -> int:
        """Defects from upstream (not production cell issues)."""
        return self.fabric_defects + self.file_defects
    
    @property
    def total_attributed_defects(self) -> int:
        """All defects that have been categorized."""
        return self.cell_attributed_defects + self.upstream_defects + self.other_defects
    
    @property
    def unattributed_defects(self) -> int:
        """Defects that couldn't be matched to any category."""
        return max(0, self.total_failed - self.total_attributed_defects)
    
    @property
    def attribution_coverage_percent(self) -> float:
        """What percentage of defects were successfully attributed?"""
        if self.total_failed == 0:
            return 100.0
        return (self.total_attributed_defects / self.total_failed) * 100.0
    
    @property
    def cell_quality_impact(self) -> Dict[str, float]:
        """
        Calculate quality impact by cell for OEE reconciliation.
        Returns quality rate (0-1) for each cell.
        """
        if self.total_components == 0:
            return {
                'printer': 1.0,
                'cutter': 1.0,
                'pick': 1.0
            }
        
        return {
            'printer': (self.total_components - self.print_defects) / self.total_components,
            'cutter': (self.total_components - self.cut_defects) / self.total_components,
            'pick': (self.total_components - self.pick_defects) / self.total_components
        }


@dataclass
class CellOEEReconciled:
    """Reconciled OEE for a single production cell after QC completion."""
    cell_name: str                    # 'Printer', 'Cut', 'Pick'
    production_batch_id: int
    production_date: date
    
    # OEE Components
    availability: float               # 0-1
    performance: float                # 0-1
    quality_attributed: float         # 0-1 (from defect attribution)
    oee_reconciled: float             # A × P × Q
    
    # Defect Details
    total_processed: int
    defects_attributed: int
    defect_rate_percent: float
    
    # Metadata
    is_reconciled: bool = True        # True = post-QC, False = real-time estimate
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy display."""
        return {
            'cell_name': self.cell_name,
            'production_batch_id': self.production_batch_id,
            'production_date': str(self.production_date),
            'availability': self.availability,
            'performance': self.performance,
            'quality_attributed': self.quality_attributed,
            'oee_reconciled': self.oee_reconciled,
            'total_processed': self.total_processed,
            'defects_attributed': self.defects_attributed,
            'defect_rate_percent': self.defect_rate_percent,
            'is_reconciled': self.is_reconciled
        }


def calculate_reconciled_cell_oee(
    cell_name: str,
    batch_id: int,
    production_date: date,
    availability: float,
    performance: float,
    breakdown: BatchQualityBreakdown
) -> CellOEEReconciled:
    """
    Calculate reconciled OEE for a specific cell using attributed defects.
    
    Args:
        cell_name: 'Printer', 'Cutter', or 'Pick'
        batch_id: Production batch ID
        production_date: Production date
        availability: Cell availability (0-1) from uptime tracking
        performance: Cell performance (0-1) from cycle time analysis
        breakdown: BatchQualityBreakdown with defect attribution
        
    Returns:
        CellOEEReconciled with calculated values
    """
    # Get defects for this cell
    defect_map = {
        'Printer': breakdown.print_defects,
        'Cutter': breakdown.cut_defects,
        'Cut': breakdown.cut_defects,  # Support both names
        'Pick': breakdown.pick_defects
    }
    
    cell_defects = defect_map.get(cell_name, 0)
    total_processed = breakdown.total_components
    
    # Calculate quality rate
    if total_processed > 0:
        quality = (total_processed - cell_defects) / total_processed
        defect_rate = (cell_defects / total_processed) * 100.0
    else:
        quality = 1.0
        defect_rate = 0.0
    
    # Calculate reconciled OEE
    oee = availability * performance * quality
    
    return CellOEEReconciled(
        cell_name=cell_name,
        production_batch_id=batch_id,
        production_date=production_date,
        availability=availability,
        performance=performance,
        quality_attributed=quality,
        oee_reconciled=oee,
        total_processed=total_processed,
        defects_attributed=cell_defects,
        defect_rate_percent=defect_rate,
        is_reconciled=True
    )


def aggregate_batch_metrics(
    jobs_data: list,
    time_column: str = 'uptime',
    downtime_column: str = 'downtime'
) -> Dict[str, float]:
    """
    Aggregate uptime/downtime metrics across multiple jobs.

    This is a simple sum aggregation - use for preparing data
    before passing to calculate_cell_oee().

    Args:
        jobs_data: List of dictionaries containing job metrics
        time_column: Column name for uptime values
        downtime_column: Column name for downtime values

    Returns:
        Dictionary with total_uptime and total_downtime

    Example:
        >>> jobs = [
        ...     {'uptime': 100, 'downtime': 20},
        ...     {'uptime': 150, 'downtime': 30},
        ...     {'uptime': 100, 'downtime': 40}
        ... ]
        >>> totals = aggregate_batch_metrics(jobs)
        >>> print(totals)
        {'total_uptime': 350, 'total_downtime': 90}
    """
    total_uptime = sum(job.get(time_column, 0) for job in jobs_data)
    total_downtime = sum(job.get(downtime_column, 0) for job in jobs_data)

    return {
        'total_uptime': total_uptime,
        'total_downtime': total_downtime
    }


def calculate_performance_ratio(
    actual_speed: float,
    nominal_speed: float
) -> float:
    """
    Calculate performance ratio for printer cell.

    Args:
        actual_speed: Actual production speed
        nominal_speed: Target/nominal speed

    Returns:
        Performance ratio (0.0-1.0)

    Example:
        >>> ratio = calculate_performance_ratio(45, 50)
        >>> print(f"Performance: {ratio:.1%}")
        Performance: 90.0%
    """
    if nominal_speed <= 0:
        return 0.0
    return min(actual_speed / nominal_speed, 1.0)


def calculate_quality_accuracy(
    successful_picks: int,
    total_picks: int
) -> float:
    """
    Calculate quality accuracy for pick cell.

    Args:
        successful_picks: Number of successful pick operations
        total_picks: Total number of pick attempts

    Returns:
        Quality accuracy ratio (0.0-1.0)

    Example:
        >>> accuracy = calculate_quality_accuracy(940, 1000)
        >>> print(f"Pick accuracy: {accuracy:.1%}")
        Pick accuracy: 94.0%
    """
    if total_picks <= 0:
        return 0.0
    return successful_picks / total_picks
