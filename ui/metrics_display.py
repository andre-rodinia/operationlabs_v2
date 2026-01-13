"""
Metrics Display UI Component

Provides UI components for displaying OEE metrics and timeline charts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
from typing import Optional, Dict

from core.calculations.oee import (
    OEEMetrics, 
    CellOEEReconciled,
    BatchQualityBreakdown
)

logger = logging.getLogger(__name__)


def get_color_for_oee(oee_value: float) -> str:
    """
    Get color code based on OEE value.
    
    Args:
        oee_value: OEE value (0.0-1.0)
        
    Returns:
        Color hex code
    """
    if oee_value >= 0.85:
        return "#28a745"  # Green - World class
    elif oee_value >= 0.60:
        return "#ffc107"  # Yellow - Acceptable
    else:
        return "#dc3545"  # Red - Needs improvement


def render_printer_metrics_detailed(
    metrics: OEEMetrics,
    print_df: pd.DataFrame
):
    """
    Display detailed printer metrics in the requested format.
    
    Args:
        metrics: OEEMetrics object with OEE components
        print_df: DataFrame with print job data containing uptime, downtime, speed, nominalSpeed
    """
    # Calculate aggregated values from print_df
    uptime_min = float(print_df['uptime (min)'].sum()) if 'uptime (min)' in print_df.columns else 0.0
    downtime_min = float(print_df['downtime (min)'].sum()) if 'downtime (min)' in print_df.columns else 0.0
    
    # Calculate theoretical and actual speeds (in m/h)
    theoretical_speed = 0.0
    actual_speed = 0.0
    
    if 'nominalSpeed' in print_df.columns:
        # Theoretical speed = average nominal speed (all jobs should have same nominal speed, but average handles variations)
        theoretical_speed = float(print_df['nominalSpeed'].mean()) if len(print_df) > 0 else 0.0
    
    if 'speed' in print_df.columns and 'length (m)' in print_df.columns:
        # Actual speed = weighted average by job length
        # Weight each job's speed by its length: sum(speed * length) / sum(length)
        valid_rows = print_df[print_df['speed'].notna() & print_df['length (m)'].notna() & (print_df['length (m)'] > 0)]
        if len(valid_rows) > 0:
            total_weighted_speed = (valid_rows['speed'] * valid_rows['length (m)']).sum()
            total_length = valid_rows['length (m)'].sum()
            if total_length > 0:
                actual_speed = float(total_weighted_speed / total_length)
        elif 'speed' in print_df.columns:
            # Fallback to simple average if length data is not available
            actual_speed = float(print_df['speed'].mean()) if len(print_df) > 0 else 0.0
    
    # Display header with styling
    st.markdown(
        """
        <div style="
            padding: 1rem 0;
            border-bottom: 2px solid #4a90e2;
            margin-bottom: 1.5rem;
        ">
            <h2 style="margin: 0; color: #4a90e2;">🖨️ Printer Cell</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Row 1: OEE Components (Availability, Performance, Quality) - using metrics for better visual
    st.markdown("#### OEE Components")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Availability</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #4a90e2;">{metrics.availability * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Performance</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #4a90e2;">{metrics.performance * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Quality</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #4a90e2;">{metrics.quality * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Row 2: Time Metrics
    st.markdown("#### Time Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Uptime</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #4a90e2;">{uptime_min:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">mins</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Downtime</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #4a90e2;">{downtime_min:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">mins</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Row 3: Speed Metrics
    st.markdown("#### Speed Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Theoretical Speed</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #4a90e2;">{theoretical_speed:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">m/h</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Actual Speed</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #4a90e2;">{actual_speed:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">m/h</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # OEE Summary - prominent display
    oee_color = get_color_for_oee(metrics.oee)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {oee_color}15 0%, {oee_color}05 100%);
            border-left: 4px solid {oee_color};
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        ">
            <h3 style="margin: 0; color: {oee_color};">
                Overall Equipment Effectiveness (OEE)
            </h3>
            <p style="font-size: 2.5em; font-weight: bold; margin: 0.5rem 0; color: {oee_color};">
                {metrics.oee * 100:.1f}%
            </p>
            <p style="margin: 0; color: #666; font-size: 0.9em;">
                Availability × Performance × Quality
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_cut_metrics_detailed(
    metrics: OEEMetrics,
    cut_df: pd.DataFrame
):
    """
    Display detailed cut metrics in the requested format.
    
    Args:
        metrics: OEEMetrics object with OEE components
        cut_df: DataFrame with cut job data containing uptime, downtime
    """
    # Calculate aggregated values from cut_df
    uptime_min = float(cut_df['uptime (min)'].sum()) if 'uptime (min)' in cut_df.columns else 0.0
    downtime_min = float(cut_df['downtime (min)'].sum()) if 'downtime (min)' in cut_df.columns else 0.0
    
    # Calculate theoretical and actual speeds if available
    theoretical_speed = 0.0
    actual_speed = 0.0
    
    if 'nominalSpeed' in cut_df.columns:
        theoretical_speed = float(cut_df['nominalSpeed'].mean()) if len(cut_df) > 0 else 0.0
    elif 'nominal_speed' in cut_df.columns:
        theoretical_speed = float(cut_df['nominal_speed'].mean()) if len(cut_df) > 0 else 0.0
    
    if 'speed' in cut_df.columns and 'length (m)' in cut_df.columns:
        valid_rows = cut_df[cut_df['speed'].notna() & cut_df['length (m)'].notna() & (cut_df['length (m)'] > 0)]
        if len(valid_rows) > 0:
            total_weighted_speed = (valid_rows['speed'] * valid_rows['length (m)']).sum()
            total_length = valid_rows['length (m)'].sum()
            if total_length > 0:
                actual_speed = float(total_weighted_speed / total_length)
        elif 'speed' in cut_df.columns:
            actual_speed = float(cut_df['speed'].mean()) if len(cut_df) > 0 else 0.0
    
    # Display header with styling
    st.markdown(
        """
        <div style="
            padding: 1rem 0;
            border-bottom: 2px solid #ff7f0e;
            margin-bottom: 1.5rem;
        ">
            <h2 style="margin: 0; color: #ff7f0e;">✂️ Cut Cell</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Row 1: OEE Components
    st.markdown("#### OEE Components")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Availability</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #ff7f0e;">{metrics.availability * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Performance</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #ff7f0e;">{metrics.performance * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Quality</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #ff7f0e;">{metrics.quality * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Row 2: Time Metrics
    st.markdown("#### Time Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Uptime</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #ff7f0e;">{uptime_min:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">mins</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Downtime</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #ff7f0e;">{downtime_min:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">mins</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Row 3: Speed Metrics (if available)
    if theoretical_speed > 0 or actual_speed > 0:
        st.markdown("#### Speed Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Theoretical Speed</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #ff7f0e;">{theoretical_speed:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">m/h</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Actual Speed</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #ff7f0e;">{actual_speed:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">m/h</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.divider()
    
    # OEE Summary
    oee_color = get_color_for_oee(metrics.oee)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {oee_color}15 0%, {oee_color}05 100%);
            border-left: 4px solid {oee_color};
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        ">
            <h3 style="margin: 0; color: {oee_color};">
                Overall Equipment Effectiveness (OEE)
            </h3>
            <p style="font-size: 2.5em; font-weight: bold; margin: 0.5rem 0; color: {oee_color};">
                {metrics.oee * 100:.1f}%
            </p>
            <p style="margin: 0; color: #666; font-size: 0.9em;">
                Availability × Performance × Quality
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pick_metrics_detailed(
    metrics: OEEMetrics,
    pick_df: pd.DataFrame
):
    """
    Display detailed pick metrics in the requested format.
    
    Args:
        metrics: OEEMetrics object with OEE components
        pick_df: DataFrame with pick job data containing uptime, downtime
    """
    # Calculate aggregated values from pick_df
    # Uptime/downtime are calculated from equipment state, not per component
    # All component rows have the same values, so take the first instead of summing
    if 'uptime_min' in pick_df.columns and len(pick_df) > 0:
        uptime_min = float(pick_df['uptime_min'].sum())
    elif 'uptime (min)' in pick_df.columns and len(pick_df) > 0:
        uptime_min = float(pick_df['uptime (min)'].iloc[0])
    else:
        uptime_min = 0.0
    
    if 'downtime_min' in pick_df.columns and len(pick_df) > 0:
        downtime_min = float(pick_df['downtime_min'].sum())
    elif 'downtime (min)' in pick_df.columns and len(pick_df) > 0:
        downtime_min = float(pick_df['downtime (min)'].iloc[0])
    else:
        downtime_min = 0.0
    
    # Calculate theoretical and actual speeds if available
    theoretical_speed = 0.0
    actual_speed = 0.0
    
    if 'nominalSpeed' in pick_df.columns:
        theoretical_speed = float(pick_df['nominalSpeed'].mean()) if len(pick_df) > 0 else 0.0
    elif 'nominal_speed' in pick_df.columns:
        theoretical_speed = float(pick_df['nominal_speed'].mean()) if len(pick_df) > 0 else 0.0
    
    if 'speed' in pick_df.columns and 'length (m)' in pick_df.columns:
        valid_rows = pick_df[pick_df['speed'].notna() & pick_df['length (m)'].notna() & (pick_df['length (m)'] > 0)]
        if len(valid_rows) > 0:
            total_weighted_speed = (valid_rows['speed'] * valid_rows['length (m)']).sum()
            total_length = valid_rows['length (m)'].sum()
            if total_length > 0:
                actual_speed = float(total_weighted_speed / total_length)
        elif 'speed' in pick_df.columns:
            actual_speed = float(pick_df['speed'].mean()) if len(pick_df) > 0 else 0.0
    
    # Display header with styling
    st.markdown(
        """
        <div style="
            padding: 1rem 0;
            border-bottom: 2px solid #2ca02c;
            margin-bottom: 1.5rem;
        ">
            <h2 style="margin: 0; color: #2ca02c;">📦 Pick Cell</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Row 1: OEE Components
    st.markdown("#### OEE Components")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Availability</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #2ca02c;">{metrics.availability * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Performance</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #2ca02c;">{metrics.performance * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        # Check if robot_accuracy is available (two-layer quality)
        if 'robot_accuracy' in pick_df.columns and len(pick_df) > 0:
            robot_acc = float(pick_df['robot_accuracy'].mean())
            combined = metrics.quality * 100
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Quality</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #2ca02c;">{combined:.1f}%</div>
                    <div style="font-size: 0.7em; color: #999; margin-top: 0.3rem;">Robot Accuracy: {robot_acc:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Quality</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #2ca02c;">{metrics.quality * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.divider()
    
    # Row 2: Time Metrics
    st.markdown("#### Time Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Uptime</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #2ca02c;">{uptime_min:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">mins</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Downtime</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #2ca02c;">{downtime_min:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">mins</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Row 3: Component Metrics
    st.markdown("#### Component Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'components_per_job' in pick_df.columns:
            total = int(pick_df['components_per_job'].sum())
        elif 'components_completed' in pick_df.columns and 'components_failed' in pick_df.columns:
            total = int(pick_df['components_completed'].sum() + pick_df['components_failed'].sum())
        else:
            total = 0
        st.markdown(f"**Total:** {total}")
    with col2:
        if 'components_completed' in pick_df.columns:
            completed = int(pick_df['components_completed'].sum())
        else:
            completed = 0
        st.markdown(f"**Completed:** {completed}")
    with col3:
        if 'components_failed' in pick_df.columns:
            failed = int(pick_df['components_failed'].sum())
        else:
            failed = 0
        st.markdown(f"**Failed:** {failed}")
    
    st.divider()
    
    # Row 4: Speed Metrics (if available)
    if theoretical_speed > 0 or actual_speed > 0:
        st.markdown("#### Speed Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Theoretical Speed</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #2ca02c;">{theoretical_speed:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">m/h</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.9em; color: #666; margin-bottom: 0.3rem;">Actual Speed</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #2ca02c;">{actual_speed:.1f} <span style="font-size: 0.6em; font-weight: normal; color: #999;">m/h</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.divider()
    
    # OEE Summary
    oee_color = get_color_for_oee(metrics.oee)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {oee_color}15 0%, {oee_color}05 100%);
            border-left: 4px solid {oee_color};
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        ">
            <h3 style="margin: 0; color: {oee_color};">
                Overall Equipment Effectiveness (OEE)
            </h3>
            <p style="font-size: 2.5em; font-weight: bold; margin: 0.5rem 0; color: {oee_color};">
                {metrics.oee * 100:.1f}%
            </p>
            <p style="margin: 0; color: #666; font-size: 0.9em;">
                Availability × Performance × Quality
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_cell_metrics_card(
    metrics: OEEMetrics,
    cell_type: str,
    key_prefix: Optional[str] = None
):
    """
    Display OEE breakdown with color coding for a manufacturing cell.
    DEPRECATED: Use render_printer_metrics_detailed, render_cut_metrics_detailed, or render_pick_metrics_detailed instead.

    Args:
        metrics: OEEMetrics object with OEE components
        cell_type: 'printer', 'cut', or 'pick'
        key_prefix: Optional unique key prefix for Streamlit widgets
    """
    if key_prefix is None:
        key_prefix = f"{cell_type}_metrics"

    cell_name = cell_type.capitalize()
    oee_pct = metrics.oee * 100
    color = get_color_for_oee(metrics.oee)

    # Create card with colored border
    st.markdown(
        f"""
        <div style="
            border-left: 5px solid {color};
            padding: 1rem;
            margin: 0.5rem 0;
            background-color: #f8f9fa;
            border-radius: 5px;
        ">
            <h3 style="margin-top: 0;">{cell_name} Cell</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Availability",
            f"{metrics.availability * 100:.1f}%",
            help="Uptime / (Uptime + Downtime)"
        )

    with col2:
        st.metric(
            "Performance",
            f"{metrics.performance * 100:.1f}%",
            help="Actual speed / Nominal speed (for printer)"
        )

    with col3:
        st.metric(
            "Quality",
            f"{metrics.quality * 100:.1f}%",
            help="Quality accuracy (for pick cell)"
        )

    with col4:
        st.metric(
            "OEE",
            f"{oee_pct:.1f}%",
            delta=None,
            help="Overall Equipment Effectiveness"
        )

    # OEE value display (removed progress bar as requested)
    # Progress bar removed - showing OEE value in metric above is sufficient


def render_reconciled_oee_comparison(
    real_time_oee: Optional[OEEMetrics],
    reconciled_oee: CellOEEReconciled,
    cell_type: str
):
    """
    Display a comparison between real-time OEE and reconciled OEE (post-QC).
    
    Args:
        real_time_oee: Real-time OEE metrics (before QC reconciliation), or None if not available
        reconciled_oee: Reconciled OEE metrics (after QC with defect attribution)
        cell_type: Cell name ('printer', 'cut', 'pick')
    """
    if real_time_oee:
        st.subheader(f"🔄 {cell_type.capitalize()} OEE: Real-Time vs Reconciled")
        
        # Create comparison table
        comparison_data = {
            'Metric': ['Availability', 'Performance', 'Quality', 'OEE'],
            'Real-Time': [
                f"{real_time_oee.availability * 100:.1f}%",
                f"{real_time_oee.performance * 100:.1f}%",
                f"{real_time_oee.quality * 100:.1f}%",
                f"{real_time_oee.oee * 100:.1f}%"
            ],
            'Reconciled (Post-QC)': [
                f"{reconciled_oee.availability * 100:.1f}%",
                f"{reconciled_oee.performance * 100:.1f}%",
                f"{reconciled_oee.quality_attributed * 100:.1f}%",
                f"{reconciled_oee.oee_reconciled * 100:.1f}%"
            ],
            'Difference': [
                f"{(reconciled_oee.availability - real_time_oee.availability) * 100:+.1f}%",
                f"{(reconciled_oee.performance - real_time_oee.performance) * 100:+.1f}%",
                f"{(reconciled_oee.quality_attributed - real_time_oee.quality) * 100:+.1f}%",
                f"{(reconciled_oee.oee_reconciled - real_time_oee.oee) * 100:+.1f}%"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.subheader(f"📊 {cell_type.capitalize()} OEE: Reconciled (Post-QC)")
        
        # Show only reconciled OEE
        oee_data = {
            'Metric': ['Availability', 'Performance', 'Quality (Attributed)', 'OEE (Reconciled)'],
            'Value': [
                f"{reconciled_oee.availability * 100:.1f}%",
                f"{reconciled_oee.performance * 100:.1f}%",
                f"{reconciled_oee.quality_attributed * 100:.1f}%",
                f"{reconciled_oee.oee_reconciled * 100:.1f}%"
            ]
        }
        
        oee_df = pd.DataFrame(oee_data)
        st.dataframe(oee_df, use_container_width=True, hide_index=True)
    
    # Display defect information
    st.info(
        f"**Defect Attribution:** {reconciled_oee.defects_attributed} defects attributed to {cell_type.capitalize()} "
        f"out of {reconciled_oee.total_processed} total components "
        f"({reconciled_oee.defect_rate_percent:.2f}% defect rate)"
    )


def render_batch_quality_breakdown(breakdown: BatchQualityBreakdown):
    """
    Display detailed quality breakdown with defect attribution for a batch.
    
    Args:
        breakdown: BatchQualityBreakdown object with quality data
    """
    st.subheader(f"📊 Quality Breakdown - Batch {breakdown.production_batch_id}")
    
    # Overall quality metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Components", breakdown.total_components)
    with col2:
        st.metric("Passed", breakdown.total_passed)
    with col3:
        st.metric("Failed", breakdown.total_failed)
    with col4:
        st.metric("Not Scanned", breakdown.not_scanned)
    
    # QC Source breakdown
    st.markdown("**QC Source Breakdown:**")
    source_data = {
        'Source': ['Handheld Scanner', 'Manual Entry', 'System/Inside'],
        'Count': [
            breakdown.scanned_handheld,
            breakdown.scanned_manual,
            breakdown.scanned_inside
        ]
    }
    source_df = pd.DataFrame(source_data)
    st.dataframe(source_df, use_container_width=True, hide_index=True)
    
    # Defect attribution breakdown
    st.markdown("**Defect Attribution by Category:**")
    defect_data = {
        'Category': [
            'Printer (Cell)', 
            'Cut (Cell)', 
            'Pick (Cell)', 
            'Fabric (Material)', 
            'File/Pre-production', 
            'Other',
            'Unattributed'
        ],
        'Defects': [
            breakdown.print_defects,
            breakdown.cut_defects,
            breakdown.pick_defects,
            breakdown.fabric_defects,
            breakdown.file_defects,
            breakdown.other_defects,
            breakdown.unattributed_defects
        ]
    }
    defect_df = pd.DataFrame(defect_data)
    defect_df['Percentage'] = (defect_df['Defects'] / breakdown.total_failed * 100).round(2) if breakdown.total_failed > 0 else 0
    st.dataframe(defect_df, use_container_width=True, hide_index=True)
    
    # Attribution coverage
    st.info(
        f"**Attribution Coverage:** {breakdown.attribution_coverage_percent:.1f}% of defects successfully attributed. "
        f"Cell-attributed: {breakdown.cell_attributed_defects}, Upstream: {breakdown.upstream_defects}, "
        f"Other: {breakdown.other_defects}, Unattributed: {breakdown.unattributed_defects}"
    )
    
    # Quality rates
    col1, col2, col3 = st.columns(3)
    with col1:
        if breakdown.quality_rate_checked_percent is not None:
            st.metric(
                "Quality Rate (Checked)", 
                f"{breakdown.quality_rate_checked_percent:.1f}%"
            )
    with col2:
        st.metric("Scan Coverage", f"{breakdown.scan_coverage_percent:.1f}%")
    with col3:
        st.metric("Unchecked Rate", f"{breakdown.unchecked_rate_percent:.1f}%")


def render_timeline_chart(
    print_df: Optional[pd.DataFrame] = None,
    cut_df: Optional[pd.DataFrame] = None,
    pick_df: Optional[pd.DataFrame] = None
):
    """
    Create a Gantt-style timeline chart showing job events across cells.

    Args:
        print_df: DataFrame with print job data (should have timing_start, timing_end columns)
        cut_df: DataFrame with cut job data (should have ts or timing_start, timing_end columns)
        pick_df: DataFrame with pick job data (should have ts or job_start, job_end columns)
    """
    try:
        fig = go.Figure()

        # Add print events
        if print_df is not None and not print_df.empty:
            if 'timing_start' in print_df.columns and 'timing_end' in print_df.columns:
                for idx, row in print_df.iterrows():
                    if pd.notna(row['timing_start']) and pd.notna(row['timing_end']):
                        fig.add_trace(go.Scatter(
                            x=[row['timing_start'], row['timing_end']],
                            y=['Print', 'Print'],
                            mode='lines+markers',
                            name='Print',
                            line=dict(width=8, color='#1f77b4'),
                            marker=dict(size=8),
                            showlegend=(idx == print_df.index[0])  # Only show legend for first
                        ))

        # Add cut events
        if cut_df is not None and not cut_df.empty:
            if 'ts' in cut_df.columns:
                for idx, row in cut_df.iterrows():
                    if pd.notna(row['ts']):
                        fig.add_trace(go.Scatter(
                            x=[row['ts']],
                            y=['Cut'],
                            mode='markers',
                            name='Cut',
                            marker=dict(size=10, symbol='diamond', color='#ff7f0e'),
                            showlegend=(idx == cut_df.index[0])
                        ))
            elif 'timing_start' in cut_df.columns and 'timing_end' in cut_df.columns:
                for idx, row in cut_df.iterrows():
                    if pd.notna(row['timing_start']) and pd.notna(row['timing_end']):
                        fig.add_trace(go.Scatter(
                            x=[row['timing_start'], row['timing_end']],
                            y=['Cut', 'Cut'],
                            mode='lines+markers',
                            name='Cut',
                            line=dict(width=8, color='#ff7f0e'),
                            marker=dict(size=8),
                            showlegend=(idx == cut_df.index[0])
                        ))

        # Add pick events
        if pick_df is not None and not pick_df.empty:
            if 'job_start' in pick_df.columns and 'job_end' in pick_df.columns:
                for idx, row in pick_df.iterrows():
                    if pd.notna(row['job_start']) and pd.notna(row['job_end']):
                        fig.add_trace(go.Scatter(
                            x=[row['job_start'], row['job_end']],
                            y=['Pick', 'Pick'],
                            mode='lines+markers',
                            name='Pick',
                            line=dict(width=8, color='#2ca02c'),
                            marker=dict(size=8),
                            showlegend=(idx == pick_df.index[0])
                        ))
            elif 'ts' in pick_df.columns:
                for idx, row in pick_df.iterrows():
                    if pd.notna(row['ts']):
                        fig.add_trace(go.Scatter(
                            x=[row['ts']],
                            y=['Pick'],
                            mode='markers',
                            name='Pick',
                            marker=dict(size=10, symbol='square', color='#2ca02c'),
                            showlegend=(idx == pick_df.index[0])
                        ))

        if len(fig.data) == 0:
            st.info("No timeline data available to display.")
            return

        fig.update_layout(
            title="Manufacturing Timeline - Print, Cut, and Pick Events",
            xaxis_title="Time",
            yaxis_title="Process",
            height=400,
            showlegend=True,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)
        logger.info("Successfully created timeline chart")

    except Exception as e:
        logger.error(f"Error creating timeline chart: {e}", exc_info=True)
        st.error(f"Error creating timeline visualization: {e}")

