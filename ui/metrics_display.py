"""
Metrics Display Functions

UI components for displaying equipment state timelines and throughput metrics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def display_cell_timeline(
    cell_name: str,
    hourly_df: pd.DataFrame,
    summary: Dict[str, float],
    start_ts: datetime,
    end_ts: datetime
):
    """
    Display one cell's equipment state timeline.
    
    Shows:
    - Four metrics at top: time window, running percent, idle percent, fault percent
    - Stacked bar chart with hours on x-axis and percentages on y-axis
    - Colors: green for running, yellow for idle, red for fault, gray for other
    - List of problematic hours with high idle or fault
    
    Args:
        cell_name: Cell name (e.g., "Print1", "Cut1", "Pick1")
        hourly_df: DataFrame from calculate_hourly_state_breakdown
        summary: Dictionary from calculate_state_summary
        start_ts: Start timestamp
        end_ts: End timestamp
    """
    st.subheader(f"📊 {cell_name} Equipment State Timeline")
    
    # Display metrics at top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        time_window_str = f"{start_ts.strftime('%H:%M')} - {end_ts.strftime('%H:%M')}"
        st.metric("Time Window", time_window_str)
    
    with col2:
        st.metric("Running", f"{summary['running_percent']:.1f}%")
    
    with col3:
        st.metric("Idle", f"{summary['idle_percent']:.1f}%")
    
    with col4:
        st.metric("Fault", f"{summary['fault_percent']:.1f}%")
    
    # Create stacked bar chart
    if not hourly_df.empty:
        fig = go.Figure()
        
        # Use datetime values directly for x-axis (Plotly handles this better)
        x_values = hourly_df['hour_start']
        
        # Add stacked bars
        fig.add_trace(go.Bar(
            x=x_values,
            y=hourly_df['running_percent'],
            name='Running',
            marker_color='#28a745',  # Green
            hovertemplate='%{x|%H:%M}<br>%{y:.1f}% Running<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            x=x_values,
            y=hourly_df['idle_percent'],
            name='Idle',
            marker_color='#ffc107',  # Yellow
            hovertemplate='%{x|%H:%M}<br>%{y:.1f}% Idle<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            x=x_values,
            y=hourly_df['fault_percent'],
            name='Fault',
            marker_color='#dc3545',  # Red
            hovertemplate='%{x|%H:%M}<br>%{y:.1f}% Fault<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            x=x_values,
            y=hourly_df['other_percent'],
            name='Other',
            marker_color='#6c757d',  # Gray
            hovertemplate='%{x|%H:%M}<br>%{y:.1f}% Other<extra></extra>'
        ))
        
        # Set x-axis range to match the actual time window
        x_min = hourly_df['hour_start'].min()
        x_max = hourly_df['hour_end'].max()
        
        fig.update_layout(
            barmode='stack',
            xaxis_title='Hour',
            yaxis_title='Percentage (%)',
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                range=[x_min, x_max],
                type='date',
                tickformat='%H:%M',
                dtick=3600000  # 1 hour in milliseconds
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # List problematic hours
        problematic_hours = hourly_df[
            (hourly_df['idle_percent'] > 30) | (hourly_df['fault_percent'] > 5)
        ]
        
        if not problematic_hours.empty:
            st.warning("⚠️ **Problematic Hours Detected:**")
            for _, row in problematic_hours.iterrows():
                hour_str = row['hour_start'].strftime('%H:%M')
                issues = []
                if row['idle_percent'] > 30:
                    issues.append(f"High Idle: {row['idle_percent']:.1f}%")
                if row['fault_percent'] > 5:
                    issues.append(f"High Fault: {row['fault_percent']:.1f}%")
                st.text(f"  • {hour_str}: {', '.join(issues)}")
    else:
        st.info("No hourly data available for this cell")


def display_all_cell_timelines(
    cell_data: Dict[str, Dict]
):
    """
    Loop through Print, Cut, Pick and display timeline for each.
    
    Args:
        cell_data: Dictionary with keys 'Print1', 'Cut1', 'Pick1', each containing:
                   - 'hourly_df': DataFrame from calculate_hourly_state_breakdown
                   - 'summary': Dictionary from calculate_state_summary
                   - 'start_ts': Start timestamp
                   - 'end_ts': End timestamp
    """
    cells = ['Print1', 'Cut1', 'Pick1']
    
    for i, cell_name in enumerate(cells):
        if cell_name in cell_data and cell_data[cell_name] is not None:
            display_cell_timeline(
                cell_name=cell_name,
                hourly_df=cell_data[cell_name]['hourly_df'],
                summary=cell_data[cell_name]['summary'],
                start_ts=cell_data[cell_name]['start_ts'],
                end_ts=cell_data[cell_name]['end_ts']
            )
            
            # Add divider between cells (except after last one)
            if i < len(cells) - 1:
                st.divider()


def display_throughput_metrics(throughput_results: Dict):
    """
    Display system-level throughput metrics and cell-level breakdown.
    
    Shows:
    - System metrics in three columns: raw output, sellable output, efficiency
    - Table with one row per cell listing active throughput, calendar throughput, utilization
    - Bottleneck identification
    - Improvement opportunity identification
    
    Args:
        throughput_results: Dictionary from calculate_system_throughput
    """
    st.subheader("📈 Throughput Analysis")
    
    system_metrics = throughput_results['system_metrics']
    cell_throughputs = throughput_results['cell_throughputs']
    
    # System-level metrics in three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Raw Output**")
        st.caption("Productive throughput (excl. breaks)")
        if system_metrics['system_components_per_hour']:
            st.metric("Components/Hour", f"{system_metrics['system_components_per_hour']:.1f}")
        if system_metrics['system_garments_per_hour']:
            st.metric("Garments/Hour", f"{system_metrics['system_garments_per_hour']:.1f}")

    with col2:
        st.markdown("**Sellable Output**")
        st.caption("Quality-adjusted throughput")
        if system_metrics['sellable_components_per_hour']:
            st.metric("Components/Hour", f"{system_metrics['sellable_components_per_hour']:.1f}")
        if system_metrics['sellable_garments_per_hour']:
            st.metric("Garments/Hour", f"{system_metrics['sellable_garments_per_hour']:.1f}")
        if not system_metrics['sellable_components_per_hour']:
            st.info("QC data not available")

    with col3:
        st.markdown("**Efficiency**")
        # Calculate quality rate from garments directly (not estimated components)
        if system_metrics['sellable_garments'] and system_metrics['total_garments']:
            quality_rate = (system_metrics['sellable_garments'] / system_metrics['total_garments']) * 100
            st.metric("Quality Rate", f"{quality_rate:.1f}%")
        elif system_metrics['sellable_components'] and system_metrics['total_components']:
            # Fallback to components if garments not available
            quality_rate = (system_metrics['sellable_components'] / system_metrics['total_components']) * 100
            st.metric("Quality Rate", f"{quality_rate:.1f}%")

        # Show productive time (excluding breaks)
        if system_metrics.get('system_productive_hours'):
            productive_h = system_metrics['system_productive_hours']
            break_h = system_metrics.get('system_break_hours', 0.0)
            if break_h > 0:
                st.metric("System Time", f"{productive_h:.1f}h", delta=f"-{break_h:.1f}h breaks", delta_color="off")
            else:
                st.metric("System Time", f"{productive_h:.1f}h")
    
    st.divider()
    
    # Cell-level throughput table
    if cell_throughputs:
        throughput_data = []
        for cell_name, throughput in cell_throughputs.items():
            row = {
                'Cell': cell_name,
                'Active Throughput (comp/hr)': f"{throughput['active_throughput_components_per_hour']:.1f}",
                'Productive Throughput (comp/hr)': f"{throughput.get('productive_throughput_components_per_hour', 0):.1f}",
                'Utilization': f"{throughput.get('productive_utilization', throughput['utilization']):.1%}"
            }

            # Add blocked hours if available (for future tandem tracking)
            if throughput.get('blocked_hours', 0) > 0:
                row['Blocked Time (h)'] = f"{throughput['blocked_hours']:.1f}"

            throughput_data.append(row)

        throughput_df = pd.DataFrame(throughput_data)
        st.dataframe(throughput_df, use_container_width=True, hide_index=True)

        # Add note about current state tracking limitations
        if any(tp.get('blocked_hours', 0) == 0 for tp in cell_throughputs.values()):
            st.caption("ℹ️ **Note**: Pick's active throughput currently includes waiting time due to tandem operation with Cut. Future state tracking improvements will separate actual picking time from blocked/waiting time.")
        
        # Bottleneck and improvement opportunity
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if throughput_results['bottleneck']:
                st.warning(f"⚠️ **Bottleneck:** {throughput_results['bottleneck']} has the lowest active throughput")
            else:
                st.info("ℹ️ No bottleneck identified")
        
        with col2:
            if throughput_results['improvement_opportunity']:
                st.info(f"💡 **Improvement Opportunity:** {throughput_results['improvement_opportunity']} has the lowest utilization")
            else:
                st.info("ℹ️ No improvement opportunity identified")
    else:
        st.warning("No throughput data available")
