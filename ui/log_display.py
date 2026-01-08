"""
Log Display UI Component

Provides a centralized log area to display all processing messages and status updates.
"""

import streamlit as st
from typing import List, Dict
from datetime import datetime


class LogCollector:
    """Collects log messages for display in a dedicated log area."""
    
    def __init__(self, session_key: str = "app_logs"):
        self.session_key = session_key
        if session_key not in st.session_state:
            st.session_state[session_key] = []
    
    def add_info(self, message: str, icon: str = "ℹ️"):
        """Add an informational message."""
        self._add_log("info", message, icon)
    
    def add_success(self, message: str, icon: str = "✅"):
        """Add a success message."""
        self._add_log("success", message, icon)
    
    def add_warning(self, message: str, icon: str = "⚠️"):
        """Add a warning message."""
        self._add_log("warning", message, icon)
    
    def add_error(self, message: str, icon: str = "❌"):
        """Add an error message."""
        self._add_log("error", message, icon)
    
    def _add_log(self, level: str, message: str, icon: str):
        """Internal method to add log entry."""
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
            "icon": icon
        }
        st.session_state[self.session_key].append(log_entry)
    
    def clear(self):
        """Clear all logs."""
        st.session_state[self.session_key] = []
    
    def get_logs(self) -> List[Dict]:
        """Get all log entries."""
        return st.session_state.get(self.session_key, [])


def render_log_area(log_collector: LogCollector, max_height: int = 400):
    """
    Render a scrollable log area displaying all collected messages.
    
    Args:
        log_collector: LogCollector instance with messages
        max_height: Maximum height of the log area in pixels
    """
    logs = log_collector.get_logs()
    
    if not logs:
        return
    
    st.subheader("📋 Processing Log")
    
    # Create a scrollable container for logs
    log_container = st.container()
    
    with log_container:
        # Display logs in reverse order (newest first) or chronological order
        for log in logs:
            timestamp = log.get("timestamp", "")
            level = log.get("level", "info")
            message = log.get("message", "")
            icon = log.get("icon", "")
            
            # Choose color based on level
            if level == "success":
                st.success(f"[{timestamp}] {icon} {message}")
            elif level == "warning":
                st.warning(f"[{timestamp}] {icon} {message}")
            elif level == "error":
                st.error(f"[{timestamp}] {icon} {message}")
            else:
                st.info(f"[{timestamp}] {icon} {message}")


def render_compact_log_area(log_collector: LogCollector):
    """
    Render a compact, collapsible log area.
    
    Args:
        log_collector: LogCollector instance with messages
    """
    logs = log_collector.get_logs()
    
    if not logs:
        return
    
    with st.expander(f"📋 View Processing Log ({len(logs)} messages)", expanded=False):
        # Show logs in reverse chronological order (newest at top)
        for log in reversed(logs):
            timestamp = log.get("timestamp", "")
            level = log.get("level", "info")
            message = log.get("message", "")
            icon = log.get("icon", "")
            
            # Use markdown for compact display
            color_map = {
                "success": "🟢",
                "warning": "🟡",
                "error": "🔴",
                "info": "🔵"
            }
            
            color_icon = color_map.get(level, "⚪")
            st.markdown(f"{color_icon} `[{timestamp}]` {icon} {message}")
        
        # Clear log button
        if st.button("Clear Log", key="clear_log_button"):
            log_collector.clear()
            st.rerun()

