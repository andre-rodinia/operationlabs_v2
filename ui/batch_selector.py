"""
Batch Selection UI Component

Provides UI components for batch selection in Streamlit.
Supports text input or multi-select dropdown for batch IDs.
"""

import streamlit as st
import pandas as pd
import logging
from typing import List, Optional, Tuple

from core.db.fetchers import fetch_production_printjob_data

logger = logging.getLogger(__name__)


def fetch_available_batches() -> List[str]:
    """
    Fetch list of available batches from production_printjob table.

    Returns:
        List of batch IDs (as strings)
    """
    try:
        production_df = fetch_production_printjob_data()
        if not production_df.empty and 'production_batch_id' in production_df.columns:
            batches = production_df['production_batch_id'].dropna().unique().tolist()
            batches = sorted([str(b) for b in batches])
            logger.info(f"Found {len(batches)} available batches")
            return batches
        else:
            logger.warning("No batches found in production_printjob table")
            return []
    except Exception as e:
        logger.error(f"Error fetching available batches: {e}")
        st.error(f"Error fetching batch list: {e}")
        return []


def validate_batch_selection(batch_ids: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate that selected batches exist in the database.

    Args:
        batch_ids: List of batch IDs to validate

    Returns:
        Tuple of (valid_batches, invalid_batches)
    """
    if not batch_ids:
        return [], []

    try:
        # Fetch all available batches
        available_batches = fetch_available_batches()
        available_batches_set = set(str(b) for b in available_batches)

        valid_batches = []
        invalid_batches = []

        for batch_id in batch_ids:
            batch_str = str(batch_id).strip()
            # Remove 'B' prefix if present for comparison
            clean_batch = batch_str.lstrip("B")
            
            if clean_batch in available_batches_set or batch_str in available_batches_set:
                valid_batches.append(batch_str)
            else:
                invalid_batches.append(batch_str)

        return valid_batches, invalid_batches

    except Exception as e:
        logger.error(f"Error validating batch selection: {e}")
        # If validation fails, assume all are invalid
        return [], batch_ids


def render_batch_input(key_prefix: str = "batch_input") -> List[str]:
    """
    Render UI for batch input with two modes:
    1. Text input (comma-separated batch IDs)
    2. Multi-select dropdown

    Args:
        key_prefix: Unique key prefix for Streamlit widgets

    Returns:
        List of selected batch IDs
    """
    st.subheader("🎯 Batch Selection")

    # Input mode selection
    input_mode = st.radio(
        "Input method:",
        options=["Text Input", "Dropdown Selection"],
        index=0,
        key=f"{key_prefix}_mode",
        help="Choose how to enter batch IDs:\n"
             "- Text Input: Enter comma-separated batch IDs\n"
             "- Dropdown Selection: Select from available batches"
    )

    selected_batches = []

    if input_mode == "Text Input":
        # Text input for batch IDs
        batch_text = st.text_input(
            "Enter batch IDs (comma-separated):",
            key=f"{key_prefix}_text",
            help="Enter batch IDs separated by commas, e.g., '123, 456, 789'"
        )

        if batch_text:
            # Parse comma-separated values
            batch_list = [b.strip() for b in batch_text.split(',') if b.strip()]
            selected_batches = batch_list

            if selected_batches:
                st.info(f"📝 Entered {len(selected_batches)} batch ID(s)")

    else:
        # Dropdown multi-select
        with st.spinner("Loading available batches..."):
            available_batches = fetch_available_batches()

        if not available_batches:
            st.warning("⚠️ No batches found in production_printjob table")
            return []

        st.info(f"📊 Found {len(available_batches)} batches in the system")

        selected_batches = st.multiselect(
            "Select batches to analyze:",
            options=available_batches,
            default=[],
            key=f"{key_prefix}_dropdown",
            help="Choose one or more batches for analysis"
        )

    # Validate selection if batches are provided
    if selected_batches:
        valid_batches, invalid_batches = validate_batch_selection(selected_batches)

        if invalid_batches:
            st.error(f"❌ Invalid batch IDs: {', '.join(invalid_batches)}")
            st.warning("These batches were not found in the database and will be excluded.")

        if valid_batches:
            st.success(f"✅ Validated {len(valid_batches)} batch ID(s)")
            return valid_batches
        else:
            st.warning("⚠️ No valid batches selected")
            return []

    return selected_batches

