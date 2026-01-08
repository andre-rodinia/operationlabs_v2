"""
Configuration Management

Simple utility for loading and validating environment configuration.
"""

import os
from typing import Optional
from dotenv import load_dotenv


def load_config(env_path: Optional[str] = None) -> bool:
    """
    Load environment configuration from .env file.

    Args:
        env_path: Optional path to .env file. If None, searches in current directory.

    Returns:
        bool: True if .env file was found and loaded, False otherwise
    """
    if env_path:
        return load_dotenv(env_path)
    return load_dotenv()


def get_database_config(database_type: str) -> dict:
    """
    Get database configuration for specified type.

    Args:
        database_type: "historian" or "inside"

    Returns:
        dict: Database configuration

    Raises:
        ValueError: If required configuration is missing
    """
    if database_type == "historian":
        config = {
            "host": os.getenv("HISTORIANDB_HOST"),
            "port": os.getenv("HISTORIANDB_PORT"),
            "database": os.getenv("HISTORIANDB_NAME"),
            "user": os.getenv("HISTORIANDB_USER"),
            "password": os.getenv("HISTORIANDB_PASS"),
        }
    elif database_type == "inside":
        config = {
            "host": os.getenv("INSIDEDB_HOST"),
            "port": os.getenv("INSIDEDB_PORT"),
            "database": os.getenv("INSIDEDB_NAME"),
            "user": os.getenv("INSIDEDB_USER"),
            "password": os.getenv("INSIDEDB_PASS"),
        }
    else:
        raise ValueError(f"Unknown database type: {database_type}")

    # Validate
    missing = [k for k, v in config.items() if not v]
    if missing:
        raise ValueError(
            f"Missing {database_type} database configuration: {missing}. "
            f"Please check your .env file."
        )

    return config


def get_app_config() -> dict:
    """
    Get application configuration settings.

    Returns:
        dict: Application settings
    """
    return {
        "timezone": os.getenv("TIMEZONE", "Europe/Copenhagen"),
        "default_time_window_hours": int(os.getenv("DEFAULT_TIME_WINDOW_HOURS", "8")),
        "cache_ttl_seconds": int(os.getenv("CACHE_TTL_SECONDS", "300")),
    }


def validate_config() -> list:
    """
    Validate all required configuration is present.

    Returns:
        list: List of missing configuration items (empty if all valid)
    """
    missing = []

    # Check HISTORIAN config
    try:
        get_database_config("historian")
    except ValueError as e:
        missing.append(f"HISTORIAN: {str(e)}")

    # Check INSIDE config
    try:
        get_database_config("inside")
    except ValueError as e:
        missing.append(f"INSIDE: {str(e)}")

    return missing
