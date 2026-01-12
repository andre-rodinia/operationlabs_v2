"""
Configuration Management
Loads and validates environment variables
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""

    # HISTORIAN Database
    HISTORIAN_HOST = os.getenv("HISTORIANDB_HOST")
    HISTORIAN_PORT = int(os.getenv("HISTORIANDB_PORT", 5432))
    HISTORIAN_DB = os.getenv("HISTORIANDB_NAME")
    HISTORIAN_USER = os.getenv("HISTORIANDB_USER")
    HISTORIAN_PASS = os.getenv("HISTORIANDB_PASS")

    # INSIDE Database
    INSIDE_HOST = os.getenv("INSIDEDB_HOST")
    INSIDE_PORT = int(os.getenv("INSIDEDB_PORT", 5432))
    INSIDE_DB = os.getenv("INSIDEDB_NAME")
    INSIDE_USER = os.getenv("INSIDEDB_USER")
    INSIDE_PASS = os.getenv("INSIDEDB_PASS")

    # Application Settings
    TIMEZONE = os.getenv("TIMEZONE", "Europe/Copenhagen")
    DEFAULT_OPERATING_HOURS = float(os.getenv("DEFAULT_OPERATING_HOURS", 24))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # MQTT Topics
    TOPICS = {
        'print': 'rg_v2/RG/CPH/Prod/ComponentLine/Print1/JobReport',
        'cut': 'rg_v2/RG/CPH/Prod/ComponentLine/Cut1/JobReport',
        'pick': 'rg_v2/RG/CPH/Prod/ComponentLine/Pick1/JobReport'
    }

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required = [
            'HISTORIAN_HOST', 'HISTORIAN_DB', 'HISTORIAN_USER', 'HISTORIAN_PASS',
            'INSIDE_HOST', 'INSIDE_DB', 'INSIDE_USER', 'INSIDE_PASS'
        ]

        missing = [field for field in required if not getattr(cls, field)]

        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

        return True

# Validate on import
Config.validate()
