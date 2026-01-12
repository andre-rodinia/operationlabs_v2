# Fresh Implementation Guide: Batch-Centric OEE Analysis System

## Overview

This guide provides a complete, from-scratch implementation of a batch-centric OEE (Overall Equipment Effectiveness) analysis system for manufacturing operations. The system handles three production cells (Print, Cut, Pick) with proper quality overlay support and three-level OEE metrics.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Database Schema](#database-schema)
3. [Core Concepts](#core-concepts)
4. [File Structure](#file-structure)
5. [Implementation Steps](#implementation-steps)
6. [Complete Code](#complete-code)
7. [Testing Strategy](#testing-strategy)
8. [Edge Cases & Handling](#edge-cases--handling)

---

## System Architecture

### High-Level Flow

```
User selects day → Discover batches → Select batches → Fetch JobReports → Calculate OEE → Apply QC overlay → Display metrics
```

### Data Sources

1. **HISTORIAN Database**
   - Table: `jobs`
   - Contains MQTT JobReport messages as JSON payloads
   - Topics:
     - `rg_v2/RG/CPH/Prod/ComponentLine/Print1/JobReport`
     - `rg_v2/RG/CPH/Prod/ComponentLine/Cut1/JobReport`
     - `rg_v2/RG/CPH/Prod/ComponentLine/Pick1/JobReport`

2. **INSIDE Database**
   - Table: `production_printjob` - Maps job IDs to batch IDs
   - Table: `production_componentorder` - Component orders per batch
   - Table: `production_componentorderqc` - QC inspection results

### Key Design Decisions

1. **Batch-Centric Approach**: Use batches as the primary unit of analysis (not time windows)
2. **JobReport as Source**: Extract all metrics directly from JobReport payloads
3. **Quality Timing Strategy**:
   - Day 0 (production day): Assume 100% quality (optimistic)
   - Day 3+ (QC available): Apply actual QC inspection results (realistic)
4. **Pick Cell Logic**:
   - Performance = 100% (robots operate at consistent nominal speed)
   - Quality = successful picks / total attempts
5. **Three-Level OEE**:
   - Job-level: Individual job OEE
   - Batch-level: Aggregated batch OEE
   - Daily-level: Batch OEE × Utilization

---

## Database Schema

### HISTORIAN Database

#### jobs table
```sql
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    time TIMESTAMP NOT NULL,
    topic VARCHAR(255) NOT NULL,
    payload JSONB NOT NULL
);

-- Indexes for performance
CREATE INDEX idx_jobs_topic_time ON jobs(topic, time);
CREATE INDEX idx_jobs_payload_jobid ON jobs((payload->>'jobId'));
```

#### JobReport Payload Structure

**Print1/JobReport:**
```json
{
  "jobId": "job_12345",
  "batchId": "batch_001",
  "sheetIndex": 1,
  "reportData": {
    "uptime": 3600,
    "downtime": 300,
    "speed": {
      "actual": 45.5,
      "nominal": 50.0
    }
  },
  "closingReport": {
    "time": "2025-01-10T14:30:00Z",
    "totalArea": {
      "value": 12.5
    }
  }
}
```

**Cut1/JobReport:**
```json
{
  "jobId": "job_12345",
  "batchId": "batch_001",
  "sheetIndex": 1,
  "reportData": {
    "uptime": 3500,
    "downtime": 400,
    "speed": {
      "actual": 42.0,
      "nominal": 48.0
    }
  },
  "closingReport": {
    "time": "2025-01-10T15:00:00Z",
    "components": {
      "count": 150
    }
  }
}
```

**Pick1/JobReport:**
```json
{
  "jobId": "job_12345",
  "batchId": "batch_001",
  "sheetIndex": 1,
  "reportData": {
    "uptime": 3400,
    "downtime": 500
  },
  "closingReport": {
    "time": "2025-01-10T15:30:00Z",
    "components": {
      "completed": 145,
      "failed": 5
    },
    "averagePickTime": {
      "value": 2.3
    }
  }
}
```

### INSIDE Database

#### production_printjob table
```sql
CREATE TABLE production_printjob (
    id SERIAL PRIMARY KEY,
    rg_id VARCHAR(255) UNIQUE NOT NULL,
    production_batch_id VARCHAR(255),
    status VARCHAR(50),
    component_order_count INTEGER,
    index INTEGER
);

CREATE INDEX idx_printjob_batch ON production_printjob(production_batch_id);
CREATE INDEX idx_printjob_rgid ON production_printjob(rg_id);
```

#### production_componentorder table
```sql
CREATE TABLE production_componentorder (
    id SERIAL PRIMARY KEY,
    production_batch_id VARCHAR(255),
    component_name VARCHAR(255),
    quantity INTEGER
);
```

#### production_componentorderqc table
```sql
CREATE TABLE production_componentorderqc (
    id SERIAL PRIMARY KEY,
    component_order_id INTEGER REFERENCES production_componentorder(id),
    result VARCHAR(50),  -- 'PASS' or 'FAIL'
    defect_type VARCHAR(255),
    inspected_at TIMESTAMP
);
```

---

## Core Concepts

### OEE Formula

```
OEE = Availability × Performance × Quality
```

**Availability** = Uptime / (Uptime + Downtime) × 100%
- Measures equipment uptime vs total time
- Includes planned downtime

**Performance** = Actual Speed / Nominal Speed × 100%
- Print/Cut: From reportData.speed.actual / reportData.speed.nominal
- Pick: Always 100% (robots operate at consistent speed)

**Quality** = Good Units / Total Units × 100%
- Day 0 (no QC): 100% assumption for Print/Cut, Pick success rate for Pick
- Day 3+ (QC available): QC pass rate for Print/Cut, Pick success × QC pass for Pick

### Three-Level OEE

**1. Job-Level OEE**
```
OEE_job = Availability_job × Performance_job × Quality_job
```

**2. Batch-Level OEE**
```
OEE_batch = Average(OEE_job for all jobs in batch)
```

**3. Daily-Level OEE**
```
Utilization = Production_Time / Available_Time × 100%
Daily_OEE = Batch_OEE × Utilization / 100
```

### Quality Overlay Strategy

**Day 0 (Production Day - No QC Data):**
- Print: Quality = 100%
- Cut: Quality = 100%
- Pick: Quality = components_completed / (components_completed + components_failed) × 100%

**Day 3+ (QC Data Available):**
- Print: Quality = QC_pass_rate
- Cut: Quality = QC_pass_rate
- Pick: Quality = (pick_success_rate × QC_pass_rate) / 100

**Rationale**: Pick quality is compound - component must be successfully picked AND pass QC inspection.

---

## File Structure

```
operationlabs_v2/
├── .env                          # Environment variables
├── config.py                     # Configuration management
├── db/
│   ├── __init__.py
│   ├── connections.py            # Database connection pooling
│   └── fetchers.py               # All data fetching functions
├── analysis/
│   ├── __init__.py
│   ├── oee_calculator.py         # OEE calculation logic
│   └── quality_overlay.py        # QC data overlay logic
├── ui/
│   ├── __init__.py
│   └── app.py                    # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── logging_config.py         # Logging setup
│   └── timezone_utils.py         # Timezone handling
└── requirements.txt              # Python dependencies
```

---

## Implementation Steps

### Step 1: Environment Setup

**Create `.env` file:**
```env
# HISTORIAN Database
HISTORIANDB_HOST=your_host
HISTORIANDB_PORT=5432
HISTORIANDB_NAME=historian
HISTORIANDB_USER=your_user
HISTORIANDB_PASS=your_password

# INSIDE Database
INSIDEDB_HOST=your_host
INSIDEDB_PORT=5432
INSIDEDB_NAME=inside
INSIDEDB_USER=your_user
INSIDEDB_PASS=your_password

# Application Settings
TIMEZONE=Europe/Copenhagen
DEFAULT_OPERATING_HOURS=24
LOG_LEVEL=INFO
```

**Create `requirements.txt`:**
```txt
streamlit==1.32.0
pandas==2.1.4
numpy==1.26.3
psycopg2-binary==2.9.9
python-dotenv==1.0.1
pytz==2024.1
plotly==5.18.0
```

### Step 2: Configuration Management

**Create `config.py`:**
```python
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
```

### Step 3: Database Connection Pool

**Create `db/connections.py`:**
```python
"""
Database Connection Pool
Manages connections to HISTORIAN and INSIDE databases
"""
import psycopg2
from psycopg2 import pool
import logging
from contextlib import contextmanager
from config import Config

logger = logging.getLogger(__name__)

# Connection pools (initialized lazily)
_historian_pool = None
_inside_pool = None

def get_historian_pool():
    """Get or create HISTORIAN connection pool"""
    global _historian_pool

    if _historian_pool is None:
        try:
            _historian_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=Config.HISTORIAN_HOST,
                port=Config.HISTORIAN_PORT,
                database=Config.HISTORIAN_DB,
                user=Config.HISTORIAN_USER,
                password=Config.HISTORIAN_PASS
            )
            logger.info("HISTORIAN connection pool created")
        except Exception as e:
            logger.error(f"Failed to create HISTORIAN pool: {e}")
            raise

    return _historian_pool

def get_inside_pool():
    """Get or create INSIDE connection pool"""
    global _inside_pool

    if _inside_pool is None:
        try:
            _inside_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=Config.INSIDE_HOST,
                port=Config.INSIDE_PORT,
                database=Config.INSIDE_DB,
                user=Config.INSIDE_USER,
                password=Config.INSIDE_PASS
            )
            logger.info("INSIDE connection pool created")
        except Exception as e:
            logger.error(f"Failed to create INSIDE pool: {e}")
            raise

    return _inside_pool

@contextmanager
def get_historian_connection():
    """Context manager for HISTORIAN database connections"""
    pool = get_historian_pool()
    conn = None

    try:
        conn = pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"HISTORIAN connection error: {e}")
        raise
    finally:
        if conn:
            pool.putconn(conn)

@contextmanager
def get_inside_connection():
    """Context manager for INSIDE database connections"""
    pool = get_inside_pool()
    conn = None

    try:
        conn = pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"INSIDE connection error: {e}")
        raise
    finally:
        if conn:
            pool.putconn(conn)

def close_all_pools():
    """Close all connection pools (call on application shutdown)"""
    global _historian_pool, _inside_pool

    if _historian_pool:
        _historian_pool.closeall()
        _historian_pool = None
        logger.info("HISTORIAN pool closed")

    if _inside_pool:
        _inside_pool.closeall()
        _inside_pool = None
        logger.info("INSIDE pool closed")
```

### Step 4: Data Fetchers

**Create `db/fetchers.py`:**
```python
"""
Data Fetchers
All database query functions for batch discovery and JobReport fetching
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from db.connections import get_historian_connection, get_inside_connection
from config import Config

logger = logging.getLogger(__name__)

# ============================================================================
# BATCH DISCOVERY
# ============================================================================

def fetch_batches_for_day(day_start: str, day_end: str) -> pd.DataFrame:
    """
    Discover batches produced on a specific day.

    Strategy:
    1. Query HISTORIAN for Print1 JobReports in time window
    2. Extract job IDs from JobReports
    3. Map job IDs to batch IDs via INSIDE database
    4. Group by batch_id with aggregated metadata

    Args:
        day_start: ISO format timestamp (start of day)
        day_end: ISO format timestamp (end of day)

    Returns:
        DataFrame with columns:
        - batch_id: Batch identifier
        - job_count: Number of jobs in batch
        - first_job_time: Earliest job timestamp
        - last_job_time: Latest job timestamp

    Edge Cases:
    - No JobReports found: Returns empty DataFrame
    - Jobs without batch mapping: Excluded from results
    - Multiple batches in same day: All returned
    """
    try:
        # Step 1: Query HISTORIAN for Print JobReports
        with get_historian_connection() as historian_conn:
            cursor = historian_conn.cursor()

            query = """
                SELECT
                    payload->>'jobId' as job_id,
                    time as job_time
                FROM jobs
                WHERE topic = %s
                AND time >= %s::timestamp
                AND time < %s::timestamp
                AND payload->>'jobId' IS NOT NULL
                ORDER BY time ASC
            """

            cursor.execute(query, (Config.TOPICS['print'], day_start, day_end))
            job_records = cursor.fetchall()
            cursor.close()

        if not job_records:
            logger.info(f"No JobReports found for {day_start} to {day_end}")
            return pd.DataFrame(columns=['batch_id', 'job_count', 'first_job_time', 'last_job_time'])

        # Extract job IDs and timestamps
        job_ids = [row[0] for row in job_records]
        job_times = {row[0]: row[1] for row in job_records}

        logger.info(f"Found {len(job_ids)} Print JobReports")

        # Step 2: Map job IDs to batch IDs via INSIDE database
        with get_inside_connection() as inside_conn:
            cursor = inside_conn.cursor()

            placeholders = ','.join(['%s'] * len(job_ids))
            query = f"""
                SELECT
                    rg_id as job_id,
                    production_batch_id as batch_id
                FROM production_printjob
                WHERE rg_id IN ({placeholders})
                AND production_batch_id IS NOT NULL
            """

            cursor.execute(query, job_ids)
            mapping_records = cursor.fetchall()
            cursor.close()

        if not mapping_records:
            logger.warning("No batch mappings found for job IDs")
            return pd.DataFrame(columns=['batch_id', 'job_count', 'first_job_time', 'last_job_time'])

        # Step 3: Build batch metadata
        batch_metadata = {}
        for job_id, batch_id in mapping_records:
            if batch_id not in batch_metadata:
                batch_metadata[batch_id] = {
                    'job_ids': [],
                    'job_times': []
                }
            batch_metadata[batch_id]['job_ids'].append(job_id)
            if job_id in job_times:
                batch_metadata[batch_id]['job_times'].append(job_times[job_id])

        # Step 4: Aggregate into DataFrame
        batch_rows = []
        for batch_id, metadata in batch_metadata.items():
            batch_rows.append({
                'batch_id': batch_id,
                'job_count': len(metadata['job_ids']),
                'first_job_time': min(metadata['job_times']) if metadata['job_times'] else None,
                'last_job_time': max(metadata['job_times']) if metadata['job_times'] else None
            })

        batches_df = pd.DataFrame(batch_rows)
        batches_df = batches_df.sort_values('first_job_time')

        logger.info(f"Discovered {len(batches_df)} batches")
        return batches_df

    except Exception as e:
        logger.error(f"Error in fetch_batches_for_day: {e}", exc_info=True)
        return pd.DataFrame(columns=['batch_id', 'job_count', 'first_job_time', 'last_job_time'])

def fetch_job_ids_for_batch(batch_ids: List[str]) -> List[str]:
    """
    Map batch IDs to job IDs via INSIDE database.

    Args:
        batch_ids: List of batch identifiers

    Returns:
        List of job IDs (rg_ids) associated with the batches

    Edge Cases:
    - Empty batch_ids list: Returns empty list
    - Batch not found: Excluded from results
    - Multiple jobs per batch: All returned
    """
    if not batch_ids:
        return []

    try:
        with get_inside_connection() as inside_conn:
            cursor = inside_conn.cursor()

            placeholders = ','.join(['%s'] * len(batch_ids))
            query = f"""
                SELECT DISTINCT rg_id
                FROM production_printjob
                WHERE production_batch_id IN ({placeholders})
                AND rg_id IS NOT NULL
                ORDER BY rg_id
            """

            cursor.execute(query, batch_ids)
            job_records = cursor.fetchall()
            cursor.close()

        job_ids = [row[0] for row in job_records]
        logger.info(f"Mapped {len(batch_ids)} batches to {len(job_ids)} job IDs")

        return job_ids

    except Exception as e:
        logger.error(f"Error in fetch_job_ids_for_batch: {e}", exc_info=True)
        return []

# ============================================================================
# JOBREPORT FETCHERS
# ============================================================================

def fetch_print_jobreports(job_ids: List[str]) -> pd.DataFrame:
    """
    Fetch Print1 JobReports from HISTORIAN.

    Topic: rg_v2/RG/CPH/Prod/ComponentLine/Print1/JobReport

    Extracts:
    - job_id, batch_id, sheet_index
    - uptime_sec, downtime_sec (from reportData)
    - speed_actual, speed_nominal (from reportData)
    - area_sqm (from closingReport)
    - job_start, job_end timestamps

    Calculates:
    - total_time_sec = uptime + downtime
    - availability = uptime / total_time × 100%
    - performance = actual_speed / nominal_speed × 100%
    - quality = 100% (initial assumption, QC overlay applied later)
    - oee = availability × performance × quality / 10000

    Returns:
        DataFrame with Print job metrics

    Edge Cases:
    - Empty job_ids: Returns empty DataFrame
    - Missing JSON fields: Returns NULL for that field
    - Division by zero: Returns 0% for that metric
    - No records found: Returns empty DataFrame
    """
    if not job_ids:
        return pd.DataFrame()

    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()

            placeholders = ','.join(['%s'] * len(job_ids))
            query = f"""
                SELECT
                    payload->>'jobId' as job_id,
                    payload->>'batchId' as batch_id,
                    CAST(payload->>'sheetIndex' AS FLOAT) as sheet_index,
                    time as job_start,
                    payload->'closingReport'->>'time' as job_end,
                    CAST(payload->'reportData'->>'uptime' AS FLOAT) as uptime_sec,
                    CAST(payload->'reportData'->>'downtime' AS FLOAT) as downtime_sec,
                    CAST(payload->'reportData'->'speed'->>'actual' AS FLOAT) as speed_actual,
                    CAST(payload->'reportData'->'speed'->>'nominal' AS FLOAT) as speed_nominal,
                    CAST(payload->'closingReport'->'totalArea'->>'value' AS FLOAT) as area_sqm
                FROM jobs
                WHERE topic = %s
                AND payload->>'jobId' IN ({placeholders})
                ORDER BY time ASC
            """

            cursor.execute(query, [Config.TOPICS['print']] + job_ids)
            records = cursor.fetchall()
            cursor.close()

        if not records:
            logger.warning(f"No Print JobReports found for {len(job_ids)} jobs")
            return pd.DataFrame()

        # Convert to DataFrame
        columns = [
            'job_id', 'batch_id', 'sheet_index', 'job_start', 'job_end',
            'uptime_sec', 'downtime_sec', 'speed_actual', 'speed_nominal', 'area_sqm'
        ]
        df = pd.DataFrame(records, columns=columns)

        # Handle NULL values
        df['uptime_sec'] = df['uptime_sec'].fillna(0)
        df['downtime_sec'] = df['downtime_sec'].fillna(0)
        df['speed_actual'] = df['speed_actual'].fillna(0)
        df['speed_nominal'] = df['speed_nominal'].fillna(0)
        df['area_sqm'] = df['area_sqm'].fillna(0)

        # Calculate derived metrics
        df['total_time_sec'] = df['uptime_sec'] + df['downtime_sec']

        # Availability (handle division by zero)
        df['availability'] = np.where(
            df['total_time_sec'] > 0,
            (df['uptime_sec'] / df['total_time_sec']) * 100,
            0.0
        )

        # Performance (handle division by zero)
        df['performance'] = np.where(
            df['speed_nominal'] > 0,
            (df['speed_actual'] / df['speed_nominal']) * 100,
            0.0
        )

        # Quality (initial 100% assumption)
        df['quality'] = 100.0

        # OEE
        df['oee'] = (df['availability'] * df['performance'] * df['quality']) / 10000

        logger.info(f"Fetched {len(df)} Print JobReports")
        return df

    except Exception as e:
        logger.error(f"Error in fetch_print_jobreports: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_cut_jobreports(job_ids: List[str]) -> pd.DataFrame:
    """
    Fetch Cut1 JobReports from HISTORIAN.

    Topic: rg_v2/RG/CPH/Prod/ComponentLine/Cut1/JobReport

    Extracts:
    - job_id, batch_id, sheet_index
    - uptime_sec, downtime_sec
    - speed_actual, speed_nominal
    - component_count (from closingReport)
    - job_start, job_end timestamps

    Calculates:
    - Same as Print cell

    Returns:
        DataFrame with Cut job metrics

    Edge Cases:
    - Same as fetch_print_jobreports
    """
    if not job_ids:
        return pd.DataFrame()

    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()

            placeholders = ','.join(['%s'] * len(job_ids))
            query = f"""
                SELECT
                    payload->>'jobId' as job_id,
                    payload->>'batchId' as batch_id,
                    CAST(payload->>'sheetIndex' AS FLOAT) as sheet_index,
                    time as job_start,
                    payload->'closingReport'->>'time' as job_end,
                    CAST(payload->'reportData'->>'uptime' AS FLOAT) as uptime_sec,
                    CAST(payload->'reportData'->>'downtime' AS FLOAT) as downtime_sec,
                    CAST(payload->'reportData'->'speed'->>'actual' AS FLOAT) as speed_actual,
                    CAST(payload->'reportData'->'speed'->>'nominal' AS FLOAT) as speed_nominal,
                    CAST(payload->'closingReport'->'components'->>'count' AS INT) as component_count
                FROM jobs
                WHERE topic = %s
                AND payload->>'jobId' IN ({placeholders})
                ORDER BY time ASC
            """

            cursor.execute(query, [Config.TOPICS['cut']] + job_ids)
            records = cursor.fetchall()
            cursor.close()

        if not records:
            logger.warning(f"No Cut JobReports found for {len(job_ids)} jobs")
            return pd.DataFrame()

        columns = [
            'job_id', 'batch_id', 'sheet_index', 'job_start', 'job_end',
            'uptime_sec', 'downtime_sec', 'speed_actual', 'speed_nominal', 'component_count'
        ]
        df = pd.DataFrame(records, columns=columns)

        # Handle NULL values
        df['uptime_sec'] = df['uptime_sec'].fillna(0)
        df['downtime_sec'] = df['downtime_sec'].fillna(0)
        df['speed_actual'] = df['speed_actual'].fillna(0)
        df['speed_nominal'] = df['speed_nominal'].fillna(0)
        df['component_count'] = df['component_count'].fillna(0)

        # Calculate derived metrics (same as Print)
        df['total_time_sec'] = df['uptime_sec'] + df['downtime_sec']
        df['availability'] = np.where(
            df['total_time_sec'] > 0,
            (df['uptime_sec'] / df['total_time_sec']) * 100,
            0.0
        )
        df['performance'] = np.where(
            df['speed_nominal'] > 0,
            (df['speed_actual'] / df['speed_nominal']) * 100,
            0.0
        )
        df['quality'] = 100.0
        df['oee'] = (df['availability'] * df['performance'] * df['quality']) / 10000

        logger.info(f"Fetched {len(df)} Cut JobReports")
        return df

    except Exception as e:
        logger.error(f"Error in fetch_cut_jobreports: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_pick_jobreports(job_ids: List[str]) -> pd.DataFrame:
    """
    Fetch Pick1 JobReports from HISTORIAN.

    Topic: rg_v2/RG/CPH/Prod/ComponentLine/Pick1/JobReport

    KEY LOGIC:
    - Performance = 100% (robots operate at consistent speed)
    - Quality = successful picks / total attempts

    Extracts:
    - job_id, batch_id, sheet_index
    - uptime_sec, downtime_sec
    - components_completed, components_failed (from closingReport)
    - average_pick_time_sec (from closingReport)
    - job_start, job_end timestamps

    Calculates:
    - availability = uptime / total_time × 100%
    - performance = 100% (FIXED - robots don't vary speed)
    - quality = completed / (completed + failed) × 100%
    - oee = availability × performance × quality / 10000

    Returns:
        DataFrame with Pick job metrics

    Edge Cases:
    - No components picked: quality = 100% (benefit of doubt)
    - All components failed: quality = 0%
    - Missing failed count: Assumes 0 failures
    """
    if not job_ids:
        return pd.DataFrame()

    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()

            placeholders = ','.join(['%s'] * len(job_ids))
            query = f"""
                SELECT
                    payload->>'jobId' as job_id,
                    payload->>'batchId' as batch_id,
                    CAST(payload->>'sheetIndex' AS FLOAT) as sheet_index,
                    time as job_start,
                    payload->'closingReport'->>'time' as job_end,
                    CAST(payload->'reportData'->>'uptime' AS FLOAT) as uptime_sec,
                    CAST(payload->'reportData'->>'downtime' AS FLOAT) as downtime_sec,
                    CAST(payload->'closingReport'->'components'->>'completed' AS INT) as components_completed,
                    CAST(payload->'closingReport'->'components'->>'failed' AS INT) as components_failed,
                    CAST(payload->'closingReport'->'averagePickTime'->>'value' AS FLOAT) as average_pick_time_sec
                FROM jobs
                WHERE topic = %s
                AND payload->>'jobId' IN ({placeholders})
                ORDER BY time ASC
            """

            cursor.execute(query, [Config.TOPICS['pick']] + job_ids)
            records = cursor.fetchall()
            cursor.close()

        if not records:
            logger.warning(f"No Pick JobReports found for {len(job_ids)} jobs")
            return pd.DataFrame()

        columns = [
            'job_id', 'batch_id', 'sheet_index', 'job_start', 'job_end',
            'uptime_sec', 'downtime_sec', 'components_completed', 'components_failed',
            'average_pick_time_sec'
        ]
        df = pd.DataFrame(records, columns=columns)

        # Handle NULL values
        df['uptime_sec'] = df['uptime_sec'].fillna(0)
        df['downtime_sec'] = df['downtime_sec'].fillna(0)
        df['components_completed'] = df['components_completed'].fillna(0)
        df['components_failed'] = df['components_failed'].fillna(0)
        df['average_pick_time_sec'] = df['average_pick_time_sec'].fillna(0)

        # Calculate derived metrics
        df['total_time_sec'] = df['uptime_sec'] + df['downtime_sec']
        df['availability'] = np.where(
            df['total_time_sec'] > 0,
            (df['uptime_sec'] / df['total_time_sec']) * 100,
            0.0
        )

        # CORRECTED: Pick performance is always 100%
        df['performance'] = 100.0

        # CORRECTED: Quality = successful picks / total attempts
        df['total_attempts'] = df['components_completed'] + df['components_failed']
        df['quality'] = np.where(
            df['total_attempts'] > 0,
            (df['components_completed'] / df['total_attempts']) * 100,
            100.0  # If no attempts, assume 100% (benefit of doubt)
        )

        # OEE
        df['oee'] = (df['availability'] * df['performance'] * df['quality']) / 10000

        logger.info(f"Fetched {len(df)} Pick JobReports")
        return df

    except Exception as e:
        logger.error(f"Error in fetch_pick_jobreports: {e}", exc_info=True)
        return pd.DataFrame()

# ============================================================================
# QUALITY / QC DATA FETCHERS
# ============================================================================

def fetch_qc_data_for_batches(batch_ids: List[str]) -> Optional[pd.DataFrame]:
    """
    Fetch QC inspection data for batches from INSIDE database.

    Queries production_componentorderqc table to get inspection results.

    Args:
        batch_ids: List of batch identifiers

    Returns:
        DataFrame with columns:
        - batch_id: Batch identifier
        - total_inspected: Total components inspected
        - passed: Number of components that passed
        - failed: Number of components that failed
        - pass_rate: Percentage of components that passed (0-100)

    Returns None if no QC data available (Day 0 scenario).

    Edge Cases:
    - No QC data: Returns None (not empty DataFrame)
    - Partial QC data: Returns only batches with QC results
    - Multiple inspections per batch: Aggregated into single pass rate
    """
    if not batch_ids:
        return None

    try:
        with get_inside_connection() as conn:
            cursor = conn.cursor()

            placeholders = ','.join(['%s'] * len(batch_ids))
            query = f"""
                SELECT
                    co.production_batch_id as batch_id,
                    COUNT(*) as total_inspected,
                    SUM(CASE WHEN qc.result = 'PASS' THEN 1 ELSE 0 END) as passed,
                    SUM(CASE WHEN qc.result = 'FAIL' THEN 1 ELSE 0 END) as failed,
                    CAST(SUM(CASE WHEN qc.result = 'PASS' THEN 1 ELSE 0 END) AS FLOAT) /
                        NULLIF(COUNT(*), 0) * 100 as pass_rate
                FROM production_componentorder co
                INNER JOIN production_componentorderqc qc ON co.id = qc.component_order_id
                WHERE co.production_batch_id IN ({placeholders})
                AND qc.result IS NOT NULL
                GROUP BY co.production_batch_id
            """

            cursor.execute(query, batch_ids)
            results = cursor.fetchall()
            cursor.close()

        if not results:
            logger.info(f"No QC data found for {len(batch_ids)} batches (Day 0 scenario)")
            return None

        df = pd.DataFrame(results, columns=['batch_id', 'total_inspected', 'passed', 'failed', 'pass_rate'])
        logger.info(f"Fetched QC data for {len(df)} batches")

        return df

    except Exception as e:
        logger.error(f"Error in fetch_qc_data_for_batches: {e}", exc_info=True)
        return None
```

### Step 5: Quality Overlay Logic

**Create `analysis/quality_overlay.py`:**
```python
"""
Quality Overlay Logic
Applies QC inspection data to job-level OEE calculations
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_qc_overlay(
    print_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    pick_df: pd.DataFrame,
    qc_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply QC inspection data overlay to cell DataFrames.

    Strategy:
    - Print/Cut: QC pass rate replaces 100% quality assumption
    - Pick: QC pass rate × pick success rate (compound quality)

    Args:
        print_df: Print job DataFrame (with quality=100%)
        cut_df: Cut job DataFrame (with quality=100%)
        pick_df: Pick job DataFrame (with quality=pick_success_rate)
        qc_df: QC data DataFrame with batch_id and pass_rate columns

    Returns:
        Tuple of (print_df, cut_df, pick_df) with updated quality and OEE

    Edge Cases:
    - Empty qc_df: Returns original DataFrames unchanged
    - Batch not in QC data: Retains 100% assumption for that batch
    - QC pass rate = 0%: Quality becomes 0%, OEE becomes 0%
    """
    if qc_df is None or qc_df.empty:
        logger.info("No QC data to apply - retaining 100% quality assumption")
        return print_df, cut_df, pick_df

    # Group QC data by batch (in case of multiple entries)
    qc_by_batch = qc_df.groupby('batch_id').agg({
        'pass_rate': 'mean'  # Average pass rate
    }).reset_index()

    # Apply to Print
    if not print_df.empty:
        print_df = print_df.merge(qc_by_batch, on='batch_id', how='left')
        # Replace quality with QC pass rate (keep 100% if no QC data for that batch)
        print_df['quality'] = print_df['pass_rate'].fillna(print_df['quality'])
        # Recalculate OEE
        print_df['oee'] = (print_df['availability'] * print_df['performance'] * print_df['quality']) / 10000
        # Drop temporary pass_rate column
        print_df = print_df.drop(columns=['pass_rate'])
        logger.info(f"Applied QC overlay to {len(print_df)} Print jobs")

    # Apply to Cut
    if not cut_df.empty:
        cut_df = cut_df.merge(qc_by_batch, on='batch_id', how='left')
        cut_df['quality'] = cut_df['pass_rate'].fillna(cut_df['quality'])
        cut_df['oee'] = (cut_df['availability'] * cut_df['performance'] * cut_df['quality']) / 10000
        cut_df = cut_df.drop(columns=['pass_rate'])
        logger.info(f"Applied QC overlay to {len(cut_df)} Cut jobs")

    # Apply to Pick (compound quality)
    if not pick_df.empty:
        pick_df = pick_df.merge(qc_by_batch, on='batch_id', how='left')
        # Store original pick quality (success rate)
        pick_original_quality = pick_df['quality'].copy()
        # Compound quality: pick_success_rate × qc_pass_rate / 100
        # (component must be successfully picked AND pass QC inspection)
        pick_df['quality'] = np.where(
            pick_df['pass_rate'].notna(),
            (pick_original_quality * pick_df['pass_rate']) / 100,
            pick_original_quality  # Keep pick success rate if no QC data
        )
        pick_df['oee'] = (pick_df['availability'] * pick_df['performance'] * pick_df['quality']) / 10000
        pick_df = pick_df.drop(columns=['pass_rate'])
        logger.info(f"Applied QC overlay to {len(pick_df)} Pick jobs")

    return print_df, cut_df, pick_df
```

### Step 6: OEE Calculator

**Create `analysis/oee_calculator.py`:**
```python
"""
OEE Calculator
Aggregates job-level OEE into batch-level and daily-level metrics
"""
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def calculate_batch_metrics(
    print_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    pick_df: pd.DataFrame,
    batch_ids: list
) -> pd.DataFrame:
    """
    Calculate batch-level OEE metrics by aggregating job-level data.

    Args:
        print_df: Print jobs DataFrame
        cut_df: Cut jobs DataFrame
        pick_df: Pick jobs DataFrame
        batch_ids: List of batch IDs to calculate metrics for

    Returns:
        DataFrame with batch-level metrics (one row per batch)

    Edge Cases:
    - Batch has no jobs in a cell: Metrics = 0 for that cell
    - Empty DataFrames: Returns zeros for all metrics
    """
    batch_metrics = []

    for batch_id in batch_ids:
        # Filter jobs for this batch
        batch_print = print_df[print_df['batch_id'] == batch_id] if not print_df.empty else pd.DataFrame()
        batch_cut = cut_df[cut_df['batch_id'] == batch_id] if not cut_df.empty else pd.DataFrame()
        batch_pick = pick_df[pick_df['batch_id'] == batch_id] if not pick_df.empty else pd.DataFrame()

        # Aggregate metrics (mean for percentages, sum for times)
        metrics = {
            'batch_id': batch_id,

            # Print metrics
            'print_job_count': len(batch_print),
            'print_oee': batch_print['oee'].mean() if not batch_print.empty else 0,
            'print_availability': batch_print['availability'].mean() if not batch_print.empty else 0,
            'print_performance': batch_print['performance'].mean() if not batch_print.empty else 0,
            'print_quality': batch_print['quality'].mean() if not batch_print.empty else 0,
            'print_total_time_sec': batch_print['total_time_sec'].sum() if not batch_print.empty else 0,

            # Cut metrics
            'cut_job_count': len(batch_cut),
            'cut_oee': batch_cut['oee'].mean() if not batch_cut.empty else 0,
            'cut_availability': batch_cut['availability'].mean() if not batch_cut.empty else 0,
            'cut_performance': batch_cut['performance'].mean() if not batch_cut.empty else 0,
            'cut_quality': batch_cut['quality'].mean() if not batch_cut.empty else 0,
            'cut_total_time_sec': batch_cut['total_time_sec'].sum() if not batch_cut.empty else 0,

            # Pick metrics
            'pick_job_count': len(batch_pick),
            'pick_oee': batch_pick['oee'].mean() if not batch_pick.empty else 0,
            'pick_availability': batch_pick['availability'].mean() if not batch_pick.empty else 0,
            'pick_performance': batch_pick['performance'].mean() if not batch_pick.empty else 0,
            'pick_quality': batch_pick['quality'].mean() if not batch_pick.empty else 0,
            'pick_total_time_sec': batch_pick['total_time_sec'].sum() if not batch_pick.empty else 0,
        }

        batch_metrics.append(metrics)

    return pd.DataFrame(batch_metrics)

def calculate_daily_metrics(
    batch_metrics_df: pd.DataFrame,
    available_hours: float
) -> Dict:
    """
    Calculate daily-level OEE metrics with utilization.

    Formula:
    - Utilization = Production Time / Available Time × 100%
    - Daily OEE = Batch OEE × Utilization / 100

    Args:
        batch_metrics_df: Batch-level metrics DataFrame
        available_hours: Total available hours in the day (e.g., 24)

    Returns:
        Dictionary with daily metrics for each cell

    Edge Cases:
    - available_hours = 0: Utilization = 0%
    - Production time > available hours: Utilization > 100% (possible in multi-shift)
    """
    # Calculate total production time (sum across batches)
    print_production_hours = batch_metrics_df['print_total_time_sec'].sum() / 3600
    cut_production_hours = batch_metrics_df['cut_total_time_sec'].sum() / 3600
    pick_production_hours = batch_metrics_df['pick_total_time_sec'].sum() / 3600

    # Calculate utilization
    print_utilization = (print_production_hours / available_hours * 100) if available_hours > 0 else 0
    cut_utilization = (cut_production_hours / available_hours * 100) if available_hours > 0 else 0
    pick_utilization = (pick_production_hours / available_hours * 100) if available_hours > 0 else 0

    # Calculate idle time
    print_idle_hours = max(0, available_hours - print_production_hours)
    cut_idle_hours = max(0, available_hours - cut_production_hours)
    pick_idle_hours = max(0, available_hours - pick_production_hours)

    # Calculate batch OEE (average across batches)
    print_batch_oee = batch_metrics_df['print_oee'].mean()
    cut_batch_oee = batch_metrics_df['cut_oee'].mean()
    pick_batch_oee = batch_metrics_df['pick_oee'].mean()

    # Calculate daily OEE
    print_daily_oee = (print_batch_oee * print_utilization) / 100
    cut_daily_oee = (cut_batch_oee * cut_utilization) / 100
    pick_daily_oee = (pick_batch_oee * pick_utilization) / 100

    daily_metrics = {
        'print': {
            'batch_oee': print_batch_oee,
            'production_hours': print_production_hours,
            'idle_hours': print_idle_hours,
            'utilization': print_utilization,
            'daily_oee': print_daily_oee
        },
        'cut': {
            'batch_oee': cut_batch_oee,
            'production_hours': cut_production_hours,
            'idle_hours': cut_idle_hours,
            'utilization': cut_utilization,
            'daily_oee': cut_daily_oee
        },
        'pick': {
            'batch_oee': pick_batch_oee,
            'production_hours': pick_production_hours,
            'idle_hours': pick_idle_hours,
            'utilization': pick_utilization,
            'daily_oee': pick_daily_oee
        }
    }

    logger.info(f"Calculated daily metrics: Print={print_daily_oee:.2f}%, Cut={cut_daily_oee:.2f}%, Pick={pick_daily_oee:.2f}%")

    return daily_metrics
```

### Step 7: Streamlit Application

**Create `ui/app.py`:**
```python
"""
Batch-Centric OEE Analysis - Streamlit Application
Main UI for selecting days, batches, and viewing OEE metrics
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import pytz
import logging

from config import Config
from db.fetchers import (
    fetch_batches_for_day,
    fetch_job_ids_for_batch,
    fetch_print_jobreports,
    fetch_cut_jobreports,
    fetch_pick_jobreports,
    fetch_qc_data_for_batches
)
from analysis.quality_overlay import apply_qc_overlay
from analysis.oee_calculator import calculate_batch_metrics, calculate_daily_metrics

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timezone
tz = pytz.timezone(Config.TIMEZONE)

# Page config
st.set_page_config(
    page_title="Batch-Centric OEE Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HEADER
# ============================================================================

st.title("🏭 Batch-Centric OEE Analysis")
st.markdown("**Manufacturing Excellence Dashboard** - Real-time batch performance tracking")
st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # Operating hours
    operating_hours = st.number_input(
        "Operating Hours per Day",
        min_value=1.0,
        max_value=24.0,
        value=Config.DEFAULT_OPERATING_HOURS,
        step=0.5,
        help="Total available hours for daily utilization calculation"
    )

    st.markdown("---")

    # Info
    st.info(f"**Timezone:** {Config.TIMEZONE}")
    st.caption("📊 Data sources: HISTORIAN + INSIDE databases")
    st.caption("🔄 Quality overlay: Auto-applies when QC data available")

# ============================================================================
# SESSION STATE
# ============================================================================

if 'batches_df' not in st.session_state:
    st.session_state.batches_df = None
if 'selected_batches' not in st.session_state:
    st.session_state.selected_batches = []
if 'print_df' not in st.session_state:
    st.session_state.print_df = None
if 'cut_df' not in st.session_state:
    st.session_state.cut_df = None
if 'pick_df' not in st.session_state:
    st.session_state.pick_df = None
if 'qc_applied' not in st.session_state:
    st.session_state.qc_applied = False

# ============================================================================
# STEP 1: DAY SELECTION
# ============================================================================

st.header("1️⃣ Select Production Day")

col1, col2 = st.columns([2, 1])

with col1:
    selected_date = st.date_input(
        "Production Date",
        value=date.today() - timedelta(days=1),
        help="Select the day to analyze"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("🔍 Load Batches", type="primary", use_container_width=True):
        # Calculate day boundaries
        day_start = tz.localize(datetime.combine(selected_date, datetime.min.time()))
        day_end = day_start + timedelta(days=1)

        with st.spinner("Discovering batches..."):
            batches_df = fetch_batches_for_day(
                day_start.isoformat(),
                day_end.isoformat()
            )
            st.session_state.batches_df = batches_df

            if not batches_df.empty:
                st.success(f"✅ Found {len(batches_df)} batches on {selected_date}")
            else:
                st.warning(f"⚠️ No batches found on {selected_date}")

# ============================================================================
# STEP 2: BATCH SELECTION
# ============================================================================

if st.session_state.batches_df is not None and not st.session_state.batches_df.empty:
    st.markdown("---")
    st.header("2️⃣ Select Batches to Analyze")

    batches_df = st.session_state.batches_df

    # Display batch overview
    st.dataframe(
        batches_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "batch_id": st.column_config.TextColumn("Batch ID", width="medium"),
            "job_count": st.column_config.NumberColumn("Jobs", width="small"),
            "first_job_time": st.column_config.DatetimeColumn("Start Time", width="medium"),
            "last_job_time": st.column_config.DatetimeColumn("End Time", width="medium")
        }
    )

    # Batch multiselect
    selected_batches = st.multiselect(
        "Select batches",
        options=batches_df['batch_id'].tolist(),
        default=batches_df['batch_id'].tolist(),
        help="Choose one or more batches for analysis"
    )

    st.session_state.selected_batches = selected_batches

    # ============================================================================
    # STEP 3: FETCH & ANALYZE
    # ============================================================================

    if selected_batches:
        st.markdown("---")
        st.header("3️⃣ Fetch & Analyze Data")

        if st.button("📊 Calculate OEE", type="primary", use_container_width=True):
            with st.spinner("Fetching JobReports and calculating OEE..."):

                # Get job IDs
                job_ids = fetch_job_ids_for_batch(selected_batches)

                if not job_ids:
                    st.error("❌ No job IDs found for selected batches")
                else:
                    st.info(f"📋 Found {len(job_ids)} jobs across {len(selected_batches)} batches")

                    # Fetch JobReports
                    print_df = fetch_print_jobreports(job_ids)
                    cut_df = fetch_cut_jobreports(job_ids)
                    pick_df = fetch_pick_jobreports(job_ids)

                    # Try to fetch QC data
                    qc_df = fetch_qc_data_for_batches(selected_batches)

                    if qc_df is not None and not qc_df.empty:
                        st.info("🔬 QC inspection data found - applying quality overlay...")
                        print_df, cut_df, pick_df = apply_qc_overlay(print_df, cut_df, pick_df, qc_df)
                        st.session_state.qc_applied = True
                    else:
                        st.info("ℹ️ No QC data available - using 100% quality assumption")
                        st.session_state.qc_applied = False

                    # Store in session state
                    st.session_state.print_df = print_df
                    st.session_state.cut_df = cut_df
                    st.session_state.pick_df = pick_df

                    st.success("✅ OEE calculation complete!")

# ============================================================================
# STEP 4: DISPLAY RESULTS
# ============================================================================

if st.session_state.print_df is not None:
    st.markdown("---")
    st.header("4️⃣ OEE Analysis Results")

    # QC Status Banner
    if st.session_state.qc_applied:
        st.success("✅ **QC Inspection Data Applied** - Quality metrics reflect actual inspection results")
    else:
        st.info("ℹ️ **No QC Data Yet** - Using 100% quality assumption (Day 0 optimistic view)")

    # Get data
    print_df = st.session_state.print_df
    cut_df = st.session_state.cut_df
    pick_df = st.session_state.pick_df

    # Calculate metrics
    batch_metrics_df = calculate_batch_metrics(
        print_df, cut_df, pick_df,
        st.session_state.selected_batches
    )

    daily_metrics = calculate_daily_metrics(batch_metrics_df, operating_hours)

    # ========================================================================
    # 4.1: BATCH-LEVEL METRICS
    # ========================================================================

    st.subheader("📦 Batch-Level OEE Metrics")

    st.dataframe(
        batch_metrics_df[[
            'batch_id',
            'print_oee', 'print_availability', 'print_performance', 'print_quality',
            'cut_oee', 'cut_availability', 'cut_performance', 'cut_quality',
            'pick_oee', 'pick_availability', 'pick_performance', 'pick_quality'
        ]].style.format({
            'print_oee': '{:.2f}%',
            'print_availability': '{:.2f}%',
            'print_performance': '{:.2f}%',
            'print_quality': '{:.2f}%',
            'cut_oee': '{:.2f}%',
            'cut_availability': '{:.2f}%',
            'cut_performance': '{:.2f}%',
            'cut_quality': '{:.2f}%',
            'pick_oee': '{:.2f}%',
            'pick_availability': '{:.2f}%',
            'pick_performance': '{:.2f}%',
            'pick_quality': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================================
    # 4.2: DAILY-LEVEL METRICS
    # ========================================================================

    st.subheader("📅 Daily-Level OEE Metrics (with Utilization)")

    daily_df = pd.DataFrame([
        {
            'Cell': '🖨️ Print',
            'Batch OEE': daily_metrics['print']['batch_oee'],
            'Production Hours': daily_metrics['print']['production_hours'],
            'Idle Hours': daily_metrics['print']['idle_hours'],
            'Utilization': daily_metrics['print']['utilization'],
            'Daily OEE': daily_metrics['print']['daily_oee']
        },
        {
            'Cell': '✂️ Cut',
            'Batch OEE': daily_metrics['cut']['batch_oee'],
            'Production Hours': daily_metrics['cut']['production_hours'],
            'Idle Hours': daily_metrics['cut']['idle_hours'],
            'Utilization': daily_metrics['cut']['utilization'],
            'Daily OEE': daily_metrics['cut']['daily_oee']
        },
        {
            'Cell': '🤖 Pick',
            'Batch OEE': daily_metrics['pick']['batch_oee'],
            'Production Hours': daily_metrics['pick']['production_hours'],
            'Idle Hours': daily_metrics['pick']['idle_hours'],
            'Utilization': daily_metrics['pick']['utilization'],
            'Daily OEE': daily_metrics['pick']['daily_oee']
        }
    ])

    st.dataframe(
        daily_df.style.format({
            'Batch OEE': '{:.2f}%',
            'Production Hours': '{:.2f}h',
            'Idle Hours': '{:.2f}h',
            'Utilization': '{:.2f}%',
            'Daily OEE': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================================
    # 4.3: JOB-LEVEL DETAILS
    # ========================================================================

    with st.expander("🔍 View Job-Level Details"):
        tab1, tab2, tab3 = st.tabs(["Print Jobs", "Cut Jobs", "Pick Jobs"])

        with tab1:
            if not print_df.empty:
                st.dataframe(
                    print_df[[
                        'job_id', 'batch_id', 'sheet_index',
                        'availability', 'performance', 'quality', 'oee',
                        'uptime_sec', 'downtime_sec', 'total_time_sec'
                    ]].style.format({
                        'availability': '{:.2f}%',
                        'performance': '{:.2f}%',
                        'quality': '{:.2f}%',
                        'oee': '{:.2f}%',
                        'uptime_sec': '{:.0f}s',
                        'downtime_sec': '{:.0f}s',
                        'total_time_sec': '{:.0f}s'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No Print data available")

        with tab2:
            if not cut_df.empty:
                st.dataframe(
                    cut_df[[
                        'job_id', 'batch_id', 'sheet_index',
                        'availability', 'performance', 'quality', 'oee',
                        'uptime_sec', 'downtime_sec', 'total_time_sec'
                    ]].style.format({
                        'availability': '{:.2f}%',
                        'performance': '{:.2f}%',
                        'quality': '{:.2f}%',
                        'oee': '{:.2f}%',
                        'uptime_sec': '{:.0f}s',
                        'downtime_sec': '{:.0f}s',
                        'total_time_sec': '{:.0f}s'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No Cut data available")

        with tab3:
            if not pick_df.empty:
                st.dataframe(
                    pick_df[[
                        'job_id', 'batch_id', 'sheet_index',
                        'availability', 'performance', 'quality', 'oee',
                        'uptime_sec', 'downtime_sec', 'total_time_sec',
                        'components_completed', 'components_failed', 'total_attempts'
                    ]].style.format({
                        'availability': '{:.2f}%',
                        'performance': '{:.2f}%',
                        'quality': '{:.2f}%',
                        'oee': '{:.2f}%',
                        'uptime_sec': '{:.0f}s',
                        'downtime_sec': '{:.0f}s',
                        'total_time_sec': '{:.0f}s'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No Pick data available")

    # ========================================================================
    # 4.4: EXPORT
    # ========================================================================

    st.subheader("💾 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        batch_csv = batch_metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Batch Metrics CSV",
            data=batch_csv,
            file_name=f"batch_metrics_{selected_date}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        daily_csv = daily_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Daily Metrics CSV",
            data=daily_csv,
            file_name=f"daily_metrics_{selected_date}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(f"🏭 Batch-Centric OEE Analysis | Timezone: {Config.TIMEZONE} | Data: HISTORIAN + INSIDE")
```

---

## Testing Strategy

### Unit Tests

Create `tests/test_fetchers.py`:

```python
import pytest
from db.fetchers import (
    fetch_batches_for_day,
    fetch_job_ids_for_batch,
    fetch_print_jobreports
)

def test_fetch_batches_empty_range():
    """Test with date range that has no batches"""
    df = fetch_batches_for_day('2020-01-01T00:00:00', '2020-01-02T00:00:00')
    assert df.empty
    assert list(df.columns) == ['batch_id', 'job_count', 'first_job_time', 'last_job_time']

def test_fetch_job_ids_empty_list():
    """Test with empty batch list"""
    job_ids = fetch_job_ids_for_batch([])
    assert job_ids == []

def test_fetch_print_jobreports_empty_list():
    """Test with empty job list"""
    df = fetch_print_jobreports([])
    assert df.empty
```

### Integration Tests

Create `tests/test_integration.py`:

```python
import pytest
from datetime import datetime, timedelta
import pytz

from db.fetchers import fetch_batches_for_day, fetch_job_ids_for_batch
from analysis.oee_calculator import calculate_batch_metrics

def test_full_workflow():
    """Test complete workflow from day selection to metrics"""
    tz = pytz.timezone('Europe/Copenhagen')

    # Use yesterday's date (more likely to have data)
    yesterday = datetime.now() - timedelta(days=1)
    day_start = tz.localize(datetime.combine(yesterday.date(), datetime.min.time()))
    day_end = day_start + timedelta(days=1)

    # Fetch batches
    batches_df = fetch_batches_for_day(day_start.isoformat(), day_end.isoformat())

    if not batches_df.empty:
        # Get job IDs
        batch_ids = batches_df['batch_id'].tolist()
        job_ids = fetch_job_ids_for_batch(batch_ids)

        assert len(job_ids) > 0
        assert isinstance(job_ids, list)
```

### Manual Testing Scenarios

**Scenario 1: Fresh Production (Day 0 - No QC)**
1. Select yesterday's date
2. Load batches
3. Select all batches
4. Calculate OEE
5. **Expected**: Banner shows "No QC Data Yet"
6. **Verify**: Print/Cut quality = 100%, Pick quality = success rate

**Scenario 2: Post-QC Production (Day 3+)**
1. Select date from 3+ days ago
2. Load batches
3. Select batches with known QC data
4. Calculate OEE
5. **Expected**: Banner shows "QC Inspection Data Applied"
6. **Verify**: Print/Cut quality = QC pass rate, Pick quality = success × QC

**Scenario 3: No Batches Found**
1. Select a weekend or shutdown day
2. Load batches
3. **Expected**: Warning "No batches found on [date]"
4. **Verify**: No errors, graceful handling

**Scenario 4: Partial Data**
1. Select batch with only Print jobs (no Cut/Pick)
2. Calculate OEE
3. **Expected**: Metrics show 0 for missing cells
4. **Verify**: No crashes, empty tables show "No data available"

**Scenario 5: Pick Performance Verification**
1. View Pick job-level details
2. **Verify**: Performance column shows exactly 100.00% for all rows
3. **Verify**: Quality varies based on success rate

---

## Edge Cases & Handling

### 1. Missing JSON Fields in JobReports

**Problem**: JobReport payload missing expected fields (e.g., `reportData.uptime`)

**Handling**:
```python
df['uptime_sec'] = df['uptime_sec'].fillna(0)
```

**Result**: Missing values treated as 0, metric calculations proceed

---

### 2. Division by Zero

**Problem**: Total time = 0, nominal speed = 0, total attempts = 0

**Handling**:
```python
df['availability'] = np.where(
    df['total_time_sec'] > 0,
    (df['uptime_sec'] / df['total_time_sec']) * 100,
    0.0
)
```

**Result**: Returns 0% instead of NaN or error

---

### 3. No QC Data Available (Day 0)

**Problem**: QC inspection not yet performed

**Handling**:
```python
if qc_df is None or qc_df.empty:
    logger.info("No QC data - retaining 100% quality assumption")
    return print_df, cut_df, pick_df
```

**Result**: Uses optimistic 100% quality assumption

---

### 4. Partial QC Data

**Problem**: Some batches have QC, others don't

**Handling**:
```python
print_df = print_df.merge(qc_by_batch, on='batch_id', how='left')
print_df['quality'] = print_df['pass_rate'].fillna(print_df['quality'])
```

**Result**: QC applied where available, 100% assumption retained otherwise

---

### 5. No Jobs Found for Batch

**Problem**: Batch exists in INSIDE but no JobReports in HISTORIAN

**Handling**:
```python
batch_print = print_df[print_df['batch_id'] == batch_id]
metrics['print_oee'] = batch_print['oee'].mean() if not batch_print.empty else 0
```

**Result**: Returns 0% OEE for that cell

---

### 6. Utilization > 100%

**Problem**: Multi-shift production exceeds configured operating hours

**Handling**: No artificial cap - displays actual utilization

**Result**: Utilization can exceed 100% (e.g., 24-hour operation with 16-hour available)

---

### 7. Pick with Zero Attempts

**Problem**: Pick job has 0 completed + 0 failed components

**Handling**:
```python
df['quality'] = np.where(
    df['total_attempts'] > 0,
    (df['components_completed'] / df['total_attempts']) * 100,
    100.0  # Benefit of doubt
)
```

**Result**: Quality = 100% (benefit of doubt if no attempts)

---

### 8. Batch Spans Multiple Days

**Problem**: Batch started on Day 1, completed on Day 2

**Handling**: Batch discovered on day with most JobReports (typically start day)

**Alternative**: Modify `fetch_batches_for_day()` to include batch if ANY job in range

---

### 9. Database Connection Failure

**Problem**: Cannot connect to HISTORIAN or INSIDE

**Handling**:
```python
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    return pd.DataFrame()
```

**Result**: Returns empty DataFrame, logs error, UI shows "No data available"

---

### 10. Large Batch Selection (Performance)

**Problem**: User selects 100+ batches, queries slow

**Handling**:
- Connection pooling (max 10 connections)
- Parameterized queries (prevents SQL injection, improves plan caching)
- Consider adding pagination or warning for large selections

**Alternative**: Add batch count limit with user confirmation

---

## Running the Application

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your database credentials

# 4. Run application
streamlit run ui/app.py

# 5. Access in browser
# Opens automatically at http://localhost:8501
```

---

## Deployment Considerations

### Production Checklist

- [ ] Set `LOG_LEVEL=WARNING` in production
- [ ] Use environment variables for all secrets (never commit `.env`)
- [ ] Configure connection pool sizes based on expected load
- [ ] Set up database read replicas for HISTORIAN/INSIDE (if available)
- [ ] Enable Streamlit authentication (if deploying publicly)
- [ ] Set up monitoring (Sentry, DataDog, etc.)
- [ ] Configure backup/disaster recovery for session data
- [ ] Review and optimize slow queries (add indexes if needed)
- [ ] Set up automated testing pipeline (CI/CD)
- [ ] Document deployment architecture

---

## Summary

This implementation provides:

✅ **Batch-centric approach** - Simplifies queries, improves performance
✅ **Corrected Pick logic** - Performance = 100%, Quality = success rate
✅ **Quality overlay strategy** - Day 0 (optimistic) vs Day 3+ (realistic)
✅ **Three-level OEE** - Job → Batch → Daily with utilization
✅ **Comprehensive edge case handling** - Graceful degradation, no crashes
✅ **Production-ready architecture** - Connection pooling, logging, error handling
✅ **Clean separation of concerns** - DB / Analysis / UI layers
✅ **Fully documented** - Inline comments, docstrings, this guide

**Estimated Implementation Time**: 8-12 hours for experienced developer

**Next Steps**: Deploy to staging, test with real data, iterate based on feedback.
