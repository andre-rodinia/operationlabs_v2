# Manufacturing OEE Analytics Platform - Technical Specifications v2.0

**Document Version:** 2.0
**Date:** 2026-01-06
**Status:** Architecture Redesign
**Author:** Based on requirements for simplified, maintainable OEE tracking

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Core Requirements](#core-requirements)
3. [Data Architecture](#data-architecture)
4. [OEE Calculation Specifications](#oee-calculation-specifications)
5. [Time Window Management](#time-window-management)
6. [Application Architecture](#application-architecture)
7. [User Interface Requirements](#user-interface-requirements)
8. [Components to Reuse](#components-to-reuse)
9. [Components to Remove](#components-to-remove)
10. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

### Purpose
Redesign the manufacturing analytics platform to provide **clear, actionable OEE metrics** for three production cells (Printer, Cut, Pick) with **simplified calculations** and **flexible time-window tracking**.

### Key Design Principles
1. **Simplicity over complexity** - Remove redundant calculation layers
2. **Per-cell focus** - Each cell has distinct OEE formula based on its operational characteristics
3. **Flexible time tracking** - Support batch operations spanning multiple days/shifts
4. **System-level OEE** - Calculate overall system effectiveness as product of cell OEEs
5. **Quality separation** - Overall quality tracked separately from cell-level OEE

### Success Metrics
- Reduce codebase from ~10,000 lines to ~3,000 lines
- Single source of truth for OEE calculations
- System OEE calculation as product of individual cell OEEs
- Support multi-day batch tracking with time segments

---

## Core Requirements

### Critical Features (MUST HAVE)

#### 1. Per-Cell OEE Metrics

**Printer Cell:**
- **Availability (A%)**: Uptime / (Uptime + Downtime)
- **Performance (P%)**: Actual Speed / Nominal Speed
- **Quality (Q%)**: Fixed at 100%
- **OEE = A × P × 100%**

**Cut Cell:**
- **Availability (A%)**: Uptime / (Uptime + Downtime)
- **Performance (P%)**: Fixed at 100%
- **Quality (Q%)**: Fixed at 100%
- **OEE = A × 100% × 100%**

**Pick Cell:**
- **Availability (A%)**: Uptime / (Uptime + Downtime)
- **Performance (P%)**: Fixed at 100%
- **Operational Quality Accuracy (Q_acc%)**: Pick Success Rate
- **OEE = A × 100% × Q_acc%**

#### 2. System OEE Calculation
- Calculate overall system OEE as the product of individual cell OEEs
- **Formula:** `OEE_system = OEE_printer × OEE_cut × OEE_pick`
- This multiplicative approach reflects that all cells must perform well for the system to achieve high overall effectiveness
- Each cell's performance directly impacts the final output
- **Exclude overall quality** (QC data from production_componentorder) from system OEE calculation
- Rationale: Time offset between production and quality inspection

#### 3. Time Window Management
- **Per-cell time windows** with configurable segments
- Support scenarios:
  - Full batch across multiple days (e.g., print Monday, cut/pick Tuesday)
  - Shift-based tracking (day shift vs. night shift)
  - Custom time ranges per cell
- **Allowable segments**: Define valid time periods to include in analysis

#### 4. Batch Selection & Analysis
- Select one or multiple batches
- View metrics across selected batches
- Compare batch performance over time

### Features to Remove (FROM CURRENT SYSTEM)

#### Removed Calculations
- ❌ Job-level OEE aggregation
- ❌ Weighted batch-level OEE
- ❌ Stage correlation OEE (Print→Cut gap analysis)
- ❌ OEE grading system (World Class/Good/Fair/Poor)
- ❌ Multi-batch comparison rankings
- ❌ Quality-adjusted cell OEE (quality tracked separately)

#### Removed Infrastructure
- ❌ Google Sheets integration (no longer needed)
- ❌ Async data processing layer
- ❌ Pagination utilities
- ❌ SQL optimizer recommendations
- ❌ Over-engineered config validation
- ❌ Time-range based fetching (replaced by batch + time window)

#### Removed Visualizations
- ❌ Ink consumption charts
- ❌ Batch comparison bar charts
- ❌ Data quality indicators dashboard
- ❌ Correlation scatter plots

---

## Data Architecture

### Database Systems

#### HISTORIAN Database (PostgreSQL + JSONB)
**Purpose:** Stores real-time MQTT messages from manufacturing equipment

**Table: `jobs`**
```sql
CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP WITH TIME ZONE,  -- Message timestamp
    topic TEXT,                     -- MQTT topic path
    payload JSONB                   -- Equipment metrics
);

-- Example topics:
-- 'rg_v2/RG/CPH/Prod/ComponentLine/Print1/JobReport'
-- 'rg_v2/RG/CPH/Prod/ComponentLine/Cut1/JobReport'
-- 'rg_v2/RG/CPH/Prod/ComponentLine/Pick1/JobReport'
```

**Payload Schema (Print1 Example):**
```json
{
  "jobId": "RG-12345",
  "data": {
    "ids": {
      "batchId": "876",
      "sheetIndex": 1
    },
    "reportData": {
      "oee": {
        "cycleTime": 10.5,
        "runtime": 8.2,
        "performance": 85.3,
        "availability": 78.2
      },
      "printSettings": {
        "speed": 45,
        "nominalSpeed": 50
      },
      "jobInfo": {
        "area": 2.5,
        "width": 1500,
        "height": 2000
      },
      "timing": {
        "start": "2024-01-15T08:00:00Z",
        "end": "2024-01-15T08:10:30Z"
      }
    },
    "time": {
      "uptime": 492,    // seconds
      "downtime": 138   // seconds
    }
  }
}
```

#### INSIDE Database (Production ERP)
**Purpose:** Tracks production jobs and quality control data

**Table: `production_printjob`**
```sql
CREATE TABLE production_printjob (
    id BIGSERIAL PRIMARY KEY,
    rg_id TEXT,                    -- Job ID (matches HISTORIAN.jobs.payload->>'jobId')
    production_batch_id BIGINT,    -- Batch number
    component_order_count INT,     -- Number of components
    status TEXT
);
```

**Table: `production_componentorder`**
```sql
CREATE TABLE production_componentorder (
    id BIGSERIAL PRIMARY KEY,
    print_job_id BIGINT REFERENCES production_printjob(id),
    production_batch_id BIGINT,
    qc_state TEXT  -- 'passed', 'failed', 'not qced'
);
```

### Data Relationships

```
Batch (e.g., "876")
  └─> Multiple Print Jobs (RG-12345, RG-12346, ...)
       ├─> Cut Job (1:1 mapping via jobId)
       ├─> Pick Job (1:1 mapping via jobId)
       └─> Quality Data (aggregated per batch)

Time Windows (per cell):
  Print: [2024-01-15 08:00 - 2024-01-15 16:00]
  Cut:   [2024-01-16 08:00 - 2024-01-16 14:00]
  Pick:  [2024-01-16 14:00 - 2024-01-16 18:00]
```

---

## OEE Calculation Specifications

### Calculation Functions

#### Cell OEE Calculator
```python
def calculate_cell_oee(cell_type: str, uptime: float, downtime: float,
                       performance_ratio: float = 1.0,
                       quality_ratio: float = 1.0) -> dict:
    """
    Calculate OEE for a specific manufacturing cell.

    Args:
        cell_type: 'printer', 'cut', or 'pick'
        uptime: Total uptime in minutes
        downtime: Total downtime in minutes
        performance_ratio: Actual speed / Nominal speed (0.0-1.0)
        quality_ratio: Success rate (0.0-1.0)

    Returns:
        {
            'availability': float,  # 0.0-1.0
            'performance': float,   # 0.0-1.0
            'quality': float,       # 0.0-1.0
            'oee': float           # 0.0-1.0
        }
    """
    total_time = uptime + downtime
    availability = uptime / total_time if total_time > 0 else 0.0

    if cell_type == 'printer':
        performance = performance_ratio
        quality = 1.0  # Fixed
    elif cell_type == 'cut':
        performance = 1.0  # Fixed
        quality = 1.0      # Fixed
    elif cell_type == 'pick':
        performance = 1.0       # Fixed
        quality = quality_ratio  # Pick success rate
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    oee = availability * performance * quality

    return {
        'availability': availability,
        'performance': performance,
        'quality': quality,
        'oee': oee
    }
```

#### System OEE Calculator
```python
def calculate_system_oee(printer_oee: float, cut_oee: float,
                         pick_oee: float) -> float:
    """
    Calculate overall system OEE as the product of individual cell OEEs.

    Args:
        printer_oee: OEE value for printer cell (0.0-1.0)
        cut_oee: OEE value for cut cell (0.0-1.0)
        pick_oee: OEE value for pick cell (0.0-1.0)

    Returns:
        System OEE value (0.0-1.0)

    Example:
        >>> system_oee = calculate_system_oee(0.901, 0.846, 0.901)
        >>> print(f"System OEE: {system_oee:.1%}")
        'System OEE: 68.7%'
    """
    return printer_oee * cut_oee * pick_oee
```

### Metric Extraction

#### Print Data Extraction
```python
# From HISTORIAN.jobs.payload (JSONB)
SELECT
    payload->>'jobId' as job_id,
    payload->'data'->'ids'->>'batchId' as batch_id,
    (payload->'data'->'time'->>'uptime')::numeric / 60.0 as uptime_minutes,
    (payload->'data'->'time'->>'downtime')::numeric / 60.0 as downtime_minutes,
    (payload->'data'->'reportData'->'printSettings'->>'speed')::numeric as actual_speed,
    (payload->'data'->'reportData'->'printSettings'->>'nominalSpeed')::numeric as nominal_speed,
    payload->'data'->'reportData'->'timing'->>'start' as start_time,
    payload->'data'->'reportData'->'timing'->>'end' as end_time
FROM jobs
WHERE topic LIKE '%Print1/JobReport%'
    AND payload->'data'->'ids'->>'batchId' = ANY(%s)  -- Batch ID filter
ORDER BY ts;
```

#### Cut Data Extraction
```python
# From HISTORIAN.jobs.payload (JSONB)
SELECT
    payload->>'jobId' as job_id,
    (payload->'data'->'time'->>'uptime')::numeric / 60.0 as uptime_minutes,
    (payload->'data'->'time'->>'downtime')::numeric / 60.0 as downtime_minutes,
    payload->'data'->'reportData'->'timing'->>'start' as start_time,
    payload->'data'->'reportData'->'timing'->>'end' as end_time
FROM jobs
WHERE topic LIKE '%Cut1/JobReport%'
    AND payload->>'jobId' = ANY(%s)  -- Job ID filter
ORDER BY ts;
```

#### Pick Data Extraction
```python
# From HISTORIAN.jobs.payload (JSONB)
SELECT
    payload->>'jobId' as job_id,
    (payload->'data'->'time'->>'uptime')::numeric / 60.0 as uptime_minutes,
    (payload->'data'->'time'->>'downtime')::numeric / 60.0 as downtime_minutes,
    (payload->'data'->'reportData'->'metrics'->>'successfulPicks')::int as successful_picks,
    (payload->'data'->'reportData'->'metrics'->>'totalPicks')::int as total_picks,
    payload->'data'->'reportData'->'timing'->>'start' as start_time,
    payload->'data'->'reportData'->'timing'->>'end' as end_time
FROM jobs
WHERE topic LIKE '%Pick1/JobReport%'
    AND payload->>'jobId' = ANY(%s)  -- Job ID filter
ORDER BY ts;
```

---

## Time Window Management

### Requirements

#### Per-Cell Time Windows
Each cell can have different time windows to accommodate:
- Multi-day batch operations
- Shift changes
- Equipment maintenance periods
- Non-continuous production

#### Allowable Segments
Define specific time ranges to include in analysis:
```python
# Example: Track a batch across two days
time_windows = {
    'printer': [
        {'start': '2024-01-15 08:00:00', 'end': '2024-01-15 16:00:00'},  # Day 1
        {'start': '2024-01-16 08:00:00', 'end': '2024-01-16 12:00:00'}   # Day 2 morning
    ],
    'cut': [
        {'start': '2024-01-16 13:00:00', 'end': '2024-01-16 18:00:00'}   # Day 2 afternoon
    ],
    'pick': [
        {'start': '2024-01-16 18:00:00', 'end': '2024-01-17 02:00:00'}   # Day 2 night shift
    ]
}
```

### Implementation Schema

#### Time Window Configuration
```python
@dataclass
class TimeSegment:
    start: datetime
    end: datetime
    description: str = ""

@dataclass
class CellTimeWindow:
    cell_type: str  # 'printer', 'cut', 'pick'
    segments: List[TimeSegment]
    timezone: str = "Europe/Copenhagen"

    def total_duration_minutes(self) -> float:
        """Calculate total time across all segments."""
        return sum(
            (seg.end - seg.start).total_seconds() / 60.0
            for seg in self.segments
        )

    def is_within_window(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within any segment."""
        return any(
            seg.start <= timestamp <= seg.end
            for seg in self.segments
        )
```

#### Data Filtering by Time Window
```python
def filter_data_by_time_window(df: pd.DataFrame,
                                time_window: CellTimeWindow) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows within time window segments.

    Args:
        df: DataFrame with 'start_time' column (datetime)
        time_window: CellTimeWindow configuration

    Returns:
        Filtered DataFrame
    """
    mask = df['start_time'].apply(time_window.is_within_window)
    return df[mask]
```

### UI for Time Window Selection

**Default Behavior:**
- Auto-detect time range from batch data (earliest start to latest end per cell)
- Display suggested time windows based on data availability

**Manual Override:**
- Date/time pickers for each cell
- Add/remove segment buttons
- Preset templates (same day, overnight batch, weekend production)

---

## Application Architecture

### Simplified Architecture

```
keen-tesla-v2/
├── app.py                      # Main Streamlit application (~800 lines)
├── .env                        # Environment variables
├── requirements.txt            # Dependencies
├── SPECIFICATIONS.md           # This document
│
├── core/
│   ├── __init__.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── pool.py            # Connection pooling (REUSE from db_pool.py)
│   │   ├── queries.py         # Secure query builder (REUSE from secure_queries.py)
│   │   └── fetchers.py        # Batch-based data retrieval (SIMPLIFY from data_fetchers.py)
│   │
│   ├── calculations/
│   │   ├── __init__.py
│   │   └── oee.py             # NEW: Simplified OEE calculations
│   │
│   └── time_windows/
│       ├── __init__.py
│       ├── models.py          # NEW: TimeSegment, CellTimeWindow classes
│       └── filters.py         # NEW: Time window filtering logic
│
├── ui/
│   ├── __init__.py
│   ├── batch_selector.py      # SIMPLIFY from batch_selection_ui.py
│   ├── time_window_selector.py # NEW: Time window configuration UI
│   └── metrics_display.py     # NEW: OEE metrics visualization
│
└── utils/
    ├── __init__.py
    ├── formatting.py          # REUSE from data_processors.py
    └── config.py              # NEW: Simple config loading
```

### Module Responsibilities

#### `core/db/pool.py` (REUSE)
- Manage connection pools for HISTORIAN and INSIDE databases
- Thread-safe connection handling
- Health checks and reconnection logic

#### `core/db/queries.py` (REUSE)
- SecureQueryBuilder class
- Parameterized query construction
- Input validation

#### `core/db/fetchers.py` (SIMPLIFY)
**Keep these functions:**
- `fetch_print_data_by_batch(batch_ids, time_window)`
- `fetch_cut_data_by_jobs(job_ids, time_window)`
- `fetch_pick_data_by_jobs(job_ids, time_window)`
- `fetch_quality_metrics_by_batch(batch_ids)` - Separate from OEE

**Remove:**
- All time-range based fetchers
- Async variants
- Multi-level aggregation fetchers

#### `core/calculations/oee.py` (NEW)
```python
# Single file with ~200 lines
- calculate_cell_oee()
- calculate_system_oee()  # Product of individual cell OEEs
- aggregate_batch_metrics()  # Simple sum of uptime/downtime across jobs
```

#### `core/time_windows/models.py` (NEW)
```python
# Dataclasses for time window management
- TimeSegment
- CellTimeWindow
- TimeWindowPreset (templates)
```

#### `ui/batch_selector.py` (SIMPLIFY)
```python
# Streamlit components for batch selection
- render_batch_input()        # Text input or dropdown
- fetch_available_batches()   # Get batch list from database
- validate_batch_selection()  # Ensure batches exist
```

#### `ui/time_window_selector.py` (NEW)
```python
# Streamlit components for time window configuration
- render_time_window_editor(cell_type)
- add_time_segment()
- remove_time_segment()
- apply_preset_template()
```

#### `ui/metrics_display.py` (NEW)
```python
# Display OEE metrics and system OEE
- render_cell_metrics_card(cell_data)
- render_system_oee_summary(system_oee, cell_oees)
- render_timeline_chart(filtered_data)  # REUSE from visualizations.py
```

---

## User Interface Requirements

### Page Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Manufacturing OEE Analytics                                     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                   │
│  📦 Batch Selection                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Select Batches: [876, 877, 878]              [Load Data]  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ⏰ Time Window Configuration                                    │
│  ┌──────────────────┬──────────────────┬──────────────────────┐ │
│  │ PRINTER          │ CUT              │ PICK                 │ │
│  │ 2024-01-15 08:00 │ 2024-01-16 13:00 │ 2024-01-16 18:00     │ │
│  │   to 16:00       │   to 18:00       │   to 02:00 (+1 day)  │ │
│  │ [+ Add Segment]  │ [+ Add Segment]  │ [+ Add Segment]      │ │
│  └──────────────────┴──────────────────┴──────────────────────┘ │
│                                                                   │
│  📊 Cell OEE Metrics                                             │
│  ┌──────────────────┬──────────────────┬──────────────────────┐ │
│  │ PRINTER          │ CUT              │ PICK                 │ │
│  │ Availability: 82%│ Availability: 95%│ Availability: 88%    │ │
│  │ Performance: 87% │ Performance: 100%│ Performance: 100%    │ │
│  │ Quality: 100%    │ Quality: 100%    │ Quality: 94%         │ │
│  │ ━━━━━━━━━━━━━━━ │ ━━━━━━━━━━━━━━━ │ ━━━━━━━━━━━━━━━━━━  │ │
│  │ OEE: 71.3%       │ OEE: 95.0%       │ OEE: 82.7%           │ │
│  └──────────────────┴──────────────────┴──────────────────────┘ │
│                                                                   │
│  📊 System OEE                                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ System OEE: 56.0%                                          │  │
│  │                                                             │  │
│  │ Calculation:                                                │  │
│  │ 71.3% × 95.0% × 82.7% = 56.0%                              │  │
│  │                                                             │  │
│  │ This represents the overall effectiveness of the entire     │  │
│  │ production system, where all cells must perform well for   │  │
│  │ high system output.                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  📈 Detailed Metrics                                             │
│  [Timeline Chart] [Download CSV]                                │
└─────────────────────────────────────────────────────────────────┘
```

### Key UI Components

#### 1. Batch Selection
- **Input method:** Text input (comma-separated) or multi-select dropdown
- **Validation:** Check batch existence in database
- **Feedback:** Show number of jobs found per batch

#### 2. Time Window Configuration
- **Per-cell tabs** or columns
- **Segment management:**
  - Date/time pickers for start/end
  - Add/remove segment buttons
  - Visual indicator of total duration
- **Presets:**
  - "Same Day" - Auto-detect from data
  - "Overnight Batch" - Day 1 evening → Day 2 morning
  - "Custom" - Manual entry

#### 3. Cell Metrics Cards
- **Large OEE number** (primary metric)
- **Component breakdown** (A%, P%, Q%)
- **Color coding:**
  - Green: OEE ≥ 85%
  - Yellow: 70% ≤ OEE < 85%
  - Red: OEE < 70%

#### 4. System OEE Summary
- **Display overall system OEE** with calculation breakdown
- **Show individual cell contributions:** Each cell's OEE and its impact
- **Visual indicator:** Color-coded based on system OEE value
- **Actionable insights:** Identify which cells need improvement to boost system performance

#### 5. Timeline Chart (REUSE)
- Gantt-style chart showing job timing per cell
- Color-coded by job status
- Hover tooltips with job details

---

## Components to Reuse

### 1. Database Infrastructure (HIGH PRIORITY)

**File: `db_pool.py` (313 lines) → `core/db/pool.py`**
- ✅ Thread-safe connection pooling
- ✅ Health checks and reconnection
- ✅ Context manager pattern
- **Minor refactoring:** Remove legacy `get_connection_direct()` method

**File: `secure_queries.py` (709 lines) → `core/db/queries.py`**
- ✅ SecureQueryBuilder class
- ✅ Parameterized queries (SQL injection prevention)
- ✅ Input validation (datetime, batch IDs, job IDs)
- **No changes needed**

### 2. Data Fetching (MEDIUM PRIORITY)

**File: `data_fetchers.py` (782 lines) → `core/db/fetchers.py`**
- ✅ `fetch_print_data_by_batch()` - Core batch retrieval
- ✅ `fetch_cut_data_by_jobs()` - Job-matched cut data
- ✅ `fetch_pick_data_by_jobs()` - Robot/pick data
- ✅ `fetch_quality_metrics_by_batch()` - QC data (separate from OEE)
- ✅ Query caching with TTL
- ✅ Retry decorators
- **Refactoring needed:**
  - Add `time_window: CellTimeWindow` parameter to all fetch functions
  - Remove time-range based variants
  - Simplify to ~400 lines

### 3. Data Processing Utilities (LOW PRIORITY)

**File: `data_processors.py` (379 lines) → `utils/formatting.py`**
- ✅ `format_timestamp()` - Datetime formatting
- ✅ `convert_all_datetime_to_str()` - DataFrame conversion
- ✅ `validate_time_range()` - Time validation
- ✅ JSON parsing functions
- **Refactoring needed:**
  - Extract only essential functions (~150 lines)
  - Remove `calculate_batch_metrics()` (replaced by new oee.py)

### 4. UI Components (MEDIUM PRIORITY)

**File: `batch_selection_ui.py` (280 lines) → `ui/batch_selector.py`**
- ✅ Batch input rendering
- ✅ Available batch fetching
- ✅ Validation logic
- **Refactoring needed:**
  - Simplify to ~150 lines
  - Remove unused visualization components

**File: `visualizations.py` (368 lines) → `ui/metrics_display.py`**
- ✅ `create_timeline_chart()` - Gantt-style job timeline
- ✅ `show_data_preview()` - DataFrame display
- **Refactoring needed:**
  - Extract only timeline chart (~100 lines)
  - Remove ink consumption, batch comparison charts

---

## Components to Remove

### 1. Overcomplicated Calculations

**File: `oee_calculator.py` (838 lines) → DELETE**
- ❌ `calculate_job_level_oee()` - Too granular
- ❌ `calculate_batch_level_oee()` - Redundant aggregation
- ❌ `calculate_multi_batch_comparison()` - Not needed
- ❌ `calculate_stage_correlation_oee()` - Over-engineered
- ❌ `get_oee_insights()` - Grading system removed
- ❌ Complex integrated Cut+Pick logic

**Replacement:** New `core/calculations/oee.py` (~200 lines)

### 2. Unnecessary Infrastructure

**File: `async_data_processor.py` (374 lines) → DELETE**
- ❌ Async fetching adds complexity
- ❌ Batch queries don't benefit from async
- Synchronous fetching is sufficient

**File: `pagination_utils.py` (177 lines) → DELETE**
- ❌ Memory optimization not needed for batch-level data
- Typical dataset: <10,000 rows (easily handled in memory)

**File: `sql_optimizer.py` (230 lines) → DELETE**
- ❌ Index recommendations not useful in application code
- Database tuning should be done separately

**File: `config_validator.py` (206 lines) → DELETE**
- ❌ Over-engineered validation
- Simple `.env` validation with `python-dotenv` is sufficient

**File: `query_cache.py` (150 lines) → DELETE**
- ❌ Caching is already built into `data_fetchers.py`
- Remove standalone cache dashboard

**File: `google_sheets_client.py` (187 lines) → DELETE**
- ❌ Google Sheets integration no longer needed
- Replaced with simple CSV download functionality

### 3. Unused Visualizations

**File: `visualizations.py` (portions) → DELETE**
- ❌ `create_ink_consumption_chart()` - Not mentioned in requirements
- ❌ `create_batch_comparison_chart()` - Replaced by simple metrics cards
- ❌ `create_data_quality_indicators()` - Over-engineered

### 4. Bloated Main Application

**File: `app.py` (4332 lines) → REDESIGN**
- ❌ Time-range mode UI (replaced by time windows)
- ❌ Multi-level OEE displays
- ❌ Redundant data processing logic
- **Target:** Reduce to ~800 lines with modular architecture

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
**Goal:** Set up clean architecture and core infrastructure

#### Tasks:
1. ✅ Create new directory structure
2. ✅ Copy and refactor database layer:
   - `core/db/pool.py` (from `db_pool.py`)
   - `core/db/queries.py` (from `secure_queries.py`)
3. ✅ Copy and refactor data fetchers:
   - `core/db/fetchers.py` (from `data_fetchers.py`)
   - Add `time_window` parameter support
4. ✅ Set up configuration:
   - `utils/config.py` - Simple `.env` loader
   - Validate required environment variables

#### Deliverables:
- Working database connections
- Functional data fetching with time window filtering
- Unit tests for core functions

### Phase 2: OEE Calculations (Week 2)
**Goal:** Implement simplified OEE logic

#### Tasks:
1. ✅ Create time window models:
   - `core/time_windows/models.py`
   - `core/time_windows/filters.py`
2. ✅ Implement OEE calculator:
   - `core/calculations/oee.py`
   - `calculate_cell_oee()` function
   - `calculate_system_oee()` function
3. ✅ Write comprehensive tests:
   - Test each cell type formula
   - Test system OEE calculation
   - Edge cases (zero uptime, 100% downtime)

#### Deliverables:
- Functional OEE calculations for all three cells
- System OEE calculation working
- Test coverage >90%

### Phase 3: User Interface (Week 3)
**Goal:** Build Streamlit UI with batch selection and time windows

#### Tasks:
1. ✅ Create batch selector:
   - `ui/batch_selector.py`
   - Text input and validation
   - Display available batches
2. ✅ Create time window selector:
   - `ui/time_window_selector.py`
   - Per-cell time segment management
   - Preset templates
3. ✅ Create metrics display:
   - `ui/metrics_display.py`
   - Cell OEE cards
   - System OEE summary
   - Timeline chart (reuse from `visualizations.py`)
4. ✅ Build main app:
   - `app.py` (~800 lines)
   - Integrate all UI components
   - Data flow: Batch selection → Time windows → Fetch → Calculate → Display

#### Deliverables:
- Functional Streamlit application
- All UI components working
- Responsive design

### Phase 4: Data Export & Polish (Week 4)
**Goal:** CSV export functionality and final refinements

#### Tasks:
1. ✅ Implement CSV export:
   - Generate CSV files for OEE metrics
   - Include system OEE calculation
   - Add quality metrics (separate from OEE)
2. ✅ Add download buttons to UI:
   - Download OEE metrics as CSV
   - Download detailed job data as CSV
3. ✅ Error handling and user feedback:
   - Connection errors
   - Invalid batch IDs
   - Empty data sets
4. ✅ Performance optimization:
   - Query caching
   - Memoization for expensive calculations
5. ✅ Documentation:
   - User guide (README.md)
   - Developer setup instructions
   - API documentation

#### Deliverables:
- Working CSV export functionality
- Polished user experience
- Complete documentation

### Phase 5: Migration & Deployment (Week 5)
**Goal:** Migrate from old system to new system

#### Tasks:
1. ✅ Side-by-side comparison:
   - Run both old and new systems
   - Validate OEE calculations match expected values
   - User acceptance testing
2. ✅ Data migration:
   - Ensure all historical batches work
   - Test edge cases (multi-day batches, missing data)
3. ✅ Production deployment:
   - Environment setup
   - Database connection strings
   - SSL/security configuration
4. ✅ Decommission old system:
   - Archive old codebase
   - Update documentation
   - Redirect users to new application

#### Deliverables:
- Production-ready application
- User training completed
- Old system archived

---

## Technical Considerations

### Performance Targets
- **Page load time:** < 2 seconds for batch selection
- **Data fetch time:** < 5 seconds for 3 batches with 500 jobs total
- **OEE calculation:** < 1 second for 1000 jobs
- **CSV export:** < 2 seconds for generating downloadable files

### Scalability
- **Concurrent users:** Support 10+ simultaneous users
- **Data volume:** Handle batches with 1000+ jobs
- **Query optimization:** Use database indexes on `batch_id`, `job_id`, `topic`, `ts`

### Error Handling
- **Database connection failures:** Retry with exponential backoff
- **Invalid batch IDs:** Display friendly error message
- **Empty data sets:** Show "No data found" message
- **Export failures:** Clear error messages and retry options

### Security
- **Database credentials:** Store in `.env` file (not in code)
- **SQL injection prevention:** Use parameterized queries only
- **Input validation:** Sanitize all user inputs
- **Connection encryption:** Use SSL for database connections

---

## Success Criteria

### Functional Requirements
- ✅ Calculate OEE for Printer (A×P×100%), Cut (A×100%×100%), Pick (A×100%×Q_acc)
- ✅ Calculate system OEE as product of individual cell OEEs
- ✅ Support per-cell time windows with multiple segments
- ✅ Export data to CSV format
- ✅ Display timeline charts and metrics cards

### Non-Functional Requirements
- ✅ Codebase reduced from ~10,000 lines to ~3,000 lines
- ✅ Maintainability: Single source of truth for OEE calculations
- ✅ Performance: Sub-5-second data fetching for typical batches
- ✅ Usability: Intuitive UI requiring minimal training
- ✅ Reliability: 99.9% uptime with proper error handling

### User Acceptance
- ✅ Manufacturing team can track OEE without manual calculations
- ✅ System OEE calculation provides clear overall performance metric
- ✅ Multi-day batch tracking works seamlessly
- ✅ CSV export enables flexible data analysis

---

## Appendix

### A. Sample Data Flow

```
1. User Input:
   Batches: [876, 877]
   Time Windows:
     Printer: 2024-01-15 08:00 - 16:00
     Cut:     2024-01-16 08:00 - 14:00
     Pick:    2024-01-16 14:00 - 20:00

2. Data Fetching:
   fetch_print_data_by_batch([876, 877], printer_window)
   → Returns 120 print jobs with uptime, downtime, speed data

   fetch_cut_data_by_jobs([job_ids], cut_window)
   → Returns 120 cut jobs matched by job_id

   fetch_pick_data_by_jobs([job_ids], pick_window)
   → Returns 120 pick jobs with success rates

3. Time Filtering:
   Filter each dataset to only include jobs within respective time windows

4. OEE Calculation:
   For each batch:
     Printer: calculate_cell_oee('printer', uptime=350, downtime=90, performance=0.87)
     → {availability: 0.795, performance: 0.87, quality: 1.0, oee: 0.692}

     Cut: calculate_cell_oee('cut', uptime=280, downtime=20, performance=1.0)
     → {availability: 0.933, performance: 1.0, quality: 1.0, oee: 0.933}

     Pick: calculate_cell_oee('pick', uptime=310, downtime=50, quality=0.94)
     → {availability: 0.861, performance: 1.0, quality: 0.94, oee: 0.809}

5. System OEE Calculation:
   calculate_system_oee(0.692, 0.933, 0.809)
   → 0.692 × 0.933 × 0.809 = 0.511 (51.1%)

6. Display:
   Render metrics cards with color coding (Printer=Red, Cut=Green, Pick=Yellow)
   Show system OEE summary: "System OEE: 51.1% (69.2% × 93.3% × 80.9%)"

7. Export:
   generate_csv(batch_metrics, system_oee_data, quality_data)
   → Returns CSV file for download
```

### B. Configuration Example

**`.env` file:**
```bash
# HISTORIAN Database (MQTT data)
HISTORIANDB_HOST=historian.production.local
HISTORIANDB_PORT=5432
HISTORIANDB_NAME=manufacturing
HISTORIANDB_USER=analytics_ro
HISTORIANDB_PASS=secure_password_123

# INSIDE Database (ERP data)
INSIDEDB_HOST=inside.production.local
INSIDEDB_PORT=5432
INSIDEDB_NAME=production_erp
INSIDEDB_USER=analytics_ro
INSIDEDB_PASS=secure_password_456

# Application Settings
TIMEZONE=Europe/Copenhagen
DEFAULT_TIME_WINDOW_HOURS=8
CACHE_TTL_SECONDS=300
```

### C. Dependencies

**`requirements.txt`:**
```
streamlit==1.32.0
pandas==2.1.4
psycopg2-binary==2.9.9
python-dotenv==1.0.1
pytz==2024.1
plotly==5.18.0  # For timeline charts
pytest==7.4.3   # For testing
```

### D. Database Indexes (Recommended)

```sql
-- HISTORIAN.jobs table
CREATE INDEX idx_jobs_topic ON jobs(topic);
CREATE INDEX idx_jobs_ts ON jobs(ts);
CREATE INDEX idx_jobs_batch_id ON jobs USING GIN ((payload->'data'->'ids'->>'batchId'));
CREATE INDEX idx_jobs_job_id ON jobs USING GIN ((payload->>'jobId'));

-- INSIDE.production_printjob table
CREATE INDEX idx_printjob_rg_id ON production_printjob(rg_id);
CREATE INDEX idx_printjob_batch_id ON production_printjob(production_batch_id);

-- INSIDE.production_componentorder table
CREATE INDEX idx_componentorder_batch_id ON production_componentorder(production_batch_id);
CREATE INDEX idx_componentorder_qc_state ON production_componentorder(qc_state);
```

---

## Document Revision History

| Version | Date       | Author | Changes                          |
|---------|------------|--------|----------------------------------|
| 1.0     | 2026-01-06 | System | Initial specification document   |
| 2.0     | 2026-01-06 | System | Added time window requirements,  |
|         |            |        | simplified OEE formulas,         |
|         |            |        | system OEE calculation           |
| 2.1     | 2026-01-06 | System | Removed Google Sheets integration,|
|         |            |        | replaced with CSV export         |

---

**End of Specifications Document**
