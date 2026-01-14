# Manufacturing OEE Analytics v2.0

Batch-centric manufacturing analytics platform providing clear, actionable OEE metrics for three production cells (Printer, Cut, Pick) with batch-level and daily-level analysis, quality overlay, and comprehensive job-level details.

## Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the project
cd operationlabs_v2

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Database Connections

```bash
# Copy the environment template (if it exists)
# Create .env file with your database credentials

# Required environment variables:
# HISTORIANDB_HOST, HISTORIANDB_PORT, HISTORIANDB_NAME, HISTORIANDB_USER, HISTORIANDB_PASS
# INSIDEDB_HOST, INSIDEDB_PORT, INSIDEDB_NAME, INSIDEDB_USER, INSIDEDB_PASS
# TIMEZONE (default: Europe/Copenhagen)
# DEFAULT_OPERATING_HOURS (default: 24)
# LOG_LEVEL (default: INFO)
```

### 3. Run the Application

```bash
# Start the Streamlit app
streamlit run ui/app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Architecture

```
operationlabs_v2/
├── ui/
│   └── app.py              # Main Streamlit application (batch-centric OEE analysis)
├── analysis/
│   ├── oee_calculator.py   # Batch-level and daily-level OEE calculations
│   └── quality_overlay.py  # QC quality overlay logic
├── core/
│   ├── analysis/
│   │   └── oee.py          # Job-level OEE calculations and QC quality application
│   ├── db/
│   │   ├── pool.py         # Database connection pooling
│   │   ├── queries.py      # Secure query builder for SQL queries
│   │   └── fetchers.py     # Advanced data retrieval functions
│   └── time_windows/
│       ├── models.py       # TimeSegment, CellTimeWindow classes
│       └── filters.py      # Time window filtering utilities
├── db/
│   ├── connections.py      # Database connection management
│   └── fetchers.py         # Core data fetching functions
├── utils/
│   └── formatting.py       # Data formatting utilities
├── config.py               # Application configuration (Config class)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Features

### Batch-Centric Analysis
- **Batch Discovery**: Automatically discover batches produced on a selected day
- **Multi-Cell Tracking**: Track Print, Cut, and Pick jobs per batch
- **Time Window Validation**: View start/end times for each cell per batch

### OEE Calculation Framework

The application calculates OEE at three levels:

1. **Job-Level OEE**: Individual job performance metrics
2. **Batch-Level OEE**: Aggregated metrics per production batch
3. **Daily-Level OEE**: Overall daily effectiveness including utilization

### Per-Cell OEE Formulas

**Printer:**
```
OEE = Availability × Performance × Quality
where:
  Availability = Uptime / (Uptime + Downtime)
  Performance  = min(Actual Speed / Nominal Speed, 100%) [capped for batch-level]
  Quality      = 100% (before QC) or QC Pass Rate (after QC)
```

**Cut:**
```
OEE = Availability × Performance × Quality
where:
  Availability = Uptime / (Uptime + Downtime)
  Performance  = 100% (fixed)
  Quality      = 100% (before QC) or QC Pass Rate (after QC)
```

**Pick:**
```
OEE = Availability × Performance × Quality
where:
  Availability    = Running Time / Total Time (from equipment states)
  Performance     = 100% (fixed - robots operate at consistent speed)
  Quality         = Pick Success Rate × QC Pass Rate (compound quality)
```

### Quality Overlay

- **Day 0 (Production)**: Quality assumed at 100% for Print/Cut, pick success rate for Pick
- **Day 3+ (Post-QC)**: Actual QC inspection results automatically applied
- **Quality Breakdown**: Detailed defect attribution by cell (Print, Cut, Fabric, File, Pick, Other)

### Batch-Level Metrics

- **Weighted Averages**: Availability and Performance calculated using time-weighted averages
- **QC-Based Quality**: Quality calculated from batch-level QC data
- **Actual vs Capped Performance**: Print performance shown both actual (uncapped) and capped (for OEE)

### Daily-Level Metrics

- **Utilization**: Production hours / Available hours
- **Daily OEE**: Batch OEE × Utilization
- **Idle Time Tracking**: Time not spent in production

## Usage Guide

### 1. Select Production Day

Choose the date you want to analyze. The application will discover all batches produced on that day.

### 2. Load Batches

Click "Load Batches" to fetch batch information including:
- Batch IDs
- Job counts per cell (Print, Cut, Pick)
- Start and end times for each cell

### 3. Select Batches to Analyze

- Use the multi-select widget to choose specific batches
- View batch details in the table to validate extraction

### 4. Fetch & Analyze Data

Click "Calculate OEE" to:
- Fetch JobReports for Print, Cut, and Pick
- Apply QC quality overlay (if available)
- Calculate job-level, batch-level, and daily-level OEE

### 5. View Results

The application displays:

**Batch-Level OEE Metrics:**
- OEE, Availability, Performance (capped and actual), Quality for each cell
- Formatted with larger, readable numbers

**Daily-Level OEE Metrics:**
- Batch OEE, Production Hours, Idle Hours, Utilization, Daily OEE

**Job-Level Details:**
- Tabs for Print Jobs, Cut Jobs, Pick Jobs
- QC Data with defect attribution
- Components per Job
- Batch Structure (styles and units)

**Export:**
- CSV export of batch metrics and daily metrics

## Data Sources

### HISTORIAN Database
Stores MQTT JobReports from manufacturing equipment:
- **Print JobReports**: Speed, timing, uptime/downtime, batch IDs
- **Cut JobReports**: Component counts, timing, uptime/downtime
- **Pick JobReports**: Component pick status, timing, robot performance
- **Equipment States**: Robot state changes for availability calculation

### INSIDE Database
Production ERP and QC data:
- **production_printjob**: Job metadata and batch associations
- **production_componentorder**: Component-level data
- **production_componentorderevent**: QC inspection results with defect attribution

## Configuration

The application uses a `Config` class in `config.py` that reads from environment variables:

- **Database Connections**: HISTORIAN and INSIDE database credentials
- **Timezone**: Default timezone for date/time operations (default: Europe/Copenhagen)
- **Operating Hours**: Default hours per day for utilization (default: 24)
- **Logging**: Log level configuration (default: INFO)

## Development Status

### ✅ Completed Features
- [x] Batch discovery and selection
- [x] Multi-cell job tracking (Print, Cut, Pick)
- [x] Job-level OEE calculations
- [x] Batch-level OEE with weighted averages
- [x] Daily-level OEE with utilization
- [x] QC quality overlay system
- [x] Quality breakdown by defect type
- [x] Database connection pooling
- [x] Secure query builder
- [x] Time window filtering
- [x] CSV export functionality
- [x] Comprehensive job-level detail views
- [x] Batch structure analysis
- [x] FPY (First Pass Yield) calculations
- [x] Component-level QC tracking

### 🔄 Current Capabilities
- Batch-centric OEE analysis workflow
- Automatic QC data application when available
- Print performance capping at 100% for batch-level calculations
- Equipment state-based Pick availability calculation
- Comprehensive logging and error handling

### 📋 Potential Future Enhancements
- [ ] Throughput metrics (components/hour, garments/hour)
- [ ] Timeline visualization
- [ ] Unit tests
- [ ] Additional export formats
- [ ] Real-time monitoring dashboard
- [ ] Historical trend analysis

## Technical Details

### OEE Calculation Flow

1. **Job-Level**: Calculate OEE for each individual job
   - Availability from uptime/downtime
   - Performance from speed ratios (Print) or fixed 100% (Cut/Pick)
   - Quality from pick success rate (Pick) or 100% assumption (Print/Cut)

2. **Batch-Level Aggregation**:
   - **Availability**: Weighted average by total_time_sec
   - **Performance**: Weighted average by uptime_sec (Print uses capped performance)
   - **Quality**: From batch-level QC data (total_components - defects) / total_components

3. **Daily-Level Calculation**:
   - **Utilization**: Sum of production hours / available hours
   - **Daily OEE**: Average batch OEE × Utilization

### Quality Overlay Logic

- **Print/Cut**: QC pass rate replaces 100% quality assumption
- **Pick**: QC pass rate × pick success rate (compound quality)
- Applied automatically when QC data is available (typically Day 3+)

### Performance Capping

Print performance is capped at 100% for batch-level OEE calculations to ensure OEE doesn't exceed 100%, while actual measured performance is also displayed for assessment purposes.

## Contributing

This is an active manufacturing analytics platform. When making changes:

1. Follow the existing code structure
2. Add logging for debugging
3. Handle edge cases (empty data, missing fields, etc.)
4. Update this README if adding new features

## License

Internal use for Rodinia manufacturing operations.
