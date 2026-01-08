# Manufacturing OEE Analytics v2.0

Redesigned manufacturing analytics platform providing clear, actionable OEE metrics for three production cells (Printer, Cut, Pick) with simplified calculations and flexible time-window tracking.

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
# Copy the environment template
cp .env.template .env

# Edit .env with your database credentials
nano .env  # or use your preferred editor
```

### 3. Run the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

## Architecture

```
operationlabs_v2/
├── core/
│   ├── db/
│   │   ├── pool.py          # Database connection pooling
│   │   ├── queries.py       # Secure query builder (TODO)
│   │   └── fetchers.py      # Data retrieval functions (TODO)
│   ├── calculations/
│   │   └── oee.py          # Simplified OEE calculations
│   └── time_windows/
│       ├── models.py       # TimeSegment, CellTimeWindow classes
│       └── filters.py      # Time window filtering utilities
├── ui/
│   ├── batch_selector.py    # Batch selection UI (TODO)
│   ├── time_window_selector.py  # Time window config UI (TODO)
│   └── metrics_display.py   # OEE metrics visualization (TODO)
├── utils/
│   ├── config.py           # Configuration management
│   └── formatting.py       # Data formatting utilities (TODO)
├── app.py                   # Main Streamlit application (TODO)
├── requirements.txt         # Python dependencies
├── .env.template           # Environment template
└── SPECIFICATIONS.md       # Detailed technical specifications
```

## OEE Calculation Formulas

### Per-Cell OEE

**Printer:**
```
OEE = Availability × Performance × 100%
where:
  Availability = Uptime / (Uptime + Downtime)
  Performance  = Actual Speed / Nominal Speed
  Quality      = 100% (fixed)
```

**Cut:**
```
OEE = Availability × 100% × 100%
where:
  Availability = Uptime / (Uptime + Downtime)
  Performance  = 100% (fixed)
  Quality      = 100% (fixed)
```

**Pick:**
```
OEE = Availability × 100% × Quality_accuracy
where:
  Availability    = Uptime / (Uptime + Downtime)
  Performance     = 100% (fixed)
  Quality_accuracy = Successful Picks / Total Picks
```

### Bottleneck Identification

```python
Bottleneck = min(OEE_printer, OEE_cut, OEE_pick)
```

**Note:** Overall quality metrics (from QC data) are tracked separately to avoid time offset issues between production and quality inspection.

## Usage Examples

### Basic OEE Calculation

```python
from core.calculations.oee import calculate_cell_oee, identify_bottleneck

# Calculate OEE for each cell
printer_metrics = calculate_cell_oee('printer', uptime=350, downtime=90, performance_ratio=0.87)
cut_metrics = calculate_cell_oee('cut', uptime=280, downtime=20)
pick_metrics = calculate_cell_oee('pick', uptime=310, downtime=50, quality_ratio=0.94)

# Identify bottleneck
bottleneck = identify_bottleneck(
    printer_metrics.oee,
    cut_metrics.oee,
    pick_metrics.oee
)

print(f"Printer OEE: {printer_metrics.oee:.1%}")
print(f"Cut OEE: {cut_metrics.oee:.1%}")
print(f"Pick OEE: {pick_metrics.oee:.1%}")
print(f"\nBottleneck: {bottleneck.limiting_factor}")
```

### Time Window Management

```python
from datetime import datetime
import pytz
from core.time_windows.models import CellTimeWindow, TimeSegment, TimeWindowPreset

# Option 1: Manual segment creation
tz = pytz.timezone("Europe/Copenhagen")
segment1 = TimeSegment(
    start=tz.localize(datetime(2024, 1, 15, 8, 0)),
    end=tz.localize(datetime(2024, 1, 15, 16, 0)),
    description="Day shift"
)

window = CellTimeWindow('printer', [segment1])

# Option 2: Use presets
window = TimeWindowPreset.same_day(
    'printer',
    date=datetime(2024, 1, 15),
    start_hour=8,
    end_hour=16
)

# Option 3: Overnight batch
window = TimeWindowPreset.overnight_batch(
    'cut',
    start_date=datetime(2024, 1, 15),
    evening_start_hour=18,
    morning_end_hour=6
)
```

### Filter Data by Time Window

```python
from core.time_windows.filters import filter_dataframe_by_time_window
import pandas as pd

# Assuming you have a DataFrame with print jobs
filtered_df = filter_dataframe_by_time_window(
    df=print_jobs_df,
    time_window=printer_window,
    timestamp_column='start_time'
)

print(f"Filtered from {len(print_jobs_df)} to {len(filtered_df)} jobs")
```

## Database Schema

### HISTORIAN Database
Stores MQTT messages from manufacturing equipment in the `jobs` table:

```json
{
  "jobId": "RG-12345",
  "data": {
    "ids": {"batchId": "876", "sheetIndex": 1},
    "reportData": {
      "oee": {"cycleTime": 10.5, "runtime": 8.2, "performance": 85.3},
      "printSettings": {"speed": 45, "nominalSpeed": 50},
      "timing": {"start": "2024-01-15T08:00:00Z", "end": "2024-01-15T08:10:30Z"}
    },
    "time": {"uptime": 492, "downtime": 138}
  }
}
```

### INSIDE Database
Production ERP data:
- `production_printjob`: Job information
- `production_componentorder`: Quality control data

## Development Status

### ✅ Completed
- [x] Project structure
- [x] OEE calculation engine
- [x] Time window management
- [x] Database connection pooling
- [x] Configuration management

### 🚧 In Progress (Ready for Cursor)
- [ ] Secure query builder (`core/db/queries.py`)
- [ ] Data fetchers with time window support (`core/db/fetchers.py`)
- [ ] Batch selector UI (`ui/batch_selector.py`)
- [ ] Time window selector UI (`ui/time_window_selector.py`)
- [ ] Metrics display UI (`ui/metrics_display.py`)
- [ ] Main Streamlit application (`app.py`)
- [ ] Data formatting utilities (`utils/formatting.py`)

### 📋 TODO
- [ ] Unit tests
- [ ] CSV export functionality
- [ ] Timeline visualization
- [ ] User documentation

## Contributing

This is a redesign from scratch. See `SPECIFICATIONS.md` for detailed technical specifications and architecture decisions.

## Migration from v1

The v2 redesign simplifies the codebase from ~10,000 lines to ~3,000 lines by:
- Removing redundant OEE calculation layers
- Eliminating async/pagination overhead
- Simplifying per-cell OEE formulas
- Replacing Google Sheets with CSV export
- Focusing on batch-based analysis

## License

Internal use for Rodinia manufacturing operations.
