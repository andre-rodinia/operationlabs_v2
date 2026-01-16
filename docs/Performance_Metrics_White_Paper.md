# Performance Metrics at Rodinia: Technical White Paper

**Version:** 2.0
**Date:** January 2026
**Author:** Rodinia Operations Analytics Team
**Status:** Production Implementation

---

## Executive Summary

This document defines the comprehensive methodology for calculating Overall Equipment Effectiveness (OEE) metrics across Rodinia's three-stage manufacturing process: Print, Cut, and Pick. The system provides batch-centric analysis with break-aware daily metrics, quality overlay from QC inspection data, and consistent production time measurement across all cells.

**Key Characteristics:**
- Batch-centric analysis (not shift-based)
- Break-aware utilization calculations
- Equipment state-based availability for robotic cells
- Automatic QC quality overlay (Day 3+ post-production)
- Support for both full production batches and rework/defect rounds

---

## Table of Contents

1. [Manufacturing Process Overview](#1-manufacturing-process-overview)
2. [OEE Framework](#2-oee-framework)
3. [Data Sources](#3-data-sources)
4. [Calculation Methodology](#4-calculation-methodology)
5. [Cell-Specific Implementation](#5-cell-specific-implementation)
6. [Quality Overlay System](#6-quality-overlay-system)
7. [Daily-Level Metrics](#7-daily-level-metrics)
8. [Edge Cases and Assumptions](#8-edge-cases-and-assumptions)
9. [Technical Implementation Notes](#9-technical-implementation-notes)
10. [Validation and Testing](#10-validation-and-testing)

---

## 1. Manufacturing Process Overview

### 1.1 Production Flow

Rodinia's manufacturing process follows a sequential three-stage flow:

```
Print Cell → Cut Cell → Pick Cell → Quality Control → Fulfillment
```

**Print Cell:**
- Digital textile printing on fabric
- Variable speed operation (actual vs nominal speed tracked)
- Uptime and downtime tracked per job
- Batch ID assigned at this stage

**Cut Cell:**
- Automated cutting of printed fabric into components
- Fixed-speed operation (100% performance assumed)
- Component-level tracking begins here
- Uptime and downtime tracked per job

**Pick Cell:**
- Robotic picking and sorting of cut components
- Fixed-speed operation (100% performance assumed)
- Equipment state tracking (running/down/idle)
- Component pick success rate tracked

**Quality Control:**
- Post-production inspection (typically 3+ days after production)
- Defect attribution by cell (print, cut, fabric, file, pick, other)
- Pass/fail status per component
- QC data overlaid on production metrics retroactively

### 1.2 Production Patterns

**Full Production Batches:**
- Complete order fulfillment
- All components for all styles produced
- Example: Batch 950 with 2 styles, 12 components total

**Rework/Defect Rounds:**
- Partial batch production
- Only failed components from previous QC are reproduced
- Actual component count < theoretical order quantity
- Example: Batch 961 with 11 components produced (not full 12)

---

## 2. OEE Framework

### 2.1 OEE Definition

Overall Equipment Effectiveness (OEE) is calculated as:

```
OEE = Availability × Performance × Quality
```

Where all factors are expressed as percentages (0-100%).

### 2.2 Three-Level Hierarchy

The system calculates OEE at three distinct levels:

**Level 1: Job-Level OEE**
- Individual job performance metrics
- Raw data from equipment JobReports
- Basis for all higher-level calculations

**Level 2: Batch-Level OEE**
- Aggregated metrics per production batch
- Time-weighted averages for availability and performance
- QC-based quality when available
- Represents "while running" effectiveness

**Level 3: Daily-Level OEE**
- Overall daily effectiveness including utilization
- Break-aware calculations
- Formula: Daily OEE = Batch OEE × Utilization / 100

### 2.3 Key Metrics Definitions

**Availability:**
- Percentage of production time the equipment was running (not down)
- Formula: `Running Time / (Running Time + Downtime) × 100%`
- Excludes idle time (equipment available but not producing)

**Performance:**
- Speed efficiency relative to nominal/expected speed
- Print: `Actual Speed / Nominal Speed × 100%` (capped at 100% for OEE)
- Cut & Pick: Fixed at 100% (consistent speed operation)

**Quality:**
- Percentage of components produced without defects
- Formula: `(Total Components - Defects) / Total Components × 100%`
- Pick: Compound quality = Pick Success Rate × QC Pass Rate

**Utilization:**
- Percentage of scheduled time used for production
- Formula: `Active Production Hours / Scheduled Hours × 100%`
- Active Production = Production Hours - Break Overlap

---

## 3. Data Sources

### 3.1 HISTORIAN Database (PostgreSQL)

**Purpose:** Real-time equipment data from MQTT JobReports

**Tables:**
- `jobs`: MQTT messages with JobReport payloads (JSON)
- `equipment`: Equipment state change events (running/down/idle)

**Print JobReport Structure:**
```json
{
  "jobId": "string",
  "data": {
    "ids": {
      "batchId": "string",
      "sheetIndex": number
    },
    "time": {
      "uptime": number (seconds),
      "downtime": number (seconds)
    },
    "reportData": {
      "printSettings": {
        "speed": number (actual),
        "nominalSpeed": number
      },
      "timing": {
        "start": "ISO timestamp",
        "end": "ISO timestamp"
      },
      "jobInfo": {
        "area": number (sqm)
      }
    }
  }
}
```

**Cut JobReport Structure:**
```json
{
  "jobId": "string",
  "data": {
    "ids": {
      "batchId": "string",
      "sheetIndex": number
    },
    "time": {
      "uptime": number (seconds),
      "downtime": number (seconds)
    },
    "componentCount": number,
    "timing": {
      "start": "ISO timestamp",
      "end": "ISO timestamp"
    }
  }
}
```

**Pick JobReport Structure:**
```json
{
  "jobId": "string",
  "data": {
    "batchId": "string",
    "sheetIndex": number,
    "components": [
      {
        "componentId": "string",
        "status": "success" | "failed"
      }
    ],
    "timing": {
      "start": "ISO timestamp",
      "end": "ISO timestamp"
    }
  }
}
```

**Equipment State Structure:**
```sql
CREATE TABLE equipment (
  ts TIMESTAMPTZ,
  cell VARCHAR,
  state VARCHAR -- 'running', 'down', 'idle', 'blocked'
)
```

### 3.2 INSIDE Database (PostgreSQL)

**Purpose:** Production ERP and quality control data

**Tables:**

**production_printjob:**
- `rg_id`: Unique garment/unit identifier
- `production_batch_id`: Batch number
- `status`: Job status (cancelled jobs excluded)
- `component_order_count`: Number of components per job

**production_componentorder:**
- `id`: Component order ID
- `production_batch_id`: Batch number
- `item_order_id`: Style number (unique design)
- `component_id`: Component type identifier
- `rg_id`: Garment identifier
- `print_job_id`: Link to print job
- `created_at`: Component creation timestamp

**production_componentorderevent:**
- `component_order_id`: Link to component
- `event_type`: 'qc', 'pick', etc.
- `state`: 'passed', 'failed'
- `reasons`: Array of defect reasons
- `source`: 'handheld-scan', 'handheld-manual', 'inside'

---

## 4. Calculation Methodology

### 4.1 Production Time Calculation

**Critical Principle:** All cells must measure production time consistently as **actual production time (active + down)**, not idle periods.

#### Print & Cut Production Time

```
Production Time = uptime_sec + downtime_sec
```

- Comes directly from JobReport data
- Only counts time during actual job processing
- Excludes gaps between jobs

#### Pick Production Time

Pick uses equipment state tracking, requiring special handling:

```sql
-- Query equipment states for batch time window
WITH relevant_states AS (
    -- Get last state before window + all states in window
    SELECT * FROM (
        SELECT ts, cell, state
        FROM equipment
        WHERE cell = 'Pick1' AND ts <= batch_start
        ORDER BY ts DESC LIMIT 1
    ) AS before_window
    UNION ALL
    SELECT ts, cell, state
    FROM equipment
    WHERE cell = 'Pick1'
      AND ts > batch_start
      AND ts < batch_end
),
state_changes AS (
    SELECT
        ts,
        state,
        LEAD(ts) OVER (ORDER BY ts ASC) AS next_ts
    FROM relevant_states
)
SELECT
    state,
    SUM(EXTRACT(EPOCH FROM (
        LEAST(COALESCE(next_ts, batch_end), batch_end) -
        GREATEST(ts, batch_start)
    ))) AS duration_seconds
FROM state_changes
WHERE ts < batch_end
  AND (next_ts IS NULL OR next_ts > batch_start)
GROUP BY state;
```

**State Classification:**
- `running`: Robot actively picking → counts as running time
- `down`: Robot failure/unavailable → counts as downtime
- `idle`: Robot available but not producing → **excluded from production time**
- `blocked`: Downstream blockage → **excluded from production time**

```
Pick Production Time = running_time_sec + downtime_sec
```

**Rationale:** This ensures Pick's production time only includes actual production (active + down), matching Print/Cut methodology and excluding idle gaps between jobs.

### 4.2 Availability Calculation

**Print & Cut:**
```
Availability = uptime_sec / (uptime_sec + downtime_sec) × 100%
```

**Pick:**
```
Availability = running_time_sec / (running_time_sec + downtime_sec) × 100%
```

**Key Assumption:** Idle time is NOT counted as downtime. The equipment is available but not scheduled for production during idle periods.

### 4.3 Performance Calculation

**Print (Job-Level):**
```
Performance = (speed_actual / speed_nominal) × 100%
```

**Print (Batch-Level - Capped):**
```
Performance = MIN((speed_actual / speed_nominal) × 100%, 100%)
```

**Rationale for Capping:**
- Print speed can exceed 100% in raw measurements
- OEE should not exceed 100% (theoretical maximum)
- Actual uncapped performance is still displayed separately for assessment

**Cut & Pick:**
```
Performance = 100% (fixed)
```

**Rationale:** These cells operate at consistent, predictable speeds without meaningful variation.

### 4.4 Quality Calculation

**Day 0 (Production Day):**

Print & Cut:
```
Quality = 100% (assumed, will be updated post-QC)
```

Pick:
```
Quality = Pick Success Rate = Successful Picks / Total Picks × 100%
```

**Day 3+ (Post-QC):**

Print & Cut:
```
Quality = QC Pass Rate = Passed Components / Total Components × 100%
```

Pick:
```
Quality = Pick Success Rate × QC Pass Rate
```

**Rationale for Pick Compound Quality:** Pick has two quality gates:
1. Immediate pick success (did robot successfully pick?)
2. Downstream QC inspection (was component defect-free?)

Both must pass for a component to be considered good quality.

---

## 5. Cell-Specific Implementation

### 5.1 Print Cell

**Availability:**
- Source: JobReport `uptime_sec` and `downtime_sec`
- Batch-level: Time-weighted average by `total_time_sec`

**Performance:**
- Source: JobReport `speed_actual` and `speed_nominal`
- Job-level: Uncapped (actual measurement)
- Batch-level: Capped at 100%, time-weighted by `uptime_sec`

**Quality:**
- Day 0: 100% assumed
- Day 3+: QC pass rate from INSIDE database

**Production Time:**
- Sum of (`uptime_sec` + `downtime_sec`) across all jobs in batch

### 5.2 Cut Cell

**Availability:**
- Source: JobReport `uptime_sec` and `downtime_sec`
- Batch-level: Time-weighted average by `total_time_sec`

**Performance:**
- Fixed at 100%

**Quality:**
- Day 0: 100% assumed
- Day 3+: QC pass rate from INSIDE database

**Production Time:**
- Sum of (`uptime_sec` + `downtime_sec`) across all jobs in batch

### 5.3 Pick Cell

**Availability:**
- Source: Equipment state transitions from HISTORIAN
- Query time window: First pick job start to last pick job end
- Calculation: `running_time / (running_time + downtime) × 100%`

**Performance:**
- Fixed at 100%

**Quality:**
- Day 0: Pick success rate from JobReports
- Day 3+: Pick success rate × QC pass rate (compound)

**Production Time:**
- `running_time_sec + downtime_sec` from equipment states
- **Critical:** NOT the full time window, only active production time

**Special Considerations:**
- Equipment states query must include state active before batch start
- State durations clipped to batch boundaries using LEAST/GREATEST
- Idle time explicitly excluded from production time calculation

---

## 6. Quality Overlay System

### 6.1 QC Timeline

```
Day 0: Production
  ↓
Day 0-2: Production continues, QC data not yet available
  ↓
Day 3+: QC inspection complete
  ↓
Metrics automatically updated with actual quality data
```

### 6.2 Defect Attribution

QC defects are attributed to specific cells based on failure reasons:

**Print Defects:**
- Reasons: `print`, `print-water-mark`

**Cut Defects:**
- Reasons: `cut`, `cut-outside-bleed`, `cut-measurement`, `cut-measurement-small`, `cut-measurement-big`, `cut-fraying`

**Fabric Defects:**
- Reasons: `fabric`

**File/Pre-Production Defects:**
- Reasons: `file`, `file-product`

**Pick Defects:**
- Reasons: `pick`

**Other Defects:**
- Reasons: `other`

### 6.3 Quality Calculation with QC Data

**Per Batch:**
```sql
SELECT
    production_batch_id,
    COUNT(*) as total_components,
    COUNT(*) FILTER (WHERE state = 'passed') as passed,
    COUNT(*) FILTER (WHERE state = 'failed') as failed
FROM production_componentorder pc
JOIN production_componentorderevent qc
  ON qc.component_order_id = pc.id
  AND qc.event_type = 'qc'
WHERE production_batch_id = ?
GROUP BY production_batch_id
```

**Quality Rate:**
```
Quality = passed / (passed + failed) × 100%
```

**Key Assumptions:**
- Components not scanned are excluded from quality calculation
- Only components with QC events (passed or failed) are counted
- Defect attribution is based on first defect reason in array

---

## 7. Daily-Level Metrics

### 7.1 Break-Aware Utilization

Traditional utilization calculations don't account for scheduled breaks, leading to inflated idle time. Rodinia's break-aware approach adjusts for this.

**Step 1: Calculate Production Hours per Cell**
```
Print Production Hours = Σ(print_uptime + print_downtime) / 3600
Cut Production Hours = Σ(cut_uptime + cut_downtime) / 3600
Pick Production Hours = Σ(pick_running + pick_downtime) / 3600
```

**Step 2: Calculate Break Overlap per Cell**

For each cell, determine if production time overlaps with scheduled break:

```python
def calculate_break_overlap(cell_start, cell_end, break_start, break_end):
    """
    Calculate hours of overlap between production and break.

    Returns:
        float: Hours of overlap (0 if no overlap)
    """
    overlap_start = max(cell_start, break_start)
    overlap_end = min(cell_end, break_end)

    if overlap_start < overlap_end:
        overlap_seconds = (overlap_end - overlap_start).total_seconds()
        return overlap_seconds / 3600
    return 0.0
```

**Step 3: Calculate Active Production Hours**
```
Active Production Hours = Production Hours - Break Overlap
```

**Step 4: Calculate Utilization**
```
Utilization = Active Production Hours / Scheduled Hours × 100%
```

**Example:**
- Scheduled hours: 7.0 (9:00-16:00)
- Break: 0.5 hours (12:00-12:30)
- Print production: 5.5 hours (9:00-14:30)
- Print break overlap: 0.5 hours
- Print active production: 5.0 hours
- Print utilization: 5.0 / 7.0 = 71.4%

### 7.2 Weighted Batch OEE

When multiple batches are produced in a day, batch OEE must be weighted by production time:

```
Weighted Batch OEE = Σ(batch_oee × batch_duration) / Σ(batch_duration)
```

**Rationale:** A batch that ran for 3 hours should have more influence on daily OEE than a batch that ran for 30 minutes.

### 7.3 Daily OEE Formula

```
Daily OEE = Weighted Batch OEE × Utilization / 100
```

**Interpretation:**
- Weighted Batch OEE: How well equipment performed while running
- Utilization: How much of scheduled time was used for production
- Daily OEE: Overall effectiveness including scheduled idle time

**Example:**
- Weighted Batch OEE: 85%
- Utilization: 70%
- Daily OEE: 85% × 70% / 100 = 59.5%

---

## 8. Edge Cases and Assumptions

### 8.1 Rework/Defect Batches

**Challenge:** Rework batches only produce specific failed components, not full order quantities.

**Solution:** Count actual component records, not calculated quantities.

**Implementation:**
```sql
-- Actual component count (works for both full and rework batches)
SELECT
    production_batch_id,
    item_order_id,
    COUNT(*) as actual_components_in_batch
FROM production_componentorder
WHERE production_batch_id = ?
  AND print_job_id IN (
    SELECT id FROM production_printjob
    WHERE status != 'cancelled'
  )
GROUP BY production_batch_id, item_order_id
```

**Assumption:** All component records in the batch represent actual production, regardless of batch type.

### 8.2 Cancelled Jobs

**Assumption:** Jobs with status = 'cancelled' are excluded from all calculations.

**Rationale:** Cancelled jobs represent orders that were never completed and should not impact OEE metrics.

### 8.3 Missing QC Data

**Day 0-2:** Quality assumed at 100% for Print/Cut, pick success rate for Pick.

**Day 3+ with no QC data:** Previous assumption retained until QC data becomes available.

**Components not scanned:** Excluded from quality calculation (neither pass nor fail).

### 8.4 Multi-Day Batches

**Scenario:** A batch starts on Day 1 and completes on Day 2.

**Assumption:** Batch is attributed to the day it was **discovered** (earliest job timestamp), not the day it completed.

**Rationale:** Provides consistency in batch-to-day mapping for reporting purposes.

### 8.5 Equipment State Gaps

**Scenario:** No equipment state recorded for a period during batch production.

**Assumption:** Last known state persists until next state change.

**Implementation:** Query includes last state before batch start to handle this case.

### 8.6 Overlapping Batches

**Scenario:** Multiple batches running simultaneously (rare but possible).

**Assumption:** Equipment states are shared across overlapping batches proportionally.

**Current Implementation:** Each batch queries equipment states independently, which may double-count time if batches overlap.

**Future Enhancement:** Implement time allocation logic for concurrent batches.

### 8.7 Zero Production Time

**Scenario:** A batch has no valid production time (all jobs cancelled or missing data).

**Assumption:**
- Availability = 0%
- OEE = 0%
- Batch excluded from daily weighted averages (zero weight)

### 8.8 Performance > 100%

**Scenario:** Print speed exceeds nominal speed.

**Job-Level:** Actual performance stored uncapped (e.g., 105%)

**Batch-Level:** Performance capped at 100% for OEE calculation

**Display:** Both uncapped (actual) and capped (for OEE) shown in UI

**Rationale:** OEE is theoretically bounded at 100%; exceeding nominal speed is positive but doesn't increase maximum possible OEE.

---

## 9. Technical Implementation Notes

### 9.1 Database Queries

**Performance Considerations:**
- Use parameterized queries to prevent SQL injection
- Batch ID validation: alphanumeric + hyphens/underscores only
- Index on `production_batch_id` for component queries
- Index on `ts` for equipment state queries
- Index on `cell` for equipment state queries

**Query Patterns:**

**Batch Discovery:**
```sql
-- Find all batches produced on a specific date
SELECT DISTINCT
    CAST(payload AS jsonb)->'data'->'ids'->>'batchId' as batch_id
FROM jobs
WHERE topic = 'rodinia/print/JobReport'
  AND ts >= date_start
  AND ts < date_end
ORDER BY batch_id ASC
```

**Equipment States for Pick:**
```sql
-- See Section 4.1 for full query
-- Key points:
-- 1. Include last state before window
-- 2. Use LEAD window function for duration
-- 3. Clip durations to batch boundaries
-- 4. Group by state and sum durations
```

### 9.2 Data Type Handling

**Timestamps:**
- All timestamps stored as `TIMESTAMPTZ` (timezone-aware)
- Default timezone: Europe/Copenhagen
- Convert to timezone before date comparisons

**Numeric Precision:**
- Use `FLOAT` for all calculated metrics (availability, performance, quality)
- Use `DECIMAL` for monetary values (not applicable in current system)
- Avoid integer division (cast to float first)

**JSON Parsing:**
```sql
-- PostgreSQL JSONB extraction
CAST(payload AS jsonb)->'data'->'time'->>'uptime' AS uptime_sec
```

### 9.3 Error Handling

**Missing Fields:**
- Use `COALESCE` for optional fields
- Default numeric fields to 0
- Log warnings for missing critical fields

**Invalid Data:**
- Filter out negative time values
- Filter out availability > 100% (data error)
- Log data quality issues for investigation

**Connection Failures:**
- Implement connection pooling
- Retry logic with exponential backoff
- Graceful degradation (show cached data if available)

### 9.4 Logging Strategy

**Essential Logs (INFO level):**
- Batch discovery results
- OEE calculation completion
- QC data application
- Equipment state query results

**Debug Logs (DEBUG level):**
- Individual job metrics
- Intermediate calculation steps
- Time window boundaries

**Error Logs (ERROR level):**
- Database connection failures
- Invalid data detection
- Query execution errors

**Avoid:**
- Logging inside tight loops
- Logging large JSON payloads
- Excessive emoji decoration

---

## 10. Validation and Testing

### 10.1 Manual Validation Queries

**Verify Production Time:**
```sql
-- Print/Cut: Sum of uptime + downtime should match production time
SELECT
    batch_id,
    SUM(uptime_sec + downtime_sec) as calculated_production_time
FROM job_reports
WHERE batch_id = ?
GROUP BY batch_id
```

**Verify Pick Availability:**
```sql
-- Manual calculation of Pick running/down time
WITH state_transitions AS (
    SELECT
        state,
        ts,
        LEAD(ts) OVER (ORDER BY ts) as next_ts,
        LEAD(state) OVER (ORDER BY ts) as next_state
    FROM equipment
    WHERE cell = 'Pick1'
      AND ts >= batch_start
      AND ts <= batch_end
)
SELECT
    state,
    SUM(EXTRACT(EPOCH FROM (next_ts - ts))) as duration_seconds
FROM state_transitions
WHERE next_ts IS NOT NULL
GROUP BY state
```

**Verify Component Count:**
```sql
-- Actual vs theoretical component count
SELECT
    production_batch_id,
    item_order_id,
    COUNT(*) as actual_count,
    COUNT(DISTINCT component_id) *
      COUNT(DISTINCT rg_id) /
      COUNT(DISTINCT component_id) as theoretical_count
FROM production_componentorder
WHERE production_batch_id = ?
GROUP BY production_batch_id, item_order_id
```

### 10.2 Test Scenarios

**Test Case 1: Full Production Batch**
- All components produced
- No QC failures
- Expected: OEE based on availability and performance only

**Test Case 2: Rework Batch**
- Partial component production (11 of 12)
- Previous QC failures
- Expected: Actual component count used, not theoretical

**Test Case 3: Multi-Style Batch**
- Multiple item_order_ids in batch
- Different component counts per style
- Expected: Correct aggregation across styles

**Test Case 4: Break Overlap**
- Production spans scheduled break time
- Expected: Break time subtracted from production hours

**Test Case 5: Pick with Idle Time**
- Equipment states include idle periods
- Expected: Idle time excluded from production time

**Test Case 6: QC Data Application**
- Day 0: Quality = 100% (Print/Cut) or pick success rate (Pick)
- Day 3+: Quality updated with actual QC pass rate
- Expected: Metrics automatically recalculated

**Test Case 7: Performance > 100%**
- Print speed exceeds nominal
- Expected: Capped at 100% for OEE, uncapped shown separately

**Test Case 8: Zero Downtime**
- Job with downtime_sec = 0
- Expected: Availability = 100%

**Test Case 9: All Jobs Cancelled**
- Batch with all cancelled jobs
- Expected: Empty dataframe, no OEE calculated

### 10.3 Data Quality Checks

**Batch Level:**
- Component count matches sum of job component counts
- Production time positive and reasonable (<24 hours per batch)
- OEE values between 0-100%

**Daily Level:**
- Production hours ≤ 24 hours per cell
- Utilization ≤ 100% (or slightly >100% for overtime)
- Weighted batch OEE matches manual calculation

**QC Level:**
- passed + failed = total_inspected
- Quality rate between 0-100%
- Defect attribution sums to total failed

---

## Appendix A: Formula Reference

### Job-Level Formulas

```
Print Availability = uptime_sec / (uptime_sec + downtime_sec) × 100%
Print Performance = speed_actual / speed_nominal × 100%
Print Quality = 100% (Day 0) or QC Pass Rate (Day 3+)
Print OEE = Availability × Performance × Quality / 10000

Cut Availability = uptime_sec / (uptime_sec + downtime_sec) × 100%
Cut Performance = 100%
Cut Quality = 100% (Day 0) or QC Pass Rate (Day 3+)
Cut OEE = Availability × Performance × Quality / 10000

Pick Availability = Successful Picks / Total Picks × 100% (job-level)
Pick Performance = 100%
Pick Quality = Pick Success Rate (Day 0) or Pick Success Rate × QC Pass Rate (Day 3+)
Pick OEE = Availability × Performance × Quality / 10000
```

### Batch-Level Formulas

```
Batch Availability = Σ(availability × total_time) / Σ(total_time)
Batch Performance (Print) = Σ(performance_capped × uptime) / Σ(uptime)
Batch Performance (Cut/Pick) = 100%
Batch Quality = (total_components - defects) / total_components × 100%
Batch OEE = Batch Availability × Batch Performance × Batch Quality / 10000

Pick Batch Availability = running_time / (running_time + downtime) × 100%
Pick Batch Quality = Pick Success Rate × QC Pass Rate
```

### Daily-Level Formulas

```
Production Hours = Σ(production_time_sec) / 3600
Break Overlap = overlap_hours_per_cell
Active Production Hours = Production Hours - Break Overlap
Scheduled Idle Hours = Scheduled Hours - Active Production Hours
Utilization = Active Production Hours / Scheduled Hours × 100%

Weighted Batch OEE = Σ(batch_oee × batch_duration) / Σ(batch_duration)
Daily OEE = Weighted Batch OEE × Utilization / 100
```

---

## Appendix B: Glossary

**Availability:** Percentage of production time equipment was running (not down)

**Batch:** A production order processed through Print → Cut → Pick stages

**Batch OEE:** OEE calculated for a specific batch (while running effectiveness)

**Break Overlap:** Time when scheduled break overlaps with production

**Component:** Individual piece/part of a garment (e.g., front panel, back panel)

**Daily OEE:** Overall daily effectiveness including utilization

**Downtime:** Time equipment is unavailable due to failures or maintenance

**Equipment State:** Current status of equipment (running/down/idle/blocked)

**Garment/Unit (rg_id):** Complete garment composed of multiple components

**Idle Time:** Time equipment is available but not scheduled for production

**Item Order (item_order_id):** Unique style/design within a batch

**Job:** Single work order for a specific sheet/unit processed by a cell

**JobReport:** MQTT message containing job completion data from equipment

**Performance:** Speed efficiency relative to nominal/expected speed

**Pick Success Rate:** Percentage of components successfully picked by robot

**Production Time:** Active production time (running + downtime, excludes idle)

**Quality:** Percentage of components produced without defects

**QC (Quality Control):** Post-production inspection of components

**Rework Batch:** Partial batch producing only previously failed components

**Running Time:** Time equipment is actively producing

**Scheduled Hours:** Total hours available for production (shift duration - break)

**Style:** Unique design/pattern (represented by item_order_id)

**Uptime:** Time equipment was running (synonymous with running time)

**Utilization:** Percentage of scheduled time used for production

**Weighted Average:** Average calculation where each value is multiplied by its duration/weight

---

## Appendix C: Change Log

### Version 2.0 (January 2026)
- Implemented break-aware daily-level OEE calculations
- Fixed Pick production time to use running + downtime (exclude idle)
- Fixed batch structure to count actual components (handles rework batches)
- Added equipment state-based Pick availability calculation
- Implemented consistent production time methodology across all cells
- Added QC quality overlay system with defect attribution
- Documented all assumptions and edge cases

### Version 1.0 (December 2025)
- Initial batch-centric OEE framework
- Job-level and batch-level calculations
- Basic daily metrics
- QC integration

---

## Document Control

**Approval:**
- [ ] Operations Manager
- [ ] Software Development Lead
- [ ] Data Analytics Team

**Next Review Date:** April 2026

**Distribution:**
- Software Development Team
- Operations Team
- Data Analytics Team
- Quality Assurance Team

---

*End of Document*
