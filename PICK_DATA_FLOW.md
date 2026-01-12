# Pick Data Flow Documentation

## Overview
The pick data flow fetches component-level pick status from the INSIDE database and combines it with equipment uptime/downtime from the historian database to calculate Pick cell OEE.

---

## Data Flow Steps

### 1. **Entry Point: `app.py` (Lines 230-242)**
   - **Trigger**: When 'pick' is in `enabled_cells`
   - **Prerequisites**: 
     - Requires `detected_batches` (from printer cell analysis or manual input)
     - Requires `pick_window` (CellTimeWindow with time segments)
   - **Action**: 
     - Calls `fetch_pick_data_by_batch(detected_batches, pick_window)`
     - Stores result in `pick_df` and session state

---

### 2. **Main Fetcher: `fetch_pick_data_by_batch()` in `core/db/fetchers.py` (Lines 211-335)**

   #### **Step 2.1: Validation**
   - Validates `batch_ids` are not empty
   - Validates `time_window` has segments (required for equipment state filtering)
   - Returns empty DataFrame if validation fails

   #### **Step 2.2: Get Components & Pick Status (INSIDE Database)**
   - **Connection**: Uses `get_inside_connection()`
   - **Query Builder**: Calls `secure_query_builder.build_pick_data_by_batch_query(batch_ids)`
   - **Query Execution**: Executes query against INSIDE database
   - **Result**: DataFrame with columns:
     - `component_id` - Component ID from production_componentorder
     - `production_batch_id` - Batch ID
     - `job_id` - Job ID (rg_id from production_printjob)
     - `print_job_id` - Print job ID
     - `created_at` - Component creation timestamp
     - `pick_status` - Calculated status: 'successful', 'failed', or 'not_picked'
     - `pick_event_state` - Raw state from production_componentorderevent
     - `pick_event_time` - When pick event occurred

   #### **Step 2.3: Get Uptime/Downtime (Historian Database)**
   - **Connection**: Uses `get_historian_connection()`
   - **Query**: Builds inline SQL query (not using query builder)
   - **Logic**:
     - Filters `equipment` table where `cell = 'Pick1'`
     - Applies time window segments: `(e.ts > start AND e.ts < end)` for each segment
     - Uses SQL window function `LEAD()` to get next state and timestamp
     - Calculates running duration: Sum of time when state='running' → transitions to 'idle' or 'down'
     - Calculates downtime duration: Sum of time when state='down' → transitions to 'idle' or 'running'
     - Converts seconds to minutes
   - **Result**: Single row with `uptime (min)` and `downtime (min)`

   #### **Step 2.4: Combine Data**
   - Adds `uptime (min)` and `downtime (min)` to every row in `component_df` (same value for all components)
   - Adds `cell = 'Pick1'` column
   - Calculates helper columns:
     - `successful_picks`: 1 if `pick_status == 'successful'`, else 0
     - `total_picks`: 1 if `pick_status` is 'successful' or 'failed', else 0
   - Returns combined DataFrame

---

### 3. **Query Builder: `build_pick_data_by_batch_query()` in `core/db/queries.py` (Lines 686-747)**

   #### **Step 3.1: Batch ID Validation**
   - Converts batch IDs from string format ('B955' or '955') to integers
   - Filters out invalid batch IDs
   - Raises error if no valid batch IDs remain

   #### **Step 3.2: SQL Query Construction**
   - **Tables Joined**:
     - `production_componentorder` (pc) - Master component list
     - `production_printjob` (pj) - Print job metadata
     - `production_componentorderevent` (pick_event) - Pick events (LEFT JOIN)
   - **Join Conditions**:
     - `pc.print_job_id = pj.id` (get job info)
     - `pick_event.component_order_id = pc.id AND pick_event.event_type = 'pick'` (get pick status)
   - **Filter**: `pc.production_batch_id = ANY(%s::bigint[])` and `pj.status != 'cancelled'`
   - **Pick Status Logic**:
     - `'successful'`: When `pick_event.state = 'passed'` OR `'successful'`
     - `'failed'`: When `pick_event.state = 'failed'`
     - `'not_picked'`: When no pick event exists (LEFT JOIN returns NULL)
   - **Returns**: Query string and parameters list

---

### 4. **OEE Calculation: `app.py` (Lines 424-467)**

   #### **Step 4.1: Extract Uptime/Downtime**
   - Gets first row's `uptime (min)` and `downtime (min)` (all rows have same value)
   - Defaults to 0.0 if columns missing

   #### **Step 4.2: Calculate Pick Quality Accuracy**
   - **Primary Method**: Uses `pick_status` column
     - `total_successful` = Count of rows where `pick_status == 'successful'`
     - `total_picks` = Count of rows where `pick_status` is 'successful' or 'failed'
     - `pick_quality_accuracy = total_successful / total_picks`
   - **Fallback Method**: Uses `successful_picks` and `total_picks` columns if `pick_status` missing
   - **Default**: 1.0 (100%) if no pick data available

   #### **Step 4.3: Calculate Pick OEE**
   - Calls `calculate_cell_oee('pick', pick_uptime, pick_downtime, quality_ratio=pick_quality_accuracy)`
   - Formula: `OEE = Availability × Performance × Quality`
     - **Availability**: Calculated from `pick_uptime` and `pick_downtime`
     - **Performance**: Always 100% for pick cell
     - **Quality**: Uses `pick_quality_accuracy` (successful picks / total picks)
   - Stores result in `st.session_state['pick_oee']`

---

## Data Sources

### **INSIDE Database**
- **Table**: `production_componentorder`
  - Contains all components for a batch
  - Links to print jobs via `print_job_id`
- **Table**: `production_printjob`
  - Contains job metadata (rg_id, status)
- **Table**: `production_componentorderevent`
  - Contains pick events with `event_type = 'pick'`
  - Links to components via `component_order_id`
  - State values: 'passed', 'successful', 'failed', or NULL (not picked)

### **Historian Database**
- **Table**: `equipment`
  - Contains equipment state changes for Pick1 cell
  - Columns: `ts` (timestamp), `cell`, `state` ('running', 'idle', 'down')
  - Used to calculate uptime/downtime within time window segments

---

## Key Design Decisions

1. **Component-Level Granularity**: Returns one row per component, not per job
   - Allows accurate counting of successful vs failed picks
   - Each component has its own pick status

2. **Separate Database Queries**: 
   - Components/pick status from INSIDE (ERP data)
   - Uptime/downtime from Historian (equipment state data)
   - Combined in Python, not SQL

3. **Time Window Filtering**: 
   - Applied only to equipment state changes (for uptime/downtime)
   - Component data is filtered by batch_id, not time window
   - All components in a batch are included regardless of when they were picked

4. **Uptime/Downtime Calculation**:
   - Uses SQL window functions to track state transitions
   - Only counts time within the provided time window segments
   - Same value applied to all components (equipment-level metric)

5. **Quality Accuracy Calculation**:
   - Component-level: Count successful vs total attempts
   - Excludes 'not_picked' components from total (only counts actual pick attempts)
   - Formula: `successful_picks / (successful_picks + failed_picks)`

---

## Error Handling

- **No batch IDs**: Returns empty DataFrame with warning
- **No time window segments**: Raises ValueError
- **No components found**: Returns empty DataFrame with warning
- **No equipment state data**: Sets uptime/downtime to 0.0 with warning
- **Database errors**: Logs error and returns empty DataFrame

---

## Data Flow Diagram

```
app.py
  │
  ├─> fetch_pick_data_by_batch(batch_ids, time_window)
  │     │
  │     ├─> Step 1: INSIDE Database Query
  │     │     │
  │     │     └─> build_pick_data_by_batch_query(batch_ids)
  │     │           │
  │     │           └─> SQL: production_componentorder 
  │     │                 JOIN production_printjob
  │     │                 LEFT JOIN production_componentorderevent
  │     │                 WHERE event_type = 'pick'
  │     │
  │     ├─> Step 2: Historian Database Query
  │     │     │
  │     │     └─> Inline SQL: equipment table
  │     │           WHERE cell = 'Pick1'
  │     │           AND ts within time_window segments
  │     │           Calculate uptime/downtime from state transitions
  │     │
  │     └─> Step 3: Combine Data
  │           │
  │           └─> Add uptime/downtime to component_df
  │           └─> Calculate successful_picks/total_picks columns
  │
  └─> Calculate OEE
        │
        ├─> Extract uptime/downtime (from pick_df)
        ├─> Calculate quality_accuracy (from pick_status column)
        └─> calculate_cell_oee('pick', uptime, downtime, quality_accuracy)
```

---

## Summary

The pick data flow is a **two-database, component-level approach**:
1. Gets component list and pick status from INSIDE (ERP)
2. Gets equipment uptime/downtime from Historian (equipment state)
3. Combines them at the component level
4. Calculates OEE using component-level quality metrics and equipment-level availability metrics
