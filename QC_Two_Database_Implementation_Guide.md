# QC Data Implementation Guide - Two-Database Approach

## Document Purpose

This guide provides implementation specifications for fetching QC (Quality Control) data using a two-database approach. The historian database `public.components` table contains richer QC data than the inside database `production_componentorderevent` table.

**Author:** Andre (COO) with Lean Manufacturing Consultant  
**Date:** 2026-01-06  
**Status:** Implementation Ready  
**Target:** Cursor AI Implementation

---

## Executive Summary

### Current State (Limited)

QC data is fetched from `production_componentorderevent` in the INSIDE database. This table:
- Only contains records for components that were scanned
- Has defect reasons in a `description` array field
- Misses components added manually or not scanned

### New Approach (Recommended)

Fetch QC data from `public.components` in the HISTORIAN database. This table:
- Contains comprehensive QC records with full payload
- Has structured JSON with `state`, `reasons`, and `source`
- Provides better data quality for defect attribution

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TWO-DATABASE FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐         ┌─────────────────────┐               │
│  │    INSIDE DATABASE  │         │  HISTORIAN DATABASE │               │
│  ├─────────────────────┤         ├─────────────────────┤               │
│  │                     │         │                     │               │
│  │ production_         │  Step 1 │ public.components   │               │
│  │ componentorder      │────────▶│ (process = 'QC')    │               │
│  │                     │  Get    │                     │               │
│  │ • id (component_id) │  IDs    │ • component_id      │               │
│  │ • production_batch_id│        │ • state             │               │
│  │ • print_job_id      │         │ • payload (JSON)    │               │
│  │                     │         │   - state           │               │
│  │ production_printjob │         │   - reasons[]       │               │
│  │ • id                │         │   - source          │               │
│  │ • status            │         │                     │               │
│  └─────────────────────┘         └─────────────────────┘               │
│            │                               │                            │
│            │ Step 2: Match                 │                            │
│            │ component_id ◀───────────────┘                            │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────────────────────────┐                           │
│  │         AGGREGATED QC METRICS           │                           │
│  ├─────────────────────────────────────────┤                           │
│  │ • total_components (from inside)        │                           │
│  │ • total_passed (from historian)         │                           │
│  │ • total_failed (from historian)         │                           │
│  │ • not_scanned (inside - historian)      │                           │
│  │ • defects by reason (from payload)      │                           │
│  └─────────────────────────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Database Schema Reference

### INSIDE Database: `production_componentorder`

```sql
-- Table: production_componentorder
-- Purpose: Master list of all components for a batch

Column              | Type      | Description
--------------------|-----------|----------------------------------
id                  | bigint    | Primary key (this is component_id)
production_batch_id | bigint    | Foreign key to batch
print_job_id        | bigint    | Foreign key to print job
created_at          | timestamp | When component was created
updated_at          | timestamp | Last update time
```

### INSIDE Database: `production_printjob`

```sql
-- Table: production_printjob
-- Purpose: Print job metadata

Column              | Type      | Description
--------------------|-----------|----------------------------------
id                  | bigint    | Primary key
rg_id               | varchar   | Rodinia Generation ID
status              | varchar   | 'completed', 'cancelled', etc.
production_batch_id | bigint    | Foreign key to batch
```

### HISTORIAN Database: `public.components`

```sql
-- Table: public.components
-- Purpose: Event log for all component processes including QC

Column              | Type      | Description
--------------------|-----------|----------------------------------
id                  | bigint    | Primary key (event ID)
ts                  | timestamp | Event timestamp
uuid                | uuid      | Unique identifier
topic               | varchar   | Message topic
component_id        | bigint    | Links to production_componentorder.id
process             | varchar   | 'QC', 'print', 'cut', etc.
state               | varchar   | 'passed', 'failed'
payload             | jsonb     | Full event data (see below)
unit_type           | varchar   | 'cell'
enterprise          | varchar   | 'RG'
site                | varchar   | 'CPH'
```

### Payload JSON Structure

```json
{
  "id": "1000059872",
  "type": "component",
  "state": "passed",
  "reasons": [],
  "source": "handheld-scan"
}
```

For failed components:
```json
{
  "id": "1000059845",
  "type": "component", 
  "state": "failed",
  "reasons": ["cut", "cut-measurement-small"],
  "source": "handheld-scan"
}
```

**Known reason values** (from production data):
- `file`, `file-product` → Pre-production issues
- `cut`, `cut-outside-bleed`, `cut-measurement`, `cut-measurement-small`, `cut-measurement-big`, `cut-fraying` → Cutter cell
- `print`, `print-water-mark` → Printer cell
- `fabric` → Incoming material defect
- `other` → Unattributed

**Known source values:**
- `handheld-scan` → Primary QC method
- `handheld-manual` → Manual entry
- `inside` → System/automated

---

## Implementation Specifications

### Step 1: Fetch Component IDs for Batch

**File:** `core/db/fetchers.py`  
**Function:** `fetch_component_ids_for_batch`

```python
def fetch_component_ids_for_batch(batch_ids: List[str]) -> List[int]:
    """
    Fetch all component IDs for specified batches from INSIDE database.
    
    This is Step 1 of the two-database QC fetch approach.
    
    Args:
        batch_ids: List of batch IDs (can be 'B955' or '955' format)
        
    Returns:
        List of component IDs (integers) that belong to these batches
    """
    logger.info(f"Fetching component IDs for batches: {batch_ids}")

    try:
        with get_inside_connection() as conn:
            cursor = conn.cursor()

            # Build query
            query, parameters = secure_query_builder.build_component_ids_for_batch_query(batch_ids)
            
            cursor.execute(query, parameters)
            results = cursor.fetchall()

            if not results:
                logger.warning(f"No components found for batches: {batch_ids}")
                return []

            # Extract component IDs from results
            component_ids = [row[0] for row in results]
            
            logger.info(f"Found {len(component_ids)} components for batches {batch_ids}")
            return component_ids

    except Exception as e:
        logger.error(f"Error fetching component IDs: {e}", exc_info=True)
        return []
```

**File:** `core/db/queries.py`  
**Function:** `build_component_ids_for_batch_query`

```python
def build_component_ids_for_batch_query(batch_ids: List[str]) -> Tuple[str, list]:
    """
    Build query to fetch component IDs for specified batches.
    
    Args:
        batch_ids: List of batch IDs (can be 'B955' or '955' format)
        
    Returns:
        Tuple of (query_string, parameters_list)
    """
    # Convert string batch IDs to integers
    validated_batches = []
    for bid in batch_ids:
        try:
            clean_id = str(bid).replace('B', '').strip()
            validated_batches.append(int(clean_id))
        except (ValueError, AttributeError):
            logger.warning(f"Invalid batch ID format: {bid}")
            continue
    
    if not validated_batches:
        raise ValueError("No valid batch IDs provided")
    
    query = """
        SELECT DISTINCT pc.id as component_id
        FROM production_componentorder pc
        JOIN production_printjob pj ON pc.print_job_id = pj.id
        WHERE pc.production_batch_id = ANY(%s::bigint[])
          AND pj.status != 'cancelled'
        ORDER BY pc.id
    """
    
    logger.info(f"Built component IDs query for {len(validated_batches)} batches")
    return query, [validated_batches]
```

---

### Step 2: Fetch QC Data from Historian

**File:** `core/db/fetchers.py`  
**Function:** `fetch_qc_data_from_historian`

```python
def fetch_qc_data_from_historian(component_ids: List[int]) -> pd.DataFrame:
    """
    Fetch QC data from HISTORIAN database public.components table.
    
    This is Step 2 of the two-database QC fetch approach.
    
    Args:
        component_ids: List of component IDs from Step 1
        
    Returns:
        DataFrame with QC data including state, reasons, and source
    """
    if not component_ids:
        logger.warning("No component IDs provided for QC data fetch")
        return pd.DataFrame()
    
    logger.info(f"Fetching QC data for {len(component_ids)} components from historian")

    try:
        with get_historian_connection() as conn:
            cursor = conn.cursor()

            query, parameters = secure_query_builder.build_qc_data_from_historian_query(component_ids)
            
            cursor.execute(query, parameters)
            results = cursor.fetchall()

            if not results:
                logger.warning(f"No QC data found for {len(component_ids)} components")
                return pd.DataFrame()

            # Get column names from cursor
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            
            # Parse JSON reasons array into Python list
            if 'qc_reasons' in df.columns:
                df['qc_reasons'] = df['qc_reasons'].apply(
                    lambda x: x if isinstance(x, list) else []
                )
            
            logger.info(f"Retrieved {len(df)} QC records from historian")
            return df

    except Exception as e:
        logger.error(f"Error fetching QC data from historian: {e}", exc_info=True)
        return pd.DataFrame()
```

**File:** `core/db/queries.py`  
**Function:** `build_qc_data_from_historian_query`

```python
def build_qc_data_from_historian_query(component_ids: List[int]) -> Tuple[str, list]:
    """
    Build query to fetch QC data from historian public.components table.
    
    Args:
        component_ids: List of component IDs to fetch QC data for
        
    Returns:
        Tuple of (query_string, parameters_list)
    """
    if not component_ids:
        raise ValueError("No component IDs provided")
    
    query = """
        SELECT 
            c.id as event_id,
            c.ts as qc_timestamp,
            c.component_id,
            c.state as event_state,
            c.payload::json->>'state' as qc_state,
            c.payload::json->>'source' as qc_source,
            c.payload::json->'reasons' as qc_reasons
        FROM public.components c
        WHERE c.process = 'QC'
          AND c.component_id = ANY(%s::bigint[])
        ORDER BY c.component_id, c.ts DESC
    """
    
    logger.info(f"Built historian QC query for {len(component_ids)} components")
    return query, [component_ids]
```

---

### Step 3: Aggregate QC Metrics

**File:** `core/db/fetchers.py`  
**Function:** `fetch_batch_quality_breakdown_v2`

```python
def fetch_batch_quality_breakdown_v2(batch_ids: List[str]) -> pd.DataFrame:
    """
    Fetch comprehensive QC breakdown using two-database approach.
    
    This function:
    1. Gets component IDs from INSIDE database
    2. Fetches QC data from HISTORIAN database
    3. Aggregates metrics with defect attribution
    
    Args:
        batch_ids: List of batch IDs to analyze
        
    Returns:
        DataFrame with quality breakdown per batch
    """
    logger.info(f"Fetching batch quality breakdown (v2) for batches: {batch_ids}")
    
    # Validate and convert batch IDs
    validated_batches = []
    for bid in batch_ids:
        try:
            clean_id = str(bid).replace('B', '').strip()
            validated_batches.append(int(clean_id))
        except (ValueError, AttributeError):
            logger.warning(f"Invalid batch ID format: {bid}")
            continue
    
    if not validated_batches:
        logger.error("No valid batch IDs provided")
        return pd.DataFrame()
    
    try:
        # Step 1: Get component IDs and batch mapping from INSIDE database
        with get_inside_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    pc.id as component_id,
                    pc.production_batch_id,
                    DATE(pc.created_at) as production_date
                FROM production_componentorder pc
                JOIN production_printjob pj ON pc.print_job_id = pj.id
                WHERE pc.production_batch_id = ANY(%s::bigint[])
                  AND pj.status != 'cancelled'
            """
            
            cursor.execute(query, [validated_batches])
            component_results = cursor.fetchall()
            
            if not component_results:
                logger.warning(f"No components found for batches: {batch_ids}")
                return pd.DataFrame()
            
            # Create DataFrame with component-to-batch mapping
            components_df = pd.DataFrame(
                component_results, 
                columns=['component_id', 'production_batch_id', 'production_date']
            )
            
            component_ids = components_df['component_id'].tolist()
            logger.info(f"Found {len(component_ids)} components across {len(validated_batches)} batches")
        
        # Step 2: Get QC data from HISTORIAN database
        with get_historian_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    c.component_id,
                    c.payload::json->>'state' as qc_state,
                    c.payload::json->>'source' as qc_source,
                    c.payload::json->'reasons' as qc_reasons
                FROM public.components c
                WHERE c.process = 'QC'
                  AND c.component_id = ANY(%s::bigint[])
            """
            
            cursor.execute(query, [component_ids])
            qc_results = cursor.fetchall()
            
            # Create QC DataFrame
            if qc_results:
                qc_df = pd.DataFrame(
                    qc_results,
                    columns=['component_id', 'qc_state', 'qc_source', 'qc_reasons']
                )
                logger.info(f"Found {len(qc_df)} QC records in historian")
            else:
                qc_df = pd.DataFrame(columns=['component_id', 'qc_state', 'qc_source', 'qc_reasons'])
                logger.warning("No QC records found in historian")
        
        # Step 3: Merge and aggregate
        merged_df = components_df.merge(qc_df, on='component_id', how='left')
        
        # Fill missing QC state as 'not_scanned'
        merged_df['qc_state'] = merged_df['qc_state'].fillna('not_scanned')
        merged_df['qc_reasons'] = merged_df['qc_reasons'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        # Aggregate by batch
        aggregated = aggregate_qc_by_batch(merged_df)
        
        logger.info(f"Successfully processed QC breakdown for {len(aggregated)} batches")
        return aggregated

    except Exception as e:
        logger.error(f"Error in batch quality breakdown v2: {e}", exc_info=True)
        return pd.DataFrame()
```

---

### Step 4: Aggregation Helper Function

**File:** `core/db/fetchers.py` (or new file `core/qc/aggregation.py`)  
**Function:** `aggregate_qc_by_batch`

```python
def aggregate_qc_by_batch(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate QC data by batch with defect attribution.
    
    Args:
        merged_df: DataFrame with component_id, production_batch_id, 
                   production_date, qc_state, qc_source, qc_reasons
                   
    Returns:
        DataFrame with aggregated metrics per batch
    """
    
    def count_reason(reasons_list, pattern: str) -> int:
        """Count occurrences of a reason pattern in list of reason arrays."""
        count = 0
        for reasons in reasons_list:
            if isinstance(reasons, list):
                for reason in reasons:
                    if pattern.lower() in str(reason).lower():
                        count += 1
                        break  # Count component once per pattern
        return count
    
    def count_reason_exact(reasons_list, patterns: List[str]) -> int:
        """Count components with any of the exact reason patterns."""
        count = 0
        for reasons in reasons_list:
            if isinstance(reasons, list):
                for reason in reasons:
                    reason_lower = str(reason).lower()
                    if any(p.lower() == reason_lower or reason_lower.startswith(p.lower()) 
                           for p in patterns):
                        count += 1
                        break
        return count
    
    results = []
    
    for (batch_id, prod_date), group in merged_df.groupby(['production_batch_id', 'production_date']):
        total = len(group)
        passed = len(group[group['qc_state'] == 'passed'])
        failed = len(group[group['qc_state'] == 'failed'])
        not_scanned = len(group[group['qc_state'] == 'not_scanned'])
        
        # Get failed reasons for defect attribution
        failed_reasons = group[group['qc_state'] == 'failed']['qc_reasons'].tolist()
        
        # Defect attribution by cell
        # Print defects: print, print-water-mark
        print_defects = count_reason_exact(failed_reasons, ['print', 'print-water-mark'])
        
        # Cut defects: cut, cut-outside-bleed, cut-measurement*, cut-fraying
        cut_defects = count_reason_exact(failed_reasons, [
            'cut', 'cut-outside-bleed', 'cut-measurement', 
            'cut-measurement-small', 'cut-measurement-big', 'cut-fraying'
        ])
        
        # Fabric defects: fabric
        fabric_defects = count_reason_exact(failed_reasons, ['fabric'])
        
        # File/Pre-production defects: file, file-product
        file_defects = count_reason_exact(failed_reasons, ['file', 'file-product'])
        
        # Pick defects: pick
        pick_defects = count_reason_exact(failed_reasons, ['pick'])
        
        # Other defects: other
        other_defects = count_reason_exact(failed_reasons, ['other'])
        
        # QC Source breakdown
        scanned_handheld = len(group[group['qc_source'] == 'handheld-scan'])
        scanned_manual = len(group[group['qc_source'] == 'handheld-manual'])
        scanned_inside = len(group[group['qc_source'] == 'inside'])
        
        # Calculate rates
        scanned_total = passed + failed
        quality_rate = (passed / scanned_total * 100) if scanned_total > 0 else None
        scan_coverage = (scanned_total / total * 100) if total > 0 else 0
        unchecked_rate = (not_scanned / total * 100) if total > 0 else 0
        
        results.append({
            'production_batch_id': batch_id,
            'production_date': prod_date,
            'total_components': total,
            'total_passed': passed,
            'total_failed': failed,
            'not_scanned': not_scanned,
            'print_defects': print_defects,
            'cut_defects': cut_defects,
            'fabric_defects': fabric_defects,
            'file_defects': file_defects,
            'pick_defects': pick_defects,
            'other_defects': other_defects,
            'scanned_handheld': scanned_handheld,
            'scanned_manual': scanned_manual,
            'scanned_inside': scanned_inside,
            'quality_rate_checked_percent': quality_rate,
            'scan_coverage_percent': scan_coverage,
            'unchecked_rate_percent': unchecked_rate
        })
    
    return pd.DataFrame(results)
```

---

## Data Models

**File:** `core/models/qc_models.py`

```python
from dataclasses import dataclass
from typing import Optional, List
from datetime import date


@dataclass
class BatchQualityBreakdownV2:
    """Quality breakdown for a batch using two-database approach."""
    production_batch_id: int
    production_date: date
    
    # Overall Counts
    total_components: int
    total_passed: int
    total_failed: int
    not_scanned: int
    
    # Defect Attribution by Cell
    print_defects: int          # Printer cell
    cut_defects: int            # Cutter cell
    pick_defects: int           # Pick cell
    
    # Non-cell defects (upstream issues)
    fabric_defects: int         # Incoming material
    file_defects: int           # Pre-production/file issues
    other_defects: int          # Unattributed
    
    # QC Source breakdown
    scanned_handheld: int
    scanned_manual: int
    scanned_inside: int
    
    # Rates
    quality_rate_checked_percent: Optional[float]
    scan_coverage_percent: float
    unchecked_rate_percent: float
    
    @property
    def cell_attributed_defects(self) -> int:
        """Defects that can be attributed to production cells."""
        return self.print_defects + self.cut_defects + self.pick_defects
    
    @property
    def upstream_defects(self) -> int:
        """Defects from upstream (not production cell issues)."""
        return self.fabric_defects + self.file_defects
    
    @property
    def total_attributed_defects(self) -> int:
        """All defects that have been categorized."""
        return self.cell_attributed_defects + self.upstream_defects + self.other_defects
    
    @property
    def unattributed_defects(self) -> int:
        """Defects that couldn't be matched to any category."""
        return max(0, self.total_failed - self.total_attributed_defects)
    
    @property
    def attribution_coverage_percent(self) -> float:
        """What percentage of defects were successfully attributed?"""
        if self.total_failed == 0:
            return 100.0
        return (self.total_attributed_defects / self.total_failed) * 100
    
    @property
    def cell_quality_rates(self) -> dict:
        """
        Calculate quality rate (0-1) for each cell for OEE reconciliation.
        """
        if self.total_components == 0:
            return {'printer': 1.0, 'cutter': 1.0, 'pick': 1.0}
        
        return {
            'printer': (self.total_components - self.print_defects) / self.total_components,
            'cutter': (self.total_components - self.cut_defects) / self.total_components,
            'pick': (self.total_components - self.pick_defects) / self.total_components
        }


@dataclass 
class ComponentQCRecord:
    """Individual component QC record from historian."""
    component_id: int
    production_batch_id: int
    production_date: date
    qc_state: str              # 'passed', 'failed', 'not_scanned'
    qc_source: Optional[str]   # 'handheld-scan', 'handheld-manual', 'inside'
    qc_reasons: List[str]      # ['cut', 'cut-measurement-small'] etc.
    qc_timestamp: Optional[str]


def dataframe_to_batch_quality_breakdown(df: pd.DataFrame) -> List[BatchQualityBreakdownV2]:
    """Convert aggregated DataFrame to list of BatchQualityBreakdownV2 objects."""
    results = []
    for _, row in df.iterrows():
        results.append(BatchQualityBreakdownV2(
            production_batch_id=int(row['production_batch_id']),
            production_date=row['production_date'],
            total_components=int(row['total_components']),
            total_passed=int(row['total_passed']),
            total_failed=int(row['total_failed']),
            not_scanned=int(row['not_scanned']),
            print_defects=int(row['print_defects']),
            cut_defects=int(row['cut_defects']),
            pick_defects=int(row.get('pick_defects', 0)),
            fabric_defects=int(row['fabric_defects']),
            file_defects=int(row['file_defects']),
            other_defects=int(row.get('other_defects', 0)),
            scanned_handheld=int(row.get('scanned_handheld', 0)),
            scanned_manual=int(row.get('scanned_manual', 0)),
            scanned_inside=int(row.get('scanned_inside', 0)),
            quality_rate_checked_percent=row.get('quality_rate_checked_percent'),
            scan_coverage_percent=float(row.get('scan_coverage_percent', 0)),
            unchecked_rate_percent=float(row.get('unchecked_rate_percent', 0))
        ))
    return results
```

---

## Integration with Existing Code

### Update `secure_query_builder` Class

**File:** `core/db/queries.py`

Add the following methods to the `SecureQueryBuilder` class:

```python
class SecureQueryBuilder:
    # ... existing methods ...
    
    def build_component_ids_for_batch_query(self, batch_ids: List[str]) -> Tuple[str, list]:
        """Build query to fetch component IDs for specified batches."""
        validated_batches = self._validate_batch_ids(batch_ids)
        
        query = """
            SELECT DISTINCT pc.id as component_id
            FROM production_componentorder pc
            JOIN production_printjob pj ON pc.print_job_id = pj.id
            WHERE pc.production_batch_id = ANY(%s::bigint[])
              AND pj.status != 'cancelled'
            ORDER BY pc.id
        """
        
        logger.info(f"Built component IDs query for {len(validated_batches)} batches")
        return query, [validated_batches]
    
    def build_qc_data_from_historian_query(self, component_ids: List[int]) -> Tuple[str, list]:
        """Build query to fetch QC data from historian."""
        if not component_ids:
            raise ValueError("No component IDs provided")
        
        query = """
            SELECT 
                c.component_id,
                c.payload::json->>'state' as qc_state,
                c.payload::json->>'source' as qc_source,
                c.payload::json->'reasons' as qc_reasons
            FROM public.components c
            WHERE c.process = 'QC'
              AND c.component_id = ANY(%s::bigint[])
        """
        
        logger.info(f"Built historian QC query for {len(component_ids)} components")
        return query, [component_ids]
    
    def _validate_batch_ids(self, batch_ids: List[str]) -> List[int]:
        """Convert and validate batch IDs."""
        validated = []
        for bid in batch_ids:
            try:
                clean_id = str(bid).replace('B', '').strip()
                validated.append(int(clean_id))
            except (ValueError, AttributeError):
                logger.warning(f"Invalid batch ID format: {bid}")
                continue
        
        if not validated:
            raise ValueError("No valid batch IDs provided")
        
        return validated
```

---

## Testing

### Unit Test: Component ID Fetch

```python
def test_fetch_component_ids_for_batch():
    """Test fetching component IDs for a batch."""
    batch_ids = ['B955']
    
    component_ids = fetch_component_ids_for_batch(batch_ids)
    
    assert len(component_ids) > 0
    assert all(isinstance(cid, int) for cid in component_ids)
    print(f"Found {len(component_ids)} components for batch 955")
```

### Unit Test: QC Data Fetch

```python
def test_fetch_qc_data_from_historian():
    """Test fetching QC data from historian."""
    # First get component IDs
    component_ids = fetch_component_ids_for_batch(['B955'])
    
    # Then fetch QC data
    qc_df = fetch_qc_data_from_historian(component_ids)
    
    assert not qc_df.empty
    assert 'qc_state' in qc_df.columns
    assert 'qc_reasons' in qc_df.columns
    print(f"Found {len(qc_df)} QC records")
    print(f"States: {qc_df['qc_state'].value_counts().to_dict()}")
```

### Integration Test: Full Flow

```python
def test_batch_quality_breakdown_v2():
    """Test the complete two-database QC fetch flow."""
    batch_ids = ['B955']
    
    result_df = fetch_batch_quality_breakdown_v2(batch_ids)
    
    assert not result_df.empty
    assert 'total_components' in result_df.columns
    assert 'print_defects' in result_df.columns
    assert 'cut_defects' in result_df.columns
    
    # Verify defect attribution
    row = result_df.iloc[0]
    total_attributed = (
        row['print_defects'] + row['cut_defects'] + row['pick_defects'] +
        row['fabric_defects'] + row['file_defects'] + row['other_defects']
    )
    
    print(f"Total components: {row['total_components']}")
    print(f"Passed: {row['total_passed']}, Failed: {row['total_failed']}, Not scanned: {row['not_scanned']}")
    print(f"Defects attributed: {total_attributed} / {row['total_failed']}")
```

---

## Validation Query

Run this query directly to validate the approach before implementing:

```sql
-- Run on HISTORIAN database
-- Replace component_ids with actual values from batch 955

WITH batch_components AS (
    -- This would come from INSIDE database in production
    SELECT unnest(ARRAY[
        1000059872, 1000059866, 1000059869, 1000059867
        -- Add more component IDs here
    ]::bigint[]) as component_id
)
SELECT 
    c.component_id,
    c.payload::json->>'state' as qc_state,
    c.payload::json->>'source' as qc_source,
    c.payload::json->'reasons' as qc_reasons
FROM public.components c
JOIN batch_components bc ON c.component_id = bc.component_id
WHERE c.process = 'QC'
ORDER BY c.component_id;
```

---

## Migration Notes

### Deprecation Path

1. Keep `fetch_batch_quality_breakdown` (v1) working during transition
2. Add `fetch_batch_quality_breakdown_v2` as the new implementation
3. Test v2 thoroughly against v1 results
4. Switch callers to v2 once validated
5. Deprecate v1 after full migration

### Backward Compatibility

The output DataFrame schema is identical between v1 and v2, so callers should not need changes. The only difference is data accuracy (v2 captures more components and has better reason attribution).

---

## Troubleshooting

### No QC Records Found

If historian returns no QC records:
1. Verify component_ids are correct (from inside database)
2. Check `process = 'QC'` filter matches actual data
3. Verify timestamp range if filtering by time

### Defect Attribution Mismatch

If `total_attributed_defects < total_failed`:
1. Run discovery query to find new reason values
2. Add new patterns to `count_reason_exact` function
3. Check for typos or variations in reason strings

### Performance Issues

For large batches (>10,000 components):
1. Consider batching the historian query
2. Add index on `public.components(component_id)` if not exists
3. Use EXPLAIN ANALYZE to identify slow operations

---

## Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-06 | Andre | Initial two-database implementation guide |
