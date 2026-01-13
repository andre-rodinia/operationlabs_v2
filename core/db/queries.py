"""
Secure Query Builder Module

This module provides secure, parameterized query building functions to prevent SQL injection attacks.
All user inputs are properly sanitized and validated before being used in database queries.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class SecureQueryBuilder:
    """Secure query builder with parameterized queries and input validation."""
    
    @staticmethod
    def validate_datetime(dt_string: str) -> bool:
        """
        Validate datetime string format.
        
        Args:
            dt_string: Datetime string to validate
            
        Returns:
            bool: True if valid datetime format
        """
        try:
            datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_topic(topic: str) -> bool:
        """
        Validate MQTT topic format.
        
        Args:
            topic: MQTT topic to validate
            
        Returns:
            bool: True if valid topic format
        """
        if not topic or len(topic) > 255:
            return False
        
        # MQTT topics should only contain specific characters
        valid_pattern = re.compile(r'^[a-zA-Z0-9/_-]+$')
        return bool(valid_pattern.match(topic))
    
    @staticmethod
    def validate_batch_id(batch_id: str) -> bool:
        """
        Validate batch ID format.
        
        Args:
            batch_id: Batch ID to validate
            
        Returns:
            bool: True if valid batch ID format
        """
        if not batch_id or len(batch_id) > 50:
            return False
        
        # Batch IDs should be alphanumeric with possible hyphens/underscores
        valid_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
        return bool(valid_pattern.match(batch_id))
    
    def build_production_query(self, selected_batches: Optional[List[str]] = None) -> Tuple[str, List[Any]]:
        """
        Build secure parameterized query for production data.
        
        Args:
            selected_batches: List of batch IDs to filter by
            
        Returns:
            Tuple[str, List[Any]]: (query_string, parameters)
        """
        base_query = """
            SELECT rg_id, status, production_batch_id, component_order_count, index
            FROM production_printjob
        """
        
        if selected_batches and len(selected_batches) > 0:
            # Validate all batch IDs
            validated_batches = []
            for batch_id in selected_batches:
                # Remove 'B' prefix if present and validate
                clean_batch = str(batch_id).lstrip("B")
                if self.validate_batch_id(clean_batch):
                    validated_batches.append(clean_batch)
                else:
                    logger.warning(f"Invalid batch ID filtered out: {batch_id}")
            
            if validated_batches:
                # Use parameterized query with placeholders
                placeholders = ','.join(['%s'] * len(validated_batches))
                query = f"{base_query} WHERE production_batch_id IN ({placeholders}) ORDER BY rg_id ASC;"
                parameters = validated_batches
            else:
                # No valid batches, return empty result
                query = f"{base_query} WHERE 1=0;"
                parameters = []
        else:
            query = f"{base_query} ORDER BY rg_id ASC;"
            parameters = []
        
        logger.info(f"Built secure production query for {len(parameters) if parameters else 'all'} batches")
        return query, parameters
    
    def build_print_data_by_batch_query(self, batch_ids: List[str]) -> Tuple[str, List[Any]]:
        """
        Build secure query to fetch print data by batch IDs (regardless of time).
        This allows fetching complete batch data even if printed on different days.

        Args:
            batch_ids: List of batch IDs to query

        Returns:
            Tuple[str, List[Any]]: (query_string, parameters)
        """
        if not batch_ids or len(batch_ids) == 0:
            raise ValueError("batch_ids cannot be empty")

        # Validate batch IDs (alphanumeric + hyphens/underscores)
        validated_batches = []
        for batch_id in batch_ids:
            if self.validate_batch_id(str(batch_id)):
                validated_batches.append(str(batch_id))
            else:
                logger.warning(f"Invalid batch_id format: {batch_id}")

        if not validated_batches:
            raise ValueError("No valid batch IDs provided")

        query = """
            SELECT
                ts,
                split_part(topic, '/', 6) AS cell,
                payload::jsonb->>'jobId' AS "jobId",
                payload::jsonb->'data'->'ids'->>'batchId' AS "batchId",
                (payload::jsonb->'data'->'ids'->>'sheetIndex')::numeric AS "sheetIndex",
                (payload::jsonb->'data'->'reportData'->'oee'->>'cycleTime')::numeric AS "cycleTime (min)",
                (payload::jsonb->'data'->'reportData'->'oee'->>'runtime')::numeric AS "runtime (min)",
                ((payload::jsonb->'data'->'reportData'->'oee'->>'performance')::numeric / 100.0) AS "performance",
                ((payload::jsonb->'data'->'reportData'->'oee'->>'availability')::numeric / 100.0) AS "availability",
                ((payload::jsonb->'data'->'reportData'->'oee'->>'oee')::numeric / 100.0) AS "oee",
                (payload::jsonb->'data'->'reportData'->'printSettings'->>'nominalSpeed')::numeric AS "nominalSpeed",
                (payload::jsonb->'data'->'reportData'->'printSettings'->>'speed')::numeric AS "speed",
                payload::jsonb->'data'->'reportData'->'printSettings'->>'resolution' AS "resolution",
                payload::jsonb->'data'->'reportData'->'printSettings'->>'quality' AS "quality",
                payload::jsonb->'data'->'reportData'->'printSettings'->>'direction' AS "direction",
                ((payload::jsonb->'data'->'reportData'->'jobInfo'->>'width')::numeric * 0.001) AS "width (m)",
                ((payload::jsonb->'data'->'reportData'->'jobInfo'->>'height')::numeric * 0.001) AS "length (m)",
                (payload::jsonb->'data'->'reportData'->'jobInfo'->>'area')::numeric AS "area (m²)",
                payload::jsonb->'data'->'reportData'->'jobStatus'->>'endReason' AS "endReason",
                payload::jsonb->'data'->'reportData'->'timing'->>'start' AS timing_start_iso,
                payload::jsonb->'data'->'reportData'->'timing'->>'end' AS timing_end_iso,
                ((payload::jsonb->'data'->'time'->>'downtime')::numeric / 60.0) AS "downtime (min)",
                ((payload::jsonb->'data'->'time'->>'uptime')::numeric / 60.0) AS "uptime (min)",
                payload::jsonb->'data'->'time'->'downtimeReasons'::text AS downtime_reasons_json,
                payload::jsonb->>'jobState' AS "jobState",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->>'totalInk')::numeric AS "totalInk (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'C')::numeric AS "ink_C (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'M')::numeric AS "ink_M (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'Y')::numeric AS "ink_Y (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'K')::numeric AS "ink_K (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'R')::numeric AS "ink_R (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'G')::numeric AS "ink_G (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'S')::numeric AS "ink_S (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'FOF')::numeric AS "ink_FOF (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->>'inkPerArea')::numeric AS "inkPerArea (mL/m²)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->>'inkPerMeter')::numeric AS "inkPerMeter (mL/m)"
            FROM jobs
            WHERE topic = %s
            AND payload::jsonb->'data'->'ids'->>'batchId' = ANY(%s)
            ORDER BY ts ASC;
        """

        topic = 'rg_v2/RG/CPH/Prod/ComponentLine/Print1/JobReport'
        parameters = [topic, validated_batches]

        logger.info(f"Built secure print data by batch query for {len(validated_batches)} batches")
        return query, parameters

    def build_quality_metrics_by_batch_query(self, batch_ids: List[str]) -> Tuple[str, List[Any]]:
        """
        Build secure query to fetch quality metrics for batches from production_componentorder.
        This provides the Quality component of OEE calculation.

        Quality Rate = (Passed Components) / (Passed + Failed Components)

        Args:
            batch_ids: List of batch IDs to query

        Returns:
            Tuple[str, List[Any]]: (query_string, parameters)
        """
        if not batch_ids or len(batch_ids) == 0:
            raise ValueError("batch_ids cannot be empty")

        # Validate and convert batch IDs to integers for INSIDE DB
        # Strip "B" prefix if present (e.g., "B955" -> 955)
        validated_batches = []
        for batch_id in batch_ids:
            try:
                # Remove "B" prefix if present
                batch_id_str = str(batch_id).lstrip('B').lstrip('b')
                # Convert to int (production_batch_id is bigint in INSIDE DB)
                int_batch_id = int(batch_id_str)
                validated_batches.append(int_batch_id)
            except (ValueError, TypeError):
                logger.warning(f"Invalid batch_id format (must be integer or 'B' prefix): {batch_id}")

        if not validated_batches:
            raise ValueError("No valid batch IDs provided")

        query = """
            SELECT
                pc.production_batch_id,
                DATE(pc.created_at) as production_date,
                COUNT(*) as total_components,
                -- Get QC state from the most recent QC event, or default to 'not qced'
                COUNT(CASE 
                    WHEN EXISTS (
                        SELECT 1 FROM production_componentorderevent pce2 
                        WHERE pce2.component_order_id = pc.id 
                        AND pce2.event_type = 'qc' 
                        AND pce2.state = 'passed'
                    ) THEN 1 
                END) as total_passed,
                COUNT(CASE 
                    WHEN EXISTS (
                        SELECT 1 FROM production_componentorderevent pce2 
                        WHERE pce2.component_order_id = pc.id 
                        AND pce2.event_type = 'qc' 
                        AND pce2.state = 'failed'
                    ) THEN 1 
                END) as total_failed,
                COUNT(CASE 
                    WHEN NOT EXISTS (
                        SELECT 1 FROM production_componentorderevent pce2 
                        WHERE pce2.component_order_id = pc.id 
                        AND pce2.event_type = 'qc'
                    ) THEN 1 
                END) as not_scanned,

                -- Quality rate for checked items only: passed / (passed + failed)
                CASE
                    WHEN (
                        COUNT(CASE WHEN EXISTS (
                            SELECT 1 FROM production_componentorderevent pce2 
                            WHERE pce2.component_order_id = pc.id 
                            AND pce2.event_type = 'qc' 
                            AND pce2.state IN ('passed', 'failed')
                        ) THEN 1 END) > 0
                    ) THEN
                        (COUNT(CASE WHEN EXISTS (
                            SELECT 1 FROM production_componentorderevent pce2 
                            WHERE pce2.component_order_id = pc.id 
                            AND pce2.event_type = 'qc' 
                            AND pce2.state = 'passed'
                        ) THEN 1 END) * 100.0) /
                        COUNT(CASE WHEN EXISTS (
                            SELECT 1 FROM production_componentorderevent pce2 
                            WHERE pce2.component_order_id = pc.id 
                            AND pce2.event_type = 'qc' 
                            AND pce2.state IN ('passed', 'failed')
                        ) THEN 1 END)
                    ELSE NULL
                END as quality_rate_checked_percent,

                -- Unchecked rate: not qc'd / total
                (COUNT(CASE WHEN NOT EXISTS (
                    SELECT 1 FROM production_componentorderevent pce2 
                    WHERE pce2.component_order_id = pc.id 
                    AND pce2.event_type = 'qc'
                ) THEN 1 END) * 100.0) /
                COUNT(*) as unchecked_rate_percent,

                -- Quality rate including unchecked (assume unchecked = good for conservative OEE)
                (COUNT(CASE WHEN EXISTS (
                    SELECT 1 FROM production_componentorderevent pce2 
                    WHERE pce2.component_order_id = pc.id 
                    AND pce2.event_type = 'qc' 
                    AND pce2.state = 'passed'
                ) OR NOT EXISTS (
                    SELECT 1 FROM production_componentorderevent pce2 
                    WHERE pce2.component_order_id = pc.id 
                    AND pce2.event_type = 'qc'
                ) THEN 1 END) * 100.0) /
                COUNT(*) as quality_rate_inclusive_percent

            FROM production_componentorder pc
            JOIN production_printjob pj ON pc.print_job_id = pj.id
            WHERE pc.production_batch_id = ANY(%s::bigint[])
              AND pj.status != 'cancelled'
            GROUP BY pc.production_batch_id, DATE(pc.created_at)
            ORDER BY DATE(pc.created_at) DESC, pc.production_batch_id DESC;
        """

        parameters = [validated_batches]

        logger.info(f"Built secure quality metrics query for {len(validated_batches)} batches")
        return query, parameters

    def build_batch_quality_breakdown_query(self, batch_ids: List[str]) -> Tuple[str, List[Any]]:
        """
        Build secure query for Phase 2: Post-QC Reconciliation with defect attribution.
        
        Updated batch quality breakdown query with refined defect categories.
        
        Based on actual defect data observed:
        - file, file-product → Pre-production/File issues
        - cut, cut-outside-bleed, cut-measurement, cut-measurement-small, cut-fraying, cut-measurement-big → Cutter
        - print, print-water-mark → Printer
        - fabric → Incoming material
        - other → Unattributed
        
        This query:
        1. Starts from production_componentorder to capture ALL components (including not scanned)
        2. LEFT JOINs to events to get QC status
        3. Attributes defects to cells based on description patterns
        4. Handles manually-added components (handheld-manual, inside sources)
        
        Args:
            batch_ids: List of batch IDs (can be 'B955' or '955' format)
            
        Returns:
            Tuple[str, List[Any]]: (query_string, parameters)
        """
        if not batch_ids or len(batch_ids) == 0:
            raise ValueError("batch_ids cannot be empty")

        # Convert string batch IDs to integers
        validated_batches = []
        for bid in batch_ids:
            try:
                # Handle 'B955' format
                clean_id = str(bid).replace('B', '').replace('b', '').strip()
                validated_batches.append(int(clean_id))
            except (ValueError, AttributeError, TypeError):
                logger.warning(f"Invalid batch ID format: {bid}")
                continue

        if not validated_batches:
            raise ValueError("No valid batch IDs provided")

        query = """
            SELECT
                pc.production_batch_id,
                DATE(pc.created_at) as production_date,
                
                -- ===================
                -- TOTAL COUNTS
                -- ===================
                COUNT(DISTINCT pc.id) as total_components,
                
                -- Passed: has a QC event with state = 'passed'
                COUNT(DISTINCT CASE 
                    WHEN pce.state = 'passed' THEN pc.id 
                END) as total_passed,
                
                -- Failed: has a QC event with state = 'failed'
                COUNT(DISTINCT CASE 
                    WHEN pce.state = 'failed' THEN pc.id 
                END) as total_failed,
                
                -- Not scanned: no QC event exists at all
                COUNT(DISTINCT CASE 
                    WHEN pce.id IS NULL THEN pc.id 
                END) as not_scanned,

                -- ===================
                -- DEFECT ATTRIBUTION BY CELL
                -- ===================
                
                -- Print defects (Printer cell): print, print-water-mark
                COUNT(DISTINCT CASE 
                    WHEN pce.state = 'failed' 
                    AND (
                        pce.description::text ILIKE '%%print%%'
                        OR pce.description::text ILIKE '%%water-mark%%'
                    )
                    -- Exclude 'file-product' from print matches
                    AND pce.description::text NOT ILIKE '%%file%%'
                    THEN pc.id 
                END) as print_defects,
                
                -- Cut defects (Cutter cell): cut, cut-outside-bleed, cut-measurement, cut-fraying, etc.
                COUNT(DISTINCT CASE 
                    WHEN pce.state = 'failed' 
                    AND (
                        pce.description::text ILIKE '%%cut%%'
                        OR pce.description::text ILIKE '%%bleed%%'
                        OR pce.description::text ILIKE '%%fraying%%'
                        OR pce.description::text ILIKE '%%measurement%%'
                    )
                    THEN pc.id 
                END) as cut_defects,
                
                -- Fabric defects (Incoming material - not cell attributed)
                COUNT(DISTINCT CASE 
                    WHEN pce.state = 'failed' 
                    AND pce.description::text ILIKE '%%fabric%%'
                    THEN pc.id 
                END) as fabric_defects,
                
                -- File/Pre-production defects (upstream - not cell attributed)
                COUNT(DISTINCT CASE 
                    WHEN pce.state = 'failed' 
                    AND (
                        pce.description::text ILIKE '%%file%%'
                        OR pce.description::text ILIKE '%%product%%'
                    )
                    THEN pc.id 
                END) as file_defects,
                
                -- Pick defects (Pick cell) - keeping for future use
                COUNT(DISTINCT CASE 
                    WHEN pce.state = 'failed' 
                    AND pce.description::text ILIKE '%%pick%%'
                    THEN pc.id 
                END) as pick_defects,
                
                -- Other/Unattributed defects
                COUNT(DISTINCT CASE 
                    WHEN pce.state = 'failed' 
                    AND pce.description::text ILIKE '%%other%%'
                    THEN pc.id 
                END) as other_defects,

                -- ===================
                -- QC SOURCE BREAKDOWN
                -- ===================
                
                -- Handheld scanner (primary method)
                COUNT(DISTINCT CASE 
                    WHEN pce.source = 'handheld-scan' THEN pc.id 
                END) as scanned_handheld,
                
                -- Manual entry
                COUNT(DISTINCT CASE 
                    WHEN pce.source = 'handheld-manual' THEN pc.id 
                END) as scanned_manual,
                
                -- System/Inside
                COUNT(DISTINCT CASE 
                    WHEN pce.source = 'inside' THEN pc.id 
                END) as scanned_inside,

                -- ===================
                -- QUALITY RATES
                -- ===================
                
                -- Quality rate (checked items only): passed / (passed + failed)
                CASE
                    WHEN COUNT(DISTINCT CASE 
                        WHEN pce.state IN ('passed', 'failed') THEN pc.id 
                    END) > 0 
                    THEN
                        (COUNT(DISTINCT CASE WHEN pce.state = 'passed' THEN pc.id END) * 100.0) /
                        COUNT(DISTINCT CASE WHEN pce.state IN ('passed', 'failed') THEN pc.id END)
                    ELSE NULL
                END as quality_rate_checked_percent,

                -- Scan coverage: (passed + failed) / total
                (COUNT(DISTINCT CASE 
                    WHEN pce.state IN ('passed', 'failed') THEN pc.id 
                END) * 100.0) /
                NULLIF(COUNT(DISTINCT pc.id), 0) as scan_coverage_percent,
                
                -- Unchecked rate: not scanned / total
                (COUNT(DISTINCT CASE 
                    WHEN pce.id IS NULL THEN pc.id 
                END) * 100.0) /
                NULLIF(COUNT(DISTINCT pc.id), 0) as unchecked_rate_percent

            FROM production_componentorder pc
            JOIN production_printjob pj ON pc.print_job_id = pj.id
            LEFT JOIN production_componentorderevent pce 
                ON pce.component_order_id = pc.id 
                AND pce.event_type = 'qc'
            WHERE pc.production_batch_id = ANY(%s::bigint[])
              AND pj.status != 'cancelled'
            GROUP BY pc.production_batch_id, DATE(pc.created_at)
            ORDER BY DATE(pc.created_at) DESC, pc.production_batch_id DESC
        """
        
        logger.info(f"Built secure batch quality breakdown query for {len(validated_batches)} batches")
        return query, [validated_batches]

    def build_component_ids_for_batch_query(self, batch_ids: List[str]) -> Tuple[str, List[Any]]:
        """
        Build query to fetch component IDs for specified batches.
        
        This is Step 1 of the two-database QC fetch approach.
        
        Args:
            batch_ids: List of batch IDs (can be 'B955' or '955' format)
            
        Returns:
            Tuple of (query_string, parameters_list)
        """
        # Convert string batch IDs to integers
        validated_batches = []
        for bid in batch_ids:
            try:
                clean_id = str(bid).replace('B', '').replace('b', '').strip()
                validated_batches.append(int(clean_id))
            except (ValueError, AttributeError, TypeError):
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

    def build_qc_data_from_historian_query(self, component_ids: List[str]) -> Tuple[str, List[Any]]:
        """
        Build query to fetch QC data from historian public.components table.
        
        This is Step 2 of the two-database QC fetch approach.
        
        Note: component_ids are actually rg_id values (text) from production_printjob,
        which match component_id (text) in the historian database.
        
        Args:
            component_ids: List of rg_id values (strings) to fetch QC data for
            
        Returns:
            Tuple of (query_string, parameters_list)
        """
        if not component_ids:
            raise ValueError("No component IDs provided")
        
        # component_id in historian is text, and we're matching with rg_id (also text)
        query = """
            SELECT 
                c.component_id,
                c.payload::json->>'state' as qc_state,
                c.payload::json->>'source' as qc_source,
                c.payload::json->'reasons' as qc_reasons
            FROM public.components c
            WHERE c.process = 'QC'
              AND c.component_id = ANY(%s::text[])
        """
        
        logger.info(f"Built historian QC query for {len(component_ids)} rg_id values")
        return query, [component_ids]

    def build_cut_data_by_job_query(self, job_ids: List[str]) -> Tuple[str, List[Any]]:
        """
        Build secure query to fetch cut data by specific job IDs.
        This is the correct way to get cut data since cut records don't have batch IDs.

        Args:
            job_ids: List of job IDs (from print jobs) to query cut data for

        Returns:
            Tuple[str, List[Any]]: (query_string, parameters)
        """
        if not job_ids or len(job_ids) == 0:
            raise ValueError("job_ids cannot be empty")

        # Validate job IDs
        validated_jobs = []
        for job_id in job_ids:
            job_str = str(job_id).strip()
            if len(job_str) > 0 and len(job_str) < 100:
                validated_jobs.append(job_str)
            else:
                logger.warning(f"Invalid job_id format: {job_id}")

        if not validated_jobs:
            raise ValueError("No valid job IDs provided")

        query = """
            SELECT
                j.ts,
                j.payload::jsonb->>'jobId' AS job_id,
                j.payload::jsonb->'data'->'ids'->>'batchId' AS "batchId",
                ((j.payload::jsonb->'data'->'time'->>'uptime')::numeric / 60.0) AS "uptime (min)",
                ((j.payload::jsonb->'data'->'time'->>'downtime')::numeric / 60.0) AS "downtime (min)",
                CASE
                    WHEN (j.payload::jsonb->'data'->'time'->>'uptime')::numeric + (j.payload::jsonb->'data'->'time'->>'downtime')::numeric > 0
                    THEN ((j.payload::jsonb->'data'->'time'->>'uptime')::numeric /
                          ((j.payload::jsonb->'data'->'time'->>'uptime')::numeric + (j.payload::jsonb->'data'->'time'->>'downtime')::numeric))
                    ELSE 0
                END AS availability,
                (j.payload::jsonb->'data'->>'componentCount')::numeric AS component_count,
                j.payload::jsonb->'data'->'reportData'->'timing'->>'start' AS timing_start,
                j.payload::jsonb->'data'->'reportData'->'timing'->>'end' AS timing_end,
                (j.payload::jsonb->'data'->'reportData'->>'patternPiecesPerHour')::numeric AS pattern_pieces_per_hour,
                split_part(j.topic, '/', 6) AS cell
            FROM jobs j
            WHERE j.topic = %s
            AND j.payload::jsonb->>'jobId' = ANY(%s)
            ORDER BY j.ts ASC;
        """

        topic = 'rg_v2/RG/CPH/Prod/ComponentLine/Cut1/JobReport'
        parameters = [topic, validated_jobs]

        logger.info(f"Built secure cut data by job query for {len(validated_jobs)} jobs")
        return query, parameters

    def build_pick_data_by_job_query(self, job_ids: List[str]) -> Tuple[str, List[Any]]:
        """
        Build secure query to fetch Pick/Robot data by specific job IDs with component-level details.
        Pick data tracks individual components within each job to ensure all printed components are picked.

        Args:
            job_ids: List of job IDs (from print jobs) to query pick data for

        Returns:
            Tuple[str, List[Any]]: (query_string, parameters)
        """
        if not job_ids or len(job_ids) == 0:
            raise ValueError("job_ids cannot be empty")

        # Validate job IDs
        validated_jobs = []
        for job_id in job_ids:
            job_str = str(job_id).strip()
            if len(job_str) > 0 and len(job_str) < 100:
                validated_jobs.append(job_str)
            else:
                logger.warning(f"Invalid job_id format: {job_id}")

        if not validated_jobs:
            raise ValueError("No valid job IDs provided")

        query = """
            SELECT
                j.ts,
                j.payload::jsonb->>'jobId' AS job_id,
                -- Extract batchId from first component in componentsCompleted or componentsEvents
                COALESCE(
                    j.payload::jsonb->'data'->'reportData'->'closingReport'->'componentsCompleted'->0->'ids'->>'productionBatchRgId',
                    j.payload::jsonb->'data'->'reportData'->'closingReport'->'componentsEvents'->0->'payload'->'data'->'ids'->>'productionBatchRgId',
                    ''
                ) AS "batchId",
                -- sheetIndex not available in Pick JobReports - will be merged from Print JobReports by job_id
                NULL AS "sheetIndex",
                split_part(j.topic, '/', 6) AS cell,

                -- Job-level timing
                j.payload::jsonb->'data'->'reportData'->'closingReport'->'jobActiveWindow'->>'startTs' AS job_start,
                j.payload::jsonb->'data'->'reportData'->'closingReport'->'jobActiveWindow'->>'endTs' AS job_end,
                (j.payload::jsonb->'data'->'reportData'->'closingReport'->'jobActiveWindow'->>'duration_s')::numeric AS job_duration_s,
                
                -- Sheet timing (for batch time window calculation)
                -- insideSummary is at data.insideSummary (same level as data.reportData)
                j.payload::jsonb->'data'->'insideSummary'->>'sheetStart_ts' AS sheet_start_ts,
                j.payload::jsonb->'data'->'insideSummary'->>'sheetEnd_ts' AS sheet_end_ts,

                -- Uptime and Downtime
                COALESCE((j.payload::jsonb->'data'->'time'->>'uptime')::numeric, 0) / 60.0 AS "uptime (min)",
                COALESCE((j.payload::jsonb->'data'->'time'->>'downtime')::numeric, 0) / 60.0 AS "downtime (min)",

                -- Pick quality metrics from closingReport
                COALESCE(
                    (j.payload::jsonb->'data'->'reportData'->'closingReport'->>'componentsPickCompleted_qty')::int,
                    0
                ) AS successful_picks,
                COALESCE(
                    (j.payload::jsonb->'data'->'reportData'->'closingReport'->>'componentsPickCompleted_qty')::int +
                    (j.payload::jsonb->'data'->'reportData'->'closingReport'->>'componentsPickFailed_qty')::int,
                    (j.payload::jsonb->'data'->'reportData'->'closingReport'->>'printJobComponentOrderCount')::int,
                    0
                ) AS total_picks,

                -- Component count from closingReport
                COALESCE(
                    (j.payload::jsonb->'data'->'reportData'->'closingReport'->>'printJobComponentOrderCount')::int,
                    jsonb_array_length(j.payload::jsonb->'data'->'reportData'->'closingReport'->'componentsEvents'),
                    0
                ) AS component_count,

                -- Component details (arrays from closingReport)
                j.payload::jsonb->'data'->'reportData'->'closingReport'->'componentsCompleted' AS components_completed,
                j.payload::jsonb->'data'->'reportData'->'closingReport'->'componentsFailed' AS components_failed,
                j.payload::jsonb->'data'->'reportData'->'closingReport'->'componentsStateUnknown' AS components_state_unknown,
                j.payload::jsonb->'data'->'reportData'->'closingReport'->'componentsEvents' AS components_events

            FROM jobs j
            WHERE j.topic = %s
            AND j.payload::jsonb->>'jobId' = ANY(%s)
            ORDER BY j.ts ASC;
        """

        topic = 'rg_v2/RG/CPH/Prod/ComponentLine/Pick1/JobReport'
        parameters = [topic, validated_jobs]

        logger.info(f"Built secure pick data by job query for {len(validated_jobs)} jobs")
        return query, parameters

    def build_pick_data_by_batch_query(self, batch_ids: List[str]) -> Tuple[str, List[Any]]:
        """
        Build simplified query to fetch Pick data by batch from INSIDE database.
        
        This query:
        1. Gets all components for the batch from production_componentorder
        2. Matches component IDs with pick status from production_componentorderevent
        3. Returns component-level data with pick status (successful/failed/not_picked)
        
        Note: Uptime/downtime is calculated separately from equipment table in the fetcher.

        Args:
            batch_ids: List of batch IDs (can be 'B955' or '955' format)

        Returns:
            Tuple[str, List[Any]]: (query_string, parameters)
        """
        # Validate batch IDs
        validated_batches = []
        for bid in batch_ids:
            try:
                clean_id = str(bid).replace('B', '').replace('b', '').strip()
                validated_batches.append(int(clean_id))
            except (ValueError, AttributeError, TypeError):
                logger.warning(f"Invalid batch ID format: {bid}")
                continue

        if not validated_batches:
            raise ValueError("No valid batch IDs provided")

        # Simple query: Get components from INSIDE and match with pick status
        query = """
            SELECT
                pc.id as component_id,
                pc.production_batch_id,
                pj.rg_id as job_id,
                pj.id as print_job_id,
                pc.created_at,
                -- Pick status from production_componentorderevent
                CASE 
                    WHEN pick_event.state = 'passed' OR pick_event.state = 'successful' THEN 'successful'
                    WHEN pick_event.state = 'failed' THEN 'failed'
                    ELSE 'not_picked'
                END as pick_status,
                pick_event.state as pick_event_state,
                pick_event.created_at as pick_event_time
            FROM production_componentorder pc
            JOIN production_printjob pj ON pc.print_job_id = pj.id
            LEFT JOIN production_componentorderevent pick_event 
                ON pick_event.component_order_id = pc.id 
                AND pick_event.event_type = 'pick'
            WHERE pc.production_batch_id = ANY(%s::bigint[])
              AND pj.status != 'cancelled'
            ORDER BY pc.id ASC
        """
        
        logger.info(
            f"Built simplified pick data by batch query for {len(validated_batches)} batches"
        )
        return query, [validated_batches]

    def build_print_data_by_time_query(self, start_time: str, end_time: str) -> Tuple[str, List[Any]]:
        """
        Build secure parameterized query for print data by time range.
        
        Args:
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
            
        Returns:
            Tuple[str, List[Any]]: (query_string, parameters)
        """
        # Validate inputs
        if not self.validate_datetime(start_time):
            raise ValueError(f"Invalid start_time format: {start_time}")
        if not self.validate_datetime(end_time):
            raise ValueError(f"Invalid end_time format: {end_time}")
        
        query = """
            SELECT
                ts,
                split_part(topic, '/', 6) AS cell,
                payload::jsonb->>'jobId' AS "jobId",
                payload::jsonb->'data'->'ids'->>'batchId' AS "batchId",
                (payload::jsonb->'data'->'ids'->>'sheetIndex')::numeric AS "sheetIndex",
                (payload::jsonb->'data'->'reportData'->'oee'->>'cycleTime')::numeric AS "cycleTime (min)",
                (payload::jsonb->'data'->'reportData'->'oee'->>'runtime')::numeric AS "runtime (min)",
                ((payload::jsonb->'data'->'reportData'->'oee'->>'performance')::numeric / 100.0) AS "performance",
                ((payload::jsonb->'data'->'reportData'->'oee'->>'availability')::numeric / 100.0) AS "availability",
                ((payload::jsonb->'data'->'reportData'->'oee'->>'oee')::numeric / 100.0) AS "oee",
                (payload::jsonb->'data'->'reportData'->'printSettings'->>'nominalSpeed')::numeric AS "nominalSpeed",
                (payload::jsonb->'data'->'reportData'->'printSettings'->>'speed')::numeric AS "speed",
                payload::jsonb->'data'->'reportData'->'printSettings'->>'resolution' AS "resolution",
                payload::jsonb->'data'->'reportData'->'printSettings'->>'quality' AS "quality",
                payload::jsonb->'data'->'reportData'->'printSettings'->>'direction' AS "direction",
                ((payload::jsonb->'data'->'reportData'->'jobInfo'->>'width')::numeric * 0.001) AS "width (m)",
                ((payload::jsonb->'data'->'reportData'->'jobInfo'->>'height')::numeric * 0.001) AS "length (m)",
                (payload::jsonb->'data'->'reportData'->'jobInfo'->>'area')::numeric AS "area (m²)",
                payload::jsonb->'data'->'reportData'->'jobStatus'->>'endReason' AS "endReason",
                payload::jsonb->'data'->'reportData'->'timing'->>'start' AS timing_start_iso,
                payload::jsonb->'data'->'reportData'->'timing'->>'end' AS timing_end_iso,
                ((payload::jsonb->'data'->'time'->>'downtime')::numeric / 60.0) AS "downtime (min)",
                ((payload::jsonb->'data'->'time'->>'uptime')::numeric / 60.0) AS "uptime (min)",
                payload::jsonb->'data'->'time'->'downtimeReasons'::text AS downtime_reasons_json,
                payload::jsonb->>'jobState' AS "jobState",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->>'totalInk')::numeric AS "totalInk (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'C')::numeric AS "ink_C (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'M')::numeric AS "ink_M (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'Y')::numeric AS "ink_Y (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'K')::numeric AS "ink_K (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'R')::numeric AS "ink_R (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'G')::numeric AS "ink_G (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'S')::numeric AS "ink_S (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->'channel'->>'FOF')::numeric AS "ink_FOF (mL)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->>'inkPerArea')::numeric AS "inkPerArea (mL/m²)",
                (payload::jsonb->'data'->'reportData'->'inkInfo'->>'inkPerMeter')::numeric AS "inkPerMeter (mL/m)"
            FROM jobs
            WHERE topic = %s
            AND (payload::jsonb->'data'->'reportData'->'timing'->>'start')::timestamp >= %s::timestamp
            AND (payload::jsonb->'data'->'reportData'->'timing'->>'start')::timestamp < %s::timestamp
            ORDER BY (payload::jsonb->'data'->'reportData'->'timing'->>'start')::timestamp ASC;
        """
        
        topic = 'rg_v2/RG/CPH/Prod/ComponentLine/Print1/JobReport'
        parameters = [topic, start_time, end_time]
        
        logger.info(f"Built secure print data query for time range {start_time} to {end_time}")
        return query, parameters


# Global instance for convenience
secure_query_builder = SecureQueryBuilder()

