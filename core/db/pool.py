"""
Database Connection Pool Manager

Provides thread-safe PostgreSQL connection pooling with health checks
and automatic reconnection for both HISTORIAN and INSIDE databases.
"""

import os
import threading
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

import psycopg2
from psycopg2 import pool


# Configure logging
logger = logging.getLogger(__name__)


class DatabasePool:
    """Thread-safe database connection pool manager."""

    def __init__(self, database_type: str = "historian"):
        """
        Initialize the database pool.

        Args:
            database_type: "historian" or "inside" to determine which DB to connect to

        Raises:
            ValueError: If database_type is not recognized
        """
        self.database_type = database_type
        self.pool: Optional[pool.AbstractConnectionPool] = None
        self.pool_lock = threading.Lock()
        self.stats = {
            "connections_created": 0,
            "connections_used": 0,
            "connections_returned": 0,
            "pool_exhausted": 0,
            "fallback_connections": 0,
            "errors": 0
        }

        # Get database configuration based on type
        if database_type == "historian":
            self.db_config = {
                "host": os.getenv("HISTORIANDB_HOST"),
                "port": os.getenv("HISTORIANDB_PORT"),
                "database": os.getenv("HISTORIANDB_NAME"),
                "user": os.getenv("HISTORIANDB_USER"),
                "password": os.getenv("HISTORIANDB_PASS"),
                "sslmode": "disable"
            }
        elif database_type == "inside":
            self.db_config = {
                "host": os.getenv("INSIDEDB_HOST"),
                "port": os.getenv("INSIDEDB_PORT"),
                "database": os.getenv("INSIDEDB_NAME"),
                "user": os.getenv("INSIDEDB_USER"),
                "password": os.getenv("INSIDEDB_PASS"),
                "sslmode": "disable"
            }
        else:
            raise ValueError(f"Unknown database type: {database_type}")

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate that all required configuration is present."""
        required_keys = ["host", "port", "database", "user", "password"]
        missing = [k for k in required_keys if not self.db_config.get(k)]

        if missing:
            raise ValueError(
                f"Missing database configuration for {self.database_type}: {missing}. "
                f"Please check your .env file."
            )

    def initialize_pool(self, min_connections: int = 2, max_connections: int = 10) -> bool:
        """
        Initialize the connection pool.

        Args:
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections allowed

        Returns:
            bool: True if pool was created successfully, False otherwise
        """
        try:
            with self.pool_lock:
                if self.pool is not None:
                    logger.info(f"{self.database_type} pool already initialized")
                    return True  # Already initialized

                self.pool = psycopg2.pool.ThreadedConnectionPool(
                    min_connections,
                    max_connections,
                    **self.db_config
                )

                self.stats["connections_created"] = min_connections

                # Test the pool with a simple query
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        cursor.fetchone()

                logger.info(
                    f"Initialized {self.database_type} pool with "
                    f"{min_connections}-{max_connections} connections"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to initialize {self.database_type} pool: {e}")
            self.stats["errors"] += 1
            return False

    @contextmanager
    def get_connection(self, timeout: int = 30):
        """
        Get a connection from the pool using context manager.

        Args:
            timeout: Maximum time to wait for a connection (seconds)

        Yields:
            psycopg2.connection: Database connection

        Example:
            >>> pool = DatabasePool("historian")
            >>> with pool.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM jobs LIMIT 1")
        """
        connection = None
        start_time = time.time()

        try:
            # Try to get connection from pool
            if self.pool is not None:
                try:
                    connection = self.pool.getconn()
                    if connection:
                        self.stats["connections_used"] += 1
                        yield connection
                        return
                except pool.PoolError:
                    self.stats["pool_exhausted"] += 1
                    logger.warning(f"{self.database_type} pool exhausted, using fallback")
                    # Fall through to direct connection

            # Fallback to direct connection
            self.stats["fallback_connections"] += 1
            connection = psycopg2.connect(**self.db_config)
            yield connection

        except Exception as e:
            self.stats["errors"] += 1
            elapsed = time.time() - start_time
            logger.error(
                f"{self.database_type} connection error after {elapsed:.2f}s: {e}"
            )
            raise

        finally:
            if connection:
                try:
                    # Return connection to pool or close direct connection
                    if self.pool is not None and hasattr(self.pool, 'putconn'):
                        self.pool.putconn(connection)
                        self.stats["connections_returned"] += 1
                    else:
                        connection.close()
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")
                    self.stats["errors"] += 1

    def close_pool(self):
        """Close all connections in the pool."""
        try:
            with self.pool_lock:
                if self.pool is not None:
                    self.pool.closeall()
                    self.pool = None
                    logger.info(f"Closed {self.database_type} connection pool")
        except Exception as e:
            logger.warning(f"Error closing {self.database_type} pool: {e}")
            self.stats["errors"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        stats = self.stats.copy()

        if self.pool is not None:
            try:
                stats["pool_initialized"] = True
                stats["pool_type"] = type(self.pool).__name__
            except:
                pass
        else:
            stats["pool_initialized"] = False

        return stats

    def health_check(self) -> bool:
        """
        Perform a health check on the pool.

        Returns:
            bool: True if pool is healthy, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result is not None
        except Exception as e:
            logger.error(f"{self.database_type} pool health check failed: {e}")
            return False


# Global pool instances
_historian_pool: Optional[DatabasePool] = None
_inside_pool: Optional[DatabasePool] = None
_pool_lock = threading.Lock()


def get_pool(database_type: str = "historian") -> DatabasePool:
    """
    Get or create a database pool instance.

    Args:
        database_type: "historian" or "inside"

    Returns:
        DatabasePool: Pool instance

    Raises:
        ValueError: If database_type is not recognized
    """
    global _historian_pool, _inside_pool

    with _pool_lock:
        if database_type == "historian":
            if _historian_pool is None:
                _historian_pool = DatabasePool("historian")
                _historian_pool.initialize_pool()
            return _historian_pool
        elif database_type == "inside":
            if _inside_pool is None:
                _inside_pool = DatabasePool("inside")
                _inside_pool.initialize_pool()
            return _inside_pool
        else:
            raise ValueError(f"Unknown database type: {database_type}")


def close_all_pools():
    """Close all database pools."""
    global _historian_pool, _inside_pool

    with _pool_lock:
        if _historian_pool is not None:
            _historian_pool.close_pool()
            _historian_pool = None
        if _inside_pool is not None:
            _inside_pool.close_pool()
            _inside_pool = None


@contextmanager
def get_historian_connection():
    """
    Get a historian database connection using context manager.

    Yields:
        psycopg2.connection: Database connection

    Example:
        >>> with get_historian_connection() as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT * FROM jobs LIMIT 1")
    """
    pool = get_pool("historian")
    with pool.get_connection() as conn:
        yield conn


@contextmanager
def get_inside_connection():
    """
    Get an inside database connection using context manager.

    Yields:
        psycopg2.connection: Database connection

    Example:
        >>> with get_inside_connection() as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT * FROM production_printjob LIMIT 1")
    """
    pool = get_pool("inside")
    with pool.get_connection() as conn:
        yield conn


def get_pool_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all pools.

    Returns:
        Dictionary with stats for each initialized pool
    """
    stats = {}

    if _historian_pool is not None:
        stats["historian"] = _historian_pool.get_stats()

    if _inside_pool is not None:
        stats["inside"] = _inside_pool.get_stats()

    return stats
