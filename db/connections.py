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
