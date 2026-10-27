"""
Universal PostgreSQL Transaction Pooler Engine
Provides zero-lock, transaction-scoped database connectivity for TorchJIT/UPAAS nodes.
Supports Supavisor, PgBouncer, Neon, AWS RDS Proxy, or any standard Postgres transaction pooler.
"""
import os
import urllib.parse
from typing import Optional, List, Dict, Any
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

class TransactionPooler:
    """
    Direct Transaction Pooler client:
    - Never uses PREPARED statements (which break transaction poolers).
    - Sets autocommit=True or rolls back immediately after statement completion.
    - Operates with ephemeral short-lived socket checkouts.
    """
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.environ.get("DATABASE_URL", "")
        self._pool = None
        if self.database_url:
            self._init_pool()

    def _init_pool(self):
        try:
            # Transaction poolers require sslmode=require and standard TCP socket
            self._pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,  # Free PaaS should use minimal local client sockets
                dsn=self.database_url
            )
        except Exception as e:
            print(f"[TransactionPooler] Warning: Failed to connect to pooler: {e}")

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Executes a query inside an isolated transaction and immediately releases connection."""
        if not self._pool:
            return []

        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or ())
                if cur.description:
                    res = cur.fetchall()
                else:
                    res = []
            conn.commit()
            return [dict(r) for r in res]
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._pool.putconn(conn)

    def register_cid_chunk(self, model_id: str, layer_idx: int, chunk_idx: int, cid: str, byte_size: int):
        """Register a 1-3 MB micro-chunk directly into the universal database."""
        query = """
        INSERT INTO model_chunks (model_id, layer_index, chunk_index, cid, byte_size)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (model_id, layer_index, chunk_index) 
        DO UPDATE SET cid = EXCLUDED.cid, byte_size = EXCLUDED.byte_size;
        """
        self.execute_query(query, (model_id, layer_idx, chunk_idx, cid, byte_size))

    def get_layer_chunks(self, model_id: str, layer_idx: int) -> List[Dict[str, Any]]:
        """Fetch all 1-3 MB chunk CIDs for a given layer."""
        query = """
        SELECT chunk_index, cid, byte_size 
        FROM model_chunks 
        WHERE model_id = %s AND layer_index = %s
        ORDER BY chunk_index ASC;
        """
        return self.execute_query(query, (model_id, layer_idx))
