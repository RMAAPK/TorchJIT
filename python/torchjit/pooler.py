"""
Universal Zero-Dependency Storage & Registry Provider for TorchJIT
Designed as a 1:1 drop-in replacement for PyTorch.

Modes:
1. Pure Local (Default): Zero server/network needed. Uses local SQLite/file cache in ~/.tjit/
2. Remote Tx Pooler (Opt-in): When DATABASE_URL is set, connects to PgBouncer/Postgres.
"""
import os
import sqlite3
from typing import Optional, List, Dict, Any

class StatePooler:
    """
    Unified state pooler.
    If no DATABASE_URL is provided, defaults silently to local embedded SQLite (~/.tjit/registry.db).
    Anyone importing torchjit locally has 0 database setup overhead!
    """
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.environ.get("DATABASE_URL", "")
        self.is_remote = bool(self.database_url and self.database_url.startswith("postgres"))
        self._pg_pool = None
        self.local_db_path = os.path.expanduser("~/.tjit/registry.db")

        if self.is_remote:
            self._init_remote_pool()
        else:
            self._init_local_db()

    def _init_local_db(self):
        """Zero-setup local SQLite fallback."""
        os.makedirs(os.path.dirname(self.local_db_path), exist_ok=True)
        with sqlite3.connect(self.local_db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS model_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                layer_index INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                cid TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_id, layer_index, chunk_index)
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_local_model_layer ON model_chunks(model_id, layer_index);")
            conn.commit()

    def _init_remote_pool(self):
        """Remote PostgreSQL Transaction Pooler (Port 6543 / PgBouncer)."""
        try:
            import psycopg2
            from psycopg2 import pool
            self._pg_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=self.database_url
            )
        except Exception as e:
            print(f"[TorchJIT Pooler] Remote pooler connection failed: {e}. Falling back to local offline DB.")
            self.is_remote = False
            self._init_local_db()

    def register_cid_chunk(self, model_id: str, layer_idx: int, chunk_idx: int, cid: str, byte_size: int):
        """Save a 1-3 MB chunk record (works offline or online)."""
        if self.is_remote and self._pg_pool:
            conn = self._pg_pool.getconn()
            try:
                with conn.cursor() as cur:
                    query = """
                    INSERT INTO model_chunks (model_id, layer_index, chunk_index, cid, byte_size)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (model_id, layer_index, chunk_index) 
                    DO UPDATE SET cid = EXCLUDED.cid, byte_size = EXCLUDED.byte_size;
                    """
                    cur.execute(query, (model_id, layer_idx, chunk_idx, cid, byte_size))
                conn.commit()
            finally:
                self._pg_pool.putconn(conn)
        else:
            with sqlite3.connect(self.local_db_path) as conn:
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO model_chunks (model_id, layer_index, chunk_index, cid, byte_size)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_id, layer_index, chunk_index)
                DO UPDATE SET cid=excluded.cid, byte_size=excluded.byte_size;
                """, (model_id, layer_idx, chunk_idx, cid, byte_size))
                conn.commit()

    def get_layer_chunks(self, model_id: str, layer_idx: int) -> List[Dict[str, Any]]:
        """Query chunk CIDs without needing any external server running."""
        if self.is_remote and self._pg_pool:
            from psycopg2.extras import RealDictCursor
            conn = self._pg_pool.getconn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                    SELECT chunk_index, cid, byte_size FROM model_chunks
                    WHERE model_id = %s AND layer_index = %s ORDER BY chunk_index ASC;
                    """, (model_id, layer_idx))
                    return [dict(r) for r in cur.fetchall()]
            finally:
                self._pg_pool.putconn(conn)
        else:
            with sqlite3.connect(self.local_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("""
                SELECT chunk_index, cid, byte_size FROM model_chunks
                WHERE model_id = ? AND layer_index = ? ORDER BY chunk_index ASC;
                """, (model_id, layer_idx))
                return [dict(r) for r in cur.fetchall()]

# Alias for backwards compatibility
TransactionPooler = StatePooler
