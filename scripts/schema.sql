-- Universal Schema for TorchJIT / UPAAS Swarm State
-- Compatible with PostgreSQL 13+, Supabase (Supavisor), Neon, SQLite
-- Notice: DB only keeps hashes (CIDs), metadata, and decryption keys.
-- All heavy tensor data lives strictly on the decentralized IPFS network!

CREATE TABLE IF NOT EXISTS models (
    model_id VARCHAR(128) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    total_layers INT NOT NULL,
    hidden_dim INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_chunks (
    id BIGSERIAL PRIMARY KEY,
    model_id VARCHAR(128) REFERENCES models(model_id) ON DELETE CASCADE,
    layer_index INT NOT NULL,
    chunk_index INT NOT NULL,
    cid VARCHAR(128) NOT NULL,             -- IPFS Content Identifier (Hash)
    decryption_key VARCHAR(255),           -- Key to decrypt chunk from IPFS
    byte_size INT NOT NULL,                -- Size of piece (1-3 MB)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_model_layer_chunk UNIQUE (model_id, layer_index, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_model_layer ON model_chunks(model_id, layer_index);
