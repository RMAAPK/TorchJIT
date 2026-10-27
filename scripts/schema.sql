-- Universal Schema for TorchJIT / UPAAS Swarm State
-- Compatible with PostgreSQL 13+, Supabase (Supavisor), Neon, AWS RDS, CockroachDB

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
    cid VARCHAR(128) NOT NULL,
    byte_size INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_model_layer_chunk UNIQUE (model_id, layer_index, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_model_layer ON model_chunks(model_id, layer_index);
