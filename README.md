# TorchJIT (TJIT): Content-Addressed, Zero-Copy Tensor Runtime

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE.md)
[![Organization: RMAA Holdings](https://img.shields.io/badge/Organization-RMAA%20Holdings-purple.svg)](https://github.com/RMAAPK)
[![Zero-Copy: POSIX mmap](https://img.shields.io/badge/Zero--Copy-mmap-brightgreen.svg)]()
[![IPFS: Content-Addressed CIDs](https://img.shields.io/badge/IPFS-CIDv1%20Tensors-teal.svg)]()

**TorchJIT (TJIT)** is a bare-metal, high-performance C++20 and Python tensor runtime engineered by **RMAA Holdings**. It decouples neural network weights, execution graphs, and transformer layers into decentralized, content-addressed cryptographic blocks (IPFS CIDs), backed by zero-copy OS memory mapping (`mmap`) and state tracking via Supabase.

---

## ⚡ Why TorchJIT Over Standard PyTorch?

| Feature | Standard PyTorch (`torch`) | **TorchJIT (`tjit`)** |
| :--- | :--- | :--- |
| **Storage Architecture** | Monolithic file (`.pt`, `.safetensors`) | **Content-Addressed Sharded CIDs** |
| **Memory Footprint** | Must load entire 250 GB model into RAM | **Streams/Pivots individual layer CIDs on demand** |
| **Startup / Paging Latency** | Minutes to deserialize weights into VRAM | **Zero-Copy POSIX `mmap` instant paging** |
| **State Registry** | None (Local files only) | **Decentralized Supabase PostgreSQL Ledger** |
| **Ecosystem Integration** | Centralized GPU servers | **Native integration with Local-JIT (LJIT), Neural-JIT (NJIT), & UPAAS** |

---

## 🚀 Quickstart

### Python API
```python
import torchjit as torch

# 1. Create tensors with deterministic IPFS CIDs
a = torch.ones([2, 4])
b = torch.ones([2, 4])
print(a.cid)  # 'bafkreieb87129436c8da65eb87cc398827109b'

# 2. Native Arithmetic & Transformer Kernels
c = a + b
A = torch.ones([2, 3])
B = torch.ones([3, 2])
C = A @ B      # Matmul
normed = C.rms_norm() # RMSNorm attention pre-norm

# 3. Content-Addressed Zero-Copy Persistence
c.save("/data/vault/layer_0_weights.tjit")
m = torch.load_mmap("/data/vault/layer_0_weights.tjit", [2, 4])
assert m.cid == c.cid  # 100% Bit-exact cryptographic identity
```

---

## 🔬 Architecture: The Cryptographic Storage Allocator

Standard PyTorch binds tensors to raw heap allocations via `malloc` or `cudaMalloc`. TorchJIT replaces this with an aligned 64-byte virtual allocator that binds memory directly to Content Identifiers:

```
[PyTorch Model Checkpoint]
            │
            ▼  (Layer Slicing & Chunking)
[Layer 0 CID: bafkrei...] [Layer 1 CID: bafkrei...] [Layer 2 CID: bafkrei...]
            │                         │                         │
            ▼                         ▼                         ▼
   [IPFS / Swarm Cache]     [Local-JIT P2P Vault]     [Supabase State Ledger]
            │
            ▼
   [Zero-Copy POSIX mmap] ──▶ Raw Pointer ──▶ Instant Tensor Execution
```

---

## 📜 License & Intellectual Property
TorchJIT is licensed under the **Business Source License 1.1 (BSL 1.1)** by **RMAA Holdings**.
See [LICENSE.md](LICENSE.md) for full terms and third-party attribution notices (PyTorch BSD-3-Clause).
