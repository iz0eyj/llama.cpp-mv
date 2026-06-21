---
language:
- multilingual
tags:
- embedding
- dense
- sparse
- colbert
- bge-m3
- sentence-transformers
- hybrid-search
- late-interaction
- rag
license: mit
datasets:
- BAAI/bge-m3
model-index:
- name: bge-m3-gguf-multivector
  results: []
---

# BGE-M3 GGUF — Dense + Sparse + ColBERT

This repository provides pre-converted GGUF files for the **BAAI BGE-M3** embedding model,
enhanced to output **all three** embedding types from a single inference call:

| Head | Output | Shape | Use Case |
|------|--------|-------|----------|
| **Dense** | CLS pooling → L2 normalization | `float[1024]` | Semantic similarity (cosine) |
| **Sparse** | `linear → ReLU → log(1+x)` per token | `[token_id, weight]` pairs | Lexical / hybrid search |
| **ColBERT** | `linear → L2 norm` per token | `n_tokens × float[1024]` | Token-level late interaction |

**A single `llama_decode()` computes all three.** No separate models, no external BM25.

Unlike the standard BGE-M3 GGUF (which only preserves the dense head), these files
contain the full `cls.sparse` and `cls.colbert` weights extracted from the original
sentence-transformers checkpoint.

---

## Available Files

| File | Quantization | Size | Heads |
|------|-------------|------|-------|
| `bge-m3-multivector-q8.gguf` | Q8_0 | ~600 MB | Dense + Sparse + ColBERT |
| `bge-m3-multivector-f16.gguf` | F16 | ~1.1 GB | Dense + Sparse + ColBERT |

Q8_0 is recommended for production — identical quality, half the size.

---

## Usage

### With llama.cpp-mv (recommended)

Use the companion [llama.cpp-mv](https://github.com/iz0eyj/llama.cpp-mv) fork, which supports
all three heads natively. Start the server:

```bash
llama-server --model bge-m3-multivector-q8.gguf \
    --pooling cls --no-mmap --embedding \
    --host 0.0.0.0 --port 8080 \
    -c 8192 -b 8192 -ub 8192 -np 1 --fit off \
    -ngl 99
```

Default call returns **dense + sparse**:

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Samarcanda è una città meravigliosa"}'
```

Add `?colbert=true` (or `"colbert": true` in the JSON body) for all three:

```bash
curl -X POST "http://127.0.0.1:8080/v1/embeddings?colbert=true" \
  -H "Content-Type: application/json" \
  -d '{"input": "Samarcanda è una città meravigliosa"}'
```

### JSON Response Format

```json
{
  "data": [{
    "embedding": [0.001, -0.023, ...],           // dense: 1024 floats
    "sparse_embedding": [                          // sparse: [token_id, weight] pairs
      [0,       0.116],
      [121283,  0.154],
      [2,       0.095]
    ],
    "colbert_embedding": [                         // colbert: n_tokens × 1024
      [-0.768, -0.048, ...],
      [ 0.312,  1.205, ...],
      [-0.015, -0.823, ...]
    ],
    "index": 0,
    "object": "embedding"
  }],
  "model": "bge-m3-multivector-q8.gguf",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

- `colbert_embedding` is only present when `?colbert=true` is set
- Sparse weights > 0 indicate lexically active tokens
- ColBERT vectors are L2-normalized per token (norm ≈ 32, use dot product for similarity)

### Per-Token ColBERT Similarity (Late Interaction)

```python
import numpy as np

def colbert_score(vecs_a, vecs_b):
    """MaxSim per token, then average."""
    sim = np.dot(vecs_a, vecs_b.T)  # [n_a, n_b]
    return np.mean(np.max(sim, axis=1))

# Example: "Samarcanda è bella" vs "Samarcanda è meravigliosa"
# score ≈ 1010 (similar) vs ≈ 330 (different topic)
```

---

## How These GGUF Files Were Created

The standard llama.cpp converter strips tensors it doesn't recognize. BGE-M3's
sparse and colbert heads (`sparse_linear.pt`, `colbert_linear.pt`) are stored
separately from the main model checkpoint and were silently skipped.

Our converter patches (in [llama.cpp-mv](https://github.com/iz0eyj/llama.cpp-mv)):

1. **`gguf-py/gguf/constants.py`** — Added `CLS_SPARSE` and `CLS_COLBERT` tensor types
2. **`gguf-py/gguf/tensor_mapping.py`** — Mapped `sparse_linear` → `cls.sparse`, `colbert_linear` → `cls.colbert`
3. **`conversion/bert.py`** — Load `.pt` files from model directory via `generate_extra_tensors()`

Command:

```bash
python convert_hf_to_gguf.py BAAI/bge-m3 --outtype q8_0 --outfile bge-m3-multivector-q8.gguf
```

---

## Inference Architecture (llama.cpp-mv)

The companion C++ fork adds three separate output paths from the same hidden states:

```
Transformer (24 layers XLM-RoBERTa)
    │
    ▼ hidden_states [1024, n_tokens]
    │
    ├── [CLS] ──► L2 normalize ──► dense [1024]
    ├── sparse_linear → ReLU → log1p ──► sparse [n_tokens] scalars
    └── colbert_linear → L2 norm ──► colbert [n_tokens × 1024]
```

Key design decisions:
- `log1p` for sparse is applied on CPU after GPU transfer (avoids new GGML op)
- ColBERT uses `ggml_norm` for per-token L2 normalization
- Output buffers are sized for max batch size, zero overhead for small requests

---

## Limitations

- **ColBERT scales with token count** — 1000 tokens = 4 MB per document. Best used on
  short text (search queries, titles, paragraphs). For long documents, pre-filter
  with dense search, then apply ColBERT to top-K candidates.
- **CPU mode is unstable** on some upstream llama.cpp versions due to F16/F32 mixed-type
  operations. Use Vulkan (`-ngl 99`) or CUDA.
- The ColBERT head is 1024×1024 (4 MB) — negligible compared to the 600 MB model body.

---

## License

MIT — matches both BAAI/bge-m3 and llama.cpp.

## Credits

- **BAAI** for [BGE-M3](https://huggingface.co/BAAI/bge-m3) model
- **llama.cpp** by [ggerganov](https://github.com/ggml-org/llama.cpp)
- GGUF conversion and inference patches by [iz0eyj](https://github.com/iz0eyj)
- Companion fork: [llama.cpp-mv](https://github.com/iz0eyj/llama.cpp-mv)
