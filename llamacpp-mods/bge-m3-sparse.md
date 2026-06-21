# BGE-M3 Sparse Embedding Support for llama.cpp

## Overview

Adds support for BGE-M3's sparse lexical embedding head in llama.cpp.
After a single `llama_decode()`, the user can retrieve:
- Dense embeddings: `llama_get_embeddings_ith(ctx, i)` (existing)
- Sparse weights: `llama_get_embeddings_sparse_ith(ctx, i)` (NEW)

The sparse head computes `log(1 + ReLU(sparse_linear(hidden_state)))` per token,
yielding a scalar weight for each input token. The user pairs weights with
token IDs to form a sparse lexical vector for hybrid search.

## Architecture

```
llama_decode(batch)
    │
    ├─ Transformer (24 layers XLM-RoBERTa)
    │       │
    │       ▼ hidden_states [n_embd=1024, n_tokens]
    │       │
    │       ├─ [CLS token] ──► t_embd_pooled [1024, 1]  (dense, existing)
    │       │
    │       └─ sparse_linear ──► ReLU ──► t_embd_sparse [1, n_tokens]
    │                                        │
    │                          (CPU post: log1p)
    │                                        │
    │                          llama_get_embeddings_sparse_ith() ──► float
    │
    ▼
```

Key design decisions:
- `log1p` is done on CPU after GPU extraction to avoid adding a new GGML op
- Only ReLU + linear are in the compute graph (Vulkan-supported)
- Sparse output is per-token (scalar), matching BGE-M3's native output

## Files Modified

### Phase 1: Python Converter (GGUF creation)
| File | Change |
|---|---|
| `gguf-py/gguf/constants.py` | Added `CLS_SPARSE = auto()` enum + `"cls.sparse"` name + added to BERT arch |
| `gguf-py/gguf/tensor_mapping.py` | Added `"sparse_linear"` → `CLS_SPARSE` mapping |
| `conversion/bert.py` | Skip `colbert_linear` in filter_tensors; load `sparse_linear.pt` in generate_extra_tensors |

### Phase 2: C++ Inference
| File | Change |
|---|---|
| `src/llama-arch.h` | Added `LLM_TENSOR_CLS_SPARSE` to enum |
| `src/llama-arch.cpp` | Added tensor name `"cls.sparse"` and tensor info |
| `src/llama-model.h` | Added `cls_sparse`, `cls_sparse_b` tensor members |
| `src/models/bert.cpp` | Load `cls.sparse.*` tensors; wire sparse head in graph |
| `src/llama-graph.h` | Added `t_embd_sparse` to `llm_graph_result` |
| `src/llama-graph.cpp` | Wire sparse head: `transpose(cls_sparse) * hidden → ReLU` |
| `src/llama-context.h` | Added `embd_sparse` buffer_view |
| `src/llama-context.cpp` | Extract sparse output (GPU→CPU) + `log1p` post-processing |
| `include/llama.h` | Added `llama_get_embeddings_sparse_ith()` API |

## GGUF Tensor Layout

```
cls.sparse.weight  [1024, 1]  F16   # sparse_linear weight (transposed in graph)
cls.sparse.bias    [1]        F32   # sparse_linear bias
```

Note: GGUF stores weight as [1024, 1] (PyTorch's [1, 1024] transposed).
The graph transposes it back to [1, 1024] for ggml_mul_mat.

## API

```c
// Dense embedding (existing, unchanged)
// Returns float[n_embd_out] per token when pooling_type == NONE
LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);

// Dense pooled embedding (existing, unchanged)
// Returns float[n_embd_out] per sequence when pooling_type != NONE
LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

// Sparse lexical weight (NEW)
// Returns a scalar float = log(1 + ReLU(sparse_linear(hidden_state[i]))).
// Returns NaN for invalid indices or when sparse head is not available.
LLAMA_API float llama_get_embeddings_sparse_ith(struct llama_context * ctx, int32_t i);
```

## Usage: Hybrid Search (Dense + Sparse)

BGE-M3 produces two complementary representations from a single `llama_decode()`:

|  | Dense | Sparse |
|---|---|---|
| **Getter** | `llama_get_embeddings_seq(ctx, seq_id)` | `llama_get_embeddings_sparse_ith(ctx, i)` |
| **Shape** | `float[1024]` (1 vettore) | `float` scalare × n_token |
| **Semantica** | Significato complessivo del testo | Presenza/esattezza lessicale dei singoli termini |
| **Matching** | Similarità coseno | Dot product su token condivisi |

```c
// Dopo llama_decode() con pooling CLS:
// 1. Dense: un vettore per sequenza
float * dense = llama_get_embeddings_seq(ctx, seq_id);
// dense[0..1023] = embedding denso, usa similarità coseno

// 2. Sparse: uno scalare per ogni token della batch
struct { llama_token id; float weight; } sparse_vec[4096];
int sparse_len = 0;
for (int i = 0; i < n_tokens; i++) {
    if (batch.logits[i]) {
        float w = llama_get_embeddings_sparse_ith(ctx, i);
        if (w > 0.0f) {  // filtra pesi nulli
            sparse_vec[sparse_len++] = (struct){batch.token[i], w};
        }
    }
}
// sparse_vec = vettore sparso {(token_id, peso), ...}
// similarity_sparse = sum(min(w1, w2)) sui token condivisi (Jaccard pesato)
```

## Usage: Per-Token Embeddings (pooling NONE)

```c
// Con pooling_type == NONE, sia denso che sparse sono per-token:
for (int i = 0; i < batch.n_tokens; i++) {
    if (batch.logits[i]) {
        float  sparse_w = llama_get_embeddings_sparse_ith(ctx, i);  // scalare
        float *dense_v  = llama_get_embeddings_ith(ctx, i);          // float[1024]
        // batch.token[i] + sparse_w → entry sparse
        // dense_v → entry denso
    }
}
```

## Future: ColBERT Head

The colbert_linear head is skipped during conversion. To add it later:
1. Add `LLM_TENSOR_CLS_COLBERT` enum + tensor name + mapping
2. Load `colbert_linear.pt` in generate_extra_tensors
3. Wire colbert head: `linear → L2-norm per token`
4. Add `t_embd_colbert` + extraction + API

## Server Output Format

The `/v1/embeddings` endpoint returns both dense and sparse embeddings in one response:

```json
{
  "data": [{
    "embedding": [-0.049, -0.0007, ...],          // 1024 float, dense vector
    "sparse_embedding": [                          // lexical sparse vector
      [0,       0.1168],   // [token_id, weight]
      [121283,  0.1548],
      [2,       0.0957]
    ],
    "tokens_evaluated": 5,
    "index": 0,
    "object": "embedding"
  }],
  "model": "model-sparse.gguf",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

`sparse_embedding` is an array of `[token_id, weight]` pairs.
Only tokens with weight > 0 are included (ReLU filtering).
Token IDs are SentencePiece tokens from the BGE-M3 vocabulary.

### curl example

```
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Samarcanda"}'
```

## Build Status

Build succeeded on 2026-06-01 with MSVC 19.44.35226.0 / Windows x64.
CMake: `cmake -B build -G Ninja -DGGML_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release`
DLL exports verified: `llama_get_embeddings_sparse_ith` @ ord 45.
Server tested: returns dense + sparse_embedding in OAI-compat JSON.

### Generated GGUF
`D:\Models\BGE-M3\model-sparse.gguf` - 1.16 GB, MOSTLY_F16, 391 tensors.
`D:\Models\BGE-M3\model-sparse-q8.gguf` - 635 MB, Q8_0, 391 tensors.

## TODO / Future Work

- [x] Quantize sparse GGUF to Q8_0 (~600 MB)
- [x] Update server embedding endpoint to return sparse
- [ ] Vulkan backend: install Vulkan SDK and rebuild with `-DGGML_VULKAN=ON`
- [ ] Update `examples/embedding/embedding.cpp` to output sparse weights
- [ ] ColBERT head support (add CLS_COLBERT tensor + colbert_linear.pt loading)
- [ ] Verify sparse weights match HuggingFace reference implementation
- [ ] Merge/rebase onto future llama.cpp releases (weekly)

## Porting to a New llama.cpp Release

Follow this checklist, in order. Each step references the files above.

### Step 1: Python Converter (Phase 1)
1. In `gguf-py/gguf/constants.py`, add `CLS_SPARSE = auto()` after `CLS_NORM`
2. Add `MODEL_TENSOR.CLS_SPARSE: "cls.sparse"` in `TENSOR_NAMES` after `CLS_NORM`
3. Add `MODEL_TENSOR.CLS_SPARSE` to the `MODEL_ARCH.BERT` tensor list
4. In `gguf-py/gguf/tensor_mapping.py`, add `"sparse_linear"` → `CLS_SPARSE` mapping
5. In `conversion/bert.py`, in `XLMRobertaModel.filter_tensors`: skip `colbert_linear`
6. In `XLMRobertaModel.generate_extra_tensors`: load `sparse_linear.pt` from model dir
7. Run the converter: `python convert_hf_to_gguf.py <HF dir> --outtype f16 --outfile <out.gguf>`
8. Verify: `python -c "from gguf.gguf_reader import GGUFReader; ..."` → should show `cls.sparse.*`

### Step 2: C++ Inference (Phase 2)
1. `src/llama-arch.h`: add `LLM_TENSOR_CLS_SPARSE` after `LLM_TENSOR_CLS_NORM`
2. `src/llama-arch.cpp`: add name `"cls.sparse"` in `LLM_TENSOR_NAMES`
3. `src/llama-arch.cpp`: add `{LLM_TENSOR_CLS_SPARSE, {LLM_TENSOR_LAYER_OUTPUT, GGML_OP_MUL_MAT}}` in `LLM_TENSOR_INFOS`
4. `src/llama-model.h`: add `cls_sparse` and `cls_sparse_b` tensor members after `cls_norm`
5. `src/models/bert.cpp`: load tensors with `create_tensor(tn(LLM_TENSOR_CLS_SPARSE, ...), {n_embd, 1}, TENSOR_NOT_REQUIRED)`
6. `src/llama-graph.h`: add `t_embd_sparse` to `llm_graph_result` + getter
7. `src/llama-graph.h`: add `cls_sparse`/`cls_sparse_b` params to `build_pooling()` signature
8. `src/llama-graph.cpp`: in `build_pooling()`, write sparse head: `ggml_mul_mat(ctx0, cls_sparse, inp)` → `ggml_add(bias)` → `ggml_relu()` → `res->t_embd_sparse`
9. `src/llama-model.cpp`: pass `cls_sparse, cls_sparse_b` to `build_pooling()` call
10. `src/llama-context.h`: add `embd_sparse` buffer_view
11. `src/llama-context.cpp`: init `embd_sparse.size = has_embd ? (size_t)n_batch : 0`, include in `new_size` and buffer allocation
12. `src/llama-context.cpp`: extract `t_embd_sparse` → `embd_sparse.data` (both process_ubatch and streaming paths)
13. `src/llama-context.cpp`: add `get_embeddings_sparse_ith()` member → `output_resolve_row(i)` → `log1pf(embd_sparse.data[j])`
14. `src/llama-context.cpp`: add public `llama_get_embeddings_sparse_ith()` → `ctx->synchronize()` → `ctx->get_embeddings_sparse_ith(i)`
15. `include/llama.h`: declare `LLAMA_API float llama_get_embeddings_sparse_ith(ctx, i)`

### Step 3: Server Output (Phase 3)
1. `tools/server/server-task.h`: add `std::vector<std::pair<llama_token, float>> sparse_embedding` to `server_task_result_embd`
2. `tools/server/server-context.cpp`: in `send_embedding()`, iterate batch and push `llama_get_embeddings_sparse_ith()` into `res->sparse_embedding`
3. `tools/server/server-task.cpp`: include `sparse_embedding` in both `to_json_oaicompat()` and `to_json_non_oaicompat()`
4. `tools/server/server-common.cpp`: in `format_embeddings_response_oaicompat()`, forward `sparse_embedding` field if present

### Gotchas Discovered

**A. DO NOT transpose cls_sparse in the graph.** The GGUF stores weight as `[n_embd, 1]` (PyTorch's `[1, n_embd]` transposed). `ggml_mul_mat` expects this convention and computes `weight^T × input` internally. Transposing it again breaks `ggml_can_mul_mat`.

**B. GGUF tensor shape must match exactly.** `create_tensor(tn(CLS_SPARSE, "weight"), {n_embd, 1}, ...)` — the `{n_embd, 1}` shape must match what the converter stored. `check_tensor_dims` does exact comparison. If the GGUF has `{1024, 1}`, specify `{n_embd, 1}`.

**C. BGE-M3 stores output heads in SEPARATE files.** `dense`, `sparse_linear`, `colbert_linear` are NOT in `pytorch_model.bin`. They're in `.pt` files or module subdirectories. Use `generate_extra_tensors()` to load them.

**D. BGE-M3 has NO trainable dense layer.** The dense embedding is just CLS token + L2 normalization. No `dense.weight` tensor exists in the model. The `llama-embedding` and `llama-server` already handle this via CLS pooling.

**E. log1p on CPU.** We apply `log1pf()` after GPU→CPU transfer, not in the graph. This avoids adding a new GGML op. The graph only does `ReLU(sparse_linear(hidden))`.

**F. Vulkan `mul_mat` crash on single-column weights (UPSTREAM BUG).**  
The weight `cls_sparse` has shape `[1024, 1]` — a single output column. The Vulkan `ggml_mul_mat` kernel assumes multi-column output and performs out-of-bounds memory access on single-column shapes. This crashes the process silently (no assertion, no stack trace — the GPU driver kills it).
- **Symptom**: process exits without warning on certain token counts after a successful warmup
- **Fix**: replaced `ggml_mul_mat(ctx0, cls_sparse, inp)` with `ggml_mul(ctx0, inp, cls_sparse)` + `ggml_sum_rows()` — standard element-wise ops with no shape restrictions
- **CPU is NOT affected** — the CPU backend handles `[N, 1]` matmul correctly
- If you encounter similar crashes with Vulkan on other models using single-column projection tensors, apply the same fix pattern

**G. The test program crash.** The test program (`test-sparse.cpp`) crashed due to outdated `common_init` API. The library works (verified via `llama-embedding.exe` and `llama-server.exe`).

**H. Upstream ggml-cpu regression (b9637).** `ggml_mul_mat` with mixed F16/F32 types crashes CPU backend. Vulkan (`-ngl 99`) is unaffected. Workaround: always use Vulkan for inference.

**I. Porting buffer_view bug.** When reallocating the output buffer, the existing code clears `.data = nullptr` but the porting agent wrote `embd_sparse = {nullptr, 0}` which also zeroes `.size`. Fix: use `.data = nullptr` to preserve the size. Affected both `embd_sparse` and `embd_colbert`.

**J. CMake source directory.** The build must use `-S <tmp_path>` explicitly when building from a different directory than the source. Otherwise CMake caches the wrong source path and compiles stale files. Symptom: tensors exist in GGUF but `create_tensor` returns nullptr silently.

**K. `n_tokens` renamed to `ubatch.n_tokens`** in upstream b9637 `process_ubatch`. The old variable `n_tokens` in the extraction code refers to the encoder function's parameter, not the ubatch token count. Use `ubatch.n_tokens` explicitly in sparse/colbert extraction.

### Upstream Tracker

Routine: ogni domenica, fetch upstream e verificare diff sui nostri file.

| Date | Base commit | Checked up to | Files touched upstr. | Outcome |
|---|---|---|---|---|
| 2026-06-07 | 399739d5c | b9549 (04eb4c446) | 13 files, 445+/172- | Portato, bit-identico |
| 2026-06-15 | b9549 | b9637 | 10 files, 427+/126- | Portato, bit-identico |
| 2026-06-16 | b9637 | b9637 | — | ColBERT head added |
| 2026-06-22 | b9637 | b9743 | 13 files, 222+/861- | Portato, bit-identico |
| Prossima | b9743 | — | — | Domenica 29 giugno |

### Generated GGUF Files

| File | Tensori | Dimensione | Contenuto |
|---|---|---|---|
| `model-sparse.gguf` | 391 | 1.16 GB | dense + sparse (F16) |
| `model-sparse-q8.gguf` | 391 | 635 MB | dense + sparse (Q8_0) |
| `model-colbert.gguf` | 393 | 1.10 GB | dense + sparse + colbert (F16) |
| `model-colbert-q8.gguf` | 393 | 600 MB | dense + sparse + colbert (Q8_0) |

**ColBERT tensors:** `cls.colbert.weight [1024, 1024]` F16, `cls.colbert.bias [1024]` F32.
Graph: `ggml_mul_mat` + `ggml_norm` (L2 per-token normalization).
API: `llama_get_embeddings_colbert_ith(ctx, i) → float[1024]`.
Server: `?colbert=true` (query param or JSON body field) enables colbert output.

### Verification After Porting
```bash
# Build
cmake -B build -G Ninja -DGGML_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8

# Test dense (must work)
.\build\bin\llama-embedding.exe --model model.gguf --pooling cls --prompt "test"

# Test server (must return sparse_embedding field)
.\build\bin\llama-server.exe --model model.gguf --embedding --pooling cls --port 8080 --fit off
curl -X POST http://127.0.0.1:8080/v1/embeddings -H "Content-Type: application/json" -d '{"input":"test"}'
# Should contain: "sparse_embedding": [[<id>, <weight>], ...]
```

