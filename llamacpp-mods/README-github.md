# llama.cpp-mv — Multi-Vector Embedding Fork

Fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds **sparse** and **ColBERT** embedding heads
to BERT-based encoder models, starting with [BGE-M3](https://huggingface.co/BAAI/bge-m3).

A single `llama_decode()` call now produces **three complementary vector types**:

| Head | Type | Example | Use Case |
|------|------|---------|----------|
| **Dense** | `float[1024]` | Semantic meaning of text | Cosine similarity search |
| **Sparse** | `[token_id, weight]` pairs | Lexical presence of terms | Hybrid keyword + semantic search |
| **ColBERT** | `n_tokens × float[1024]` | Token-level multi-vectors | Late interaction fine-grained matching |

## Why Sparse?

Standard dense embeddings lose rare terms. If you search "Samarcanda", a dense vector may miss it
because the concept is diluted in the semantic soup. Sparse gives each token a **lexical weight** —
exact term matching inside your vector search, no external BM25 needed.

## Why ColBERT?

Dense gives you one vector per document. ColBERT gives you **one vector per token**.
Instead of comparing document-level summaries, you compare token-level representations
(late interaction). This excels at question answering, exact phrase matching, and
disambiguation between closely related concepts.

## Quick Start

```bash
# Build (Vulkan)
cmake -B build -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8

# Start server
./build/bin/llama-server \
  --model bge-m3-multivector-q8.gguf \
  --pooling cls --embedding --host 0.0.0.0 --port 8080 \
  -c 8192 -ngl 99

# Dense + Sparse (default)
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -d '{"input": "Hello world"}'

# Dense + Sparse + ColBERT
curl -X POST "http://127.0.0.1:8080/v1/embeddings?colbert=true" \
  -d '{"input": "Hello world"}'
```

## Pre-Converted GGUF Models

Download from [Hugging Face](https://huggingface.co/Uncino/bge-m3-multivector-gguf):

| File | Size | Quantization |
|------|------|-------------|
| `bge-m3-multivector-q8.gguf` | ~600 MB | Q8_0 (recommended) |
| `bge-m3-multivector-f16.gguf` | ~1.1 GB | F16 |

## Documentation

Full technical details, porting guide, and gotchas in [`llamacpp-mods/bge-m3-sparse.md`](llamacpp-mods/bge-m3-sparse.md).

## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggerganov & contributors
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) by BAAI
- GGUF conversion & inference patches by [iz0eyj](https://github.com/iz0eyj)

## License

MIT — same as llama.cpp and BGE-M3.
