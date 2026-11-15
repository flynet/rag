# RAG Gateway — OpenAI-compatible API with retrieval

A local RAG stack for documents: automatic indexing, semantic search over Qdrant, and an OpenAI-compatible gateway for agents.

The generation model is **not** part of this stack. The gateway forwards chat requests to an external OpenAI-compatible server (vLLM, LM Studio, llama.cpp, a cloud endpoint); agents only talk to the gateway, and the upstream key never leaves it.

```text
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_API_KEY=<key from API_KEYS>
```

There is no chat UI — the service exposes an API only. Any OpenAI-compatible client (OpenAI SDK, LangChain, Continue, Cline, etc.) can use it.

## Current state

This is the state of the code in this repository, not a target design.

| Area | Current implementation |
|---|---|
| Embedding model | `Qwen/Qwen3-Embedding-0.6B`, 1024 dimensions, fp16 on GPU |
| Query encoding | Retrieval instruction from the model's `query` prompt (`EMBEDDING_QUERY_PROMPT=query`) |
| Document encoding | Plain text, no instruction |
| Retrieval | Dense cosine search in Qdrant; `RAG_TOP_K=3`, `RAG_MIN_SCORE=0.5` |
| Retrieval evaluation | `scripts/eval_retrieval.py` (recall@k / MRR@10 / nDCG@10); labelled dataset and baseline pending |
| Reranking | None |
| Hybrid / sparse search | None |
| Chunking | Fixed 1024 characters with 128-character overlap |
| Indexing | Startup reconciliation + scan, `watchdog`, signature-based incremental updates |
| Deletion handling | Live watcher events + startup reconciliation of files deleted or renamed while the stack was stopped |
| Qdrant | `qdrant/qdrant:latest`, no snapshot schedule |
| Healthchecks | Gateway only; `api` and Qdrant have none |
| Embedding deps | `sentence-transformers>=5.0,<6`, `transformers>=4.51,<5` |
| Target hardware | One GPU, tested on RTX 4060 Laptop with 8 GB VRAM |

The gaps above and the improvement plan are tracked in **[TODO.md](TODO.md)**.

## Features

- OpenAI-compatible `/v1/chat/completions` (streaming and non-streaming), `/v1/models`, `/v1/embeddings`
- Automatic document indexing with file-change monitoring
- RAG enabled by default: retrieved chunks are injected into the current user turn
- Transparent passthrough of `tools`, `tool_calls`, `response_format`, `temperature`, and other OpenAI fields
- Bearer authentication with static keys
- The upstream key stays in the gateway; the agent's `Authorization` header is never forwarded
- Per-request RAG control via `X-RAG-*` headers
- A single embedding-model instance shared by search and indexing; the indexer has no torch/CUDA
- Incremental indexing: unchanged files are skipped based on a stored signature
- Retrieval evaluation harness (`recall@k`, `MRR`, `nDCG`) with a JSONL question set
- Qdrant vector database

## Architecture

```text
┌──────────────┐
│ Agent / LLM  │  OpenAI SDK, LangChain, Continue, Cline, ...
│   client     │
└──────┬───────┘
       │  POST /v1/chat/completions
       v
┌─────────────────────────┐
│ Gateway      :8001      │  single public entry point
│ • agent bearer auth     │
│ • retrieval via api     │
│ • context injection     │
│ • upstream auth swap    │
└────┬───────────────┬────┘
     │               │
     v               v
┌──────────────────┐   ┌───────────────────┐
│ LLM server (LAN) │   │ API       :8000   │
│ vLLM / LM Studio │   │ 1× embedding      │
│ llama.cpp / cloud│   │ model (fp16)      │
└──────────────────┘   └────┬─────────┬────┘
                            │         ▲
                            v         │ POST /embed
                      ┌──────────┐    │
                      │  Qdrant  │    │
                      │  :6333   │    │
                      └────▲─────┘    │
                           │          │
                      ┌────┴──────────┴┐
                      │  Embedder      │
                      │  (watchdog)    │
                      └────────┬───────┘
                               v
                       data/documents/
```

## Project structure

```text
.
├── docker-compose.yaml          # service orchestration
├── .env.example                 # environment template
├── data/
│   └── documents/               # put documents here for indexing
├── models/
│   └── hf-cache/                # Hugging Face cache (created automatically)
├── embedder/                    # indexer: watchdog + HTTP client (no torch)
│   ├── Dockerfile
│   ├── main.py                  # watcher, worker, incremental indexing
│   ├── embedder.py              # HTTP client for api
│   ├── vector_db.py             # Qdrant client wrapper
│   └── requirements.txt
├── api/                         # embeddings + search (owns the model)
│   ├── Dockerfile
│   ├── main.py                  # /search, /embed, /v1/embeddings
│   ├── embedder.py              # SentenceTransformer wrapper
│   └── requirements.txt
├── gateway/                     # public OpenAI-compatible gateway
│   ├── Dockerfile
│   ├── main.py                  # /v1/chat/completions + RAG injection
│   └── requirements.txt
├── scripts/
│   ├── smoke_test.py            # end-to-end API check
│   └── eval_retrieval.py        # retrieval quality evaluation
├── eval/
│   ├── README.md                # dataset format and metrics
│   └── questions.example.jsonl  # schema examples
├── qdrant_data/                 # vector DB data (created automatically)
├── README.md
└── TODO.md
```

## Services

### Gateway — public OpenAI-compatible API

- **Port:** `8001` → `http://localhost:8001/v1`
- **Purpose:** the only entry point for agents
- **Behavior:**
  - validates `Authorization: Bearer <key>` against `API_KEYS`
  - retrieves context for the last user message
  - injects retrieved documents into the text of that user message
  - forwards the request to the upstream OpenAI-compatible server with full body passthrough
  - authenticates to the upstream with `UPSTREAM_API_KEY`; the agent's key is never sent upstream
  - maps upstream `401/403` to `502 upstream_auth_failed`
  - streams SSE without buffering
- **Swagger:** http://localhost:8001/docs

### API — embeddings and search

- **Port:** `8000` (debug only; agents use the gateway)
- **Endpoints:**
  - `POST /search` — semantic search (used by the gateway)
  - `POST /embed` — internal batch endpoint for the indexer
  - `POST /v1/embeddings` — OpenAI-compatible embeddings
  - `GET /collection/info` — collection statistics
  - `GET /collection/sample` — sample points
  - `GET /health` — model, device, dimension, and collection status
- **Model:** loaded here and only here, fp16 on GPU
- **Swagger:** http://localhost:8000/docs

### Embedder — indexing

- Watches `./data/documents` (polling every second)
- Splits text into chunks
- Requests vectors from `api` (`POST /embed`) and writes them to Qdrant
- Re-indexes only files whose signature changed (`mtime`, size, chunking config, model)
- **GPU:** not required; the image is based on `python:3.10-slim` and has no torch/CUDA

### Qdrant — vector database

- **Port:** `6333`
- **Dashboard:** http://localhost:6333/dashboard

### LLM server — external upstream

- Not part of this stack; usually another machine or a cloud endpoint
- Any OpenAI-compatible server: vLLM, LM Studio, llama.cpp server, TGI, OpenAI
- Configure with `UPSTREAM_BASE_URL` (with or without the trailing `/v1`)
- Put its key in `UPSTREAM_API_KEY`; leave empty if the server has no auth

## Quick start

### Prerequisites

1. Docker with the Compose plugin
2. NVIDIA GPU with CUDA 12.4+ and the NVIDIA Container Toolkit (for embeddings), or CPU only
3. An OpenAI-compatible LLM server reachable over the network

The embedding model (~1.2 GB) is downloaded from Hugging Face into `./models/hf-cache` on first start.

### Setup

**1. Create the environment file:**

```bash
cp .env.example .env
```

On PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least:

```env
API_KEYS=sk-local-change-me
UPSTREAM_BASE_URL=http://192.168.1.50:8000/v1
```

**2. Build and start:**

```bash
docker compose up -d --build
```

The API service must download and load Qwen3-Embedding-0.6B before it can serve. The indexer waits for it for up to 10 minutes; watch the progress with:

```bash
docker compose logs -f api embedder
```

**3. Check the gateway:**

```bash
curl http://localhost:8001/v1/models -H "Authorization: Bearer sk-local-change-me"
```

An empty list or `502` means `UPSTREAM_BASE_URL` is wrong — it must point to your LLM server, not to the gateway.

**4. Run the smoke test:**

```bash
python scripts/smoke_test.py --api-key sk-local-change-me
```

**5. Use it from an agent:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="sk-local-change-me")

response = client.chat.completions.create(
    model="qwen2.5-7b-instruct",   # an id returned by GET /v1/models
    messages=[{"role": "user", "content": "Who is Pupkin?"}],
)
print(response.choices[0].message.content)
```

## Authentication

Keys are configured as a comma-separated list:

```env
API_KEYS=sk-local-change-me,sk-agent-2
```

- The gateway checks `Authorization: Bearer <key>`.
- Comparison is constant-time (`secrets.compare_digest`).
- An empty `API_KEYS` disables authentication (local development only).
- Errors use the OpenAI error envelope:

```json
{"error": {"message": "Invalid or missing API key...", "type": "invalid_request_error", "param": null, "code": null}}
```

### Upstream key

`UPSTREAM_API_KEY` is the **model server's** key, not the agent's. It lives only in the gateway container:

- the agent's incoming `Authorization` header is never forwarded;
- if the upstream key is empty, no `Authorization` header is sent at all;
- if the upstream still returns `401/403`, the agent gets `502 upstream_auth_failed`, not `401` — otherwise it would assume its own key is wrong.

## API

### `GET /v1/models`

Transparent proxy of the upstream model list.

```bash
curl http://localhost:8001/v1/models -H "Authorization: Bearer sk-local-change-me"
```

```json
{
  "object": "list",
  "data": [
    {"id": "qwen2.5-7b-instruct", "object": "model", "created": 1730000000, "owned_by": "vllm"}
  ]
}
```

### `POST /v1/chat/completions`

The main endpoint. The request body is standard OpenAI Chat Completions, including `tools`, `tool_choice`, `response_format`, `temperature`, `stream`, and `stream_options`.

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer sk-local-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "Who is Pupkin?"}],
    "stream": false
  }'
```

**Streaming** uses the standard OpenAI SSE format (`data: {...}` … `data: [DONE]`):

```python
stream = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "Who is Pupkin?"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

#### Per-request RAG control

| Header | Value | Effect |
|---|---|---|
| `X-RAG` | `on` / `off` | enable or disable retrieval for the request |
| `X-RAG-TOP-K` | `1..20` | number of chunks to inject |
| `X-RAG-MIN-SCORE` | `-1.0..1.0` | minimum cosine score |

```bash
# answer without the knowledge base
curl ... -H "X-RAG: off" ...
```

#### Sources in the response

The response body stays strictly OpenAI-compatible, so retrieved sources are returned in headers:

| Header | Value |
|---|---|
| `X-RAG-Hits` | number of injected chunks |
| `X-RAG-Sources` | unpadded base64url-encoded JSON: `[{"file","chunk","score"}]` |

Decoding:

```python
import base64, json
raw = response.headers["X-RAG-Sources"]
sources = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
```

With the OpenAI SDK, headers are available via `with_raw_response`:

```python
raw = client.chat.completions.with_raw_response.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "Who is Pupkin?"}],
)
print(raw.headers.get("X-RAG-Hits"))
print(raw.parse())
```

#### When retrieval is skipped

Context is injected only for a "fresh" user turn — when the last message in `messages` has `role: "user"`. Inside an agent tool loop (the last message is `tool`, or an `assistant` message with `tool_calls`), retrieval is skipped to avoid bloating every step with the same documents.

If the `api` service is unavailable, the gateway does **not** break chat: the request goes to the LLM without context and a `WARNING` is logged.

**Where the context goes.** The `<knowledge_base>` block is prepended to the `content` of the last user message; the actual question stays at the end. No separate `system` message is added, so the `system + history` prefix stays byte-identical and upstream prompt caching keeps working. Chat templates that require `system` to be the first message (ChatML, Llama 3) are not broken.

### `POST /v1/embeddings`

OpenAI-compatible embeddings from the same local model.

```bash
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Authorization: Bearer sk-local-change-me" \
  -H "Content-Type: application/json" \
  -d '{"input": ["first text", "second text"]}'
```

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.01, -0.02, "..."]}],
  "model": "rag-embeddings",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

- `input`: a string or an array of strings (token arrays are not supported)
- `encoding_format`: `float` (default) or `base64`
- `usage` is an estimate (~4 characters per token)
- texts are encoded **as documents** (no retrieval instruction); use `/search` for query-style retrieval

### `GET /health`

Public, no authentication. Checks the model server and the `api` service.

```json
{
  "status": "ok",
  "rag_enabled": true,
  "auth_required": true,
  "defaults": {"top_k": 3, "min_score": 0.5},
  "dependencies": {
    "upstream": {"url": "http://192.168.1.50:8000/v1", "ok": true, "status": 200},
    "api": {"url": "http://api:8000", "ok": true, "status": 200}
  }
}
```

`status` is `degraded` when at least one dependency is unavailable. If `UPSTREAM_BASE_URL` is empty, the gateway logs `CRITICAL`, reports `upstream.ok: false`, and returns `503` for chat requests instead of crashing.

## Configuration

Variables are read from `.env` (see [`.env.example`](.env.example)).

### User-facing variables

| Variable | Default | Description |
|---|---|---|
| `API_KEYS` | — | comma-separated bearer keys; empty disables auth |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | embedding model (one per deployment) |
| `EMBEDDING_DEVICE` | `cuda` | `cuda` or `cpu` |
| `EMBEDDING_DTYPE` | `float16` | `float16`, `bfloat16`, `float32`, `auto`; CPU always uses fp32 |
| `EMBEDDING_QUERY_PROMPT` | `query` | model prompt used for queries; empty = encode queries like documents |
| `EMBEDDING_MAX_SEQ_LENGTH` | `8192` | maximum tokens per text (the model supports 32k) |
| `EMBED_BATCH_SIZE` | `32` | texts per embedding request (both indexing and API batching) |
| `ENABLE_RAG` | `true` | global RAG switch |
| `RAG_TOP_K` | `3` | default number of chunks in the context |
| `RAG_MIN_SCORE` | `0.5` | default cosine-score threshold |
| `UPSTREAM_BASE_URL` | — | OpenAI-compatible server URL (`.../v1` or without) |
| `UPSTREAM_API_KEY` | — | model-server key; empty = no header |
| `UPSTREAM_HEADERS` | — | extra upstream headers as a JSON object |
| `UPSTREAM_TIMEOUT` | `600` | generation timeout in seconds, including streaming |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` |

### Gateway variables

These are set in `docker-compose.yaml` and normally do not need to be changed:

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://api:8000` | internal embedding/search service URL |
| `RAG_MAX_TOP_K` | `20` | hard cap for `top_k` and `X-RAG-TOP-K` |
| `RAG_TIMEOUT` | `30` | timeout for the retrieval call |
| `CONNECT_TIMEOUT` | `10` | timeout for upstream connection probes |
| `MAX_REQUEST_BYTES` | `8000000` | maximum accepted request-body size |

### API variables

Set in `docker-compose.yaml`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `${EMBEDDING_MODEL}` | model id or local path |
| `HF_HOME` | `/models/hf-cache` | Hugging Face cache inside the container |
| `EMBEDDING_BATCH_SIZE` | `${EMBED_BATCH_SIZE}` | texts per forward pass |
| `EMBEDDING_NORMALIZE` | `true` | L2-normalize output vectors |
| `EMBEDDING_MODEL_NAME` | `rag-embeddings` | `model` field returned by `/v1/embeddings` |
| `MAX_BATCH_SIZE` | `256` | maximum texts per API request |
| `QDRANT_HOST` / `QDRANT_PORT` | `qdrant` / `6333` | Qdrant location |
| `COLLECTION_NAME` | `documents` | Qdrant collection |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | reduces VRAM fragmentation |

### Embedder variables

Set in `docker-compose.yaml`:

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://api:8000` | embedding service URL |
| `WATCH_PATH` | `/data/documents` | directory to watch |
| `DEBOUNCE_SEC` | `1.0` | file-event debounce |
| `CHUNK_SIZE` | `1024` | chunk size in characters |
| `CHUNK_OVERLAP` | `128` | overlap in characters |
| `EMBED_BATCH_SIZE` | `32` | texts per `/embed` request |
| `EMBEDDING_MODEL` | `unknown` (set by Compose) | model label stored in each chunk's signature |

## Embedding model

The stack currently uses **`Qwen/Qwen3-Embedding-0.6B`**:

- 0.6B parameters, 1024-dimensional embeddings (MRL: 32–1024), 32k context
- multilingual (100+ languages)
- instruction-aware: queries can be prefixed with a retrieval instruction

Qwen3-Embedding expects the instruction on the **query** side only:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: <user query>
```

The instruction comes from the model's own `query` prompt and is applied by `api/embedder.py` through `prompt_name="query"`. Set `EMBEDDING_QUERY_PROMPT` to change the prompt name, or empty to disable it. According to the model card, the query-side instruction improves retrieval by roughly 1–5%.

Documents are encoded without an instruction. `/embed` and `/v1/embeddings` encode text as documents; `/search` encodes the query with the instruction.

### Changing the model

The model is configured by one variable, read by both services (`api` loads the weights; `embedder` uses it as an index-invalidation label):

```env
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
```

| Model | Dim | fp16 weights | Notes |
|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | ≈ 1.2 GB | default: multilingual, instruction-aware |
| `google/embeddinggemma-300m` | 768 | ≈ 0.6 GB | lighter and faster |
| `intfloat/multilingual-e5-small` | 384 | ≈ 0.24 GB | fastest option |
| `BAAI/bge-m3` | 1024 | ≈ 1.1 GB | strong multilingual model, no instructions |

Changing the model invalidates the index because each chunk stores the model name in its payload. If the dimension changes (768 → 1024 when switching from embeddinggemma), the Qdrant collection is recreated automatically (`recreating the collection` in the log); if the dimension is the same, the collection is kept but all documents are still re-indexed. The full pass starts when `embedder` starts.

Different models produce different cosine-score distributions, so re-check `RAG_MIN_SCORE` after a switch (0.5 is a reasonable starting point for Qwen3-Embedding). Use the evaluation harness from [TODO.md](TODO.md) to tune it.

The model is downloaded into `./models/hf-cache` on first start and is reused across container restarts. Old weights are not deleted automatically; after switching, you can remove `models--google--embeddinggemma-300m` to free ~0.6 GB.

### Dependency pins

Qwen3-Embedding requires `transformers>=4.51` (Qwen3 architecture support; older versions fail with `KeyError: 'qwen3'`). `api/requirements.txt` also pins `sentence-transformers<6` because sentence-transformers 6.x requires transformers 5.x, which this stack has not been tested with.

## Document indexing

Put UTF-8 text files into `data/documents`:

```powershell
Copy-Item .\mydocument.txt .\data\documents\
```

The indexer:

1. reconciles the index with the filesystem at startup: vectors for files that no longer exist under `data/documents` (deleted or renamed while the stack was stopped) are removed;
2. scans `data/documents` and queues every existing file;
3. watches for changes with `PollingObserver` and a 1-second debounce;
4. skips files whose signature matches the stored one (`mtime`, size, chunk size, overlap, model);
5. splits changed files into 1024-character chunks with 128-character overlap;
6. deletes the file's old vectors and upserts the new ones.

Logs:

```bash
docker compose logs -f embedder
```

```text
2026-11-15 13:14:33 INFO [embedder] Removed stale vectors for 1 file(s) (5 chunks)
2026-11-15 13:14:33 INFO [embedder] Stable event for /data/documents/mydocument.txt
2026-11-15 13:14:35 INFO [embedder] Upserted 5 chunks for /data/documents/mydocument.txt
2026-11-15 13:14:40 INFO [embedder] Unchanged, skipping /data/documents/other.txt
```

Reconciliation walks the `file` payload field of every point, so startup cost grows with the number of chunks in the collection. For very large collections this should be replaced with a facet query or a dedicated file manifest.

## Memory and performance

The layout targets a single machine with one GPU (tested on an RTX 4060 Laptop, 8 GB VRAM).

The embedding model used to be loaded twice — in `embedder` and in `api`: two sets of weights, two torch runtimes, two CUDA contexts on one GPU. Now only `api` loads the weights; `embedder` talks to it over HTTP and contains no torch.

| Component | Before | Now |
|---|---|---|
| `embedder`: torch + CUDA | ~1.5 GB | — |
| `embedder`: model weights | gemma-2-2b fp32 ≈ 10.4 GB | — |
| `api`: torch + CUDA | ~1.5 GB | ~1.5 GB |
| `api`: model weights | 10.4 GB | Qwen3-Embedding-0.6B fp16 ≈ 1.2 GB |
| `qdrant` | ~0.2 GB | ~0.2 GB |
| **Total host memory** | **≈ 26 GB** | **≈ 5.6 GB** |

The LLM lives on another host, so the local GPU is dedicated to embeddings.

Other decisions:

- **fp16 instead of fp32** — half the weights and faster on tensor cores. On CPU, precision is forced to fp32 because fp16 is slow and unsupported by many ops.
- **Vector normalization** (`EMBEDDING_NORMALIZE`, default `true`) — more stable cosine search.
- **Incremental indexing** — every chunk payload carries a file signature (`mtime`, `size`, `chunk_size`, `chunk_overlap`, `model`); unchanged files are skipped at startup and on watcher events.
- **Batching** — texts are sent to `api` in `EMBED_BATCH_SIZE` batches instead of the whole file at once.
- **Memory limits** — `mem_limit` of 1 GB for `embedder` and 512 MB for `gateway` prevents runaway indexing from consuming the host.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — less VRAM fragmentation.

To free the GPU entirely, move embeddings to CPU:

```env
EMBEDDING_DEVICE=cpu
EMBEDDING_DTYPE=float32
```

Qwen3-Embedding-0.6B is noticeably slower on CPU than 300M models, but background indexing and single queries are usually still tolerable.

## Known limitations

- **No labelled evaluation set yet** — `scripts/eval_retrieval.py` is in place, but `RAG_TOP_K` and `RAG_MIN_SCORE` are not yet tuned against real questions.
- **Dense search only** — exact identifiers, product codes, and rare names can be missed; there is no sparse/BM25 hybrid.
- **No reranker** — the top-`k` dense results go straight into the prompt.
- **Fixed-size chunking** — no heading/paragraph awareness and no section metadata for citations.
- **No API/Qdrant healthchecks** — `depends_on` does not wait for readiness; Qdrant is pulled as `latest`.
- **Single GPU service** — bulk indexing and user queries compete for the same model; there is no dynamic batching (each request calls `model.encode` in a worker thread).
- **Multi-turn follow-ups** — only the last user message is embedded, so elliptical questions ("what about the second one?") retrieve poorly.
- **Text only** — only UTF-8 text files are supported; no PDF/DOCX parsing.
- **Model-dependent threshold** — `RAG_MIN_SCORE` must be re-tuned after changing the embedding model.

## Troubleshooting

```bash
docker compose ps                 # container status
docker compose logs -f            # all logs
docker compose logs -f gateway
curl http://localhost:8001/health
python scripts/smoke_test.py --api-key <key>
```

**401 Unauthorized** — the key is missing or does not match `API_KEYS`; after editing `.env`, recreate the container (`docker compose up -d`).

**`status: degraded`** — one dependency is unavailable; inspect `dependencies` in `/health`.

**`503 No model server is configured`** — `UPSTREAM_BASE_URL` is empty. Set the LLM server URL in `.env` and recreate the gateway container.

**`502 upstream_auth_failed`** — the model server rejected the **gateway's** key. Check `UPSTREAM_API_KEY` (and whether the server expects Bearer auth). The agent's key is irrelevant here.

**Empty answers without context** — check `X-RAG-Hits` in the response: `0` means either the collection is empty (`/collection/info`) or all results scored below `RAG_MIN_SCORE`.

**Model fails to load** — check Hugging Face access and disk space: `docker compose logs api | Select-String "Loading model"`. Qwen3-Embedding needs `transformers>=4.51`; older versions fail with `KeyError: 'qwen3'`.

**`CUDA out of memory`** — the embedding model does not fit in VRAM. Set `EMBEDDING_DEVICE=cpu`; the LLM is on another host, so the GPU is needed only for embeddings. Qwen3-Embedding-0.6B takes ≈ 1.2 GB in fp16.

**Indexing takes long to start** — a full pass runs on an empty collection and after a model or chunking change; otherwise `embedder` skips files with a matching signature.

**Vector dimension error** — the collection was created for a different model (for example, 768 for embeddinggemma while the current model produces 1024). Delete `qdrant_data/*` while the stack is stopped and restart. A normal model switch recreates the collection automatically.

**Qdrant crashes** — clear its data:

```powershell
docker compose down
Remove-Item -Path .\qdrant_data\* -Recurse -Force
docker compose up -d
```

## Monitoring

- Qdrant dashboard: http://localhost:6333/dashboard
- API Swagger: http://localhost:8000/docs
- Gateway Swagger: http://localhost:8001/docs
- Gateway health: http://localhost:8001/health

## Development

```bash
docker compose up -d --build      # rebuild after changes
docker compose restart gateway
```

Code map:

- `gateway/main.py` — `require_api_key`, `_rag_search`, `_inject_context`, `/v1/chat/completions`, `/v1/embeddings`
- `api/main.py` — `/search` (Qdrant `query_points`, with a fallback to the older `search`), `/embed`, `/v1/embeddings`
- `api/embedder.py` — `Embedder`: fp16 loading, batching, normalization, query prompt
- `embedder/main.py` — `DebounceHandler`, `ingestion_worker`, `file_signature`, `split_text_into_chunks`
- `embedder/embedder.py` — `RemoteEmbedder`: HTTP client for `api` with retries and a readiness wait
- `embedder/vector_db.py` — `VectorDB`: upsert with a file signature, `is_up_to_date`, collection recreation on dimension change
- `scripts/smoke_test.py` — end-to-end check of health, models, embeddings, chat, and RAG headers
- `scripts/eval_retrieval.py` — retrieval-quality evaluation (recall@k, MRR, nDCG) against a JSONL question set

There is no unit-test suite yet; `scripts/smoke_test.py` is the end-to-end check and `scripts/eval_retrieval.py` measures retrieval quality. See [eval/README.md](eval/README.md) for the dataset format.

## Roadmap

The actionable, prioritized plan is in **[TODO.md](TODO.md)**. Summary:

| Priority | Item |
|---|---|
| P0 | Retrieval evaluation harness (recall@k, MRR) |
| P0 | Startup reconciliation of stale vectors |
| P0 | Tune `RAG_TOP_K` / `RAG_MIN_SCORE` against the evaluation set |
| P1 | Reranker (retrieve 30–50, pass 5–8) |
| P1 | Structure-aware chunking and citation metadata |
| P1 | Multi-turn query rewriting |
| P2 | Hybrid BM25 + dense search with RRF |
| P2 | Healthchecks, pinned Qdrant version, snapshots |
| P3 | TEI/vLLM backend, query cache, quantization, larger model, pgvector |

## Useful links

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [LM Studio local server](https://lmstudio.ai/docs/app/api)
- [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Sentence Transformers](https://www.sbert.net/)
- [pgvector Extension](https://github.com/pgvector/pgvector)

## License

MIT

## Support

1. Check the logs: `docker compose logs`
2. Verify the GPU is visible: `nvidia-smi`
3. Run the smoke test: `python scripts/smoke_test.py`
