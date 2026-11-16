# RAG Gateway — OpenAI-compatible API with retrieval

A local RAG stack for documents: automatic indexing, semantic search over Qdrant, and an OpenAI-compatible gateway for agents.

The generation model is **not** part of this stack. The gateway forwards chat requests to an external OpenAI-compatible server (vLLM, LM Studio, llama.cpp, a cloud endpoint); agents only talk to the gateway, and the upstream key never leaves it.

```text
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_API_KEY=<key from API_KEYS>
```

There is no chat UI — the service exposes an API only. Any OpenAI-compatible client (OpenAI SDK, LangChain, Continue, Cline, etc.) can use it.

## Features

- OpenAI-compatible `/v1/chat/completions` (streaming and non-streaming), `/v1/models`, `/v1/embeddings`
- Automatic document indexing: startup reconciliation, `watchdog` file monitoring, incremental updates (unchanged files are skipped by a stored signature)
- Structure-aware chunking (Markdown headings → paragraphs → sentences, 1024-character chunks, 128-character overlap) with section paths and character offsets for citations
- RAG enabled by default: retrieved chunks are injected into the current user turn; retrieval is skipped inside agent tool loops
- Per-request RAG control via `X-RAG-*` headers, sources returned in `X-RAG-Sources`
- Follow-up query rewriting for short messages (`RAG_QUERY_REWRITE=off|concat|llm`, `concat` by default)
- Bearer authentication with static keys; the upstream key never reaches the agent
- One embedding-model instance shared by search and indexing; the indexer has no torch/CUDA
- GPU (fp16) or CPU embeddings
- Qdrant vector database; snapshot backup and restore without re-embedding
- Healthchecks on every service, `/health` with dependency status
- Retrieval evaluation harness (`recall@k`, `MRR`, `nDCG`) with a JSONL question set
- Plain-text documents (UTF-8): `.txt`, `.md`, `.csv`, `.json`

## Architecture

```text
┌──────────────┐
│ Agent / LLM  │  OpenAI SDK, LangChain, Continue, Cline
│   client     │
└──────┬───────┘
       │ POST /v1/chat/completions
       v
┌──────────────────────────────┐
│ Gateway            :8001     │  the gateway is the only entry point for agents
│ • bearer auth                │
│ • retrieval via api          │
│ • context injection          │
│ • upstream auth swap         │
└─────┬────────────┬───────────┘
      │ chat       │ POST /search, /v1/embeddings
      v            v
┌─────────────┐ ┌────────────────────────┐
│ LLM server  │ │ API            :8000   │
│ (LAN/cloud) │ │ embedding model        │
│ vLLM / LM   │ │ fp16 on GPU (only here)│
│ llama.cpp   │
└─────────────┘ └──┬───────────▲─────────┘
                   │ search    │ POST /embed
                   v           │ (vectors)
              ┌──────────────┐
              │ Qdrant       │
              │ :6333        │
              └────▲─────────┘
                   │ upsert    │
              ┌────┴───────────┴─────────┐
              │ Embedder                 │
              │ watchdog, no torch       │
              └────────▲─────────────────┘
                       │ watches files, reads text
              data/documents/
```

The arrows are the calls the services actually make: `gateway → upstream` (chat), `gateway → api` (`POST /search`, `POST /v1/embeddings`), `api → qdrant` (search), `embedder → api` (`POST /embed` — vectors), `embedder → qdrant` (upsert), `embedder → data/documents` (watch and read).

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
│   ├── chunking.py              # structure-aware splitter + citation metadata
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
│   ├── eval_retrieval.py        # retrieval quality evaluation + top_k/min_score sweep
│   ├── latency_probe.py         # p50/p95 of search while the indexer is busy
│   ├── reconcile_check.py        # index vs watched folder: orphaned and un-indexed files
│   └── qdrant_snapshot.py       # backup and restore of the Qdrant index
├── eval/
│   ├── README.md                # dataset format, modes, sweep, metrics, tuning record
│   ├── questions.jsonl          # the 60-question baseline dataset
│   ├── questions.multiturn.jsonl# 9 follow-up conversations (history + short question)
│   ├── questions.example.jsonl  # schema examples
│   ├── corpus/                  # the sample documents the dataset is measured on
│   ├── baseline.json            # committed baseline numbers + sweep grid
│   ├── baseline-prompt-off.json # the same set with the query prompt disabled
│   ├── gateway-singleturn-off.json      # 60 single-turn questions, rewrite off
│   ├── gateway-singleturn-concat.json   # the same set, rewrite concat
│   ├── multiturn-off.json       # 9 follow-up conversations, rewrite off
│   ├── multiturn-gateway.json   # the same set, rewrite concat
│   ├── multiturn-llm.json       # the same set, rewrite llm
│   ├── sweep.md                 # top_k × min_score tables
│   ├── sweep-prompt-off.md      # the same grid without the query instruction
│   ├── sweep-multiturn.md       # the same grid on follow-up conversations
│   ├── sweep-multiturn-llm.md   # the same grid with the llm rewrite
│   └── perf-probe.json          # measured search latency under ingest
├── tests/
│   ├── test_eval_metrics.py     # unit tests for the eval math (no GPU/API needed)
│   └── test_gateway_rewrite.py  # unit tests for follow-up query rewriting
├── qdrant_snapshots/            # snapshots created on the Qdrant node (bind-mounted into the container)
├── backups/                     # downloaded Qdrant snapshots (created by the tool)
├── README.md
└── TODO.md
```

Qdrant storage is a named Docker volume (`qdrant_data`), not a folder in this repo: Qdrant needs a filesystem it can guarantee data safety on, which a Windows bind mount is not. The snapshots directory is bind-mounted so snapshot files created via REST survive container recreation.

`data/documents/` is git-ignored. The sample documents used by the evaluation dataset are committed under `eval/corpus/`; see [eval/README.md](eval/README.md) for the dataset format and how to reproduce the baseline.

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
- **Version:** pinned to `v1.20.1` via `QDRANT_VERSION` in `.env`. Bump it deliberately: take a snapshot and verify the storage format before changing the tag.
- **Healthcheck:** TCP connect to `127.0.0.1:6333` (the official image has no `curl`/`wget`); `api` and `embedder` wait for it with `depends_on: condition: service_healthy`.

#### Backups and restore (snapshots)

A Qdrant snapshot is a complete copy of the collection: points, vectors and metadata. Restoring it rebuilds the index **without re-embedding** the corpus, so a full re-index (GPU time for every chunk) is not needed.

```bash
# create a snapshot on the Qdrant node and download it into ./backups
python scripts/qdrant_snapshot.py create          # keeps the 5 newest snapshots on the node
python scripts/qdrant_snapshot.py list            # snapshots stored on the node
python scripts/qdrant_snapshot.py status          # collection status, points, dimension
python scripts/qdrant_snapshot.py prune --keep 3  # delete the oldest node snapshots
python scripts/qdrant_snapshot.py delete --snapshot NAME
```

Snapshots created via REST are stored by the node in `/qdrant/snapshots/documents/`, bind-mounted to `qdrant_snapshots/documents/` on the host. They are not inside the storage volume, so a snapshot created without that bind mount disappears when the container is recreated. `create` downloads a copy into `./backups/` so it survives a container rebuild. Keep the downloaded copy off the machine (cloud, NAS, external disk): a backup inside the folder that gets deleted with the container is not a backup.

Restore replaces the collection data, so it requires an explicit confirmation. Restoring into
a running stack:

```bash
# restore into a running stack (overwrites the collection)
docker compose stop embedder                      # the indexer must not write during restore
docker compose up -d qdrant api                   # qdrant and api must be up and healthy
python scripts/qdrant_snapshot.py restore --file backups/documents-....snapshot --yes
docker compose up -d embedder
python scripts/smoke_test.py                      # search must return results
```

Restoring into a fresh Qdrant instance — `docker compose down -v` removes the containers **and the
storage volume** (this destroys the indexed data; only `./backups/` remains):

```bash
docker compose down -v                              # containers and the storage volume are removed
docker compose up -d qdrant                         # qdrant starts with an empty volume
python scripts/qdrant_snapshot.py status            # collection 'documents': missing
python scripts/qdrant_snapshot.py restore --file backups/documents-....snapshot --yes
python scripts/qdrant_snapshot.py status            # status=green, points and dim as expected
docker compose up -d                                  # api and embedder join
```

The restore uploads the snapshot with its sha256 checksum; Qdrant creates the collection and writes
the data onto the volume. `status` prints `points` and `dim` after restore, and `python
scripts/smoke_test.py` checks the retrieval path end to end.

The upload endpoint (`POST /collections/documents/snapshots/upload`) verifies the sha256 checksum, creates the collection if it does not exist, and overwrites everything stored on the node for that collection. The tool prints the collection state before and after (`points`, `dim`), and refuses to run without `--yes`.

Caveats:

- **Version:** restore into the same Qdrant image version (the pinned `QDRANT_VERSION`); older servers cannot read storage written by newer ones.
- **Re-index comparison:** a snapshot restore takes seconds; a full re-index of a corpus costs one embedding pass over every chunk on GPU.
- **After restore:** `embedder` re-verifies file signatures on startup, so unchanged files are not re-indexed. Files changed while the index was stale remain out of date; check `embedder` logs.
- **Full-storage snapshots** (`POST /snapshots`, `GET /snapshots`) cover all collections at once; they are restored by starting Qdrant with the `--snapshot <path>` flag at boot, not through REST.
- **Offline:** `embedder` and `api` must be stopped while restoring, and retrieval is unavailable during restore.
- The script is stdlib-only (no extra dependencies) and needs only `6333` reachable.

#### Index integrity (watched folder vs index)

The indexer deletes vectors when a file disappears, and on startup removes vectors of files that
are no longer under `WATCH_PATH` (`deleted`/`renamed` while the stack was stopped), so the index
and the folder stay in sync. To check that from the outside:

```bash
python scripts/reconcile_check.py audit                 # every file on disk is indexed, no orphaned paths
python scripts/reconcile_check.py count --file NAME     # chunks stored for that file
python scripts/reconcile_check.py search --query "text" --expect NAME --absent NAME
```

`audit` exits non-zero if any file on disk is not indexed (`NOT INDEXED`) or any indexed path no
longer exists on disk (`ORPHANED`). `search --expect/--absent` checks whether a file appears in the
top hits, which is how a delete or a rename is verified end to end. The container mount defines the
indexed path: `./data` is mounted as `/data`, so `data/documents/report.md` is stored as
`/data/documents/report.md` (`--documents` and `--prefix`).

### Startup order

Compose starts Qdrant, then `api` (which loads the embedding model), then `embedder`. `api` is considered healthy only when `/health` reports `embedder: "ready"`, so the first start may take several minutes while the model downloads. The gateway starts early and serves chat without retrieval until `api` is ready; `/health` shows `degraded` during that window.

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
| `X-RAG-Sources` | unpadded base64url-encoded JSON: `[{"file","chunk","score","title?","section?","char_start?","char_end?"}]` |

`title`, `section`, and character offsets are included when the indexed chunk has them.

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

#### Follow-up questions

`RAG_QUERY_REWRITE` controls whether retrieval sees the conversation history:

- `off` — only the last user message is embedded.
- `concat` (default) — for short or elliptical messages ("and the second one?"), the last `RAG_REWRITE_HISTORY` user messages are joined with the current one.
- `llm` — the gateway asks the upstream model to rewrite the message into a standalone query; this adds one upstream call, is bounded by `RAG_REWRITE_TIMEOUT`, and falls back to `concat` on failure.

The rewrite fires only when the request carries a history and the last user message is short or elliptical; requests without history and long questions are sent unchanged. Mode comparison and measured numbers: [eval/README.md](eval/README.md).

**Where the context goes.** The `<knowledge_base>` block is prepended to the `content` of the last user message; the actual question stays at the end. Each `<document>` carries `source`, `relevance`, and, when available, `title`, `section`, and `chars="start-end"` attributes. No separate `system` message is added, so the `system + history` prefix stays byte-identical and upstream prompt caching keeps working. Chat templates that require `system` to be the first message (ChatML, Llama 3) are not broken.

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
  "defaults": {"top_k": 8, "min_score": 0.3, "query_rewrite": "concat", "rewrite_history": 3},
  "dependencies": {
    "upstream": {"url": "http://192.168.1.50:8000/v1", "ok": true, "status": 200},
    "api": {"url": "http://api:8000", "ok": true, "status": 200}
  }
}
```

`status` is `degraded` when at least one dependency is unavailable. If `UPSTREAM_BASE_URL` is empty, `upstream.ok` is `false` and chat requests return `503`.

## Configuration

Variables are read from `.env` (see [`.env.example`](.env.example)).

### User-facing variables

| Variable | Default | Description |
|---|---|---|
| `API_KEYS` | — | comma-separated bearer keys; empty disables auth |
| `QDRANT_VERSION` | `v1.20.1` | pinned Qdrant image tag; bump after a snapshot |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | embedding model (one per deployment) |
| `EMBEDDING_DEVICE` | `cuda` | `cuda` or `cpu` |
| `EMBEDDING_DTYPE` | `float16` | `float16`, `bfloat16`, `float32`, `auto`; CPU always uses fp32 |
| `EMBEDDING_QUERY_PROMPT` | `query` | model prompt used for queries; empty = encode queries like documents |
| `EMBEDDING_MAX_SEQ_LENGTH` | `8192` | maximum tokens per text (the model supports 32k) |
| `EMBED_BATCH_SIZE` | `32` | texts per embedding request (both indexing and API batching) |
| `ENABLE_RAG` | `true` | global RAG switch |
| `RAG_TOP_K` | `8` | chunks injected into the prompt |
| `RAG_MIN_SCORE` | `0.3` | cosine-score threshold; model-dependent, re-tune after changing the model |
| `RAG_QUERY_REWRITE` | `concat` | follow-up query mode: `off`, `concat`, or `llm` |
| `RAG_REWRITE_HISTORY` | `3` | previous user turns considered by the rewrite |
| `RAG_REWRITE_TIMEOUT` | `10` | timeout for the `llm` rewrite call, seconds |
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
| `EMBEDDING_MAX_CONCURRENCY` | `${EMBEDDING_MAX_CONCURRENCY}` = `1` | encode calls allowed at once; `1` serializes them on the GPU so indexing and chat queue instead of competing for VRAM |
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

Different models produce different cosine-score distributions, so `RAG_MIN_SCORE` and `RAG_TOP_K` must be re-measured after a switch, not copied. One retrieval pass plus the sweep grid is enough:

```bash
python scripts/eval_retrieval.py --questions eval/questions.jsonl --corpus eval/corpus \
    --sweep --json-out eval/baseline.json
```

The shipped `RAG_TOP_K=8` / `RAG_MIN_SCORE=0.3` come from that sweep on the sample corpus; the recorded grids are in [eval/README.md](eval/README.md).

The model is downloaded into `./models/hf-cache` on first start and is reused across container restarts. Old weights are not deleted automatically; after switching, you can remove `models--google--embeddinggemma-300m` to free ~0.6 GB.

### Dependency pins

Qwen3-Embedding requires `transformers>=4.51` (Qwen3 architecture support). `api/requirements.txt` pins `sentence-transformers>=5.0,<6`.

## Document indexing

Put UTF-8 text files into `data/documents`:

```powershell
Copy-Item .\mydocument.txt .\data\documents\
```

The indexer:

1. reconciles the index with the filesystem at startup: vectors for files that no longer exist under `data/documents` (deleted or renamed while the stack was stopped) are removed;
2. scans `data/documents` and queues every existing file;
3. watches for changes with `PollingObserver` and a 1-second debounce;
4. skips files whose signature matches the stored one (`mtime`, size, chunk size, overlap, chunker version, model);
5. splits changed files with the structure-aware chunker: Markdown ATX headings and blank-line paragraphs first, then sentences, then whitespace; chunks are capped at 1024 characters and overlap by up to 128;
6. prepends the document title (first heading or first line) to each chunk before embedding;
7. stores the original chunk text plus `title`, `section`, `char_start`, and `char_end` in the payload;
8. deletes the file's old vectors and upserts the new ones.

The chunker version is part of the file signature, so changing the splitting algorithm re-indexes existing documents. Character offsets refer to the normalized text (CRLF/CR converted to LF).

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

## Memory and performance

The layout targets a single machine with one GPU (tested on an RTX 4060 Laptop, 8 GB VRAM). The LLM lives on another host, so the local GPU is dedicated to embeddings.

- **One model instance** — only `api` loads the weights; `embedder` requests vectors over HTTP and has no torch/CUDA. Host memory: `api` ≈ 1.5 GB (torch + CUDA) + Qwen3-Embedding-0.6B fp16 ≈ 1.2 GB, `qdrant` ≈ 0.2 GB.
- **fp16** — half the weights and faster on tensor cores; on CPU, precision is forced to fp32.
- **Vector normalization** (`EMBEDDING_NORMALIZE`, default `true`).
- **Incremental indexing** — every chunk payload carries a file signature (`mtime`, `size`, `chunk_size`, `chunk_overlap`, `chunking_version`, `model`); unchanged files are skipped at startup and on watcher events.
- **Batching** — texts are sent to `api` in `EMBED_BATCH_SIZE` batches.
- **Memory limits** — `mem_limit` of 1 GB for `embedder` and 512 MB for `gateway`.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — less VRAM fragmentation.
- **Serialized encode calls** (`EMBEDDING_MAX_CONCURRENCY=1`) — bulk indexing and chat queries queue instead of competing for VRAM.

To free the GPU entirely, move embeddings to CPU:

```env
EMBEDDING_DEVICE=cpu
EMBEDDING_DTYPE=float32
```

On CPU a smaller model (`google/embeddinggemma-300m` or `intfloat/multilingual-e5-small`) is the usual choice.

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

**Model fails to load** — check Hugging Face access and disk space: `docker compose logs api | Select-String "Loading model"`. `api/requirements.txt` pins the versions the model needs.

**`CUDA out of memory`** — the embedding model does not fit in VRAM. Set `EMBEDDING_DEVICE=cpu` (`EMBEDDING_DTYPE=float32`), or use a smaller model.

**Indexing takes long to start** — a full pass runs on an empty collection and after a model or chunking change; otherwise `embedder` skips files with a matching signature.

**Vector dimension error** — the collection was created for a different model (for example, 768 for embeddinggemma while the current model produces 1024). Delete the storage volume while the stack is stopped and restart. A normal model switch recreates the collection automatically.

> `docker volume rm` destroys the indexed data. Take a snapshot first (`python scripts/qdrant_snapshot.py create`) and restore it afterwards if you need the index.

**Qdrant crashes** — clear its data:

```powershell
docker compose down -v            # removes the containers and the qdrant_data volume
docker compose up -d
```

If you want to keep the volume, remove only the container and start again; the data stays on the volume.

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
- `api/main.py` — `/search` (Qdrant `query_points`, with a fallback to the older `search`) returning citation metadata, `/embed`, `/v1/embeddings`
- `api/embedder.py` — `Embedder`: fp16 loading, batching, normalization, query prompt
- `embedder/main.py` — `DebounceHandler`, `ingestion_worker`, `file_signature`
- `embedder/chunking.py` — `split_text_into_chunks`, `TextChunk`, heading/section metadata and offsets
- `embedder/embedder.py` — `RemoteEmbedder`: HTTP client for `api` with retries and a readiness wait
- `embedder/vector_db.py` — `VectorDB`: upsert with chunk metadata and a file signature, startup reconciliation, `is_up_to_date`, collection recreation on dimension change
- `scripts/smoke_test.py` — end-to-end check of health, models, embeddings, chat, and RAG headers
- `scripts/eval_retrieval.py` — retrieval-quality evaluation (recall@k, MRR@10, nDCG@10) against a JSONL question set; `search` and `gateway` modes, offline `top_k` × `min_score` sweep, offline corpus validation, one keep-alive connection for the whole run
- `scripts/latency_probe.py` — p50/p95/p99 of `POST /search` when idle, while synthetic documents are being indexed, and after indexing settles; the latency multiplier under ingest, and `chunks_added`
- `scripts/reconcile_check.py` — compares the files under the watched folder with the `payload.file` values in Qdrant (`audit`), counts chunks per file (`count`), and checks whether a file is in the top hits of a query (`search --expect/--absent`); stdlib only
- `scripts/qdrant_snapshot.py` — create, list, download, prune, delete snapshots; restore a collection from a local file (stdlib only)

Unit tests for the evaluation math (matching, metrics, sweep grid, dataset and corpus validation) run without GPU, API, or Docker:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

`scripts/smoke_test.py` is the end-to-end check of the live stack, and
`scripts/eval_retrieval.py` measures retrieval quality. See
[eval/README.md](eval/README.md) for the dataset format, the corpus snapshot, and
how to read the sweep tables.

## Roadmap

The prioritized development plan is in **[TODO.md](TODO.md)**.

## Useful links

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant snapshots (backup and restore)](https://qdrant.tech/documentation/snapshots/)
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
