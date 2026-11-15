"""OpenAI-compatible RAG gateway.

Public surface (the only API agents need):

    GET  /v1/models
    GET  /v1/models/{model_id}
    POST /v1/chat/completions      # RAG-augmented, streaming + non-streaming
    POST /v1/embeddings            # proxied to api
    GET  /health

Chat requests are forwarded verbatim to an OpenAI-compatible upstream
(`UPSTREAM_BASE_URL`), so tools / tool_calls / response_format / temperature and
any other OpenAI field pass through untouched. The only thing this service
changes is that it prepends the documents retrieved from the vector store to the
text of the current user turn.

The upstream credentials (`UPSTREAM_API_KEY`) never leave this process: the
`Authorization` header sent by an agent is only used to authenticate against
this gateway and is never forwarded.

Per-request RAG overrides (headers):
    X-RAG: off|on
    X-RAG-TOP-K: 5
    X-RAG-MIN-SCORE: 0.6

Retrieved sources are returned in response headers so they do not pollute the
OpenAI response schema:
    X-RAG-Hits: 3
    X-RAG-Sources: <base64url(unpadded) of JSON
                    [{file, chunk, score, title?, section?, char_start?, char_end?}]>
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("gateway")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _env_headers(name: str) -> Dict[str, str]:
    """Parse an optional JSON object of extra upstream headers."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("Ignoring %s: not valid JSON (%s)", name, exc)
        return {}

    if not isinstance(value, dict):
        log.warning("Ignoring %s: expected a JSON object", name)
        return {}

    return {str(key): str(item) for key, item in value.items()}


UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "").strip().rstrip("/")
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "").strip()
UPSTREAM_HEADERS: Dict[str, str] = _env_headers("UPSTREAM_HEADERS")
API_URL = os.environ.get("API_URL", "http://api:8000").rstrip("/")

ENABLE_RAG = _env_bool("ENABLE_RAG", True)
RAG_TOP_K = _env_int("RAG_TOP_K", 8)
RAG_MAX_TOP_K = _env_int("RAG_MAX_TOP_K", 20)
RAG_MIN_SCORE = _env_float("RAG_MIN_SCORE", 0.3)
RAG_TIMEOUT = _env_float("RAG_TIMEOUT", 30.0)

# Follow-up query handling: off | concat | llm. "concat" joins previous user
# turns for short/elliptical messages; "llm" asks the upstream model to rewrite
# the query, falling back to concatenation when the call fails.
RAG_QUERY_REWRITE = os.environ.get("RAG_QUERY_REWRITE", "off").strip().lower() or "off"
if RAG_QUERY_REWRITE not in {"off", "concat", "llm"}:
    log.warning(
        "Unknown RAG_QUERY_REWRITE=%r - using 'off' (expected off|concat|llm)",
        RAG_QUERY_REWRITE,
    )
    RAG_QUERY_REWRITE = "off"
RAG_REWRITE_HISTORY = max(1, _env_int("RAG_REWRITE_HISTORY", 3))
RAG_REWRITE_TIMEOUT = _env_float("RAG_REWRITE_TIMEOUT", 10.0)
RAG_REWRITE_MAX_TOKENS = max(16, _env_int("RAG_REWRITE_MAX_TOKENS", 128))
RAG_REWRITE_CACHE_SIZE = max(0, _env_int("RAG_REWRITE_CACHE_SIZE", 256))

UPSTREAM_TIMEOUT = _env_float("UPSTREAM_TIMEOUT", 600.0)
CONNECT_TIMEOUT = _env_float("CONNECT_TIMEOUT", 10.0)
MAX_REQUEST_BYTES = _env_int("MAX_REQUEST_BYTES", 8_000_000)

API_KEYS = tuple(
    key.strip() for key in os.environ.get("API_KEYS", "").split(",") if key.strip()
)

if not UPSTREAM_BASE_URL:
    log.critical(
        "UPSTREAM_BASE_URL is empty - chat, models and health calls will fail. "
        "Point it at your OpenAI-compatible model server, e.g. "
        "http://192.168.1.50:8000/v1"
    )
if not API_KEYS:
    log.warning("API_KEYS is empty - the gateway accepts unauthenticated requests")
if RAG_QUERY_REWRITE == "llm" and not UPSTREAM_BASE_URL:
    log.warning(
        "RAG_QUERY_REWRITE=llm requires UPSTREAM_BASE_URL - falling back to 'concat'"
    )
    RAG_QUERY_REWRITE = "concat"

app = FastAPI(
    title="RAG Gateway",
    version="2.1.0",
    description=(
        "OpenAI-compatible gateway in front of a self-hosted model server. "
        "Every chat completion is automatically grounded in the local "
        "document collection."
    ),
)

# --------------------------------------------------------------------------- #
# Auth + error handling
# --------------------------------------------------------------------------- #

_bearer = HTTPBearer(auto_error=False)


def _openai_error(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


async def require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    """Validate `Authorization: Bearer <key>` against API_KEYS.

    Authentication is disabled when API_KEYS is empty (local development only).
    """
    if not API_KEYS:
        return

    token = credentials.credentials if credentials is not None else None
    if token and any(secrets.compare_digest(token, key) for key in API_KEYS):
        return

    raise StarletteHTTPException(
        status_code=401,
        detail="Invalid or missing API key. Send 'Authorization: Bearer <key>'.",
    )


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Keep the OpenAI error envelope for every error an agent can hit.
    error_type = "server_error" if exc.status_code >= 500 else "invalid_request_error"
    return _openai_error(exc.status_code, str(exc.detail), error_type=error_type)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error while serving %s %s", request.method, request.url.path)
    return _openai_error(500, "Internal gateway error", error_type="server_error")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_TRUTHY = {"1", "true", "yes", "on", "enabled"}
_FALSY = {"0", "false", "no", "off", "disabled"}


async def _read_json(request: Request) -> Dict[str, Any]:
    raw = await request.body()
    if not raw:
        raise StarletteHTTPException(400, "Request body must not be empty")
    if len(raw) > MAX_REQUEST_BYTES:
        raise StarletteHTTPException(413, "Request body is too large")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StarletteHTTPException(400, f"Invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise StarletteHTTPException(400, "Request body must be a JSON object")
    return body


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI message content (string or list of content parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(part for part in parts if part)
    return ""


def _last_user_query(messages: List[Any]) -> str:
    """Return the text of the final user turn, or "" when RAG should be skipped.

    Context is only injected for a fresh user turn. During an agent tool loop the
    last message is `tool`/`assistant`, and re-injecting documents every step
    would only bloat the prompt.
    """
    if not messages:
        return ""
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return ""
    return _content_to_text(last.get("content")).strip()


# --------------------------------------------------------------------------- #
# Follow-up query rewriting
# --------------------------------------------------------------------------- #

_FOLLOWUP_PREFIXES = (
    "and ",
    "but ",
    "so ",
    "then ",
    "also ",
    "what about ",
    "how about ",
    "why ",
    "а ",
    "и ",
    "но ",
    "тогда ",
    "ещё ",
    "еще ",
    "что ",
    "как ",
    "почему ",
)

_FOLLOWUP_WORDS = frozenset(
    {
        "it",
        "this",
        "that",
        "these",
        "those",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "its",
        "their",
        "first",
        "second",
        "third",
        "last",
        "previous",
        "above",
        "это",
        "этот",
        "эта",
        "эти",
        "он",
        "она",
        "они",
        "его",
        "её",
        "ее",
        "их",
        "первый",
        "второй",
        "третий",
        "последний",
        "предыдущий",
        "выше",
    }
)

_rewrite_cache: Dict[Tuple[str, str, str], str] = {}


def _history_user_texts(messages: List[Any], limit: int) -> List[str]:
    """User messages before the final turn, oldest first."""
    texts: List[str] = []
    for message in messages[:-1]:
        if isinstance(message, dict) and message.get("role") == "user":
            text = _content_to_text(message.get("content")).strip()
            if text:
                texts.append(text)
    return texts[-limit:]


def _format_history(messages: List[Any], limit: int) -> str:
    """Recent user/assistant turns as plain text for the rewrite prompt."""
    turns: List[str] = []
    for message in messages[:-1]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _content_to_text(message.get("content")).strip()
        if text:
            turns.append(f"{role}: {text[:1000]}")
    return "\n".join(turns[-limit * 2 :])


def _is_follow_up(query: str) -> bool:
    """Heuristic: short/elliptical messages and anaphora usually need history."""
    text = query.strip()
    if not text:
        return False
    words = re.findall(r"[\w']+", text.casefold())
    if len(words) <= 5:
        return True
    if text.casefold().startswith(_FOLLOWUP_PREFIXES):
        return True
    return any(word in _FOLLOWUP_WORDS for word in words)


def _concat_query(history: List[str], query: str) -> str:
    """Cheap rewrite: prepend recent user turns to the current question."""
    parts = [text[-500:] for text in history[-RAG_REWRITE_HISTORY:]]
    parts.append(query)
    return "\n".join(parts)[-2000:]


def _cache_rewrite(key: Tuple[str, str, str], value: str) -> None:
    if RAG_REWRITE_CACHE_SIZE <= 0:
        return
    if len(_rewrite_cache) >= RAG_REWRITE_CACHE_SIZE:
        _rewrite_cache.pop(next(iter(_rewrite_cache)))
    _rewrite_cache[key] = value


async def _rewrite_query_llm(history_text: str, query: str, model: str) -> str:
    """Ask the upstream model for a standalone retrieval query."""
    system = (
        "Rewrite the user's latest message as a standalone search query. "
        "Use the conversation history only to resolve references such as "
        "'it', 'the second one', or 'that'. Do not answer the question. "
        "Return only the rewritten query, with no quotes, labels, or explanation."
    )
    user_content = (
        f"Conversation history:\n{history_text}\n\n"
        f"Latest user message:\n{query[:4000]}\n\n"
        "Standalone search query:"
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": RAG_REWRITE_MAX_TOKENS,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=RAG_REWRITE_TIMEOUT) as client:
        response = await client.post(
            _upstream_url("/chat/completions"),
            json=body,
            headers=_upstream_headers(False),
        )
    response.raise_for_status()

    payload = response.json()
    content = (
        (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    )
    content = " ".join(str(content).split()).strip().strip('"').strip("'").strip()
    for prefix in ("search query:", "query:"):
        if content.casefold().startswith(prefix):
            content = content[len(prefix) :].strip()
    if not content:
        raise ValueError("the model returned an empty rewrite")
    return content[:500]


async def _prepare_retrieval_query(
    messages: List[Any], query: str, model: str
) -> str:
    """Return the query to embed, optionally resolved against the history."""
    if RAG_QUERY_REWRITE == "off":
        return query

    history = _history_user_texts(messages, RAG_REWRITE_HISTORY)
    if not history:
        return query

    if RAG_QUERY_REWRITE == "concat":
        if not _is_follow_up(query):
            return query
        return _concat_query(history, query)

    # llm: only spend an extra call on messages that look like follow-ups.
    if not _is_follow_up(query):
        return query

    history_text = _format_history(messages, RAG_REWRITE_HISTORY)
    cache_key = (model, history_text, query)
    cached = _rewrite_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        rewritten = await _rewrite_query_llm(history_text, query, model)
    except Exception as exc:  # noqa: BLE001 - retrieval must never break chat
        log.warning(
            "Query rewrite failed, falling back to concatenation: %s", exc
        )
        return _concat_query(history, query)

    _cache_rewrite(cache_key, rewritten)
    return rewritten


def _rag_settings(
    x_rag: Optional[str],
    x_rag_top_k: Optional[str],
    x_rag_min_score: Optional[str],
) -> tuple[bool, int, float]:
    enabled = ENABLE_RAG

    if x_rag is not None:
        value = x_rag.strip().lower()
        if value in _FALSY:
            enabled = False
        elif value in _TRUTHY:
            enabled = True
        else:
            raise StarletteHTTPException(400, "X-RAG must be one of: on, off")

    top_k = RAG_TOP_K
    if x_rag_top_k is not None and x_rag_top_k.strip():
        try:
            top_k = int(x_rag_top_k)
        except ValueError as exc:
            raise StarletteHTTPException(400, "X-RAG-TOP-K must be an integer") from exc
    top_k = max(1, min(top_k, RAG_MAX_TOP_K))

    min_score = RAG_MIN_SCORE
    if x_rag_min_score is not None and x_rag_min_score.strip():
        try:
            min_score = float(x_rag_min_score)
        except ValueError as exc:
            raise StarletteHTTPException(400, "X-RAG-MIN-SCORE must be a number") from exc
    min_score = max(-1.0, min(min_score, 1.0))

    return enabled, top_k, min_score


async def _rag_search(query: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=RAG_TIMEOUT) as client:
        response = await client.post(
            f"{API_URL}/search",
            json={"query": query, "top_k": top_k, "min_score": min_score},
        )
        response.raise_for_status()
        hits = response.json()

    if not isinstance(hits, list):
        return []

    results: List[Dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        score = hit.get("score")
        if not isinstance(score, (int, float)) or score < min_score:
            continue
        results.append(
            {
                "file": hit.get("file", "unknown"),
                "chunk": hit.get("chunk", 0),
                "score": round(float(score), 4),
                "text": hit.get("text", ""),
                "title": hit.get("title"),
                "section": hit.get("section"),
                "char_start": hit.get("char_start"),
                "char_end": hit.get("char_end"),
            }
        )
    return results[:top_k]


def _xml_attr(value: Any) -> str:
    """Escape a value used in a double-quoted attribute of the context block."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_context_block(hits: List[Dict[str, Any]]) -> str:
    lines = [
        "The following excerpts were retrieved from the local knowledge base.",
        "Use them when they are relevant to the user's request and cite the "
        "source file when you rely on them. Ignore them if they do not help.",
        "<knowledge_base>",
    ]
    for index, hit in enumerate(hits, 1):
        attrs = [
            f'index="{index}"',
            f'source="{_xml_attr(hit["file"])}"',
            f'relevance="{hit["score"]}"',
        ]
        if hit.get("title"):
            attrs.append(f'title="{_xml_attr(hit["title"])}"')
        if hit.get("section"):
            attrs.append(f'section="{_xml_attr(hit["section"])}"')
        if hit.get("char_start") is not None:
            attrs.append(f'chars="{hit["char_start"]}-{hit.get("char_end", "")}"')
        lines.append(f'<document {" ".join(attrs)}>')
        lines.append(hit.get("text", ""))
        lines.append("</document>")
    lines.append("</knowledge_base>")
    return "\n".join(lines)


def _inject_context(messages: List[Any], block: str) -> List[Any]:
    """Prepend the context block to the text of the final user message.

    The prefix (system message + history) stays byte-identical, so upstream
    prompt caching keeps working, and no `system` message is inserted in the
    middle of the conversation - many self-hosted chat templates reject that.
    The user's actual question is kept at the very end of the message.
    """
    prepared: List[Any] = [dict(m) if isinstance(m, dict) else m for m in messages]
    if not prepared:
        return prepared

    last = prepared[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return prepared

    content = last.get("content")
    if isinstance(content, str):
        last["content"] = f"{block}\n\n{content}" if content.strip() else block
    elif isinstance(content, list):
        last["content"] = [{"type": "text", "text": f"{block}\n\n"}, *content]
    else:
        last["content"] = block

    return prepared


def _rag_headers(hits: List[Dict[str, Any]]) -> Dict[str, str]:
    headers = {"X-RAG-Hits": str(len(hits))}
    if hits:
        # Only metadata goes into the header - document text stays in the prompt.
        public = []
        for hit in hits:
            item = {
                "file": hit["file"],
                "chunk": hit["chunk"],
                "score": hit["score"],
            }
            for key in ("title", "section", "char_start", "char_end"):
                if hit.get(key) is not None:
                    item[key] = hit[key]
            public.append(item)
        payload = json.dumps(public, ensure_ascii=True, separators=(",", ":")).encode()
        headers["X-RAG-Sources"] = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return headers


def _upstream_url(suffix: str) -> str:
    """Build an upstream URL without duplicating `/v1`.

    `UPSTREAM_BASE_URL` may be given either as `http://host:8000` or
    `http://host:8000/v1`; both produce the same result.
    """
    if not UPSTREAM_BASE_URL:
        raise StarletteHTTPException(
            503,
            "No model server is configured. Set UPSTREAM_BASE_URL to the "
            "OpenAI-compatible endpoint of your LLM server "
            "(for example http://192.168.1.50:8000/v1).",
        )

    base = UPSTREAM_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}{suffix}"


def _upstream_headers(stream: bool = False) -> Dict[str, str]:
    """Headers for the upstream call. The agent's Authorization is never reused."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    headers.update(UPSTREAM_HEADERS)
    return headers


def _upstream_error(status_code: int, text: str) -> Optional[JSONResponse]:
    """Translate upstream auth failures so agents do not blame their own key."""
    if status_code not in (401, 403):
        return None

    log.error(
        "Upstream rejected the gateway credentials (status=%d): %s",
        status_code,
        text[:300],
    )
    return _openai_error(
        502,
        "The model server rejected the gateway's credentials. "
        "Check UPSTREAM_API_KEY on the gateway.",
        error_type="server_error",
        code="upstream_auth_failed",
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


@app.get("/v1/models")
async def list_models(_: None = Depends(require_api_key)):
    url = _upstream_url("/models")
    try:
        async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT) as client:
            response = await client.get(url, headers=_upstream_headers())
    except httpx.HTTPError as exc:
        raise StarletteHTTPException(
            502, f"Cannot list models from the model server: {exc}"
        ) from exc

    mapped = _upstream_error(response.status_code, response.text)
    if mapped is not None:
        return mapped
    if response.status_code >= 400:
        raise StarletteHTTPException(
            response.status_code,
            f"Model server refused the model list: {response.text[:300]}",
        )

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, _: None = Depends(require_api_key)):
    listing = await list_models(_=None)
    if isinstance(listing, JSONResponse):
        return listing

    try:
        payload = json.loads(listing.body)
    except (TypeError, ValueError):
        payload = {}
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise StarletteHTTPException(502, "Model server returned an unexpected model list")

    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") == model_id:
            return entry
    raise StarletteHTTPException(404, f"Model '{model_id}' not found")


# --------------------------------------------------------------------------- #
# Chat completions
# --------------------------------------------------------------------------- #


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_rag: Optional[str] = Header(default=None),
    x_rag_top_k: Optional[str] = Header(default=None),
    x_rag_min_score: Optional[str] = Header(default=None),
    _: None = Depends(require_api_key),
):
    body = await _read_json(request)

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise StarletteHTTPException(400, "'model' is required")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise StarletteHTTPException(400, "'messages' must be a non-empty array")

    rag_enabled, top_k, min_score = _rag_settings(x_rag, x_rag_top_k, x_rag_min_score)

    hits: List[Dict[str, Any]] = []
    if rag_enabled:
        query = _last_user_query(messages)
        if query:
            try:
                retrieval_query = await _prepare_retrieval_query(messages, query, model)
            except Exception as exc:  # noqa: BLE001 - never break chat over a rewrite
                log.warning("Query rewrite failed, using the raw query: %s", exc)
                retrieval_query = query
            if retrieval_query != query:
                log.info(
                    "Retrieval query rewritten (%s): %r -> %r",
                    RAG_QUERY_REWRITE,
                    query[:80],
                    retrieval_query[:120],
                )
            try:
                hits = await _rag_search(retrieval_query, top_k, min_score)
            except Exception as exc:  # noqa: BLE001 - RAG must never break chat
                log.warning("RAG search failed, answering without context: %s", exc)
            if hits:
                log.info(
                    "RAG injected %d/%d documents (model=%s, query=%r)",
                    len(hits),
                    top_k,
                    model,
                    retrieval_query[:120],
                )
                body = {
                    **body,
                    "messages": _inject_context(messages, _format_context_block(hits)),
                }

    stream = bool(body.get("stream"))
    url = _upstream_url("/chat/completions")
    headers = _upstream_headers(stream)
    response_headers = _rag_headers(hits)

    if not stream:
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                upstream = await client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise StarletteHTTPException(
                502, f"The model server is unavailable: {exc}"
            ) from exc

        mapped = _upstream_error(upstream.status_code, upstream.text)
        if mapped is not None:
            return mapped

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers=response_headers,
        )

    client = httpx.AsyncClient(timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=CONNECT_TIMEOUT))
    try:
        upstream = await client.send(
            client.build_request("POST", url, json=body, headers=headers),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        raise StarletteHTTPException(
            502, f"The model server is unavailable: {exc}"
        ) from exc

    if upstream.status_code >= 400:
        payload = await upstream.aread()
        content_type = upstream.headers.get("content-type", "application/json")
        await upstream.aclose()
        await client.aclose()
        mapped = _upstream_error(upstream.status_code, payload.decode("utf-8", "replace"))
        if mapped is not None:
            return mapped
        return Response(
            content=payload,
            status_code=upstream.status_code,
            media_type=content_type,
            headers=response_headers,
        )

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    response_headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return StreamingResponse(
        relay(), media_type="text/event-stream", headers=response_headers
    )


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


@app.post("/v1/embeddings")
async def create_embeddings(request: Request, _: None = Depends(require_api_key)):
    body = await _read_json(request)

    input_value = body.get("input")
    if not isinstance(input_value, (str, list)) or not input_value:
        raise StarletteHTTPException(400, "'input' must be a string or a non-empty array")

    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            upstream = await client.post(f"{API_URL}/v1/embeddings", json=body)
    except httpx.HTTPError as exc:
        raise StarletteHTTPException(502, f"Embedding service is unavailable: {exc}") from exc

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


# --------------------------------------------------------------------------- #
# Health / root
# --------------------------------------------------------------------------- #


async def _probe(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url, headers=headers or {})
        return {
            "ok": response.status_code < 500 and response.status_code not in (401, 403),
            "status": response.status_code,
        }
    except Exception as exc:  # noqa: BLE001 - health must always answer
        return {"ok": False, "error": str(exc)}


@app.get("/health")
async def health():
    if UPSTREAM_BASE_URL:
        upstream = await _probe(_upstream_url("/models"), _upstream_headers())
    else:
        upstream = {"ok": False, "error": "UPSTREAM_BASE_URL is not set"}

    api = await _probe(f"{API_URL}/health")

    return {
        "status": "ok" if upstream["ok"] and api["ok"] else "degraded",
        "rag_enabled": ENABLE_RAG,
        "auth_required": bool(API_KEYS),
        "defaults": {"top_k": RAG_TOP_K, "min_score": RAG_MIN_SCORE},
        "dependencies": {
            "upstream": {"url": UPSTREAM_BASE_URL or None, **upstream},
            "api": {"url": API_URL, **api},
        },
    }


@app.get("/")
async def root():
    return {
        "name": "RAG Gateway",
        "version": app.version,
        "openai_base_url": "/v1",
        "endpoints": [
            "GET /v1/models",
            "GET /v1/models/{model}",
            "POST /v1/chat/completions",
            "POST /v1/embeddings",
            "GET /health",
        ],
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
