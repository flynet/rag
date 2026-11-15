# main.py
"""Indexing service: watch documents, ask `api` for vectors, store in Qdrant.

The embedding model is NOT loaded here. This service is a thin client, so a
single machine keeps one copy of the weights and one CUDA context (in `api`).
"""

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Any, Dict

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from embedder import RemoteEmbedder
from vector_db import VectorDB

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("embedder")

# Параметры (можно настроить через env)
API_URL = os.environ.get("API_URL", "http://api:8000")
WATCH_PATH = os.environ.get("WATCH_PATH", "/data/documents")
DEBOUNCE_SEC = float(os.environ.get("DEBOUNCE_SEC", "1.0"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1024"))   # символы
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "128"))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))

# Метка модели попадает в payload: при её смене индекс пересобирается
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "unknown")


# --- helper chunking ---
def split_text_into_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks


def file_signature(path: Path) -> Dict[str, Any]:
    """Всё, что влияет на результат индексации конкретного файла."""
    stat = path.stat()
    return {
        "mtime": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "model": EMBEDDING_MODEL,
    }


# --- Watcher with debounce ---
class DebounceHandler(FileSystemEventHandler):
    """
    Watchdog handler that debounces events and pushes stabilized files to asyncio.Queue.
    Watchdog events happen in a separate thread, so we use loop.call_soon_threadsafe to schedule
    the debounce resets in the asyncio loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, delay: float = DEBOUNCE_SEC):
        self.loop = loop
        self.queue = queue
        self.delay = delay
        # mapping Path -> asyncio.TimerHandle
        self.timers = {}

    def on_any_event(self, event):
        # Ignore directories
        if event.is_directory:
            return

        path = Path(event.src_path)
        # schedule / reset timer for this path in event loop
        self.loop.call_soon_threadsafe(self._reset_timer, path)

    def _reset_timer(self, path: Path):
        # cancel existing timer
        handle = self.timers.get(path)
        if handle:
            handle.cancel()

        # create new timer
        handle = self.loop.call_later(self.delay, lambda: asyncio.create_task(self._notify(path)))
        self.timers[path] = handle

    async def _notify(self, path: Path):
        # remove timer handle
        self.timers.pop(path, None)
        # push to queue (only if file exists)
        if path.exists():
            log.info("Stable event for %s", path)
            await self.queue.put({"file": str(path), "event": "modified"})
        else:
            # If file deleted - still notify so worker can remove from DB
            log.info("File disappeared (deleted?) %s", path)
            await self.queue.put({"file": str(path), "event": "deleted"})


# --- ingestion worker ---
async def ingestion_worker(queue: asyncio.Queue, embedder: RemoteEmbedder, vector_db: VectorDB):
    log.info("Worker started")

    while True:
        task = await queue.get()
        file_path = task.get("file")
        ev = task.get("event")

        try:
            path = Path(file_path)

            if ev == "deleted":
                vector_db.delete_by_file(file_path)
                log.info("Deleted vectors for %s", file_path)
                continue

            if not path.exists():
                log.warning("File not found: %s", file_path)
                continue

            try:
                signature = file_signature(path)
            except OSError as exc:
                log.warning("Cannot stat %s: %s", file_path, exc)
                continue

            # Инкрементальная индексация: неизменившиеся файлы не пересчитываем
            if vector_db.is_up_to_date(file_path, signature):
                log.info("Unchanged, skipping %s", file_path)
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except Exception as exc:
                log.warning("Error reading file %s: %s", file_path, exc)
                continue

            chunks = split_text_into_chunks(text)
            if not chunks:
                log.info("No chunks to process for %s", file_path)
                continue

            # Удаляем старые векторы, чтобы не остались чанки от прошлой версии файла
            try:
                vector_db.delete_by_file(file_path)
            except Exception as exc:
                log.warning("Error deleting old vectors for %s: %s", file_path, exc)

            try:
                vectors = await embedder.embed(chunks)
            except Exception as exc:
                log.error("Embedding error for %s: %s", file_path, exc)
                continue

            try:
                vector_db.upsert(file_path, chunks, vectors, meta=signature)
                log.info("Upserted %d chunks for %s", len(chunks), file_path)
            except Exception as exc:
                log.error("Upsert error for %s: %s", file_path, exc)
        finally:
            queue.task_done()


# --- main runner ---
async def main():
    # create queue
    queue = asyncio.Queue()

    # HTTP client to the embedding service (model lives there)
    embedder = RemoteEmbedder(API_URL, batch_size=EMBED_BATCH_SIZE)

    # connect to vector DB
    vector_db = VectorDB()

    # Wait for the embedding service and create/migrate the collection
    log.info("Waiting for embedding service at %s", API_URL)
    dim = await embedder.wait_ready()
    vector_db.ensure_collection(dim=dim)
    log.info("Collection '%s' ready (dim=%d)", vector_db.collection_name, dim)

    # Ensure watch path exists
    watch_path = Path(WATCH_PATH)
    watch_path.mkdir(parents=True, exist_ok=True)

    # Initial scan: queue everything, the worker skips unchanged files
    log.info("Scanning %s for existing files...", WATCH_PATH)
    file_count = 0
    for file_path in watch_path.rglob("*"):
        if file_path.is_file():
            await queue.put({"file": str(file_path), "event": "modified"})
            file_count += 1
    log.info("Queued %d existing files (unchanged ones will be skipped)", file_count)

    # start worker task(s)
    worker_task = asyncio.create_task(ingestion_worker(queue, embedder, vector_db))

    # start watcher in separate thread via watchdog Observer
    loop = asyncio.get_running_loop()
    event_handler = DebounceHandler(loop, queue, delay=DEBOUNCE_SEC)
    observer = PollingObserver()  # Use PollingObserver for Docker volumes
    observer.schedule(event_handler, WATCH_PATH, recursive=True)
    observer.start()
    log.info("Started watcher on %s (debounce %.1fs)", WATCH_PATH, DEBOUNCE_SEC)

    # trap signals to stop cleanly
    stop = asyncio.Event()

    def _signal_handler():
        log.info("Received stop signal")
        stop.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, _signal_handler)

    # wait until signal
    await stop.wait()

    # shutdown
    log.info("Stopping observer and worker...")
    observer.stop()
    observer.join(timeout=2)
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
