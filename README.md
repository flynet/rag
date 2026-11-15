# RAG Gateway — OpenAI-совместимый API с RAG

Локальная RAG-система на базе Docker: автоматическая индексация документов, семантический поиск и **OpenAI-совместимый API** для агентов.

Модель генерации в стек не входит: шлюз ходит к **внешнему OpenAI-совместимому серверу** (vLLM, LM Studio, llama.cpp, облако), а агенты — только к шлюзу. Ключ модельного сервера остаётся внутри шлюза.

Веб-интерфейса для чата больше нет (Open WebUI удалён) — сервис предоставляет только API. Любой агент или клиент, умеющий работать с OpenAI API, может использовать его, указав `base_url`.

```
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_API_KEY=<ваш ключ из API_KEYS>
```

## 🚀 Возможности

- ✅ **OpenAI-compatible API**: `/v1/chat/completions` (stream + non-stream), `/v1/models`, `/v1/embeddings`
- ✅ **Автоматическая индексация** документов с мониторингом изменений
- ✅ **RAG по умолчанию**: контекст из базы знаний подмешивается в каждый запрос
- ✅ **Прозрачный passthrough**: `tools`, `tool_calls`, `response_format`, `temperature` и любые другие поля OpenAI уходят в LLM без изменений
- ✅ **Bearer-аутентификация** по статическим ключам
- ✅ **Ключ апстрима не покидает шлюз**: агент видит только свой ключ, `Authorization` наружу не пробрасывается
- ✅ **Управление RAG на уровне запроса** через заголовки
- ✅ **Один экземпляр embedding-модели** (fp16 на GPU): индексатор работает без torch и CUDA
- ✅ **Инкрементальная индексация**: неизменившиеся файлы не пересчитываются
- ✅ **Qdrant** векторная база данных

## 📁 Структура проекта

```
.
├── docker-compose.yaml          # Оркестрация всех сервисов
├── .env.example                 # Шаблон переменных окружения
├── data/
│   └── documents/               # 📂 Сюда помещайте документы для индексации
├── models/
│   └── hf-cache/                # 🤖 HF-кэш embedding-модели (создаётся автоматически)
├── embedder/                    # 📝 Индексатор: watchdog + HTTP-клиент (без torch)
│   ├── Dockerfile
│   ├── main.py                  # Watcher + worker, инкрементальная индексация
│   ├── embedder.py              # HTTP-клиент к api
│   ├── vector_db.py             # Qdrant client wrapper
│   └── requirements.txt
├── api/                         # 🔍 Эмбеддинги + поиск (единственный владелец модели)
│   ├── Dockerfile
│   ├── main.py                  # /search, /embed, /v1/embeddings
│   ├── embedder.py              # SentenceTransformer (fp16)
│   └── requirements.txt
├── gateway/                     # 🌐 Публичный OpenAI-совместимый шлюз
│   ├── Dockerfile
│   ├── main.py                  # /v1/chat/completions + инжекция RAG
│   └── requirements.txt
├── scripts/
│   └── smoke_test.py            # Проверка API после запуска
├── qdrant_data/                 # 💾 Векторная БД (создается автоматически)
```

## 🏗️ Архитектура

```
┌──────────────┐
│ Агент / LLM  │  (OpenAI SDK, LangChain, Continue, Cline, ...)
│   клиент     │
└──────┬───────┘
       │  OpenAI API: POST /v1/chat/completions
       v
┌─────────────────────────┐
│  Gateway      :8001     │  ← единственная публичная точка входа
│  • Bearer auth агента   │
│  • retrieval в API      │
│  • контекст в последнее │
│    user-сообщение       │
│  • подмена Authorization│
└────┬───────────────┬────┘
     │               │
     v               v
┌──────────────────────┐     ┌───────────────────┐
│ LLM-сервер (LAN)     │     │ API       :8000   │
│ vLLM / LM Studio /   │     │ 1× embedding      │
│ llama.cpp :8000/v1   │     │ модель (fp16)     │
└──────────────────────┘     └────┬─────────┬────┘
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

## 🐳 Сервисы Docker

### 1. **Gateway** — публичный OpenAI-совместимый API
- **Порт**: `8001` → `http://localhost:8001/v1`
- **Назначение**: единственная точка входа для агентов
- **Функции**:
  - проверка `Authorization: Bearer <key>` (ключ агента)
  - retrieval по последнему user-сообщению
  - инжекция найденных документов в текст последнего user-сообщения
  - проксирование в OpenAI-совместимый апстрим с полным passthrough тела запроса
  - собственная авторизация у апстрима (`UPSTREAM_API_KEY`); ключ агента наружу не уходит
  - `401/403` апстрима превращаются в `502 upstream_auth_failed`
  - SSE-стриминг без буферизации
- **Swagger**: http://localhost:8001/docs

### 2. **API** — эмбеддинги и поиск (единственный владелец модели)
- **Порт**: `8000` (только для отладки)
- **Endpoints**:
  - `POST /search` — семантический поиск (используется шлюзом)
  - `POST /embed` — внутренний батч-эндпоинт для индексатора
  - `POST /v1/embeddings` — OpenAI-совместимые эмбеддинги
  - `GET /collection/info` — статистика коллекции
  - `GET /collection/sample` — примеры документов
  - `GET /health` — проверка состояния, устройство и размерность
- **Модель**: загружается здесь и только здесь, `fp16` на GPU
- **Swagger**: http://localhost:8000/docs

### 3. **Embedder** — индексация
- **Назначение**: автоматическая индексация документов
- **Функции**:
  - мониторит `./data/documents` (polling каждую секунду)
  - режет текст на чанки
  - запрашивает векторы у `api` (`POST /embed`) и пишет их в Qdrant
  - переиндексирует только изменившиеся файлы (`mtime` + размер + конфиг чанков + модель)
- **GPU**: не требуется — образ `python:3.10-slim` без torch/CUDA, ~150 МБ RAM

### 4. **LLM-сервер** — внешний OpenAI-совместимый апстрим
- **Где**: другая машина в сети (или облако), в этот стек не входит
- **Что подходит**: vLLM, LM Studio, llama.cpp server, TGI, облачный OpenAI API
- **Как подключить**: `UPSTREAM_BASE_URL` в `.env` (с `/v1` на конце или без)
- **Авторизация**: `UPSTREAM_API_KEY` (пусто — заголовок не отправляется, как у LM Studio/llama.cpp)
- **Модели**: шлюз отдаёт список из `GET /v1/models` апстрима как есть

### 5. **Qdrant** — векторная БД
- **Порт**: `6333`
- **Dashboard**: http://localhost:6333/dashboard

## 🚀 Быстрый старт

### Предварительные требования

1. **Docker и Docker Compose**
2. **NVIDIA GPU** с CUDA 12.4+ и **NVIDIA Container Toolkit** (только для embeddings)
3. **OpenAI-совместимый сервер с LLM**, доступный по сети (vLLM / LM Studio / llama.cpp) — модель эмбеддингов скачается автоматически из HuggingFace в `./models/hf-cache` при первом запуске

### Установка

**1. Создайте конфиг и задайте ключ:**

```powershell
Copy-Item .env.example .env
# откройте .env и впишите API_KEYS=sk-local-change-me
# и UPSTREAM_BASE_URL=http://192.168.1.50:8000/v1
```

**2. Запустите сервисы:**

```bash
docker-compose up -d --build
```

**3. Убедитесь, что шлюз видит модель:**

```bash
curl http://localhost:8001/v1/models -H "Authorization: Bearer sk-local-change-me"
```

Если список пуст или приходит `502`, проверьте `UPSTREAM_BASE_URL` — это адрес вашего LLM-сервера, а не gateway.

**4. Проверьте API:**

```bash
python scripts/smoke_test.py --api-key sk-local-change-me
```

**5. Используйте из агента:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="sk-local-change-me")

response = client.chat.completions.create(
    model="qwen2.5-7b-instruct",   # id из GET /v1/models
    messages=[{"role": "user", "content": "Кто такой Пупкин?"}],
)
print(response.choices[0].message.content)
```

> Если обновляетесь с версии с Open WebUI: `docker-compose rm -sf open-webui` и удалите папку `webui_data/`.
> Если обновляетесь с версии с Ollama: удалите папку `ollama_data/` — сервис больше не используется.

## 🔑 Аутентификация

Ключи задаются списком в `.env`:

```env
API_KEYS=sk-local-change-me,sk-agent-2
```

- Проверяется заголовок `Authorization: Bearer <key>`.
- Сравнение — constant-time (`secrets.compare_digest`).
- Если `API_KEYS` пустой, аутентификация **отключена** (только для localhost).
- Ошибка возвращается в формате OpenAI:

```json
{"error": {"message": "Invalid or missing API key...", "type": "invalid_request_error", "param": null, "code": null}}
```

### Ключ модели

`UPSTREAM_API_KEY` — это ключ **модельного сервера**, а не клиента. Он живёт только в контейнере `gateway`:

- входящий `Authorization` агента никогда не пересылается на апстрим;
- если ключ апстрима пуст, заголовок `Authorization` не отправляется вовсе;
- если апстрим всё же ответил `401/403`, агент получает `502 upstream_auth_failed`, а не `401` — иначе он решил бы, что неверен его собственный ключ.

## 📖 API

### `GET /v1/models`

Прозрачное проксирование списка моделей апстрима.

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

Основной эндпоинт. Тело запроса — стандартный OpenAI Chat Completions, включая `tools`, `tool_choice`, `response_format`, `temperature`, `stream`, `stream_options`.

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer sk-local-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "Кто такой Пупкин?"}],
    "stream": false
  }'
```

**Стриминг** — обычный OpenAI SSE (`data: {...}` … `data: [DONE]`):

```python
stream = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "Кто такой Пупкин?"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

#### Управление RAG через заголовки

| Заголовок | Значение | Действие |
|---|---|---|
| `X-RAG` | `on` / `off` | включить/выключить retrieval для запроса |
| `X-RAG-TOP-K` | `1..20` | сколько чанков подмешать |
| `X-RAG-MIN-SCORE` | `0.0..1.0` | минимальный cosine score |

```bash
# ответить без базы знаний
curl ... -H "X-RAG: off" ...
```

#### Источники в ответе

Схема ответа остаётся строго OpenAI-совместимой, поэтому найденные документы отдаются в заголовках:

| Заголовок | Значение |
|---|---|
| `X-RAG-Hits` | количество подмешанных чанков |
| `X-RAG-Sources` | `base64url(JSON)` без паддинга: `[{"file","chunk","score"}]` |

Декодирование:

```python
import base64, json
raw = response.headers["X-RAG-Sources"]
sources = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
```

В OpenAI SDK заголовки доступны через `response.with_raw_response`:

```python
raw = client.chat.completions.with_raw_response.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "Кто такой Пупкин?"}],
)
print(raw.headers.get("X-RAG-Hits"))
print(raw.parse())
```

#### Когда retrieval не выполняется

Контекст подмешивается только для «свежего» user-сообщения — то есть когда последнее сообщение в `messages` имеет `role: "user"`. Внутри агентского tool-loop (последнее сообщение — `tool` или `assistant` с `tool_calls`) retrieval пропускается, чтобы не раздувать промпт на каждом шаге.

Если сервис `api` недоступен, шлюз **не ломает** чат: запрос уходит в LLM без контекста, в логе остаётся `WARNING`.

**Куда именно попадает контекст.** Блок `<knowledge_base>` дописывается в начало `content` последнего user-сообщения, а сам вопрос остаётся в конце. Отдельное `system`-сообщение не добавляется: префикс `system + история` не меняется, поэтому prompt caching апстрима продолжает работать, а chat-шаблоны (ChatML, Llama 3), которые требуют `system` только первым сообщением, не ломаются.

### `POST /v1/embeddings`

OpenAI-совместимые эмбеддинги на той же локальной модели.

```bash
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Authorization: Bearer sk-local-change-me" \
  -H "Content-Type: application/json" \
  -d '{"input": ["первый текст", "второй текст"]}'
```

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.01, -0.02, "..."]}],
  "model": "rag-embeddings",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

- `input`: строка или массив строк (массивы токенов не поддерживаются).
- `encoding_format`: `float` (по умолчанию) или `base64`.
- `usage` — приблизительная оценка (~4 символа на токен).

### `GET /health`

Публичный, без аутентификации. Проверяет доступность модельного сервера и `api`.

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

`status` = `degraded`, если хотя бы одна зависимость недоступна. Если `UPSTREAM_BASE_URL` пуст, в логе будет `CRITICAL`, `upstream` — `ok: false`, а чат-запросы вернут `503` вместо падения контейнера.

## ⚙️ Конфигурация

Переменные читаются из `.env` (см. `.env.example`).

| Переменная | По умолчанию | Описание |
|---|---|---|
| `API_KEYS` | — | список Bearer-ключей через запятую; пусто = без auth |
| `EMBEDDING_MODEL` | `google/embeddinggemma-300m` | модель эмбеддингов (одна на всю систему) |
| `EMBEDDING_DEVICE` | `cuda` | `cuda` или `cpu` |
| `EMBEDDING_DTYPE` | `float16` | `float16`, `bfloat16`, `float32`, `auto`; на CPU всегда fp32 |
| `EMBED_BATCH_SIZE` | `32` | текстов на один запрос эмбеддинга |
| `ENABLE_RAG` | `true` | глобальный выключатель RAG |
| `RAG_TOP_K` | `3` | чанков в контексте по умолчанию |
| `RAG_MIN_SCORE` | `0.5` | порог релевантности по умолчанию |
| `UPSTREAM_BASE_URL` | — | адрес OpenAI-совместимого сервера (`.../v1` или без) |
| `UPSTREAM_API_KEY` | — | ключ модельного сервера; пусто = без заголовка |
| `UPSTREAM_HEADERS` | — | доп. заголовки апстрима JSON-объектом |
| `UPSTREAM_TIMEOUT` | `600` | таймаут генерации, сек (включая стрим) |
| `LOG_LEVEL` | `INFO` | уровень логов |

Остальные настройки `api` и `embedder` задаются прямо в `docker-compose.yaml`: `QDRANT_HOST`, `COLLECTION_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `DEBOUNCE_SEC`.

## 🧠 Память и производительность

Раскладка рассчитана на одну машину с одной GPU (проверялось на RTX 4060 Laptop, 8 ГБ VRAM).

Раньше модель эмбеддингов загружалась дважды — в `embedder` и в `api`: два экземпляра весов, два torch-рантайма и два CUDA-контекста на одной карте. Теперь веса грузит только `api`, а `embedder` ходит к нему по HTTP и вообще не содержит torch.

| Компонент | Раньше | Сейчас |
|---|---|---|
| `embedder`: torch + CUDA | ~1.5 ГБ | — |
| `embedder`: веса модели | gemma-2-2b fp32 ≈ 10.4 ГБ | — |
| `api`: torch + CUDA | ~1.5 ГБ | ~1.5 ГБ |
| `api`: веса модели | 10.4 ГБ | embeddinggemma-300m fp16 ≈ 0.6 ГБ |
| `qdrant` | ~0.2 ГБ | ~0.2 ГБ |
| **Итого** | **≈ 26 ГБ** | **≈ 5 ГБ** |

Сама LLM теперь живёт на другом хосте, поэтому локальная GPU целиком занята эмбеддингами и больше не конкурирует с моделью генерации.

Что ещё сделано:

- **fp16 вместо fp32** — вдвое меньше весов и быстрее на тензорных ядрах. На CPU точность принудительно понижается до fp32: `fp16` там медленный и поддержан не всеми операми.
- **Нормализация векторов** (`EMBEDDING_NORMALIZE`, по умолчанию `true`) — стабильнее cosine-поиск.
- **Инкрементальная индексация** — в payload каждого чанка лежит подпись файла (`mtime`, `size`, `chunk_size`, `chunk_overlap`, `model`). При рестарте и по событиям watcher неизменившиеся файлы пропускаются.
- **Батчи** — тексты уходят в `api` порциями по `EMBED_BATCH_SIZE`, а не всем файлом сразу.
- **Лимиты памяти** — `mem_limit` у `embedder` (1 ГБ) и `gateway` (512 МБ), чтобы runaway-индексация не съела хост.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — меньше фрагментации VRAM.

Если нужно освободить GPU полностью, переведите эмбеддинги на CPU:

```env
EMBEDDING_DEVICE=cpu
EMBEDDING_DTYPE=float32
```

300M-модель на CPU даёт ~10–30 мс на чанк: для индексации и одиночных запросов этого достаточно, а GPU остаётся свободной.

### Смена модели embeddings

Модель задаётся одной переменной — её читают оба сервиса (`api` грузит веса, `embedder` использует значение как метку для инвалидации индекса):

```env
EMBEDDING_MODEL=google/embeddinggemma-300m
```

| Модель | dim | fp16 | Комментарий |
|---|---|---|---|
| `google/embeddinggemma-300m` | 768 | ≈ 0.6 ГБ | оптимум для 8 ГБ GPU |
| `intfloat/multilingual-e5-small` | 384 | ≈ 0.24 ГБ | самый быстрый вариант |
| `BAAI/bge-m3` | 1024 | ≈ 1.1 ГБ | лучшее качество, мультиязычность |

> При смене модели меняется размерность векторов. Коллекция Qdrant пересоздаётся автоматически (`recreating the collection` в логе), после чего все документы индексируются заново. Если хотите начать с чистого листа вручную — удалите `qdrant_data/` при остановленном стеке.

При первом запуске модель скачивается в `./models/hf-cache` (том `HF_HOME`), поэтому перезапуск контейнера не тянет её заново.

## 📥 Индексация документов

Скопируйте текстовые файлы в `data/documents`:

```powershell
Copy-Item .\mydocument.txt .\data\documents\
```

`embedder` обнаружит файл через ~1 секунду, порежет на чанки, запросит векторы у `api` и запишет их в Qdrant. Старые чанки файла заменяются, при удалении файла — удаляются из базы, а неизменившиеся файлы при рестарте пропускаются.

Логи:

```bash
docker-compose logs -f embedder
```

```
2026-11-15 13:14:33 INFO [embedder] Stable event for /data/documents/mydocument.txt
2026-11-15 13:14:35 INFO [embedder] Upserted 5 chunks for /data/documents/mydocument.txt
2026-11-15 13:14:40 INFO [embedder] Unchanged, skipping /data/documents/other.txt
```

## 🐛 Troubleshooting

```bash
docker-compose ps              # статус контейнеров
docker-compose logs -f         # все логи
docker-compose logs -f gateway
curl http://localhost:8001/health
python scripts/smoke_test.py --api-key <ключ>
```

**401 Unauthorized** — ключ не задан или не совпадает с `API_KEYS`; после правки `.env` нужен `docker-compose up -d` (пересоздание контейнера).

**`status: degraded`** — одна из зависимостей недоступна; смотрите блок `dependencies` в `/health`.

**`503 No model server is configured`** — пуст `UPSTREAM_BASE_URL`. Укажите адрес LLM-сервера в `.env` и пересоздайте контейнер gateway.

**`502 upstream_auth_failed`** — модельный сервер отклонил ключ **шлюза**. Проверьте `UPSTREAM_API_KEY` (и что сервер вообще ждёт Bearer). Ключ агента тут ни при чём, менять его не нужно.

**Пустые ответы без контекста** — проверьте `X-RAG-Hits` в ответе: `0` означает, что либо в коллекции нет документов (`/collection/info`), либо все результаты ниже `RAG_MIN_SCORE`.

**Модель не загружается (embeddings)** — проверьте доступ к HuggingFace и место на диске: `docker-compose logs api | Select-String "Loading model"`.

**`CUDA out of memory`** — модель эмбеддингов не помещается в VRAM. Поставьте `EMBEDDING_DEVICE=cpu`: LLM живёт на другом хосте, так что GPU нужна только для эмбеддингов.

**Индексация долго стартует** — полный проход выполняется только на пустой коллекции; дальше `embedder` пропускает файлы с совпадающей подписью (`mtime`/размер/конфиг чанков/модель).

**Ошибка размерности вектора** — коллекция создана под другую модель; удалите `qdrant_data/*` при остановленном стеке и перезапустите.

**Qdrant падает** — очистите данные:

```powershell
docker-compose down
Remove-Item -Path .\qdrant_data\* -Recurse -Force
docker-compose up -d
```

**WebUI: порт 3000 занят** — сервис удалён; если остался старый контейнер, выполните `docker-compose rm -sf open-webui`.

## 📊 Мониторинг

- Qdrant Dashboard: http://localhost:6333/dashboard
- API Swagger: http://localhost:8000/docs
- Gateway Swagger: http://localhost:8001/docs
- Gateway health: http://localhost:8001/health

## 🛠️ Разработка

```bash
docker-compose up -d --build      # пересборка после изменений
docker-compose restart gateway
```

Структура кода:

- `gateway/main.py` — `require_api_key`, `_rag_search`, `_inject_context`, `/v1/chat/completions`, `/v1/embeddings`
- `api/main.py` — `/search` (Qdrant `query_points`, с фолбэком на старый `search`), `/embed`, `/v1/embeddings`
- `api/embedder.py` — `Embedder`: загрузка модели в fp16, батчи, нормализация
- `embedder/main.py` — `DebounceHandler`, `ingestion_worker`, `file_signature`, `split_text_into_chunks`
- `embedder/embedder.py` — `RemoteEmbedder`: HTTP-клиент к `api` с ретраями
- `embedder/vector_db.py` — `VectorDB`: upsert с подписью файла, `is_up_to_date`, пересоздание коллекции при смене размерности

## 📝 TODO / Планы развития

### Переход на PostgreSQL с pgvector

**Зачем:**
- Снижение затрат на инфраструктуру (один сервис вместо отдельной Qdrant БД)
- Унификация хранения данных (метаданные + векторы в одной СУБД)
- Возможность использования стандартных PostgreSQL инструментов для backup/restore
- Поддержка сложных SQL-запросов для фильтрации

**Что нужно:**

1. **Добавить PostgreSQL контейнер в docker-compose.yaml:**
   ```yaml
   postgres:
     image: pgvector/pgvector:pg16
     environment:
       POSTGRES_DB: rag_db
       POSTGRES_USER: rag_user
       POSTGRES_PASSWORD: rag_password
     volumes:
       - ./postgres_data:/var/lib/postgresql/data
     ports:
       - "5432:5432"
   ```

2. **Обновить embedder/vector_db.py:**
   - Заменить `qdrant-client` на `psycopg2` / `asyncpg`
   - Использовать pgvector extension для векторного поиска
   - Создать таблицу с колонкой типа `vector(2304)` для embeddings
   - Использовать HNSW или IVFFlat индексы для быстрого поиска

3. **Схема таблицы:**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;

   CREATE TABLE documents (
     id BIGINT PRIMARY KEY,
     file_path TEXT NOT NULL,
     chunk_index INTEGER NOT NULL,
     content TEXT NOT NULL,
     embedding vector(2304),
     created_at TIMESTAMP DEFAULT NOW()
   );

   CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
   ```

4. **Обновить сервис `api`:**
   - Использовать SQL-запросы для поиска с `<=>` оператором (cosine distance)
   - Пример: `SELECT *, 1 - (embedding <=> $1) AS similarity FROM documents ORDER BY embedding <=> $1 LIMIT 5`

**Преимущества:**
- ✅ Меньше памяти (нет отдельного Qdrant процесса)
- ✅ Проще backup/восстановление
- ✅ Поддержка транзакций
- ✅ Интеграция с существующими PostgreSQL инструментами

**Недостатки:**
- ⚠️ Может быть медленнее для очень больших датасетов (миллионы векторов)
- ⚠️ Требует ручной настройки индексов для оптимизации

## 📚 Полезные ссылки

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [LM Studio local server](https://lmstudio.ai/docs/app/api)
- [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Sentence Transformers](https://www.sbert.net/)
- [pgvector Extension](https://github.com/pgvector/pgvector)

## 📄 Лицензия

MIT

## 🤝 Поддержка

При возникновении проблем:
1. Проверьте логи: `docker-compose logs`
2. Убедитесь что GPU доступен: `nvidia-smi`
3. Запустите smoke-тест: `python scripts/smoke_test.py`
