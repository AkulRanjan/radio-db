# radio-db

A database for **conversational voice stored in its natural form**. The original audio is the record. Who spoke when, what they said, *how* they said it, and voice identity are time-aligned layers on top of it.

- Lossless, immutable, content-addressed audio (FLAC; lossy uploads kept byte-for-byte)
- Speaker diarization **and identification**: the same voice gets the same identity across recordings, and named voice profiles relabel past recordings
- English transcripts with word timestamps
- Search by words, by meaning, by voice, by **how something was said** (pitch, pace, volume and expressiveness, each relative to the speaker's own normal), and by **emotion heard in the voice** (neutral / happy / angry / sad)
- `radiodb calibrate`: measure accuracy on your own labelled recordings and get recommended settings
- Per-user data, 1-year retention with automatic purge
- Local development on SQLite + local files; production on **PostgreSQL** + **S3-compatible object storage** (both implemented and tested)

Design and research background: [architecture.md](architecture.md) (§0 requirements, §12 what is implemented and how it was validated).

## Quick start (local)

Requires Python 3.10–3.13. An NVIDIA GPU is optional but strongly recommended: on an RTX 3060
Laptop, diarization runs at 0.02× and transcription at 0.03× real time (5 h of audio ≈ 15 min).

```powershell
python -m venv .venv
.venv\Scripts\activate
# GPU build of PyTorch (skip for CPU-only)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[ml,dev]"
```

The first processing run downloads model weights (~0.7 GB) into `./data/models`. No Hugging Face token is needed.

```powershell
radiodb ingest meeting.wav                       # store + diarize + identify + transcribe + layers
radiodb show <recording-id>                      # transcript with speakers and style tags
radiodb speakers list                            # voices found so far
radiodb speakers enroll "Alice" alice_sample.wav # name a voice (relabels past recordings)
radiodb speakers rename <speaker-id> "Bob"       # or name an auto-detected speaker
radiodb speakers merge <keep-id> <drop-id>       # fix one person split in two
radiodb speakers profile alice                   # how Alice usually sounds
radiodb search "quarterly budget" --speaker alice              # hybrid: words + meaning
radiodb search "where did they grow up" --mode semantic         # meaning only
radiodb search --speaker alice --pace fast --volume loud        # how it was said
radiodb search --emotion angry                                  # emotion heard in the voice
radiodb enrich --all                                            # backfill layers, no re-transcription
radiodb search-voice unknown_clip.wav            # who is this?
radiodb clip <recording-id> 63.5 71.0 -o out.wav
radiodb purge                                    # delete expired recordings now
radiodb serve                                    # HTTP API on http://127.0.0.1:8000/docs (+ inline worker)
radiodb ingest big.wav --queue                   # queue instead of processing now...
radiodb worker                                   # ...and process the queue (run one or more)
radiodb jobs                                     # queued / running / done / failed jobs
radiodb rttm <recording-id> -o rec.rttm          # export diarization (start point for labelling)
radiodb calibrate labelled/ -o report.json       # accuracy on your data + recommended settings
```

All commands take `--user/-u` (default `local`, or set `RADIODB_USER`).

Supported input: WAV, FLAC, AIFF, OGG/Vorbis, Opus, MP3. (M4A/AAC needs converting first.)

## HTTP API

Interactive docs are at `/docs` once `radiodb serve` is running. Every request except `/health`
must identify the user: an `X-User-Id` header in local development (`RADIODB_AUTH=header`), or
`Authorization: Bearer <JWT>` in production (`RADIODB_AUTH=jwt`, see below).

| Method & path | Purpose |
|---|---|
| `POST /recordings` (multipart `file`, `title?`, `process?`) | store audio and queue processing; returns the job |
| `GET /recordings`, `GET /recordings/{id}` | list / status (`pending → processing → done/failed`) |
| `GET /recordings/{id}/transcript` | segments with speaker, text, word timings |
| `GET /recordings/{id}/audio?start=&end=` | original file, or a WAV clip of a time range |
| `POST /recordings/{id}/process` | queue a full rebuild (diarization, ASR, layers) → 202 + job |
| `POST /recordings/{id}/enrich?layer=` | queue a rebuild of enrichment layers only → 202 + job |
| `GET /jobs?status=`, `GET /jobs/{id}` | job status: `queued → running → done / failed`, attempts, last error |
| `DELETE /recordings/{id}` | delete recording, its layers, and its blob if unshared |
| `GET /speakers` · `POST /speakers/enroll` · `PATCH /speakers/{id}` · `POST /speakers/merge` · `DELETE /speakers/{id}` | speaker identities |
| `GET /speakers/{id-or-name}/segments` | everything a speaker said |
| `GET /speakers/{id-or-name}/profile` | voice baseline: median pitch, range, rate, pauses |
| `GET /search?q=&mode=&speaker=&recording=&pitch=&pace=&volume=&expressiveness=&emotion=` | `mode`: `hybrid` (default) / `keyword` / `semantic`; style filters are relative to each speaker; `q` optional |
| `POST /search/voice` (multipart `file`) | identify a voice and find where it appears |
| `GET /stats` | hours stored, speakers, status counts |
| `GET /health` | unauthenticated readiness: catalog + blob store reachable (200 / 503), queue depth |

## Configuration

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `RADIODB_HOME` | `./data` | catalog, local blobs, model cache |
| `RADIODB_DATABASE_URL` | — (SQLite in `RADIODB_HOME`) | `postgresql://user:pass@host:port/db` for production (`pip install -e ".[postgres]"`) |
| `RADIODB_DATABASE_SCHEMA` | — (`public`) | Postgres schema to use (created if missing) |
| `RADIODB_AUTH` | `header` | `header` (trust `X-User-Id`; local only) or `jwt` |
| `RADIODB_JWT_SECRET` / `RADIODB_JWT_JWKS_URL` | — | HS256 shared secret, or your identity provider's JWKS URL (RS256/ES256) |
| `RADIODB_JWT_AUDIENCE` / `RADIODB_JWT_ISSUER` | — | checked when set |
| `RADIODB_JWT_USER_CLAIM` | `sub` | claim used as the user id |
| `RADIODB_MAX_UPLOAD_MB` | `1024` | larger uploads get 413 |
| `RADIODB_USER_QUOTA_HOURS` | `0` (unlimited) | audio hours per user; uploads beyond it get 403 |
| `RADIODB_INLINE_WORKER` | `0` (`radiodb serve` sets `1`) | run a job worker inside the API process |
| `RADIODB_JOB_MAX_ATTEMPTS` | `3` | attempts before a job is marked failed |
| `RADIODB_ASR_COMPUTE` | auto (`float16` GPU, `int8` CPU) | Whisper precision; CPU int8 vs float32 measured at 17.0 % vs 16.9 % WER, 2× faster |
| `RADIODB_ASR_ISOLATE` | `auto` | run Whisper in its own process (`auto` = on CPU; see architecture.md §12.10) |
| `RADIODB_BLOB_BACKEND` | `local` | `local` or `s3` |
| `RADIODB_S3_BUCKET` / `RADIODB_S3_PREFIX` / `RADIODB_S3_ENDPOINT_URL` | — / `blobs/` / AWS | production object storage (`pip install -e ".[s3]"`; credentials via the standard AWS env/config) |
| `RADIODB_RETENTION_DAYS` | `365` | retention period |
| `RADIODB_DEVICE` | `auto` | `cuda`, `cpu` or `auto` |
| `RADIODB_ASR_MODEL` | `small.en` | any faster-whisper model (`base.en` for speed, `medium.en` for accuracy) |
| `RADIODB_SPEAKER_MATCH_THRESHOLD` | `0.58` | cosine similarity for "same person" (calibrated on AMI) |
| `RADIODB_DIARIZATION_THRESHOLD` | `0.70` | clustering distance; lower means more speakers (calibrated on AMI) |
| `RADIODB_MIN_SPEAKER_S` | `20` | less speech than this (capped at 10 % of the recording's speech) is folded into the nearest speaker |
| `RADIODB_LAYERS` | `prosody,semantic,emotion` | enrichment layers to build (empty = none) |
| `RADIODB_EMOTION_MODEL` | `superb/hubert-base-superb-er` | Hugging Face id or a local folder with the downloaded model |
| `RADIODB_EMOTION_MIN_CONFIDENCE` | `0.5` | emotion labels below this probability are stored but not used for search/tags |

Enrichment layers are optional: if one can't load or run, the recording still completes and the
reason is shown in its `error`; fix it, then `radiodb enrich --layer <name>`.
| `RADIODB_TEXT_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | text encoder for semantic search |

## Calibrating on your own recordings

The defaults were calibrated on 4.1 h of real English meetings (AMI corpus; see architecture.md
§12.13) and checked on a telephone call. Your recordings may differ (channel, number of people,
call length), so confirm on your own data.

1. Pick 20–30 minutes of real conversations with recurring speakers, in several files.
2. Ingest them, then `radiodb rttm <id> -o name.rttm` next to each audio file and correct the
   speaker labels in a text editor. **Use the same name for the same person in every file**,
   since that is what calibrates speaker identification. Optionally add `name.txt` with the true transcript.
3. `radiodb calibrate that_folder/ -o report.json`

You get DER (missed speech, false alarm, speaker confusion) across a sweep of diarization
settings, speaker-identification EER and a recommended `RADIODB_SPEAKER_MATCH_THRESHOLD`, WER if
transcripts were given, the real-time factor on your hardware, and the env settings to use.

## Production backends

| | Local default | Production |
|---|---|---|
| Catalog | SQLite file | PostgreSQL: pooled connections, `tsvector` + GIN full-text search, `timestamptz`/`jsonb`/`bytea` |
| Audio | `./data/blobs` | Any S3-compatible store (AWS S3, R2, MinIO, …), SSE-AES256, key prefix |

```powershell
docker compose up -d   # local Postgres on 127.0.0.1:55432 (see docker-compose.yml)
$env:RADIODB_DATABASE_URL = "postgresql://radiodb:radiodb@127.0.0.1:55432/radiodb"
$env:RADIODB_BLOB_BACKEND = "s3"; $env:RADIODB_S3_BUCKET = "my-bucket"   # + AWS credentials
radiodb ingest meeting.wav
```

The catalog records a schema version, migrates older schemas forward automatically, and
refuses to open a schema newer than the code.

### Deploying

One image serves both roles (`Dockerfile`; CPU by default, `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126` for GPU):

```powershell
docker compose --profile app up -d --build   # postgres + API (:8000) + worker
```

Production checklist: `RADIODB_AUTH=jwt` with your identity provider's JWKS URL, audience and
issuer; `RADIODB_DATABASE_URL` → managed Postgres; `RADIODB_BLOB_BACKEND=s3` with bucket
encryption and a lifecycle rule as a retention backstop; a GPU worker (`RADIODB_DEVICE=cuda`);
TLS in front of the API; `/health` wired to your load balancer.

In production, run the API (`uvicorn radiodb.api:app`, inline worker off) and any number of
`radiodb worker` processes against the same catalog and blob store. Jobs are leased,
heartbeated, retried with backoff and taken over if a worker dies, and retention purging is
scheduled automatically.

## Tests

```powershell
pytest                 # fast suite with fake models (no weights needed); S3 runs against a local moto server
$env:RADIODB_TEST_DATABASE_URL = "postgresql://radiodb:radiodb@127.0.0.1:55432/radiodb"
pytest                 # ...and every catalog-backed test again on Postgres (isolated schema per test)
pytest -m models -s    # end-to-end with real models on a real two-speaker clip
```

The model test expects `samples/sample.wav` (pyannote's 30 s tutorial clip):
`curl -L -o samples/sample.wav https://raw.githubusercontent.com/pyannote/pyannote-audio/main/tutorials/assets/sample.wav`

## Layout

```
radiodb/
  audio.py        decode, lossless canonical encoding, 16 kHz model input, clipping
  blobstore.py    content-addressed blob storage: local filesystem / S3
  catalog/        recordings, speakers, segments, layers
    base.py       backend-neutral queries, transactions, schema version
    sqlite.py     local: SQLite + FTS5
    postgres.py   production: PostgreSQL + tsvector, connection pool
  pipeline/
    interfaces.py model contracts (VAD, speaker embedder, transcriber)
    backends.py   Silero VAD, SpeechBrain ECAPA, faster-whisper
    asr_process.py Whisper in an isolated child process (CPU)
    diarize.py    windowed embeddings → clustering → turns → word alignment
    identify.py   matching voices to persistent speakers
    layers.py     enrichment layers: prosody (Praat), semantic (sentence embeddings), emotion (HuBERT)
  query.py        speaker-relative style labels, rank fusion
  service.py      all operations (ingest, process, enrich, enroll, search, purge, …)
  worker.py       job worker: leases, heartbeats, retries, scheduled purge
  calibrate.py    DER / speaker-ID EER / WER on labelled data, parameter sweeps
  api.py / cli.py thin HTTP and command-line wrappers
```

## Privacy

Voice is biometric data. Every recording carries an expiry date and is purged after it. Deleting a speaker removes their voice profile, every stored voice embedding, and the pitch, loudness and emotion features of their speech. Users only ever see their own data. Production must replace the trusted `X-User-Id` header with real authentication.

## Not built yet

Sound-event search (CLAP), transcription-free semantic search (semantic search currently embeds the transcript), neural codec tokens, live stream capture, schema migrations (the version check exists; no migration has been needed yet). See architecture.md §10 and §12.
