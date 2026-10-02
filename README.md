# radio-db

A database for **conversational voice stored in its natural form**. The original audio is the record. Who spoke when, what they said, and voice identity are time-aligned layers on top of it.

- Lossless, immutable, content-addressed audio (FLAC; lossy uploads kept byte-for-byte)
- Speaker diarization **and identification**: the same voice gets the same identity across recordings, and named voice profiles relabel past recordings
- English transcripts with word timestamps, full-text search, and search by voice
- Per-user data, 1-year retention with automatic purge
- Local storage for development; S3-compatible object storage for production

Design and research background: [architecture.md](architecture.md) (§0 requirements, §12 what v1 implements).

## Quick start (local)

Requires Python 3.10–3.13. An NVIDIA GPU is optional but strongly recommended.

```powershell
python -m venv .venv
.venv\Scripts\activate
# GPU build of PyTorch (skip for CPU-only)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[ml,dev]"
```

The first processing run downloads model weights (~0.6 GB) into `./data/models`. No Hugging Face token is needed.

```powershell
radiodb ingest meeting.wav                       # store + diarize + identify + transcribe
radiodb show <recording-id>                      # speaker-attributed transcript
radiodb speakers list                            # voices found so far
radiodb speakers enroll "Alice" alice_sample.wav # name a voice (relabels past recordings)
radiodb speakers rename <speaker-id> "Bob"       # or name an auto-detected speaker
radiodb speakers merge <keep-id> <drop-id>       # fix one person split in two
radiodb search "quarterly budget" --speaker alice
radiodb search-voice unknown_clip.wav            # who is this?
radiodb clip <recording-id> 63.5 71.0 -o out.wav
radiodb purge                                    # delete expired recordings (run daily)
radiodb serve                                    # HTTP API on http://127.0.0.1:8000/docs
```

All commands take `--user/-u` (default `local`, or set `RADIODB_USER`).

Supported input: WAV, FLAC, AIFF, OGG/Vorbis, Opus, MP3. (M4A/AAC needs converting first.)

## HTTP API

Interactive docs are at `/docs` once `radiodb serve` is running. Every request needs an `X-User-Id` header.

| Method & path | Purpose |
|---|---|
| `POST /recordings` (multipart `file`, `title?`, `process?`) | store audio; processing runs in the background |
| `GET /recordings`, `GET /recordings/{id}` | list / status (`pending → processing → done/failed`) |
| `GET /recordings/{id}/transcript` | segments with speaker, text, word timings |
| `GET /recordings/{id}/audio?start=&end=` | original file, or a WAV clip of a time range |
| `POST /recordings/{id}/process` | rebuild derived layers |
| `DELETE /recordings/{id}` | delete recording, its layers, and its blob if unshared |
| `GET /speakers` · `POST /speakers/enroll` · `PATCH /speakers/{id}` · `POST /speakers/merge` · `DELETE /speakers/{id}` | speaker identities |
| `GET /speakers/{id-or-name}/segments` | everything a speaker said |
| `GET /search?q=&speaker=` | full-text search |
| `POST /search/voice` (multipart `file`) | identify a voice and find where it appears |
| `GET /stats` | hours stored, speakers, status counts |

## Configuration

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `RADIODB_HOME` | `./data` | catalog, local blobs, model cache |
| `RADIODB_BLOB_BACKEND` | `local` | `local` or `s3` |
| `RADIODB_S3_BUCKET` / `RADIODB_S3_PREFIX` / `RADIODB_S3_ENDPOINT_URL` | — / `blobs/` / AWS | production object storage (`pip install -e ".[s3]"`; credentials via the standard AWS env/config) |
| `RADIODB_RETENTION_DAYS` | `365` | retention period |
| `RADIODB_DEVICE` | `auto` | `cuda`, `cpu` or `auto` |
| `RADIODB_ASR_MODEL` | `small.en` | any faster-whisper model (`base.en` for speed, `medium.en` for accuracy) |
| `RADIODB_SPEAKER_MATCH_THRESHOLD` | `0.60` | cosine similarity for "same person" |
| `RADIODB_DIARIZATION_THRESHOLD` | `0.6` | clustering distance; lower means more speakers |
| `RADIODB_MIN_SPEAKER_S` | `3.0` | less speech than this in a recording is folded into the nearest speaker |

## Tests

```powershell
pytest                 # fast suite with fake models (no weights needed)
pytest -m models -s    # end-to-end with real models on a real two-speaker clip
```

The model test expects `samples/sample.wav` (pyannote's 30 s tutorial clip):
`curl -L -o samples/sample.wav https://raw.githubusercontent.com/pyannote/pyannote-audio/main/tutorials/assets/sample.wav`

## Layout

```
radiodb/
  audio.py        decode, lossless canonical encoding, 16 kHz model input, clipping
  blobstore.py    content-addressed blob storage: local filesystem / S3
  catalog.py      SQLite catalog: recordings, speakers, segments, FTS5 index
  pipeline/
    interfaces.py model contracts (VAD, speaker embedder, transcriber)
    backends.py   Silero VAD, SpeechBrain ECAPA, faster-whisper
    diarize.py    windowed embeddings → clustering → turns → word alignment
    identify.py   matching voices to persistent speakers
  service.py      all operations (ingest, process, enroll, search, purge, …)
  api.py / cli.py thin HTTP and command-line wrappers
```

## Privacy

Voice is biometric data. Every recording carries an expiry date and is purged after it. Deleting a speaker removes their voice profile and every stored voice embedding. Users only ever see their own data. Production must replace the trusted `X-User-Id` header with real authentication.

## Not built yet

Prosody and emotion layers, sound-event search (CLAP), transcription-free semantic search, neural codec tokens, live stream capture. See architecture.md §10.
