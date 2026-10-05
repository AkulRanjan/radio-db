# radio-db — Architecture

> Status: implemented for local development (see §12). Sections 1–10 are the full target
> design; §12 says exactly which parts exist today and how production differs.

## 0. Requirements (decided)

| Question | Decision |
|---|---|
| Content | Conversations, English only |
| Volume | ~5 hours of audio per user |
| Retention | 1 year from upload, then automatic deletion |
| Environment | This repo is the **local** environment; production adds cloud object storage |
| Speaker identification | **Required**: who spoke when, consistent identities across recordings, named voice profiles |

## 1. What "natural form" means here

Most voice systems throw the audio away after running speech-to-text and keep only the transcript. That loses most of what makes speech speech: who is talking, tone, emotion, emphasis, pauses, hesitation, laughter, overlap, background sound.

radio-db treats **the audio as the record** and everything else as a derived, rebuildable index:

| Principle | Consequence |
|---|---|
| Audio is the source of truth | Original waveform is stored losslessly, immutably, content-addressed. |
| Text is *an* index, not *the* data | Transcripts are one of several parallel annotation layers, never a replacement. |
| Paralinguistics are first-class | Speaker, emotion, prosody (pitch/energy/rate), non-speech events are stored as time-aligned layers. |
| Everything is time-aligned | Every annotation points to `(recording_id, channel, start, end)` — the same model Lhotse uses for supervisions. |
| Derived data is versioned | Each layer records the model + version that produced it, so it can be re-computed when better models appear. |
| Search works on sound, not just words | Queries can be text, a voice clip, a speaker, or an emotion — retrieval runs over audio embeddings, not only over transcripts. |

## 2. Prior art this design borrows from

| Source | What we take |
|---|---|
| **Lhotse** (k2/icefall) — Recording / Supervision / Cut manifests | Core data model: recordings + time-aligned supervision segments + "cuts" as virtual views. |
| **Lance / LanceDB** | Columnar multimodal storage: blobs, embeddings and metadata in one table, fast random access, add derived columns without rewriting audio, built-in vector index + versioning. |
| **Neural audio codecs** — EnCodec (RVQ), SoundStream, Mimi (12.5 Hz, 8×2048 codebooks, ~1.1 kbps, WavLM-distilled semantic level) | Optional compact *token* representation of the audio for speech-LM use, dedup and cheap archiving tiers. |
| **SpeechRAG** (ICASSP 2025), **VoxRAG**, SpeechDPR | Transcription-free retrieval: embed speech passages directly and search them with text queries via an aligned text↔speech embedding space. |
| **CLAP** (LAION) | Joint text↔audio embeddings for "find a clip that sounds like ‹description›" and non-speech event search. |
| **pyannote.audio 3.x** + **ECAPA-TDNN** (SpeechBrain, 192-d) | Voice activity detection, diarization, and speaker embeddings for "who spoke when" and query-by-voice. |
| **emotion2vec**, **Prosody2Vec** | Emotion / prosody embeddings — the paralinguistic layer. |
| **Whisper / WhisperX** | Transcripts with word-level timestamps (text layer). |
| **AudioVec** (VGGish + TimescaleDB), Milvus/Zilliz audio-search examples | Validation that "embeddings + vector index + relational metadata" is the working pattern for audio similarity search. |

## 3. High-level architecture

```
                ┌──────────────────────────────────────────────────────────┐
  sources       │ files · radio/stream capture (Icecast/RTSP/SDR) · mic/API│
                └───────────────────────────┬──────────────────────────────┘
                                            ▼
┌──────────────────────────── INGEST ─────────────────────────────────────┐
│ 1. receive → 2. probe (ffprobe) → 3. normalise container, keep samples  │
│ 4. hash (sha256 of PCM) → dedupe → 5. write original to Blob Store      │
│ 6. create Recording row → 7. enqueue enrichment jobs                    │
└───────────────────────────┬─────────────────────────────────────────────┘
                            ▼
┌──────────────────── ENRICHMENT PIPELINE (async workers, GPU) ───────────┐
│  VAD ─► diarization ─► segmentation into utterances (Segments)          │
│   │                                                                     │
│   ├─► speaker embedding (ECAPA)      ─► speaker layer + clustering      │
│   ├─► ASR + word alignment (WhisperX)─► text layer                      │
│   ├─► prosody features (F0, energy,  ─► prosody layer                   │
│   │     speaking rate, pauses)                                          │
│   ├─► emotion (emotion2vec)          ─► affect layer                    │
│   ├─► sound events (CLAP / PANNs)    ─► event layer (music, laugh, …)   │
│   ├─► semantic speech embedding      ─► retrieval layer (SpeechRAG-style│
│   └─► codec tokens (Mimi/EnCodec)    ─► token layer (optional)          │
└───────────────────────────┬─────────────────────────────────────────────┘
                            ▼
┌──────────────────────────── STORAGE ────────────────────────────────────┐
│ Blob Store (S3/MinIO/local FS)   : original audio, immutable, CAS       │
│ Catalog (PostgreSQL)             : recordings, segments, speakers,      │
│                                    layers, provenance, ACLs             │
│ Vector/columnar store (LanceDB   : embeddings + per-segment features,   │
│   or pgvector for small installs)  ANN indexes                          │
│ Full-text index (Postgres FTS /  : transcripts, word timings            │
│   OpenSearch)                                                           │
└───────────────────────────┬─────────────────────────────────────────────┘
                            ▼
┌──────────────────────────── QUERY / API ────────────────────────────────┐
│ REST/gRPC API  ·  query planner (hybrid: filters + vectors + FTS)       │
│ playback service: byte-range / time-range streaming, on-the-fly clip    │
│ export: Lhotse manifests, HF datasets, WebDataset shards                │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4. Data model

The model is a superset of Lhotse's Recording/Supervision/Cut. Every derived row carries **provenance** (`producer`, `producer_version`, `created_at`, `params_hash`).

```
Recording            one physical audio asset (immutable)
  id (uuid) · sha256 · blob_uri · codec · sample_rate · channels
  duration_s · bit_depth · source · captured_at · station/channel meta
  license · consent · retention_policy

Segment              time span within a recording (utterance / turn)
  id · recording_id · channel · start_s · end_s
  speaker_id? · kind {speech, music, noise, silence, overlap}

Speaker              clustered or enrolled identity
  id · display_name? · centroid_embedding · enrolled (bool)

Layer rows           time-aligned annotations, one table per layer type
  transcript   (segment_id, text, language, words[{w, start, end, conf}])
  prosody      (segment_id, f0_mean, f0_std, f0_contour_ref, energy,
                speech_rate, pause_stats)
  affect       (segment_id, emotion_label, valence, arousal, embedding)
  event        (recording_id, start_s, end_s, label, score)
  tokens       (segment_id, codec, codec_version, n_codebooks, token_blob_uri)

Embedding            (owner_type, owner_id, space, model, dim, vector)
  spaces: speaker | semantic_speech | clap_audio | emotion

Cut                  saved virtual view = (recording, start, end, filters);
                     no audio copied — resolved at read time
```

## 5. Storage decisions

### 5.1 Audio blobs
- **Format:** FLAC (lossless, ~50–60 % of WAV) for anything ≤ 48 kHz; keep the original bytes as well if the upload was already lossy (MP3/AAC/Opus) — never transcode lossy → lossy.
- **Addressing:** `blobs/<sha256[0:2]>/<sha256>.flac`. Dedup is free; integrity can be re-verified.
- **Immutability:** blobs are write-once. Corrections happen in layers, never in audio.
- **Long streams (radio):** chunk continuous capture into fixed windows (e.g. 10 min) with overlap-free boundaries; a `Recording` = one chunk, and a `stream_id` + monotonic `captured_at` stitches them back together.
- **Tiering:**
  | Tier | Content | Use |
  |---|---|---|
  | Hot | FLAC originals + all layers | active search & playback |
  | Warm | Opus 32–64 kbps proxy for browsing | fast preview streaming |
  | Cold | FLAC in object-store archive class; codec tokens kept hot | cheap long-term retention; tokens still allow speech-LM use |

### 5.2 Why codec tokens are optional, not primary
Neural codec tokens (Mimi ≈ 1.1 kbps) are ~500× smaller than FLAC and are what modern speech LMs consume, but they are **lossy** and tied to a specific codec version. They are stored as a derived layer, never as the record.

### 5.3 Catalog vs. vector store
- PostgreSQL holds the relational truth (ids, time spans, provenance, permissions).
- LanceDB holds per-segment embeddings and bulky numeric features (F0 contours, frame-level features) — it handles blob + vector columns natively, supports adding a new embedding column without rewriting data, and versions tables.
- Small deployments can collapse both into **Postgres + pgvector**; the API does not change.

## 6. Retrieval

All queries resolve to a list of `(recording_id, start_s, end_s, score)` hits that are streamed back as audio clips.

| Query type | Mechanism |
|---|---|
| Keyword / phrase | Full-text on transcript layer → word timestamps give exact offsets. |
| Meaning ("segments discussing the flood") | Text query → text encoder of an aligned speech/text space (SpeechRAG-style) → ANN over `semantic_speech` embeddings. Works even where ASR failed. |
| Query-by-example (voice clip in) | Embed the clip → ANN over `semantic_speech` (same content) or `speaker` (same person). |
| Speaker ("everything this voice said") | ANN over `speaker` space, or join on `speaker_id`. |
| Sound description ("applause", "phone ringing") | CLAP text embedding → ANN over `clap_audio`. |
| Paralinguistic filter ("angry, fast speech by host") | Structured filters on affect/prosody columns combined with any of the above. |

**Hybrid planner:** apply cheap metadata filters first (time range, station, language, speaker) → run ANN / FTS in parallel → fuse with reciprocal-rank fusion → optionally rerank with a cross-modal model → return clips.

## 7. Pipeline mechanics

- **Job queue:** each `(recording, layer, model_version)` is an idempotent job (Celery/Arq/Temporal). Re-running with a new model version creates new layer rows; the old ones stay until garbage-collected.
- **DAG:** VAD → diarization → segmentation must run first; all other layers fan out per segment in parallel.
- **Backfill:** adding a new model = enqueue jobs for all recordings; readers choose which `producer_version` to use (default: latest complete).
- **Streaming mode (live radio):** a lightweight online path (VAD + streaming ASR + speaker embedding) writes provisional layers within seconds; the batch path overwrites them later with higher-quality results.

## 8. API sketch

```
POST   /recordings                    upload / register audio → {recording_id}
GET    /recordings/{id}               metadata + layer status
GET    /recordings/{id}/audio?start=&end=&format=   byte/time-range playback
GET    /recordings/{id}/segments      segments with requested layers
POST   /search                        {text? | audio? | speaker_id? | filters, top_k}
POST   /speakers/enroll               name a voice from example clips
POST   /cuts                          save a virtual clip set
GET    /export/lhotse?cut_id=         export manifests for training
```

## 9. Non-functional concerns

- **Privacy & consent:** voice is biometric data (GDPR Art. 9, BIPA etc.). Store consent/licence per recording, allow deleting a speaker's embeddings and segments, support redaction layers (mute spans at playback without altering the blob), encrypt blobs at rest.
- **Reproducibility:** every derived value is traceable to model + version + params.
- **Scale guide (rough):** 1 h of 16 kHz mono FLAC ≈ 50–60 MB; 1 h of 48 kHz stereo FLAC ≈ 0.5–0.7 GB; one 192-d float32 speaker embedding per 3–5 s segment ≈ 0.5 MB/h. Audio dominates storage; embeddings dominate RAM for ANN indexes.
- **Observability:** per-layer coverage (% of recordings enriched per model version), queue lag, ASR confidence distribution.

## 10. Suggested build order

1. Ingest + blob store + Recording catalog + playback by time range.
2. VAD + diarization + segments + speaker embeddings → "who spoke when" and query-by-voice.
3. ASR with word timestamps + full-text search.
4. Semantic speech embeddings + CLAP → transcription-free and sound search.
5. Prosody + emotion layers, hybrid planner.
6. Live stream capture path, codec-token layer, cold tiering, exports.

## 11. Open questions

All of the original open questions are answered in §0. Remaining ones:

- Production identity provider (the API currently trusts an `X-User-Id` header).
- Which cloud provider for production object storage (any S3-compatible store works as-is).
- Whether users must explicitly consent before a named voice profile is created for a third party.

## 12. Implementation status (what exists today)

Built so far: build-order steps 1–3 (§10), speaker identification, and the first parts of
steps 4–5: a **prosody layer** (how something was said) and **semantic search** over the
transcript (what it meant), all sized for the workload decided in §0.

**Not built yet:** CLAP sound-event search, *transcription-free* speech
retrieval (SpeechRAG-style), codec tokens, live capture. Semantic search currently embeds the
ASR transcript, not the audio. For clean English conversations that is the pragmatic choice,
but it inherits ASR errors.

### 12.1 Local vs. production

| Concern | Local (this repo, default) | Production |
|---|---|---|
| Audio blobs | `LocalBlobStore` → `./data/blobs/` | `S3BlobStore` (`RADIODB_BLOB_BACKEND=s3`, SSE-AES256), any S3-compatible service |
| Catalog | SQLite (WAL) + FTS5 | **PostgreSQL**: `RADIODB_DATABASE_URL`; psycopg 3 connection pool; generated `tsvector` + GIN index (English stemming, `ts_rank_cd`, `ts_headline`); `timestamptz`, `jsonb`, `bytea` |
| Vectors | float32 blobs in the catalog, brute-force cosine in NumPy | same, or pgvector if per-user data grows far past 5 h |
| Jobs | `jobs` table in the catalog; `radiodb serve` runs an inline worker thread | same table; N × `radiodb worker` processes on any hosts (§12.10) |
| Auth | `X-User-Id` header, trusted | verified token → user id at the gateway |
| Retention | `radiodb purge` | the same command on a daily scheduler, plus an S3 lifecycle rule as a backstop |

Both catalogs share one implementation (`catalog/base.py`): SQL is written once and each dialect
supplies only the connection, schema, placeholder style and full-text search. In Postgres a
transaction pins one pooled connection to the calling thread, with nested transactions as
savepoints. Both backends record `schema_version` in `radiodb_meta` and refuse to open a
schema that is newer or older than the code. The test suite runs every catalog-backed test on
both backends.

### 12.2 Models and tools (all non-gated, no Hugging Face token)

| Layer | Model / tool | Notes |
|---|---|---|
| VAD | Silero VAD | speech regions |
| Speaker embeddings | SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`, 192-d) | diarization windows, per-segment voiceprints, enrollment |
| ASR | faster-whisper `small.en` (`RADIODB_ASR_MODEL`) | English-only, word timestamps; float16 on GPU, int8 on CPU |
| Prosody | Praat via `praat-parselmouth` | F0, intensity; rate and pauses come from ASR word timings |
| Semantic | `sentence-transformers/all-MiniLM-L6-v2` (`RADIODB_TEXT_MODEL`) | 384-d, ~90 MB, CPU-friendly |
| Emotion | `superb/hubert-base-superb-er` (`RADIODB_EMOTION_MODEL`) | Apache-2.0, 378 MB, 4 classes (IEMOCAP); SUPERB ~65 % accuracy |

Models load **lazily, one at a time**, so a semantic search query only loads MiniLM and not
Whisper or ECAPA. pyannote's diarization pipeline handles overlapping speech better, but it is
gated behind an HF token and licence acceptance. Swapping it in only means implementing the
`diarize()` contract (turns + per-cluster centroids).

### 12.3 Processing flow per recording

```
blob ──decode──► mono 16 kHz
   ├─ Silero VAD ─► speech spans ─► 1.5 s windows (0.75 s hop) ─► ECAPA ─► agglomerative
   │                                    clustering (cosine, avg linkage, thr 0.6) ─► fold clusters with
   │                                    < 3 s of speech into nearest large cluster ─► turns + centroids
   ├─ faster-whisper ─► words with timestamps
   └─ words × turns ─► each word to the turn it overlaps most (zero-length words at a boundary
                       stay with the previous word's speaker) ─► segments
                       (split on speaker change or >1 s pause; voiced turns with no words kept)
segments ─► ECAPA voiceprint per segment (first 30 s)
clusters ─► identification against the user's speakers (§12.4)
segments ─► enrichment layers (§12.5): prosody, semantic
```

### 12.4 Speaker identification

- **Per-user speaker registry.** A `Speaker` has a centroid (L2-normalised ECAPA embedding),
  an optional name, and an `enrolled` flag.
- **Matching.** Each cluster centroid in a new recording (longest talker first) is compared by
  cosine similarity with all of the user's speakers. Score ≥ `RADIODB_SPEAKER_MATCH_THRESHOLD`
  (default 0.60) → same person. Otherwise a new, unnamed speaker is created. So the same voice
  gets the same id across recordings without anyone naming it.
- **Learning.** Unnamed speakers' centroids are refined with a running mean as they reappear.
  Enrolled profiles stay anchored to the user's own samples so they can't drift.
- **Enrollment.** `enroll(name, clips)` builds a voiceprint from speech-only audio (VAD,
  ≤10 s chunks averaged), then **retroactively** relabels every unnamed speaker that matches
  it, so past recordings pick up the name.
- **Corrections.** `rename`, `merge` (fixes one person split into two clusters) and `delete`.
  Delete removes the profile, every per-segment voice embedding *and* the prosody features of
  that speaker's segments (biometric erasure). The transcript text stays, unattributed.
- **Query by voice.** A clip → voiceprint → ranked speakers plus the closest segments.
- **Speaker profile.** Median pitch, pitch range, articulation rate and pause rate: how the
  person usually sounds, and the baseline their style labels are measured against.

### 12.5 Enrichment layers

Layers live in `segment_layers (segment_id, layer, producer, data JSON, vector)`, one row per
segment per layer. That makes them:

- **Backfillable.** `radiodb enrich --all [--layer X]` builds a new or upgraded layer on
  existing recordings from the stored audio + segments. ASR and diarization don't re-run, and
  segment ids are untouched.
- **Traceable.** Each row and the recording's `producers.layers` record the tool and version.
- **Cleaned up.** Rows cascade with their segments; reprocessing rebuilds them.
- **Optional.** A layer that fails to load (model can't be downloaded) or to run (out of memory)
  never fails the recording. Diarization, identification, the transcript and the other layers
  are kept, the failure is recorded in `producers.layer_errors` and the recording's `error`, and
  `radiodb enrich --layer X` backfills it once fixed. This came out of the emotion work: on this
  machine the emotion model's hub download failed, and with it every recording failed.

**Prosody** (per segment): `f0_median_hz`, `f0_mean_st` / `f0_std_st` / `f0_range_st`
(semitones re 100 Hz, p5–p95 range), `voiced_frac`, `intensity_db` (energy-averaged),
`speech_rate_sps` and `articulation_rate_sps` (syllables/s with and without pauses; syllables
estimated from vowel groups), `n_pauses`, `pause_total_s`, `pause_max_s` (gaps ≥ 0.25 s).

**Speaker-relative style labels** are computed at query time from prosody. Raw numbers aren't
searchable ("what is a high pitch?") and vary hugely between people, so each segment is
compared with *that speaker's own* baseline: median and MAD-scaled robust z. |z| ≥ 1 gets a
label, otherwise `typical`. At least 5 segments are needed before a baseline is trusted.

| Dimension | Feature | Baseline scope | Labels |
|---|---|---|---|
| pitch | `f0_mean_st` | speaker, all recordings | low / typical / high |
| pace | `articulation_rate_sps` | speaker, all recordings | slow / typical / fast |
| volume | `intensity_db` | speaker **within one recording** (mic gain varies) | quiet / typical / loud |
| expressiveness | `f0_range_st` | speaker, all recordings | flat / typical / animated |

**Semantic**: segment vector = normalise(embed(own text) + 0.35 × embed(previous segment's
last 25 words)). Short replies keep some conversational context but can't outrank the turn
that actually says the thing. Segments under 3 words are skipped.

### 12.6 Search

One entry point: `search(q?, mode, speaker?, recording?, style?)`.

- **keyword**: FTS5 with Porter stemming, all words must match, BM25-ranked, highlighted snippets.
- **semantic**: cosine over semantic vectors; keep hits ≥ max(0.15, best − 0.15).
  On MiniLM, paraphrases score around 0.2 and unrelated text 0–0.1, so a fixed 0.25 floor
  rejected correct matches. Measured, see §12.9.
- **hybrid** (default): keyword retrieval with *any* word matching (BM25 still favours more
  matches) plus semantic, merged with reciprocal-rank fusion (k = 60). Any-word matters: with
  all-words matching, a query mixing an exact word and a concept got no keyword hits at all.
- **No `q`**: filter only, in time order, e.g. "every segment where Alice was unusually fast and loud".

### 12.7 Sizing at the decided workload

| Item | Per user (5 h) |
|---|---|
| FLAC audio | ~0.3 GB at 16 kHz mono, up to ~2 GB at 48 kHz stereo |
| Segments | roughly 1,500–3,000 |
| Voice embeddings | ~2 MB (768 B each) |
| Semantic vectors | ~4 MB (1.5 KB each) |
| Processing time | not yet measured at 5 h scale (see §12.9) |

At this scale exact brute-force cosine search over a user's segments, and per-request style
baselines over all their segments, take milliseconds, so an ANN index or materialised
baselines would add complexity for no gain. Revisit if a single user's data gets past ~100 h.

### 12.8 Storage rules as implemented

- WAV/AIFF/FLAC input → re-encoded to FLAC at the source bit depth (bit-exact; tested).
- MP3/Vorbis/Opus input → original bytes kept untouched (no lossy→lossless bloat, no generation loss).
- Blob key = SHA-256 of the stored bytes. Re-uploads are deduplicated per user. A blob shared by
  several users is deleted only when its last recording goes.
- `expires_at = created_at + 365 days` (`RADIODB_RETENTION_DAYS`). `radiodb purge` deletes rows, cascades segments and layers, removes unshared blobs.
- Every recording records the models that built its layers (`producers`) so it can be reprocessed when models improve.

### 12.9 Validation so far

Measured on pyannote's 30 s two-speaker telephone clip (`samples/sample.wav` + reference RTTM),
CPU only (the CUDA PyTorch wheel has not been installed yet):

| Check | Result |
|---|---|
| Speakers found | 2 of 2. Before small-cluster folding it was 6, which is why folding exists |
| Frame-level speaker accuracy vs. reference | 87 % (fast turn-taking with overlap; errors sit at turn boundaries) |
| Transcript (Whisper `small.en`) | correct on manual reading, incl. "I heard a beep" (regression-tested) |
| Voice ID after enrolling from one segment | same speaker 0.78, other speaker 0.50 (threshold 0.60) |
| Semantic: "where did they grow up" | top 2 = both "originally from Chicago" turns (cosine 0.21) |
| Prosody / style | plausible: "Hello? Hello? Oh, hello." tagged high + loud; baselines only ~5 segments each here |
| Wall time, 30 s clip incl. model load | ~15 s on CPU |
| Production stack: real models + PostgreSQL 16 catalog + S3 (moto server over HTTP) | same transcript, speakers and search results as SQLite; one FLAC object in the bucket; purge deletes it |
| Concurrency | 12 parallel ingest/search requests across 8 threads on the pooled Postgres catalog |

Diarization parameters were swept on this one clip (window 1.5/2/3 s × linkage × threshold
0.5–0.9 × min-speaker 2/3/5 s). With 1.5 s windows the result was stable for thresholds 0.6–0.7 and
min-speaker 2–5 s. **One 30 s clip is not a calibration set.** Before production, label 20–30
minutes of real user conversations and re-check `DIARIZATION_THRESHOLD`,
`SPEAKER_MATCH_THRESHOLD`, the semantic cutoffs and GPU throughput at 5 h scale.

**Gotcha found and fixed: torch vs. CTranslate2 floating-point state.** After lazy model
loading went in, transcripts got worse ("I heard a beep" → "I heard of deep"). Bisection showed
that batched ECAPA inference (torch, oneDNN convolutions) switches the CPU to
flush-denormals-to-zero and leaves it on. CTranslate2's int8 CPU kernels then compute
slightly differently in the same process, and borderline beam-search choices flip.
`torch.set_flush_denormal(False)` before every Whisper call restores the exact output. The real-model test
asserts the phrase, so this can't silently come back. The same risk applies to any future
model mixing torch and CTranslate2 (or other native libraries) in one process.

### 12.10 Job queue and workers

Processing is queued work, not request-time work. The queue is a `jobs` table in the catalog
itself, so there is no Redis or broker to run or keep consistent with the data.

| Concern | How |
|---|---|
| Claiming | One statement: `UPDATE jobs ... WHERE id = (SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *` on Postgres; the same `UPDATE ... RETURNING` is atomic on SQLite |
| Dedupe | Partial unique index: at most one queued/running job per (recording, kind); re-enqueueing returns the pending job |
| Crash recovery | Claimed jobs carry a lease (`locked_until`, default 10 min). A heartbeat thread extends it while the job runs; if the worker dies, another worker re-claims after expiry. A stale owner can no longer complete or fail the job |
| Retries | Transient errors: exponential backoff (30 s × 2ⁿ), up to `RADIODB_JOB_MAX_ATTEMPTS` (3). Permanent errors (bad audio, missing recording, invalid request) fail at once. The recording shows `queued` with the last error while a retry is pending |
| Retention | A singleton `purge` job runs `purge_expired()` and reschedules itself 24 h later, so no external cron is needed |
| Deletion | Jobs reference their recording `ON DELETE CASCADE` |
| GPU | One inference job at a time per worker process; scale with more processes/GPUs |

Kinds: `process` (diarization, identification, ASR, layers), `enrich` (layers only), `purge`.
API: `POST /recordings` and `/process` and `/enrich` return a job (202), polled via `GET /jobs/{id}`.
CLI: `radiodb worker [--once] [--kind ...]`, `radiodb jobs`, `radiodb ingest --queue`.

Validated on both backends: idempotent enqueue, backoff retry then success, give-up after max
attempts, permanent errors not retried, crashed-worker takeover (attempts 2, stale owner
rejected), heartbeat keeping a job past its original lease, 4 concurrent workers × 8 jobs with
each job run exactly once, purge rescheduling, cascade on delete, v1→v2 schema migration, and
the API's inline worker. Real models: a recording queued with `ingest --queue` was processed by
a separate `radiodb worker` process on Postgres.

**ASR process isolation (CPU).** The worker exposed a nondeterministic version of the
torch/CTranslate2 floating-point bug (§12.9). Both libraries load the same OpenMP runtime, and
oneDNN leaves flush-denormals-to-zero on in shared worker threads. In a long-lived worker the
real clip came out degraded in **5/5** runs even with the main-thread reset. On CPU, Whisper now
runs in a dedicated child process that never imports torch (`RADIODB_ASR_ISOLATE=auto`): **5/5**
correct, with no measurable speed cost (ASR 4.4 s vs 4.7 s on the 30 s clip). On GPU, Whisper runs
in-process as before. A real-model regression test covers the long-lived worker case.

**Performance.** On the RTX 3060 Laptop GPU (CUDA torch 2.14 + CTranslate2 float16), measured
over 4.1 h of AMI meetings: diarization embedding **0.02×** and ASR **0.03×** real time, so a
user's 5 h of audio takes about 15 minutes. On this laptop's CPU, diarization varied between 0.2× and
1.3× real time with machine load, so CPU-only deployments are only viable for small volumes.
On Windows, CTranslate2 finds cuBLAS/cuDNN through torch's bundled DLLs (`ensure_cuda_dlls`,
registered without importing torch so the isolated ASR process stays torch-free).

### 12.11 Emotion layer

**Model choice.** Constraints: commercial-use licence, fits next to the other models in a few
GB of RAM, no extra framework. `superb/hubert-base-superb-er` (Apache-2.0, 378 MB, plain
`transformers`) was chosen. Rejected: audeering's dimensional model (CC-BY-NC-SA, non-commercial)
and emotion2vec+ (1.1 GB, unclear licence). Four classes (neutral/happy/angry/sad) at ~65 %
benchmark accuracy means **labels are hints**: every segment stores the full probability
vector, and a label only drives search, tags and profiles at ≥ `RADIODB_EMOTION_MIN_CONFIDENCE` (0.5).

**Mechanics.** Segments under 1 s are skipped. Longer ones are scored in windows of ≤10 s, and
probabilities are averaged weighted by window length. The emotion is from the *voice*, not the
words. It is voice-derived data, so deleting a speaker deletes it. Speaker profiles report each
person's confident emotion mix.

**On the real clip** (calm, friendly phone call): 5 of 6 scored turns neutral (0.65–0.75), the
cheerful closing question happy (0.56), and "Hello? Hello?" split neutral 0.49 / happy 0.47, so
it is correctly left untagged. The full pipeline with all layers took 39 s including model
loads on CPU with ~2 GB RAM free.

### 12.12 Calibration (`radiodb calibrate`)

Input: audio + same-named RTTM (+ optional `.txt`), speaker names consistent across files.
`radiodb rttm <id>` exports radio-db's own diarization as a labelling starting point.

| Measures | How |
|---|---|
| Diarization | NIST-style DER on a 10 ms grid, Hungarian speaker mapping, 0.25 s collar, overlap counted per speaker; split into miss / false alarm / confusion. Sweeps threshold × min-speaker; window embeddings are computed once per file, so the sweep is cheap. Recommends the **middle of the near-best plateau** (within 0.5 % DER), not its edge |
| Identification | One voiceprint per (file, speaker) from the parts of their speech with nobody else talking; same name across files = genuine pair, else impostor. EER, threshold at 1 % false accepts, and a recommended threshold (gap midpoint if separable; else the stricter of EER and 1 %-FAR) |
| ASR | WER after lowercasing/stripping punctuation |
| Speed | real-time factor of diarization embedding and ASR on this hardware |

Smoke test on the real clip split into two halves (real models): best DER 6.5 % vs 27.3 % for the
current defaults; identification separable (same speaker 0.68–0.79, different 0.31–0.35 → 0.517
recommended); diarization at 0.19× real time. That is 30 s of audio and 2 + 4 pairs, which
**proves the tool, not the settings**. Defaults are unchanged until it runs on real user data.
Building it surfaced two fixes: voiceprints had skipped any segment with *any* overlap (now only
the overlapping parts are cut), and a separable score set had put the threshold on the lowest
genuine score (now the gap midpoint).

### 12.13 Calibration on real conversations (AMI) and the defaults

Data: 8 AMI meetings (ES2004a–d, IS1009a–d; CC-BY-4.0), Mix-Headset audio, **4.1 h**, 4 speakers
each, the same people across each series. Reference speaker labels come from pyannote's AMI setup
(`only_words` RTTMs); reference words come from the AMI manual annotations (fillers removed from
both sides). Run: `radiodb calibrate` on the GPU.

DER (collar 0.25 s) by threshold × minimum speech per speaker:

| thr \ min s | 3 | 10 | 20 | 30 | 60 | 120 |
|---|---|---|---|---|---|---|
| 0.60 (old default) | **36.1 %** | 25.3 % | 18.8 % | 17.2 % | 16.6 % | 19.3 % |
| 0.70 (new default) | 20.0 % | 15.9 % | **14.5 %** | 14.3 % | 15.1 % | 18.0 % |
| 0.75 | 17.5 % | 14.7 % | 14.3 % | 14.3 % | 15.1 % | 18.0 % |
| 0.80 | 15.6 % | 14.6 % | 14.2 % | 14.4 % | 14.7 % | 17.4 % |
| 0.85 | 22.1 % | 21.9 % | 21.6 % | 21.5 % | 21.5 % | 24.5 % |
| 0.90 | 31.3 % | 31.3 % | 31.3 % | 31.4 % | 31.4 % | 33.5 % |

- **Old defaults (0.6 / 3 s) were badly tuned for real recordings: 36.1 % DER**, e.g. 43 "speakers"
  found in a 4-person meeting. They had been set on one 30 s phone clip.
- A broad plateau (0.7–0.8 × 10–30 s ≈ 14.2–15.1 %) shows the choice is robust. At the best point
  most of the error is **missed speech (12.2 %)**, mainly overlapping talk that a one-speaker-at-a-time
  hypothesis can't cover. Confusion is only 1.6 %, and the speaker count is off by 0.12 on average.
- The tool's own pick was 0.75 / 20 s, but **0.75 merged the two speakers of the narrowband
  telephone clip**, where different voices look more alike. **0.70 / 20 s** costs 0.2 points on AMI
  and handles both, so it is the default.
- A fixed 20 s minimum would delete brief participants in short calls, so the minimum is
  `min(20 s, 10 % of the recording's speech)`. Unchanged on AMI, ~2.2 s on the 30 s call.
- **Identification:** EER **0 %** over 48 same-person / 448 different-person pairs (same person
  0.78–0.97, different ≤ 0.39) → default threshold **0.58** (gap midpoint). On the telephone clip the
  other speaker scores 0.50: a smaller but still clean margin. Phone-only deployments should
  calibrate.
- **ASR:** Whisper `small.en` WER **24.1 %** over 38,022 words (17.9–33.3 % per meeting): far-field-ish
  meeting audio with overlap, a hard case. Use `medium.en` if accuracy matters more than speed.

Two scaling bugs showed up only on hour-long real audio and are fixed:
1. **GPU out of memory (12 GB requested on a 6 GB card).** ECAPA batched 32 clips padded to the
   longest. The embedder now sorts by length, caps padded audio per batch (120 s), and splits
   clips over 30 s into averaged pieces. Peak 1.5 GiB for a 200 s clip plus 300 windows.
2. **Diarization post-processing was O(clusters² × windows)** (minutes per setting on a 39-min
   meeting). Small-cluster absorption now keeps durations and centroid sums incrementally: 0.03–0.7 s.

### 12.14 API security and limits

| Concern | Implementation |
|---|---|
| Authentication | `RADIODB_AUTH=jwt`: bearer tokens verified with a shared secret (HS256) or the identity provider's JWKS (RS256/ES256), `exp` required, `aud`/`iss` checked when set, user = configurable claim. Tested against forged signatures, expired tokens, wrong audience/issuer, `alg=none`, unknown key id and RS→HS algorithm confusion. `header` mode (trust `X-User-Id`) is for local development and logs a warning |
| Isolation | every query is scoped to the authenticated user id |
| Health | `GET /health` (no auth): catalog query + blob store write/`HeadBucket`, 200 or 503, queue depth |
| Abuse limits | uploads streamed with a cap (`RADIODB_MAX_UPLOAD_MB`, 413); per-user audio quota (`RADIODB_USER_QUOTA_HOURS`, 403); duplicates don't count twice |
| Deployment | `Dockerfile` (one image; API default CMD, `radiodb worker` for workers; non-root; healthcheck; CPU or CUDA torch via build arg), `docker compose --profile app`, GitHub Actions CI running the suite on SQLite + Postgres + S3 (moto) |

### 12.15 Container deployment, verified

The CPU image (2.7 GB) was built and run as the compose stack (Postgres + API + worker) in JWT mode:
`/health` 200 (catalog + blobs ok); no token or the dev header alone → 401; an upload with a
valid token was queued, processed by the **worker container** and read back through the API
(transcript, speakers, emotion, semantic search). Fixes from that run: the image healthcheck
marked worker containers unhealthy (disabled for workers in compose); the API host port is
configurable (`RADIODB_API_PORT`; 8000 was taken on this machine); dependencies got their own
Docker layer so code changes rebuild in seconds.

**Whisper precision on CPU.** In the Linux container, int8 transcribed the test clip's "heard a
beep" as "heard of deep" with no torch in the process, so this is quantization, not the
OpenMP interference of §12.10. Measured on 10 min of AMI (IS1009b) in the container: int8
**17.0 %** WER at 0.08× real time vs float32 **16.9 %** at 0.16×. int8 stays the CPU default;
`RADIODB_ASR_COMPUTE=float32` is available. The real-model regression test's exact-phrase check
is therefore specific to the Windows/int8 numerics it was written on.

## References

- Lhotse — https://github.com/lhotse-speech/lhotse
- Lance format — https://github.com/lance-format/lance · LanceDB — https://www.lancedb.com/
- Défossez et al., *High Fidelity Neural Audio Compression* (EnCodec), 2022
- Kyutai, *Neural audio codecs explainer* (Mimi) — https://kyutai.org/codec-explainer/
- *Speech Retrieval-Augmented Generation without ASR* (SpeechRAG), ICASSP 2025 — https://arxiv.org/abs/2412.16500
- *VoxRAG: Transcription-Free RAG in Spoken QA* — https://arxiv.org/html/2505.17326v1
- Wu et al., *Large-scale Contrastive Language-Audio Pretraining* (LAION-CLAP) — https://arxiv.org/abs/2211.06687 · https://github.com/LAION-AI/CLAP
- *emotion2vec* (ACL 2024) — https://arxiv.org/abs/2312.15185 · https://github.com/ddlBoJack/emotion2vec
- *Prosody2Vec: Disentangling Prosody Representations* — https://arxiv.org/abs/2212.06972
- *ECAPA-TDNN Embeddings for Speaker Diarization* — https://arxiv.org/pdf/2104.01466
- pyannote.audio — https://github.com/pyannote/pyannote-audio
- Praat / Parselmouth — https://parselmouth.readthedocs.io/
- Sentence-Transformers (all-MiniLM-L6-v2) — https://www.sbert.net/
- Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009
- AudioVec (VGGish + TimescaleDB) — https://github.com/jamesfreire/TimescaleDB-AudioVec
- *Speaker Retrieval in the Wild* — https://arxiv.org/pdf/2504.18950
