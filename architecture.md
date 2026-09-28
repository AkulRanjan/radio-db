# radio-db — Architecture

> Status: design sketch (v0). Nothing here is implemented yet.

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

- Primary content: broadcast radio, conversational recordings, or both? (Affects chunking and live path priority.)
- Languages to support? (Drives ASR / embedding model choice.)
- Expected volume (hours/day) and retention period?
- Self-hosted only, or is cloud object storage acceptable?
- Is speaker identification (naming people) in scope, given the legal implications?

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
- AudioVec (VGGish + TimescaleDB) — https://github.com/jamesfreire/TimescaleDB-AudioVec
- *Speaker Retrieval in the Wild* — https://arxiv.org/pdf/2504.18950
