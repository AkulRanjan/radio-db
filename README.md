# radio-db

A database for **voice in its natural form**: the original audio is the record, and transcripts, speakers, emotion, prosody and sound events are time-aligned layers on top of it.

> Status: design phase. See [architecture.md](architecture.md) for the full design. No code yet.

## Why

Typical pipelines turn speech into text and discard the audio. That loses who was speaking, how they said it (tone, emphasis, pace, pauses, emotion) and everything that isn't words. radio-db keeps the audio losslessly and makes it searchable directly.

## Core ideas

- **Audio is the source of truth.** Lossless (FLAC), immutable, content-addressed by SHA-256.
- **Everything else is a derived layer.** Transcript, speaker, prosody, emotion, sound events and neural-codec tokens are each stored with the model and version that produced them, and can be recomputed.
- **Time-aligned.** Every annotation points to `(recording, channel, start, end)`, following Lhotse's Recording/Supervision/Cut model.
- **Search by sound, not only by words.** You can query by text, by an example voice clip, by speaker, by a sound description or by paralinguistic filters. Results come back as playable clips.

## Planned stack

| Concern | Choice (initial) |
|---|---|
| Audio blobs | S3-compatible object store (MinIO locally), FLAC |
| Catalog | PostgreSQL |
| Vectors / features | LanceDB (or pgvector for small setups) |
| Text search | Postgres FTS, with OpenSearch later |
| VAD + diarization | pyannote.audio |
| Speaker embeddings | ECAPA-TDNN (SpeechBrain) |
| ASR + word timestamps | Whisper / WhisperX |
| Emotion | emotion2vec |
| Text↔audio search | CLAP; SpeechRAG-style speech embeddings |
| Codec tokens (optional) | Mimi / EnCodec |
| Jobs | Python workers on a queue (Celery / Arq) |
| API | FastAPI |

## Example queries (target)

```text
"segments where the host talks about the election"   → semantic speech search
<upload 5 s clip of a voice>                         → everything that speaker said
"phone ringing" / "applause"                         → CLAP sound-event search
speaker=host AND emotion=angry AND date=2026-09      → paralinguistic filter
```

## Roadmap

1. Ingest, blob storage, catalog, time-range playback
2. VAD, diarization, speaker embeddings, query-by-voice
3. ASR with word timestamps, keyword search
4. Transcription-free semantic search and sound search
5. Prosody and emotion layers, hybrid query planner
6. Live radio stream capture, codec tokens, cold storage tiers, dataset export (Lhotse / Hugging Face)

## Privacy

Voice is biometric data. The design stores consent and licence per recording, supports deleting everything tied to a speaker, and uses redaction layers that mute spans at playback without altering the original audio.

## Further reading

The prior work this design draws on (Lhotse, Lance, EnCodec/Mimi, SpeechRAG, VoxRAG, CLAP, emotion2vec, pyannote, ECAPA-TDNN) is listed with links in [architecture.md § References](architecture.md#references).
