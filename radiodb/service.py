"""High-level operations. The CLI and HTTP API are thin wrappers over this class."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import audio
from .blobstore import BlobStore, blob_key, make_blob_store
from .catalog import BaseCatalog, open_catalog
from .config import Settings
from .pipeline.diarize import assign_words, build_segments, diarize
from .pipeline.identify import best_match, running_mean
from .pipeline.interfaces import VAD, SpeakerEmbedder, Transcriber
from .pipeline.layers import Layer
from .query import STYLE_VALUES, rrf, style_table

log = logging.getLogger(__name__)

MAX_EMBED_S = 30.0  # longer segments are embedded from their first 30 s
CANDIDATES = 200  # per-retriever candidates before rank fusion
# Semantic matches: absolute cosine floor (MiniLM scores paraphrases ~0.2, unrelated text ~0-0.1)
# plus a window below the best hit, so weak matches don't trail after strong ones.
MIN_SEMANTIC_SIM = 0.15
SEMANTIC_WINDOW = 0.15


class NotFound(LookupError):
    pass


class QuotaExceeded(Exception):
    pass


@dataclass
class Models:
    vad: VAD
    embedder: SpeakerEmbedder
    transcriber: Transcriber
    layers: list[Layer] = field(default_factory=list)


class LazyModels:
    """Same shape as ``Models`` but each model loads on first use, so e.g. a semantic search
    only loads the small text encoder, not Whisper and ECAPA."""

    def __init__(self, settings: Settings):
        from .pipeline.backends import resolve_device

        self.settings = settings
        self.device = resolve_device(settings.device)
        settings.model_dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def vad(self) -> VAD:
        from .pipeline.backends import SileroVAD

        return SileroVAD()

    @cached_property
    def embedder(self) -> SpeakerEmbedder:
        from .pipeline.backends import EcapaEmbedder

        return EcapaEmbedder(self.settings.model_dir, self.device)

    @cached_property
    def transcriber(self) -> Transcriber:
        iso = self.settings.asr_isolate
        if iso == "on" or (iso == "auto" and self.device == "cpu"):
            from .pipeline.asr_process import IsolatedWhisperTranscriber

            return IsolatedWhisperTranscriber(self.settings.asr_model, self.settings.model_dir, self.device,
                                              self.settings.asr_compute or None)
        from .pipeline.backends import WhisperTranscriber

        return WhisperTranscriber(self.settings.asr_model, self.settings.model_dir, self.device,
                                  self.settings.asr_compute or None)

    @cached_property
    def layers(self) -> list[Layer]:
        """Enabled layers that loaded. A layer that fails to load (e.g. its model can't be
        downloaded) is left out and its error kept in ``layer_errors``: core processing still
        works, and the layer can be backfilled later with ``enrich``."""
        from .pipeline import layers as L

        builders = {
            "prosody": lambda: L.ProsodyLayer(),
            "semantic": lambda: L.SemanticLayer(
                L.SentenceEmbedder(self.settings.text_model, self.settings.model_dir, self.device)),
            "emotion": lambda: L.EmotionLayer(
                L.HubertEmotionClassifier(self.settings.emotion_model, self.settings.model_dir, self.device)),
        }
        out: list[Layer] = []
        for name in self.settings.layers:
            if name not in builders:
                self.layer_errors[name] = "unknown layer"
                continue
            try:
                out.append(builders[name]())
            except Exception as e:  # noqa: BLE001
                log.warning("layer %s unavailable: %s", name, e)
                self.layer_errors[name] = f"failed to load: {type(e).__name__}: {e}"
        return out

    @cached_property
    def layer_errors(self) -> dict[str, str]:
        return {}


class RadioDB:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        catalog: BaseCatalog | None = None,
        blobs: BlobStore | None = None,
        models: Models | None = None,
    ):
        self.settings = settings or Settings()
        self.catalog = catalog or open_catalog(self.settings)
        self.blobs = blobs or make_blob_store(self.settings)
        self._models = models
        # One inference job at a time: models load once and a 6 GB GPU isn't oversubscribed.
        self._gpu = threading.RLock()

    @property
    def models(self) -> Models | LazyModels:
        with self._gpu:
            if self._models is None:
                self._models = LazyModels(self.settings)
        return self._models

    def layer(self, name: str) -> Layer | None:
        with self._gpu:
            return next((l for l in self.models.layers if l.name == name), None)

    # ---- ingest & processing ----------------------------------------------------------------

    def ingest(self, user_id: str, data: bytes, title: str | None = None) -> tuple[dict, bool]:
        """Store audio as the immutable record. Returns (recording, created)."""
        decoded = audio.decode(data)
        stored, ext = audio.canonical_bytes(data, decoded)
        sha, key = blob_key(stored, ext)
        existing = self.catalog.find_recording_by_hash(user_id, sha)
        if existing:
            return existing, False
        quota_s = self.settings.user_quota_hours * 3600
        if quota_s:
            used = self.catalog.total_duration(user_id)
            if used + decoded.duration_s > quota_s:
                raise QuotaExceeded(
                    f"storage quota is {self.settings.user_quota_hours:g} h of audio; {used / 3600:.2f} h used, "
                    f"this recording is {decoded.duration_s / 3600:.2f} h")
        self.blobs.put(key, stored)
        rec = self.catalog.add_recording(
            user_id=user_id, title=title, sha256=sha, blob_key=key, format=ext,
            sample_rate=decoded.sample_rate, channels=decoded.channels,
            duration_s=decoded.duration_s, retention_days=self.settings.retention_days,
        )
        return rec, True

    def process(self, user_id: str, rec_id: str) -> dict:
        """Build derived layers: diarization, speaker identity, transcript, voice embeddings."""
        rec = self._recording(user_id, rec_id)
        with self._gpu:
            return self._process(user_id, rec)

    def _process(self, user_id: str, rec: dict) -> dict:
        rec_id = rec["id"]
        self.catalog.set_status(rec_id, "processing")
        try:
            decoded = audio.decode(self.blobs.get(rec["blob_key"]))
            a16 = audio.to_model_input(decoded)
            m = self.models

            turns, centroids = diarize(
                a16, m.vad, m.embedder, self.settings.diarization_threshold, self.settings.min_speaker_s
            )
            transcript = m.transcriber.transcribe(a16)
            labels = assign_words(transcript.words, turns)
            segs = build_segments(transcript.words, labels, turns)

            talk = {c: sum(t.end - t.start for t in turns if t.cluster == c) for c in centroids}
            cluster_to_speaker = self._identify_clusters(user_id, centroids, talk)

            embs = self._embed_spans(a16, [(s["start_s"], s["end_s"]) for s in segs])
            for s, e in zip(segs, embs):
                s["embedding"] = e
                s["speaker_id"] = cluster_to_speaker.get(s.pop("cluster"))

            self.catalog.replace_segments(rec_id, user_id, segs)
            self.catalog.delete_orphan_speakers(user_id)  # e.g. left behind by reprocessing
            layer_producers, layer_errors = self._run_layers(user_id, rec_id, a16, m.layers)
            layer_errors = {**getattr(m, "layer_errors", {}), **layer_errors}
            producers = {
                "vad": m.vad.name, "speaker": m.embedder.name, "asr": transcript.model,
                "layers": layer_producers, **({"layer_errors": layer_errors} if layer_errors else {}),
                "diarization_threshold": self.settings.diarization_threshold,
                "min_speaker_s": self.settings.min_speaker_s,
                "speaker_match_threshold": self.settings.speaker_match_threshold,
            }
            self.catalog.set_status(rec_id, "done", error=self._layer_error_note(layer_errors), producers=producers)
        except Exception as e:
            log.exception("processing %s failed", rec_id)
            self.catalog.set_status(rec_id, "failed", error=f"{type(e).__name__}: {e}")
            raise
        return self._recording(user_id, rec_id)

    # ---- queued work (run by `radiodb worker` / the API's inline worker) ----------------------

    def enqueue_process(self, user_id: str, rec_id: str) -> dict:
        """Queue diarization + identification + ASR + layers. Idempotent while one is pending."""
        rec = self._recording(user_id, rec_id)
        job, created = self.catalog.enqueue_job(
            user_id=user_id, kind="process", recording_id=rec_id, max_attempts=self.settings.job_max_attempts)
        if created and rec["status"] != "processing":
            self.catalog.set_status(rec_id, "queued", error=rec["error"])
        return job

    def enqueue_enrich(self, user_id: str, rec_id: str, layers: list[str] | None = None) -> dict:
        rec = self._recording(user_id, rec_id)
        if rec["status"] != "done":
            raise ValueError(f"recording is {rec['status']}; process it first")
        known = set(self.settings.layers)
        if layers and set(layers) - known:
            raise ValueError(f"unknown or disabled layer(s): {', '.join(sorted(set(layers) - known))}")
        job, _ = self.catalog.enqueue_job(
            user_id=user_id, kind="enrich", recording_id=rec_id, payload={"layers": layers},
            max_attempts=self.settings.job_max_attempts)
        return job

    def get_job(self, user_id: str, job_id: str) -> dict:
        job = self.catalog.get_job(user_id, job_id)
        if not job:
            raise NotFound(f"job {job_id!r}")
        return job

    def list_jobs(self, user_id: str, status: str | None = None, limit: int = 100) -> list[dict]:
        return self.catalog.list_jobs(user_id, status, limit)

    def enrich(self, user_id: str, rec_id: str, layers: list[str] | None = None) -> dict:
        """(Re)build derived layers on existing segments, without redoing ASR or diarization."""
        rec = self._recording(user_id, rec_id)
        if rec["status"] != "done":
            raise ValueError(f"recording is {rec['status']}; process it first")
        with self._gpu:
            todo = [l for l in self.models.layers if layers is None or l.name in layers]
            unknown = set(layers or []) - set(self.settings.layers)
            if unknown:
                raise ValueError(f"unknown or disabled layer(s): {', '.join(sorted(unknown))}")
            unavailable = set(layers or []) - {l.name for l in todo}
            if unavailable:
                errs = getattr(self.models, "layer_errors", {})
                raise RuntimeError("layer(s) unavailable: " + "; ".join(f"{n} ({errs.get(n, '?')})" for n in sorted(unavailable)))
            a16 = audio.to_model_input(audio.decode(self.blobs.get(rec["blob_key"])))
            produced, errors = self._run_layers(user_id, rec_id, a16, todo)
        producers = json.loads(rec["producers"] or "{}")
        producers.setdefault("layers", {}).update(produced)
        remaining = {k: v for k, v in producers.get("layer_errors", {}).items() if k not in produced}
        remaining.update(errors)
        if layers:  # explicitly requested layers that couldn't even load
            remaining.update({k: v for k, v in getattr(self.models, "layer_errors", {}).items() if k in layers})
        producers.pop("layer_errors", None)
        if remaining:
            producers["layer_errors"] = remaining
        self.catalog.set_status(rec_id, "done", error=self._layer_error_note(remaining), producers=producers)
        if errors and layers and set(errors) >= set(layers):
            raise RuntimeError(self._layer_error_note(errors))  # nothing requested succeeded: let the job retry
        return self._recording(user_id, rec_id)

    def _run_layers(
        self, user_id: str, rec_id: str, a16: np.ndarray, layers: list[Layer]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Run enrichment layers; one failing layer never discards the others or the core
        results. Returns ({layer: producer}, {layer: error})."""
        segs = self.catalog.segments(user_id, recording_id=rec_id)
        produced, errors = {}, {}
        for layer in layers:
            try:
                rows = layer.compute(a16, segs)
                self.catalog.put_layer(
                    layer.name, layer.producer,
                    [(s["id"], d, v) for s, (d, v) in zip(segs, rows) if d is not None or v is not None],
                )
                produced[layer.name] = layer.producer
            except Exception as e:  # noqa: BLE001
                log.exception("layer %s failed on %s", layer.name, rec_id)
                errors[layer.name] = f"{type(e).__name__}: {e}"
        return produced, errors

    def _layer_error_note(self, errors: dict[str, str]) -> str | None:
        if not errors:
            return None
        return "layers not built: " + "; ".join(f"{k} ({v})" for k, v in errors.items()) + \
            " — fix and run `radiodb enrich`"

    def _identify_clusters(
        self, user_id: str, centroids: dict[int, np.ndarray], talk: dict[int, float]
    ) -> dict[int, str]:
        speakers = {s["id"]: s for s in self.catalog.list_speakers(user_id)}
        out: dict[int, str] = {}
        # Longest talkers first: their centroids are the most reliable.
        for c in sorted(centroids, key=lambda c: -talk.get(c, 0)):
            cand = [(sid, s["centroid"]) for sid, s in speakers.items()]
            match = best_match(centroids[c], cand, self.settings.speaker_match_threshold)
            if match.speaker_id:
                s = speakers[match.speaker_id]
                if not s["enrolled"]:  # enrolled profiles stay anchored to the user's samples
                    s["centroid"] = running_mean(s["centroid"], s["n_samples"], centroids[c])
                    s["n_samples"] += 1
                    self.catalog.update_speaker(
                        s["id"], centroid=s["centroid"], n_samples=s["n_samples"]
                    )
                out[c] = match.speaker_id
            else:
                s = self.catalog.add_speaker(user_id, centroids[c])
                speakers[s["id"]] = s
                out[c] = s["id"]
        return out

    def _embed_spans(self, a16: np.ndarray, spans: list[tuple[float, float]]) -> list[np.ndarray | None]:
        clips, idx = [], []
        for i, (a, b) in enumerate(spans):
            b = min(b, a + MAX_EMBED_S)
            if b - a >= 0.5:
                clips.append(a16[int(a * audio.MODEL_SR): int(b * audio.MODEL_SR)])
                idx.append(i)
        out: list[np.ndarray | None] = [None] * len(spans)
        if clips:
            for i, e in zip(idx, self.models.embedder.embed(clips)):
                out[i] = e
        return out

    def _voiceprint(self, clips: list[bytes]) -> tuple[np.ndarray, int]:
        """One normalised voice embedding from one or more example clips (speech only)."""
        pieces: list[np.ndarray] = []
        for data in clips:
            a16 = audio.to_model_input(audio.decode(data))
            spans = self.models.vad.speech(a16) or []
            speech = [a16[int(s.start * audio.MODEL_SR): int(s.end * audio.MODEL_SR)] for s in spans]
            joined = np.concatenate(speech) if speech else np.zeros(0, np.float32)
            step = int(10 * audio.MODEL_SR)
            pieces += [joined[i: i + step] for i in range(0, len(joined), step)
                       if len(joined[i: i + step]) >= audio.MODEL_SR]
        if not pieces:
            raise ValueError("no speech found in the sample(s); provide at least ~3 s of speech")
        emb = self.models.embedder.embed(pieces)
        v = emb.mean(axis=0)
        return (v / max(np.linalg.norm(v), 1e-9)).astype(np.float32), len(pieces)

    # ---- speakers ---------------------------------------------------------------------------

    def enroll(self, user_id: str, name: str, clips: list[bytes]) -> dict:
        """Create or refresh a named voice profile, then relabel matching unnamed speakers."""
        with self._gpu:
            vp, n = self._voiceprint(clips)
        spk = self.catalog.find_speaker_by_name(user_id, name)
        if spk:
            if spk["enrolled"]:  # add the new samples to the existing profile
                centroid, n = running_mean(spk["centroid"], spk["n_samples"], vp, n), spk["n_samples"] + n
            else:  # a speaker named by hand: replace the auto-learned voice with the samples
                centroid = vp
            self.catalog.update_speaker(spk["id"], centroid=centroid, n_samples=n, enrolled=True)
        else:
            spk = self.catalog.add_speaker(user_id, vp, name=name, enrolled=True, n_samples=n)
        target = self.catalog.get_speaker(user_id, spk["id"])
        for other in self.catalog.list_speakers(user_id):
            if other["id"] == target["id"] or other["enrolled"] or other["name"]:
                continue
            if float(other["centroid"] @ target["centroid"]) >= self.settings.speaker_match_threshold:
                self.catalog.reassign_segments(other["id"], target["id"])
                self.catalog.delete_speaker(other["id"])
        return self._public_speaker(self.catalog.get_speaker(user_id, spk["id"]))

    def rename_speaker(self, user_id: str, spk_id: str, name: str) -> dict:
        self._speaker(user_id, spk_id)
        clash = self.catalog.find_speaker_by_name(user_id, name)
        if clash and clash["id"] != spk_id:
            raise ValueError(f"a speaker named {name!r} already exists; merge instead")
        self.catalog.update_speaker(spk_id, name=name)
        return self._public_speaker(self.catalog.get_speaker(user_id, spk_id))

    def merge_speakers(self, user_id: str, keep_id: str, drop_id: str) -> dict:
        """Fix a split: everything said by ``drop_id`` is attributed to ``keep_id``."""
        keep, drop = self._speaker(user_id, keep_id), self._speaker(user_id, drop_id)
        if keep_id == drop_id:
            raise ValueError("cannot merge a speaker into itself")
        if not keep["enrolled"]:
            c = running_mean(keep["centroid"], keep["n_samples"], drop["centroid"], drop["n_samples"])
            self.catalog.update_speaker(keep_id, centroid=c, n_samples=keep["n_samples"] + drop["n_samples"])
        self.catalog.reassign_segments(drop_id, keep_id)
        self.catalog.delete_speaker(drop_id)
        return self._public_speaker(self.catalog.get_speaker(user_id, keep_id))

    def delete_speaker(self, user_id: str, spk_id: str) -> None:
        self._speaker(user_id, spk_id)
        self.catalog.delete_speaker(spk_id)

    def list_speakers(self, user_id: str) -> list[dict]:
        return [self._public_speaker(s) for s in self.catalog.list_speakers(user_id)]

    def resolve_speaker(self, user_id: str, ref: str) -> dict:
        """Accept a speaker id or a (case-insensitive) name."""
        s = self.catalog.get_speaker(user_id, ref) or self.catalog.find_speaker_by_name(user_id, ref)
        if not s:
            raise NotFound(f"speaker {ref!r}")
        return s

    # ---- reading & search -------------------------------------------------------------------

    def list_recordings(self, user_id: str) -> list[dict]:
        return self.catalog.list_recordings(user_id)

    def get_recording(self, user_id: str, rec_id: str) -> dict:
        return self._recording(user_id, rec_id)

    def transcript(self, user_id: str, rec_id: str) -> list[dict]:
        self._recording(user_id, rec_id)
        return self.search(user_id, recording=rec_id, limit=None)

    def speaker_segments(self, user_id: str, speaker: str) -> list[dict]:
        return self.search(user_id, speaker=speaker, limit=None)

    def speaker_profile(self, user_id: str, speaker: str) -> dict:
        """How this person usually sounds: their baseline for the speaker-relative style labels."""
        spk = self.resolve_speaker(user_id, speaker)
        segs = self.catalog.segments(user_id, speaker_id=spk["id"])
        pros = [v["data"] for v in self.catalog.layers(user_id, "prosody", speaker_id=spk["id"]).values() if v["data"]]

        def med(feat):
            vals = [p[feat] for p in pros if p.get(feat) is not None]
            return round(float(np.median(vals)), 2) if vals else None

        talk_s = sum(s["end_s"] - s["start_s"] for s in segs)
        pauses = sum(p.get("n_pauses", 0) for p in pros)
        emos = [e["data"] for e in self.catalog.layers(user_id, "emotion", speaker_id=spk["id"]).values() if e["data"]]
        confident = [e["label"] for e in emos if e["confidence"] >= self.settings.emotion_min_confidence]
        return self._public_speaker(spk) | {
            "segments": len(segs),
            "recordings": len({s["recording_id"] for s in segs}),
            "talk_time_s": round(talk_s, 1),
            "voice": {
                "f0_median_hz": med("f0_median_hz"),
                "f0_range_st": med("f0_range_st"),
                "articulation_rate_sps": med("articulation_rate_sps"),
                "pauses_per_min": round(pauses / (talk_s / 60), 2) if talk_s and pros else None,
                "based_on_segments": len(pros),
            },
            "emotion": {
                "share": {lab: round(confident.count(lab) / len(confident), 3) for lab in sorted(set(confident))},
                "confident_segments": len(confident),
                "scored_segments": len(emos),
            },
        }

    def search(
        self,
        user_id: str,
        q: str | None = None,
        *,
        mode: str = "hybrid",
        speaker: str | None = None,
        recording: str | None = None,
        style: dict[str, str] | None = None,
        emotion: str | None = None,
        limit: int | None = 20,
    ) -> list[dict]:
        """Find segments by words (keyword), meaning (semantic) or both (hybrid, rank-fused),
        optionally narrowed by speaker, recording and speaker-relative speaking style
        (e.g. ``{"pace": "fast", "volume": "loud"}``) and emotion heard in the voice
        (``"angry"``; counts only at or above ``emotion_min_confidence``). Without ``q``, returns
        matching segments in time order."""
        if mode not in ("hybrid", "keyword", "semantic"):
            raise ValueError("mode must be hybrid, keyword or semantic")
        for dim, val in (style or {}).items():
            if val not in STYLE_VALUES.get(dim, ()):
                raise ValueError(f"invalid style {dim}={val}; options: {_style_help()}")
        spk_id = self.resolve_speaker(user_id, speaker)["id"] if speaker else None

        all_segs = self.catalog.segments(user_id)
        prosody = {k: v["data"] for k, v in self.catalog.layers(user_id, "prosody").items()}
        emotions = {k: v["data"] for k, v in self.catalog.layers(user_id, "emotion").items()}
        styles = style_table(all_segs, prosody)
        min_conf = self.settings.emotion_min_confidence
        by_id = {s["id"]: s for s in all_segs}
        pool = {
            s["id"] for s in all_segs
            if (not spk_id or s["speaker_id"] == spk_id) and (not recording or s["recording_id"] == recording)
        }

        snippets: dict[int, str] = {}
        scores: dict[int, float] = {}
        if q and q.strip():
            rankings = []
            if mode in ("hybrid", "keyword"):
                hits = [h for h in self.catalog.search_text(
                            user_id, q, speaker_id=spk_id, limit=CANDIDATES, any_terms=(mode == "hybrid"))
                        if h["id"] in pool]
                snippets = {h["id"]: h["snippet"] for h in hits}
                rankings.append([h["id"] for h in hits])
            if mode in ("hybrid", "semantic"):
                sem = self._semantic_ranking(user_id, q, pool)
                if sem is None and mode == "semantic":
                    raise ValueError("semantic search is not enabled (RADIODB_LAYERS)")
                if sem:
                    rankings.append(sem)
            scores = rrf(*rankings)
            ordered = sorted(scores, key=lambda i: -scores[i])
        else:
            ordered = [s["id"] for s in all_segs if s["id"] in pool]

        out = []
        for sid in ordered:
            st = styles.get(sid, {})
            if style and any(st.get(dim) != val for dim, val in style.items()):
                continue
            emo = emotions.get(sid)
            if emotion and not (emo and emo["label"] == emotion and emo["confidence"] >= min_conf):
                continue
            hit = _public_segment(by_id[sid]) | {
                "prosody": prosody.get(sid), "style": st or None,
                "emotion": emo | {"confident": emo["confidence"] >= min_conf} if emo else None,
            }
            if sid in snippets:
                hit["snippet"] = snippets[sid]
            if scores:
                hit["score"] = round(scores[sid], 5)
            out.append(hit)
            if limit and len(out) >= limit:
                break
        return out

    def search_text(self, user_id: str, query: str, speaker: str | None = None, limit: int = 20) -> list[dict]:
        return self.search(user_id, query, mode="keyword", speaker=speaker, limit=limit)

    def _semantic_ranking(self, user_id: str, q: str, pool: set[int]) -> list[int] | None:
        vecs = {
            k: v["vector"] for k, v in self.catalog.layers(user_id, "semantic").items()
            if k in pool and v["vector"] is not None
        }
        layer = self.layer("semantic")
        if layer is None:
            return None
        if not vecs:
            return []
        with self._gpu:
            qv = layer.embed_query(q)
        ids = list(vecs)
        sims = np.stack([vecs[i] for i in ids]) @ qv
        order = np.argsort(-sims)[:CANDIDATES]
        floor = max(MIN_SEMANTIC_SIM, float(sims[order[0]]) - SEMANTIC_WINDOW)
        return [ids[i] for i in order if sims[i] >= floor]

    def clip(self, user_id: str, rec_id: str, start_s: float, end_s: float) -> bytes:
        rec = self._recording(user_id, rec_id)
        return audio.slice_wav(audio.decode(self.blobs.get(rec["blob_key"])), start_s, end_s)

    def original(self, user_id: str, rec_id: str) -> tuple[bytes, str]:
        rec = self._recording(user_id, rec_id)
        return self.blobs.get(rec["blob_key"]), rec["format"]

    def search_voice(self, user_id: str, clip: bytes, limit: int = 20) -> dict:
        """Who does this voice belong to, and where else does it appear?"""
        with self._gpu:
            vp, _ = self._voiceprint([clip])
        speakers = sorted(
            ({**self._public_speaker(s), "score": round(float(s["centroid"] @ vp), 4)}
             for s in self.catalog.list_speakers(user_id)),
            key=lambda s: -s["score"],
        )
        segs = [s for s in self.catalog.segments(user_id) if s["embedding"] is not None]
        if segs:
            sims = np.stack([s["embedding"] for s in segs]) @ vp
            order = np.argsort(-sims)[:limit]
            seg_hits = [_public_segment(segs[i]) | {"score": round(float(sims[i]), 4)} for i in order]
        else:
            seg_hits = []
        best = speakers[0] if speakers and speakers[0]["score"] >= self.settings.speaker_match_threshold else None
        return {"identified": best, "speakers": speakers[:5], "segments": seg_hits}

    def health(self) -> dict:
        """Readiness of the catalog and blob store (for load balancers / orchestration)."""
        checks: dict[str, str] = {}
        try:
            self.catalog._q("SELECT 1 AS ok")
            checks["catalog"] = "ok"
        except Exception as e:  # noqa: BLE001
            checks["catalog"] = f"error: {type(e).__name__}: {e}"
        try:
            self.blobs.ping()
            checks["blobs"] = "ok"
        except Exception as e:  # noqa: BLE001
            checks["blobs"] = f"error: {type(e).__name__}: {e}"
        ok = all(v == "ok" for v in checks.values())
        out = {"status": "ok" if ok else "degraded", "checks": checks}
        if checks["catalog"] == "ok":
            out["jobs"] = self.catalog.job_counts()
        return out

    def stats(self, user_id: str) -> dict:
        recs = self.catalog.list_recordings(user_id)
        return {
            "recordings": len(recs),
            "hours": round(self.catalog.total_duration(user_id) / 3600, 3),
            "speakers": len(self.catalog.list_speakers(user_id)),
            "by_status": {st: sum(r["status"] == st for r in recs) for st in {r["status"] for r in recs}},
            "jobs": self.catalog.job_counts(user_id),
        }

    # ---- deletion & retention ---------------------------------------------------------------

    def delete_recording(self, user_id: str, rec_id: str) -> None:
        rec = self._recording(user_id, rec_id)
        self._delete(rec)

    def purge_expired(self) -> int:
        """Delete recordings past their retention date (default 365 days). Run daily."""
        expired = self.catalog.expired_recordings()
        for rec in expired:
            self._delete(rec)
        return len(expired)

    def _delete(self, rec: dict) -> None:
        self.catalog.delete_recording(rec["id"])  # segments cascade
        self.catalog.delete_orphan_speakers(rec["user_id"])
        if not self.catalog.blob_in_use(rec["blob_key"]):
            self.blobs.delete(rec["blob_key"])

    # ---- helpers ----------------------------------------------------------------------------

    def _recording(self, user_id: str, rec_id: str) -> dict:
        rec = self.catalog.get_recording(user_id, rec_id)
        if not rec:
            raise NotFound(f"recording {rec_id!r}")
        return rec

    def _speaker(self, user_id: str, spk_id: str) -> dict:
        s = self.catalog.get_speaker(user_id, spk_id)
        if not s:
            raise NotFound(f"speaker {spk_id!r}")
        return s

    @staticmethod
    def _public_speaker(s: dict) -> dict:
        return {k: v for k, v in s.items() if k not in ("centroid", "user_id")}


def _public_segment(s: dict) -> dict:
    return {
        "id": s["id"], "recording_id": s["recording_id"], "speaker_id": s["speaker_id"],
        "speaker": s.get("speaker_name"), "start_s": round(s["start_s"], 3),
        "end_s": round(s["end_s"], 3), "text": s["text"], "words": s["words"],
    }


def _style_help() -> str:
    return "; ".join(f"{d}={'|'.join(sorted(v))}" for d, v in STYLE_VALUES.items())
