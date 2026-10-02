"""High-level operations. The CLI and HTTP API are thin wrappers over this class."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from . import audio
from .blobstore import BlobStore, blob_key, make_blob_store
from .catalog import Catalog
from .config import Settings
from .pipeline.diarize import assign_words, build_segments, diarize
from .pipeline.identify import best_match, running_mean
from .pipeline.interfaces import VAD, SpeakerEmbedder, Transcriber

log = logging.getLogger(__name__)

MAX_EMBED_S = 30.0  # longer segments are embedded from their first 30 s


class NotFound(LookupError):
    pass


@dataclass
class Models:
    vad: VAD
    embedder: SpeakerEmbedder
    transcriber: Transcriber


def load_models(settings: Settings) -> Models:
    from .pipeline.backends import EcapaEmbedder, SileroVAD, WhisperTranscriber, resolve_device

    device = resolve_device(settings.device)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    return Models(
        vad=SileroVAD(),
        embedder=EcapaEmbedder(settings.model_dir, device),
        transcriber=WhisperTranscriber(settings.asr_model, settings.model_dir, device),
    )


class RadioDB:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        catalog: Catalog | None = None,
        blobs: BlobStore | None = None,
        models: Models | None = None,
    ):
        self.settings = settings or Settings()
        self.catalog = catalog or Catalog(self.settings.db_path)
        self.blobs = blobs or make_blob_store(self.settings)
        self._models = models
        # One inference job at a time: models load once and a 6 GB GPU isn't oversubscribed.
        self._gpu = threading.RLock()

    @property
    def models(self) -> Models:
        with self._gpu:
            if self._models is None:
                self._models = load_models(self.settings)
        return self._models

    # ---- ingest & processing ----------------------------------------------------------------

    def ingest(self, user_id: str, data: bytes, title: str | None = None) -> tuple[dict, bool]:
        """Store audio as the immutable record. Returns (recording, created)."""
        decoded = audio.decode(data)
        stored, ext = audio.canonical_bytes(data, decoded)
        sha, key = blob_key(stored, ext)
        existing = self.catalog.find_recording_by_hash(user_id, sha)
        if existing:
            return existing, False
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
            producers = {
                "vad": m.vad.name, "speaker": m.embedder.name, "asr": transcript.model,
                "diarization_threshold": self.settings.diarization_threshold,
                "min_speaker_s": self.settings.min_speaker_s,
                "speaker_match_threshold": self.settings.speaker_match_threshold,
            }
            self.catalog.set_status(rec_id, "done", producers=producers)
        except Exception as e:
            log.exception("processing %s failed", rec_id)
            self.catalog.set_status(rec_id, "failed", error=f"{type(e).__name__}: {e}")
            raise
        return self._recording(user_id, rec_id)

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
        return [_public_segment(s) for s in self.catalog.segments(user_id, recording_id=rec_id)]

    def speaker_segments(self, user_id: str, speaker: str) -> list[dict]:
        spk = self.resolve_speaker(user_id, speaker)
        return [_public_segment(s) for s in self.catalog.segments(user_id, speaker_id=spk["id"])]

    def clip(self, user_id: str, rec_id: str, start_s: float, end_s: float) -> bytes:
        rec = self._recording(user_id, rec_id)
        return audio.slice_wav(audio.decode(self.blobs.get(rec["blob_key"])), start_s, end_s)

    def original(self, user_id: str, rec_id: str) -> tuple[bytes, str]:
        rec = self._recording(user_id, rec_id)
        return self.blobs.get(rec["blob_key"]), rec["format"]

    def search_text(self, user_id: str, query: str, speaker: str | None = None, limit: int = 20) -> list[dict]:
        spk_id = self.resolve_speaker(user_id, speaker)["id"] if speaker else None
        hits = self.catalog.search_text(user_id, query, speaker_id=spk_id, limit=limit)
        return [_public_segment(h) | {"snippet": h["snippet"]} for h in hits]

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

    def stats(self, user_id: str) -> dict:
        recs = self.catalog.list_recordings(user_id)
        return {
            "recordings": len(recs),
            "hours": round(self.catalog.total_duration(user_id) / 3600, 3),
            "speakers": len(self.catalog.list_speakers(user_id)),
            "by_status": {st: sum(r["status"] == st for r in recs) for st in {r["status"] for r in recs}},
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
