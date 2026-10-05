"""faster-whisper in a dedicated child process.

Why: on CPU, torch (ECAPA, Silero, oneDNN) and CTranslate2 (Whisper) load the same OpenMP
runtime and share its worker threads. oneDNN leaves flush-denormals-to-zero switched on in those
threads, which changes CTranslate2's int8 results and degrades transcripts. Which thread ends up
with the flag depends on scheduling, so it is nondeterministic, and resetting the flag on the
calling thread is not enough (measured: 5/5 runs degraded in a long-lived worker). A child
process that never imports torch has its own clean OpenMP runtime.

This module must not import torch (directly or via radiodb modules that do).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

from .interfaces import Transcript, Word

log = logging.getLogger(__name__)

def ensure_cuda_dlls() -> None:
    """Let CTranslate2 find cuBLAS/cuDNN on Windows.

    CTranslate2 needs cublas64_12 / cudnn64_9, which the CUDA build of torch ships in
    ``torch/lib``. Register that folder for DLL loading *without importing torch* (this module
    must stay torch-free). No-op elsewhere or when torch isn't installed.
    """
    if os.name != "nt":
        return
    import importlib.util

    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        lib = Path(spec.origin).parent / "lib"
        if (lib / "cublas64_12.dll").exists():
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")


TRANSCRIBE_KW = dict(language="en", word_timestamps=True, vad_filter=True,
                     condition_on_previous_text=False, beam_size=5)


def _serve(conn, model: str, model_dir: str, device: str, compute: str) -> None:
    """Child process main loop: load Whisper once, then transcribe .npy files on request."""
    try:
        from faster_whisper import WhisperModel

        if device == "cuda":
            ensure_cuda_dlls()
        m = WhisperModel(model, device=device, compute_type=compute, download_root=model_dir)
        conn.send(("ready", None))
    except Exception as e:  # noqa: BLE001
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return
    while True:
        try:
            path = conn.recv()
        except EOFError:
            return
        if path is None:
            return
        try:
            audio = np.load(path)
            segments, _ = m.transcribe(audio, **TRANSCRIBE_KW)
            words = [(w.word, float(w.start), float(w.end), float(w.probability))
                     for seg in segments for w in (seg.words or [])]
            conn.send(("ok", words))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", f"{type(e).__name__}: {e}"))


class IsolatedWhisperTranscriber:
    def __init__(self, model: str, model_dir: Path, device: str, compute: str | None = None):
        self.name = f"faster-whisper/{model}"
        self._args = (model, str(model_dir / "whisper"), device, compute or ("float16" if device == "cuda" else "int8"))
        self._lock = threading.Lock()
        self._proc = None
        self._conn = None

    def _start(self) -> None:
        ctx = mp.get_context("spawn")  # fresh interpreter: nothing inherited from this process
        parent, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, *self._args), daemon=True, name="radiodb-asr")
        self._proc.start()
        child.close()
        self._conn = parent
        status, detail = self._conn.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"ASR process failed to start: {detail}")

    def transcribe(self, audio16k: np.ndarray) -> Transcript:
        with self._lock:
            if self._proc is None or not self._proc.is_alive():
                self._start()
            fd, path = tempfile.mkstemp(suffix=".npy", prefix="radiodb-asr-")
            os.close(fd)
            try:
                np.save(path, np.ascontiguousarray(audio16k, dtype=np.float32))
                self._conn.send(path)
                try:
                    status, payload = self._conn.recv()
                except EOFError:  # child died (e.g. out of memory); restart next time
                    self.close()
                    raise RuntimeError("ASR process died during transcription")
            finally:
                Path(path).unlink(missing_ok=True)
        if status != "ok":
            raise RuntimeError(f"ASR failed: {payload}")
        return Transcript(words=[Word(*w) for w in payload], model=self.name)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(None)
            except (OSError, BrokenPipeError):
                pass
            self._conn.close()
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
        self._proc = self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
