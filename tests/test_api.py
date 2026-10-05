import pytest
from fastapi.testclient import TestClient

from radiodb.api import create_app
from radiodb.worker import Worker
from tests.conftest import VOCAB, conversation, script_words, wav_bytes

CONVO = [("A", 4.0), (None, 0.6), ("B", 4.0)]


@pytest.fixture
def client(db):
    return TestClient(create_app(db, inline_worker=False))


@pytest.fixture
def drain(db):
    """Run queued jobs to completion, as `radiodb worker` would."""
    return lambda: Worker(db, lease_s=60).run_until_idle()


def H(user="u1"):
    return {"X-User-Id": user}


def test_upload_process_search_flow(client, asr, drain):
    asr.next_words = script_words(CONVO, VOCAB)
    r = client.post("/recordings", headers=H(), files={"file": ("c.wav", wav_bytes(conversation(CONVO)), "audio/wav")})
    assert r.status_code == 201, r.text
    rec_id, job_id = r.json()["recording"]["id"], r.json()["job"]["id"]
    assert r.json()["recording"]["status"] == "queued"
    assert drain() == 1
    assert client.get(f"/jobs/{job_id}", headers=H()).json()["status"] == "done"

    assert client.get(f"/recordings/{rec_id}", headers=H()).json()["status"] == "done"
    assert len(client.get(f"/recordings/{rec_id}/transcript", headers=H()).json()) == 2

    r = client.post("/speakers/enroll", headers=H(), data={"name": "Bob"},
                    files=[("files", ("b.wav", wav_bytes(conversation([("B", 5.0)])), "audio/wav"))])
    assert r.status_code == 201, r.text

    hits = client.get("/search", params={"q": "we saved"}, headers=H()).json()
    assert hits[0]["speaker"] == "Bob"
    assert client.get("/speakers/bob/segments", headers=H()).json()[0]["text"].startswith("pretty")

    r = client.get(f"/recordings/{rec_id}/audio", params={"start": 1, "end": 2}, headers=H())
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    r = client.get(f"/recordings/{rec_id}/audio", headers=H())
    assert r.headers["content-type"] == "audio/flac"

    r = client.post("/search/voice", headers=H(), files={"file": ("q.wav", wav_bytes(conversation([("B", 2.0)])), "audio/wav")})
    assert r.json()["identified"]["name"] == "Bob"


def test_user_isolation_and_errors(client):
    r = client.post("/recordings", headers=H(), data={"process": "false"},
                    files={"file": ("c.wav", wav_bytes(conversation(CONVO)), "audio/wav")})
    rec_id = r.json()["recording"]["id"]
    assert client.get(f"/recordings/{rec_id}", headers=H("u2")).status_code == 404
    assert client.get("/recordings", headers=H("u2")).json() == []
    assert client.get("/recordings").status_code == 401  # identity required
    r = client.post("/recordings", headers=H(), files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    assert client.delete(f"/recordings/{rec_id}", headers=H()).status_code == 204


def test_search_modes_style_and_profile(client, asr, drain):
    script = [("A", 5.0), (None, 0.6), ("B", 3.0)]
    asr.next_words = script_words(script, VOCAB)
    files = {"file": ("c.wav", wav_bytes(conversation(script)), "audio/wav")}
    rec_id = client.post("/recordings", headers=H(), files=files).json()["recording"]["id"]
    drain()

    r = client.get("/search", params={"q": "finances", "mode": "semantic"}, headers=H())
    assert r.status_code == 200 and r.json()[0]["text"].startswith("hello")
    assert client.get("/search", params={"q": "finances", "mode": "keyword"}, headers=H()).json() == []
    assert client.get("/search", params={"pace": "warp"}, headers=H()).status_code == 422
    assert client.get("/search", params={"emotion": "smug"}, headers=H()).status_code == 422
    assert client.get("/search", params={"emotion": "neutral"}, headers=H()).json()[0]["emotion"]["label"] == "neutral"
    assert len(client.get("/search", params={"recording": rec_id}, headers=H()).json()) == 2

    spk = client.get("/speakers", headers=H()).json()[0]["id"]
    prof = client.get(f"/speakers/{spk}/profile", headers=H()).json()
    assert prof["segments"] == 1 and "voice" in prof

    r = client.post(f"/recordings/{rec_id}/enrich", params={"layer": "prosody"}, headers=H())
    assert r.status_code == 202 and r.json()["kind"] == "enrich"
    assert drain() == 1
    assert "prosody" in client.get(f"/recordings/{rec_id}", headers=H()).json()["producers"]
    assert client.post(f"/recordings/{rec_id}/enrich", params={"layer": "nope"}, headers=H()).status_code == 409
