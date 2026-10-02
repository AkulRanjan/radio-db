import pytest
from fastapi.testclient import TestClient

from radiodb.api import create_app
from tests.conftest import VOCAB, conversation, script_words, wav_bytes

CONVO = [("A", 4.0), (None, 0.6), ("B", 4.0)]


@pytest.fixture
def client(db):
    return TestClient(create_app(db))


def H(user="u1"):
    return {"X-User-Id": user}


def test_upload_process_search_flow(client, asr):
    asr.next_words = script_words(CONVO, VOCAB)
    r = client.post("/recordings", headers=H(), files={"file": ("c.wav", wav_bytes(conversation(CONVO)), "audio/wav")})
    assert r.status_code == 201, r.text
    rec_id = r.json()["recording"]["id"]

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
    assert client.get("/recordings").status_code == 422  # header required
    r = client.post("/recordings", headers=H(), files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    assert client.delete(f"/recordings/{rec_id}", headers=H()).status_code == 204
