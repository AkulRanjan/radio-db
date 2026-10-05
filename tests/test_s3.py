"""S3BlobStore against moto's standalone S3 server: real HTTP, custom endpoint_url — the same
path used for MinIO / R2 / AWS in production."""

import socket

import pytest

from radiodb.blobstore import S3BlobStore, blob_key, make_blob_store
from radiodb.config import Settings
from radiodb.service import Models, RadioDB
from tests.conftest import (VOCAB, FakeEmbedder, FakeVAD, conversation, default_layers, script_words,
                            wav_bytes)

moto_server = pytest.importorskip("moto.server")
boto3 = pytest.importorskip("boto3")

BUCKET = "radiodb-test"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def s3_endpoint():
    port = _free_port()
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture
def s3(s3_endpoint, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    client = boto3.client("s3", endpoint_url=s3_endpoint)
    client.create_bucket(Bucket=BUCKET)
    yield client
    for obj in client.list_objects_v2(Bucket=BUCKET).get("Contents", []):
        client.delete_object(Bucket=BUCKET, Key=obj["Key"])
    client.delete_bucket(Bucket=BUCKET)


def _keys(client):
    return [o["Key"] for o in client.list_objects_v2(Bucket=BUCKET).get("Contents", [])]


def test_put_get_exists_delete_with_prefix_and_encryption(s3, s3_endpoint):
    store = S3BlobStore(BUCKET, prefix="blobs/", endpoint_url=s3_endpoint)
    data = b"\x00\x01audio-bytes" * 1000
    _, key = blob_key(data, "flac")

    assert not store.exists(key)
    store.put(key, data)
    store.put(key, data)  # content-addressed: second put is a no-op
    assert store.exists(key) and store.get(key) == data
    assert _keys(s3) == [f"blobs/{key}"]
    assert s3.head_object(Bucket=BUCKET, Key=f"blobs/{key}")["ServerSideEncryption"] == "AES256"

    store.delete(key)
    assert not store.exists(key) and _keys(s3) == []


def test_make_blob_store_from_settings(s3, s3_endpoint, tmp_path):
    st = Settings(home=tmp_path, blob_backend="s3", s3_bucket=BUCKET, s3_endpoint_url=s3_endpoint)
    assert isinstance(make_blob_store(st), S3BlobStore)
    with pytest.raises(ValueError):
        make_blob_store(Settings(home=tmp_path, blob_backend="s3", s3_bucket=""))


def test_service_end_to_end_on_s3(s3, s3_endpoint, tmp_path, catalog, asr):
    from datetime import timedelta

    from radiodb.catalog import _iso, now_utc

    st = Settings(home=tmp_path, blob_backend="s3", s3_bucket=BUCKET, s3_endpoint_url=s3_endpoint,
                  speaker_match_threshold=0.8, diarization_threshold=0.3, min_speaker_s=1.0)
    db = RadioDB(st, catalog=catalog,
                 models=Models(FakeVAD(), FakeEmbedder(), asr, layers=default_layers()))
    script = [("A", 4.0), (None, 0.6), ("B", 4.0)]
    asr.next_words = script_words(script, VOCAB)

    rec, _ = db.ingest("u1", wav_bytes(conversation(script)), "c.wav")
    assert _keys(s3) == [f"blobs/{rec['blob_key']}"]
    rec = db.process("u1", rec["id"])
    assert rec["status"] == "done" and len(db.transcript("u1", rec["id"])) == 2
    assert len(db.clip("u1", rec["id"], 1.0, 2.0)) > 1000

    catalog._x("UPDATE recordings SET expires_at=? WHERE id=?", (_iso(now_utc() - timedelta(days=1)), rec["id"]))
    assert db.purge_expired() == 1
    assert _keys(s3) == []  # retention reaches the cloud store
