"""Tests for data/download_dataset.py using a fake in-memory server (no network)."""
import io
import json

import numpy as np
import pytest

from data import download_dataset as dl


def make_npy_bytes(rows=1000, seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, (rows, 784), dtype=np.uint8)
    buf = io.BytesIO()
    np.save(buf, arr)
    return arr, buf.getvalue()


class FakeResponse:
    def __init__(self, content, status=200):
        self._content, self.status_code, self.closed = content, status, False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise dl.requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1024):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    """Serves one file; optionally ignores Range headers like a bad server."""

    def __init__(self, files, honor_range=True):
        self.files, self.honor_range, self.requests = files, honor_range, []

    def get(self, url, headers=None, stream=False, timeout=None):
        self.requests.append((url, headers))
        name = url.rsplit("/", 1)[-1]
        if name not in self.files:
            return FakeResponse(b"", status=404)
        data = self.files[name]
        if self.honor_range and headers and "Range" in headers:
            end = int(headers["Range"].split("-")[1])
            return FakeResponse(data[:end + 1], status=206)
        return FakeResponse(data)


def test_class_url():
    assert dl.class_url("cat").endswith("/numpy_bitmap/cat.npy")
    assert dl.class_url("hot dog").endswith("hot%20dog.npy")


def test_parse_header():
    _, raw = make_npy_bytes(10)
    shape, offset = dl.parse_npy_header(raw[:1024])
    assert shape == (10, 784) and offset == 128


def test_parse_header_rejects_wrong_layout():
    buf = io.BytesIO()
    np.save(buf, np.zeros((5, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        dl.parse_npy_header(buf.getvalue())


def test_download_class_saves_first_n_rows(tmp_path):
    arr, raw = make_npy_bytes(1000)
    session = FakeSession({"cat.npy": raw})
    entry = dl.download_class("cat", 100, tmp_path, session=session)
    saved = np.load(tmp_path / "cat.npy")
    assert saved.shape == (100, 784) and saved.dtype == np.uint8
    assert np.array_equal(saved, arr[:100])
    assert entry["num_samples"] == 100 and entry["total_available"] == 1000
    assert entry["bytes_downloaded"] == 128 + 100 * 784 < len(raw)


def test_works_when_server_ignores_range(tmp_path):
    arr, raw = make_npy_bytes(500)
    session = FakeSession({"dog.npy": raw}, honor_range=False)
    dl.download_class("dog", 50, tmp_path, session=session)
    assert np.array_equal(np.load(tmp_path / "dog.npy"), arr[:50])


def test_fewer_rows_available_than_requested(tmp_path):
    arr, raw = make_npy_bytes(30)
    dl.download_class("star", 500, tmp_path, session=FakeSession({"star.npy": raw}))
    assert np.load(tmp_path / "star.npy").shape == (30, 784)


def test_existing_file_skipped_unless_forced(tmp_path):
    _, raw = make_npy_bytes(200)
    session = FakeSession({"cat.npy": raw})
    dl.download_class("cat", 100, tmp_path, session=session)
    calls = len(session.requests)
    entry = dl.download_class("cat", 100, tmp_path, session=session)
    assert entry["skipped"] and len(session.requests) == calls
    dl.download_class("cat", 100, tmp_path, force=True, session=session)
    assert len(session.requests) > calls


def test_http_error_raises(tmp_path):
    with pytest.raises(dl.requests.HTTPError):
        dl.download_class("missing", 10, tmp_path, session=FakeSession({}))
    assert not (tmp_path / "missing.npy").exists()


def test_download_all_writes_manifest(tmp_path):
    files = {f"{n}.npy": make_npy_bytes(60, seed=i)[1] for i, n in enumerate(["cat", "dog"])}
    dl.download_all(["cat", "dog"], 20, tmp_path, session=FakeSession(files))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["license"] == "CC BY 4.0"
    assert [c["class"] for c in manifest["classes"]] == ["cat", "dog"]
