"""LineCorpus: same passages whether read from disk (pread) or from RAM (CORPUS_IN_RAM=1)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retriever"))
rs = pytest.importorskip("retrieval_server")


def test_in_ram_matches_disk(tmp_path, monkeypatch):
    p = tmp_path / "c.jsonl"
    rows = [{"id": str(i), "contents": f"passage {i} é"} for i in range(50)]
    p.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode() + b"\x00" * 10)
    off = str(tmp_path / "o.npy")
    monkeypatch.delenv("CORPUS_IN_RAM", raising=False)
    disk = rs.LineCorpus(str(p), off)
    monkeypatch.setenv("CORPUS_IN_RAM", "1")
    ram = rs.LineCorpus(str(p), off)
    assert ram.buf is not None and disk.buf is None
    assert len(ram) == len(disk) == 50
    assert [ram[i] for i in range(50)] == [disk[i] for i in range(50)] == rows
