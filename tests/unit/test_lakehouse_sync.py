from __future__ import annotations

from src.lakehouse.delta_sink import merge_sql, partition_cols, to_record, to_records
from src.lakehouse.sync import CdcRow, apply_cdc, cdc_to_embedded_chunks, partition_cdc
from src.rag.types import Chunk, EmbeddedChunk
from src.rag.vector_stores.inmemory_store import InMemoryVectorStore


def test_delta_record_schema():
    ec = EmbeddedChunk(chunk=Chunk(id="x", text="hi", source="s", metadata={}), embedding=[0.1])
    rec = to_record(ec)
    assert rec["chunk_id"] == "x"
    assert rec["embedding_dim"] == 1


def test_to_records_preserves_order():
    items = [EmbeddedChunk(Chunk(id=str(i), text=str(i), source="s"), [0.0]) for i in range(3)]
    recs = to_records(items)
    assert [r["chunk_id"] for r in recs] == ["0", "1", "2"]


def test_partition_cols_includes_date():
    assert "ingested_date" in partition_cols()


def test_merge_sql_structure():
    sql = merge_sql()
    assert "MERGE INTO doc_chunks" in sql
    assert "ON t.chunk_id = s.chunk_id" in sql


def test_partition_cdc_classifies_changes():
    rows = [
        CdcRow("1", "s", "a", {}, [0.1], "insert"),
        CdcRow("2", "s", "b", {}, [0.2], "update_postimage"),
        CdcRow("3", "s", "c", {}, [0.3], "delete"),
    ]
    upserts, deletes = partition_cdc(rows)
    assert {r.chunk_id for r in upserts} == {"1", "2"}
    assert deletes == ["3"]


def test_cdc_to_embedded_chunks_roundtrip():
    rows = [CdcRow("1", "s", "a", {"k": "v"}, [0.1, 0.2], "insert")]
    out = cdc_to_embedded_chunks(rows)
    assert out[0].chunk.id == "1"
    assert out[0].chunk.metadata == {"k": "v"}
    assert out[0].embedding == [0.1, 0.2]


def test_apply_cdc_updates_store():
    store = InMemoryVectorStore()
    rows = [
        CdcRow("1", "s", "first", {}, [1.0, 0.0], "insert"),
        CdcRow("2", "s", "second", {}, [0.0, 1.0], "insert"),
        CdcRow("1", "s", "first", {}, [1.0, 0.0], "delete"),
    ]
    stats = apply_cdc(store, rows)
    assert stats == {"upserts": 2, "deletes": 1}
    assert store.count() == 1  # id "1" deleted after insert
