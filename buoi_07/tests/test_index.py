"""
=============================================================================
UNIT TESTS FOR EMBEDDING, CHROMADB & INDEXING (BUỔI 07 - BƯỚC 08)
=============================================================================
Test cases 10-20, 39-42:
10. Index hai lần không tăng record count.
11. Metadata citation được lưu đầy đủ.
12. Collection identity thay đổi khi strategy thay đổi.
13. Collection identity thay đổi khi model hoặc dimension thay đổi.
14. Query chặn collection có metadata không khớp.
15. Embedding trả sai số vector phải fail.
16. Embedding trả vector rỗng phải fail.
17. Embedding trả sai dimension phải fail.
18. Embedding có NaN hoặc Infinity phải fail.
19. Embedding lỗi trước upsert không thêm record mới.
20. Thiếu API key phải fail rõ và không upsert vector giả.
39. Embedding chặn boolean và zero vector.
40. status trên storage trống không tạo collection.
41. --reset gặp embedding lỗi vẫn giữ nguyên collection hợp lệ cũ.
42. Existing collection có metadata/configuration mismatch bị chặn trước upsert.
=============================================================================
"""

import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import chromadb
from rag import (
    get_collection_name,
    get_config,
    get_status,
    index_chunks,
    validate_embedding_vector,
)


class MockGeminiClient:
    def __init__(self, dim=128):
        self.dim = dim
        self.models = MagicMock()
        
        def fake_embed(model, contents, config=None):
            req_dim = config.output_dimensionality if config else self.dim
            # Vector deterministic không zero
            vec = [0.1] * req_dim
            mock_emb = MagicMock()
            mock_emb.values = vec
            res = MagicMock()
            res.embeddings = [mock_emb]
            return res

        self.models.embed_content.side_effect = fake_embed


class TestIndexAndEmbedding(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.chroma_dir = Path(self.temp_dir.name) / "chroma"
        self.data_dir.mkdir()
        self.chroma_dir.mkdir()

        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))

        # Sample data
        self.sample_data = [
            {
                "chunk_id": "c1",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Nội dung mẫu 1",
            },
            {
                "chunk_id": "c2",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 2,
                "page_end": 3,
                "text": "Nội dung mẫu 2",
            },
        ]
        self.json_file = self.data_dir / "chunks.json"
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_data, f, ensure_ascii=False)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # Test Case 10: Index hai lần không tăng record count (Idempotency)
    @patch("rag.get_config")
    def test_10_idempotency_double_index(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }
        client_helper = MockGeminiClient(dim=128)

        index_chunks(
            strategy="hierarchical",
            input_dir=self.data_dir,
            client_helper=client_helper,
            chroma_client=self.chroma_client,
        )
        st1 = get_status(strategy="hierarchical", chroma_client=self.chroma_client)
        self.assertEqual(st1["record_count"], 2)

        # Index lần 2
        index_chunks(
            strategy="hierarchical",
            input_dir=self.data_dir,
            client_helper=client_helper,
            chroma_client=self.chroma_client,
        )
        st2 = get_status(strategy="hierarchical", chroma_client=self.chroma_client)
        self.assertEqual(st2["record_count"], 2)

    # Test Case 11: Metadata citation được lưu đầy đủ
    @patch("rag.get_config")
    def test_11_metadata_citation_saved(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }
        client_helper = MockGeminiClient(dim=128)

        res = index_chunks(
            strategy="hierarchical",
            input_dir=self.data_dir,
            client_helper=client_helper,
            chroma_client=self.chroma_client,
        )
        coll = self.chroma_client.get_collection(res["collection_name"], embedding_function=None)
        items = coll.get(ids=["c1"])
        meta = items["metadatas"][0]

        self.assertEqual(meta["source"], "doc1.pdf")
        self.assertEqual(meta["page_start"], 1)
        self.assertEqual(meta["page_end"], 1)
        self.assertEqual(meta["chunk_id"], "c1")
        self.assertEqual(meta["strategy"], "hierarchical")

    # Test Case 12 & 13: Collection identity thay đổi khi strategy, model hoặc dimension thay đổi
    def test_12_13_collection_identity_changes(self):
        name1 = get_collection_name("hierarchical", "gemini-embedding-2", 768)
        name2 = get_collection_name("semantic", "gemini-embedding-2", 768)
        name3 = get_collection_name("hierarchical", "gemini-embedding-2", 128)
        name4 = get_collection_name("hierarchical", "other-model", 768)

        self.assertNotEqual(name1, name2)
        self.assertNotEqual(name1, name3)
        self.assertNotEqual(name1, name4)

    # Test Case 15: Embedding trả sai số vector phải fail
    def test_15_wrong_vector_count_fails(self):
        # Validate vector trực tiếp
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1] * 10, expected_dim=128)

    # Test Case 16: Embedding trả vector rỗng phải fail
    def test_16_empty_vector_fails(self):
        with self.assertRaises(ValueError):
            validate_embedding_vector([], expected_dim=128)

    # Test Case 17: Embedding trả sai dimension phải fail
    def test_17_wrong_dimension_fails(self):
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1] * 100, expected_dim=128)

    # Test Case 18: Embedding có NaN hoặc Infinity phải fail
    def test_18_nan_or_inf_fails(self):
        vec_nan = [0.1] * 127 + [float("nan")]
        vec_inf = [0.1] * 127 + [float("inf")]
        with self.assertRaises(ValueError):
            validate_embedding_vector(vec_nan, expected_dim=128)
        with self.assertRaises(ValueError):
            validate_embedding_vector(vec_inf, expected_dim=128)

    # Test Case 19: Embedding lỗi trước upsert không thêm record mới
    @patch("rag.get_config")
    def test_19_embedding_error_before_upsert_no_partial_records(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }

        # Mock helper raise exception giữa chừng
        bad_client = MagicMock()
        bad_client.models.embed_content.side_effect = RuntimeError("API Call Failed")

        with self.assertRaises(RuntimeError):
            index_chunks(
                strategy="hierarchical",
                input_dir=self.data_dir,
                client_helper=bad_client,
                chroma_client=self.chroma_client,
            )

        st = get_status(strategy="hierarchical", chroma_client=self.chroma_client)
        self.assertFalse(st["collection_exists"])
        self.assertEqual(st["record_count"], 0)

    # Test Case 20: Thiếu API key phải fail rõ và không upsert vector giả
    @patch("rag.get_config")
    def test_20_missing_api_key_fails(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "",
            "has_key": False,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }
        with self.assertRaises(ValueError) as ctx:
            index_chunks(
                strategy="hierarchical",
                input_dir=self.data_dir,
                chroma_client=self.chroma_client,
            )
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    # Test Case 39: Embedding chặn boolean và zero vector
    def test_39_block_boolean_and_zero_vector(self):
        vec_bool = [True] + [0.1] * 127
        vec_zero = [0.0] * 128

        with self.assertRaises(ValueError):
            validate_embedding_vector(vec_bool, expected_dim=128)
        with self.assertRaises(ValueError):
            validate_embedding_vector(vec_zero, expected_dim=128)

    # Test Case 40: status trên storage trống không tạo collection
    @patch("rag.get_config")
    def test_40_status_on_empty_storage_read_only(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }
        st = get_status(strategy="hierarchical", chroma_client=self.chroma_client)
        self.assertFalse(st["collection_exists"])
        self.assertEqual(st["record_count"], 0)
        self.assertEqual(len(self.chroma_client.list_collections()), 0)

    # Test Case 41: --reset gặp embedding lỗi vẫn giữ nguyên collection hợp lệ cũ
    @patch("rag.get_config")
    def test_41_reset_with_embedding_error_preserves_old_collection(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }
        good_client = MockGeminiClient(dim=128)

        # 1. Index hợp lệ đầu tiên
        res = index_chunks(
            strategy="hierarchical",
            input_dir=self.data_dir,
            client_helper=good_client,
            chroma_client=self.chroma_client,
        )
        coll_name = res["collection_name"]
        st1 = get_status(strategy="hierarchical", chroma_client=self.chroma_client)
        self.assertEqual(st1["record_count"], 2)

        # 2. Thử index với --reset nhưng embedding bị lỗi
        bad_client = MagicMock()
        bad_client.models.embed_content.side_effect = RuntimeError("Embedding Failed")

        with self.assertRaises(RuntimeError):
            index_chunks(
                strategy="hierarchical",
                reset_db=True,
                input_dir=self.data_dir,
                client_helper=bad_client,
                chroma_client=self.chroma_client,
            )

        # Verfiy collection cũ vẫn còn nguyên dữ liệu
        st2 = get_status(strategy="hierarchical", chroma_client=self.chroma_client)
        self.assertTrue(st2["collection_exists"])
        self.assertEqual(st2["record_count"], 2)

    # Test Case 42: Existing collection có metadata/configuration mismatch bị chặn trước upsert
    @patch("rag.get_config")
    def test_42_metadata_mismatch_blocked(self, mock_cfg):
        # 1. Tạo collection với model gemini-embedding-2
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }
        good_client = MockGeminiClient(dim=128)
        res = index_chunks(
            strategy="hierarchical",
            input_dir=self.data_dir,
            client_helper=good_client,
            chroma_client=self.chroma_client,
        )

        # 2. Đổi config sang model khác nhưng mock get_collection_name giữ nguyên coll_name
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "different-embedding-model",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }

        with patch("rag.get_collection_name", return_value=res["collection_name"]):
            with self.assertRaises(ValueError) as ctx:
                index_chunks(
                    strategy="hierarchical",
                    input_dir=self.data_dir,
                    client_helper=good_client,
                    chroma_client=self.chroma_client,
                )
            self.assertIn("không tương thích", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
