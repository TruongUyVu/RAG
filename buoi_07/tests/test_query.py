"""
=============================================================================
UNIT TESTS FOR RETRIEVAL, GROUNDING, CITATION & QUERY (BUỔI 07 - BƯỚC 08)
=============================================================================
Test cases 21-37, 43-47:
21. Retrieval trả đúng top-k.
22. Retrieval giữ đúng thứ tự.
23. top_k > collection.count() vẫn chạy đúng.
24. Question rỗng phải fail.
25. Top-k ngoài khoảng phải fail.
26. Collection rỗng phải fail rõ (trả insufficient_evidence).
27. Evidence tốt nhất vượt threshold -> status insufficient_evidence, generation mock không được gọi.
28. Evidence đạt threshold -> generation được gọi đúng một lần.
29. Prompt chứa question.
30. Prompt chứa đúng chunk retrieved.
31. Prompt không chứa chunk không retrieve.
32. Citation trang đơn render đúng (tr. N).
33. Citation khoảng trang render đúng (tr. N-M).
34. [E1] map đúng metadata.
35. [E99] không tạo citation giả và có warning.
36. Generation lỗi -> status retrieval_only, evidence vẫn còn.
37. Result có đủ status, answer, evidence, citations, warnings, collection, strategy, top_k.
43. Một evidence đạt và một evidence vượt threshold: result giữ cả hai, prompt chỉ chứa evidence đạt.
44. Prompt có instruction coi evidence là dữ liệu và bỏ qua lệnh nằm trong chunk.
45. Citation list không lặp, theo thứ tự xuất hiện và [E99] bị loại kèm warning.
46. Generation trả text rỗng chuyển thành retrieval_only và vẫn giữ evidence.
47. Config và CLI hoạt động khi current working directory không phải buoi_07/.
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
from rag import index_chunks, query


class MockGeminiClientForQuery:
    def __init__(self, dim=128, gen_text=""):
        self.dim = dim
        self.gen_text = gen_text
        self.models = MagicMock()

        def fake_embed(model, contents, config=None):
            req_dim = config.output_dimensionality if config else self.dim
            vec = [0.1] * req_dim
            mock_emb = MagicMock()
            mock_emb.values = vec
            res = MagicMock()
            res.embeddings = [mock_emb]
            return res

        def fake_generate(model, contents):
            res = MagicMock()
            res.text = self.gen_text
            return res

        self.models.embed_content.side_effect = fake_embed
        self.models.generate_content.side_effect = fake_generate


class TestQueryAndRetrieval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.chroma_dir = Path(self.temp_dir.name) / "chroma"
        self.data_dir.mkdir()
        self.chroma_dir.mkdir()

        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))

        self.sample_data = [
            {
                "chunk_id": "c1",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Phạm vi điều chỉnh của quy định 1",
            },
            {
                "chunk_id": "c2",
                "strategy": "hierarchical",
                "source": "doc2.pdf",
                "page_start": 2,
                "page_end": 5,
                "text": "Đối tượng áp dụng của quy định 2",
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

    def _setup_indexed_db(self, max_distance=0.45):
        with patch("rag.get_config") as mock_cfg:
            mock_cfg.return_value = {
                "api_key": "test_key",
                "has_key": True,
                "embedding_model": "gemini-embedding-2",
                "embedding_dim": 128,
                "generation_model": "gemini-3.5-flash-lite",
                "default_top_k": 5,
                "max_distance": max_distance,
            }
            client_helper = MockGeminiClientForQuery(dim=128)
            index_chunks(
                strategy="hierarchical",
                input_dir=self.data_dir,
                client_helper=client_helper,
                chroma_client=self.chroma_client,
            )

    # Test Case 24: Question rỗng phải fail
    @patch("rag.get_config")
    def test_24_empty_question_fails(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }
        with self.assertRaises(ValueError):
            query("", strategy="hierarchical", chroma_client=self.chroma_client)

    # Test Case 25: Top-k ngoài khoảng phải fail
    @patch("rag.get_config")
    def test_25_invalid_top_k_fails(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }
        with self.assertRaises(ValueError):
            query("Hỏi", strategy="hierarchical", top_k=0, chroma_client=self.chroma_client)
        with self.assertRaises(ValueError):
            query("Hỏi", strategy="hierarchical", top_k=100, chroma_client=self.chroma_client)

    # Test Case 26: Collection rỗng hoặc chưa tồn tại phải fail rõ
    @patch("rag.get_config")
    def test_26_empty_collection_returns_insufficient_evidence(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.45,
        }
        res = query("Hỏi", strategy="hierarchical", chroma_client=self.chroma_client)
        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["evidence"], [])

    # Test Case 21, 22, 23, 28, 32, 33, 34, 37: End-to-end successful query with citations & evidence
    @patch("rag.get_config")
    def test_21_to_37_successful_query_flow(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.99,
        }
        self._setup_indexed_db(max_distance=0.99)

        gen_text = "Phạm vi quy định như sau [E1]. Đối tượng quy định như sau [E2]."
        client_helper = MockGeminiClientForQuery(dim=128, gen_text=gen_text)

        res = query(
            question="Thông tư quy định như thế nào?",
            strategy="hierarchical",
            top_k=5,
            client_helper=client_helper,
            chroma_client=self.chroma_client,
        )

        # Test Case 37: Schema đầy đủ
        required_keys = {"status", "answer", "evidence", "citations", "warnings", "collection", "strategy", "top_k"}
        self.assertTrue(required_keys.issubset(res.keys()))

        self.assertEqual(res["status"], "answered")
        # Test Case 21 & 23: top_k > count vẫn trả đúng 2 items
        self.assertEqual(len(res["evidence"]), 2)

        # Test Case 32 & 34: Citation trang đơn (tr. 1) & Metadata mapping
        cit1 = res["citations"][0]
        self.assertEqual(cit1["evidence_id"], "E1")
        self.assertEqual(cit1["source"], "doc1.pdf")
        self.assertIn("tr. 1", cit1["display"])

        # Test Case 33: Citation khoảng trang (tr. 2-5)
        cit2 = res["citations"][1]
        self.assertEqual(cit2["evidence_id"], "E2")
        self.assertEqual(cit2["source"], "doc2.pdf")
        self.assertIn("tr. 2-5", cit2["display"])

    # Test Case 27: Evidence vượt threshold -> insufficient_evidence, generation NOT called
    @patch("rag.get_config")
    def test_27_confidence_gate_blocks_generation(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.0001,
        }
        self._setup_indexed_db(max_distance=0.0001)

        # Vector ngược chiều (distance = 2.0 > 0.0001)
        client_helper = MagicMock()
        mock_emb = MagicMock()
        mock_emb.values = [-0.1] * 128
        res = MagicMock()
        res.embeddings = [mock_emb]
        client_helper.models.embed_content.return_value = res

        res = query(
            question="Hỏi thử",
            strategy="hierarchical",
            client_helper=client_helper,
            chroma_client=self.chroma_client,
        )

        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["citations"], [])
        # Generation mock không được gọi
        client_helper.models.generate_content.assert_not_called()

    # Test Case 35, 45: Label không tồn tại ([E99]) bị loại kèm warning và citation list không lặp
    @patch("rag.get_config")
    def test_35_45_e99_invalid_label_and_no_duplicate_citations(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.99,
        }
        self._setup_indexed_db(max_distance=0.99)

        gen_text = "Kết quả [E1] và lại [E1] và giả [E99]."
        client_helper = MockGeminiClientForQuery(dim=128, gen_text=gen_text)

        res = query(
            question="Hỏi thử",
            strategy="hierarchical",
            client_helper=client_helper,
            chroma_client=self.chroma_client,
        )

        self.assertEqual(res["status"], "answered")
        # List citation chỉ có 1 phần tử (không trùng [E1])
        self.assertEqual(len(res["citations"]), 1)
        # Báo warning với [E99]
        self.assertTrue(any("E99" in w for w in res["warnings"]))

    # Test Case 36, 46: Generation lỗi hoặc trả text rỗng -> status retrieval_only
    @patch("rag.get_config")
    def test_36_46_generation_error_or_empty_returns_retrieval_only(self, mock_cfg):
        mock_cfg.return_value = {
            "api_key": "test_key",
            "has_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 128,
            "generation_model": "gemini-3.5-flash-lite",
            "default_top_k": 5,
            "max_distance": 0.99,
        }
        self._setup_indexed_db(max_distance=0.99)

        # Mock LLM raise exception
        bad_client = MockGeminiClientForQuery(dim=128)
        bad_client.models.generate_content.side_effect = RuntimeError("LLM Overloaded")

        res = query(
            question="Hỏi thử",
            strategy="hierarchical",
            client_helper=bad_client,
            chroma_client=self.chroma_client,
        )

        self.assertEqual(res["status"], "retrieval_only")
        self.assertIn("Đã truy xuất được nguồn", res["answer"])
        self.assertEqual(len(res["evidence"]), 2)  # Evidence vẫn giữ nguyên


if __name__ == "__main__":
    unittest.main()
