"""
=============================================================================
UNIT TESTS FOR CROSS-ENCODER RERANKER (BUỔI 08 - BƯỚC 07)
=============================================================================
"""

import sys
import os
import math
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from advanced_rag import (
    rerank_candidates,
    retrieve_hybrid_rerank,
    _RERANKER_CACHE,
)


class TestCrossEncoderReranker(unittest.TestCase):

    def setUp(self):
        self.fused_candidates = [
            {"chunk_id": "chk_001", "text": "Đoạn 1 quy định về bảo đảm tiền vay", "source": "A.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1, "rrf_score": 0.03},
            {"chunk_id": "chk_002", "text": "Đoạn 2 quy định về vệ sinh an toàn", "source": "B.pdf", "page_start": 2, "page_end": 2, "fused_rank": 2, "rrf_score": 0.02},
            {"chunk_id": "chk_003", "text": "Đoạn 3 quy định về lãi suất cho vay", "source": "C.pdf", "page_start": 3, "page_end": 3, "fused_rank": 3, "rrf_score": 0.01},
        ]

    def test_01_lazy_loading(self):
        # Đảm bảo import module không tự động nạp model vào RAM
        self.assertNotIn(("BAAI/bge-reranker-v2-m3", "auto"), _RERANKER_CACHE)

    def test_02_mock_reranker_fn_pair_formatting_and_count(self):
        passed_pairs = []

        def fake_reranker(question, texts):
            for t in texts:
                passed_pairs.append((question, t))
            return [2.0, -1.0, 0.5]

        res = rerank_candidates(
            question="bảo đảm tiền vay",
            candidates=self.fused_candidates,
            top_k=3,
            reranker_fn=fake_reranker,
        )

        self.assertEqual(len(passed_pairs), 3)
        self.assertEqual(passed_pairs[0][0], "bảo đảm tiền vay")
        self.assertEqual(passed_pairs[0][1], self.fused_candidates[0]["text"])

    def test_03_sigmoid_score_math(self):
        def fake_reranker(question, texts):
            return [0.0]  # sigmoid(0.0) = 0.5

        res = rerank_candidates(
            question="Hỏi",
            candidates=[self.fused_candidates[0]],
            top_k=1,
            reranker_fn=fake_reranker,
        )

        self.assertEqual(res[0]["rerank_raw_score"], 0.0)
        self.assertAlmostEqual(res[0]["rerank_score"], 0.5, places=5)

    def test_04_sort_and_rank_change(self):
        # Raw logits: chk_001 -> -2.0, chk_002 -> 3.0, chk_003 -> 0.0
        # Sau rerank: chk_002 (logit 3.0) lên #1, chk_003 (logit 0.0) lên #2, chk_001 (logit -2.0) xuống #3
        def fake_reranker(question, texts):
            return [-2.0, 3.0, 0.0]

        res = rerank_candidates(
            question="Lãi suất",
            candidates=self.fused_candidates,
            top_k=3,
            reranker_fn=fake_reranker,
        )

        # Top 1: chk_002
        self.assertEqual(res[0]["chunk_id"], "chk_002")
        self.assertEqual(res[0]["rerank_rank"], 1)
        # chk_002 ban đầu fused_rank=2 -> rerank_rank=1 => rank_change = 2 - 1 = +1
        self.assertEqual(res[0]["rank_change"], 1)

        # Top 3: chk_001
        self.assertEqual(res[2]["chunk_id"], "chk_001")
        self.assertEqual(res[2]["rerank_rank"], 3)
        # chk_001 ban đầu fused_rank=1 -> rerank_rank=3 => rank_change = 1 - 3 = -2
        self.assertEqual(res[2]["rank_change"], -2)

    def test_05_limit_candidate_reranking(self):
        def fake_reranker(question, texts):
            return [1.0] * len(texts)

        res = rerank_candidates(
            question="Hỏi",
            candidates=self.fused_candidates,
            top_k=2,
            rerank_candidates_limit=2,
            reranker_fn=fake_reranker,
        )
        self.assertEqual(len(res), 2)

    def test_06_returns_only_final_top_k(self):
        def fake_reranker(question, texts):
            return [1.0, 2.0, 3.0]

        res = rerank_candidates(
            question="Hỏi",
            candidates=self.fused_candidates,
            top_k=1,
            reranker_fn=fake_reranker,
        )
        self.assertEqual(len(res), 1)

    @patch("advanced_rag.get_reranker_model")
    def test_07_model_failure_raises_error(self, mock_get_model):
        mock_get_model.side_effect = RuntimeError("reranker_unavailable: Cannot download model")

        with self.assertRaises(RuntimeError):
            rerank_candidates(
                question="Hỏi",
                candidates=self.fused_candidates,
                top_k=2,
                reranker_fn=None,
            )

    def test_08_empty_question_fails(self):
        with self.assertRaises(ValueError):
            rerank_candidates(
                question="",
                candidates=self.fused_candidates,
                top_k=2,
            )

    @patch("advanced_rag.retrieve_hybrid")
    def test_09_pipeline_hybrid_rerank_integration(self, mock_hybrid):
        mock_hybrid.return_value = (self.fused_candidates, {
            "bm25_candidate_count": 3,
            "semantic_candidate_count": 3,
            "union_count": 3,
            "overlap_count": 0,
            "fused_count": 3,
            "latency_ms": {"bm25": 1.0, "semantic": 10.0, "fusion": 1.0, "total": 12.0}
        })

        def fake_reranker(question, texts):
            return [1.0, 0.5, 0.1]

        results, trace = retrieve_hybrid_rerank(
            question="Q",
            strategy="hierarchical",
            reranker_fn=fake_reranker,
        )

        self.assertIn("rerank", trace["latency_ms"])
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
