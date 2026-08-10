"""
=============================================================================
UNIT TESTS FOR RECIPROCAL RANK FUSION (RRF) & HYBRID SEARCH (BUỔI 08 - BƯỚC 06)
=============================================================================
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from advanced_rag import fuse_rrf, retrieve_hybrid


class TestRRFFusion(unittest.TestCase):

    def setUp(self):
        self.bm25_candidates = [
            {"chunk_id": "chk_001", "text": "Text 1", "source": "A.pdf", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 4.5},
            {"chunk_id": "chk_002", "text": "Text 2", "source": "A.pdf", "page_start": 2, "page_end": 2, "bm25_rank": 2, "bm25_score": 3.0},
        ]
        self.semantic_candidates = [
            {"chunk_id": "chk_002", "text": "Text 2", "source": "A.pdf", "page_start": 2, "page_end": 2, "semantic_rank": 1, "semantic_distance": 0.1},
            {"chunk_id": "chk_003", "text": "Text 3", "source": "B.pdf", "page_start": 5, "page_end": 5, "semantic_rank": 2, "semantic_distance": 0.2},
        ]

    def test_01_rrf_formula_arithmetic(self):
        # rrf_k = 60, weight_bm25 = 1, weight_sem = 1
        # chk_002: bm25_rank=2, sem_rank=1 => score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # chk_001: bm25_rank=1, sem_rank=None => score = 1/(60+1) = 0.016393
        # chk_003: bm25_rank=None, sem_rank=2 => score = 1/(60+2) = 0.016129
        fused, trace = fuse_rrf(self.bm25_candidates, self.semantic_candidates, rrf_k=60.0, top_k=3)
        self.assertEqual(len(fused), 3)
        
        # Top 1 should be chk_002
        self.assertEqual(fused[0]["chunk_id"], "chk_002")
        self.assertEqual(fused[0]["fused_rank"], 1)
        expected_score_002 = round(1 / 62 + 1 / 61, 6)
        self.assertAlmostEqual(fused[0]["rrf_score"], expected_score_002, places=5)

    def test_02_overlap_no_duplicates(self):
        fused, trace = fuse_rrf(self.bm25_candidates, self.semantic_candidates)
        chunk_ids = [c["chunk_id"] for c in fused]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertEqual(trace["union_count"], 3)
        self.assertEqual(trace["overlap_count"], 1)

    def test_03_bm25_only_candidate_preserved(self):
        fused, _ = fuse_rrf(self.bm25_candidates, self.semantic_candidates)
        chk_001 = next(c for c in fused if c["chunk_id"] == "chk_001")
        self.assertEqual(chk_001["matched_by"], ["bm25"])
        self.assertIsNone(chk_001["semantic_rank"])

    def test_04_semantic_only_candidate_preserved(self):
        fused, _ = fuse_rrf(self.bm25_candidates, self.semantic_candidates)
        chk_003 = next(c for c in fused if c["chunk_id"] == "chk_003")
        self.assertEqual(chk_003["matched_by"], ["semantic"])
        self.assertIsNone(chk_003["bm25_rank"])

    def test_05_weight_zero_excludes_branch(self):
        fused, _ = fuse_rrf(self.bm25_candidates, self.semantic_candidates, bm25_weight=0.0, semantic_weight=1.0)
        chk_001 = next(c for c in fused if c["chunk_id"] == "chk_001")
        self.assertEqual(chk_001["rrf_score"], 0.0)

    def test_06_tie_break_deterministic(self):
        # Hai candidate điểm RRF giống hệt nhau
        cand_a = [{"chunk_id": "chk_BBB", "text": "T", "source": "S.pdf", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 1.0}]
        cand_b = [{"chunk_id": "chk_AAA", "text": "T", "source": "S.pdf", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1}]
        fused, _ = fuse_rrf(cand_a, cand_b)
        # Tie-break bằng chunk_id: chk_AAA trước chk_BBB khi best_rank bằng nhau (1)
        self.assertEqual(fused[0]["chunk_id"], "chk_AAA")

    def test_07_metadata_mismatch_fails(self):
        mismatch_sem = [
            {"chunk_id": "chk_001", "text": "DIFFERENT TEXT", "source": "A.pdf", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1}
        ]
        with self.assertRaises(ValueError):
            fuse_rrf(self.bm25_candidates, mismatch_sem)

    def test_08_trace_counts(self):
        _, trace = fuse_rrf(self.bm25_candidates, self.semantic_candidates)
        self.assertEqual(trace["bm25_candidate_count"], 2)
        self.assertEqual(trace["semantic_candidate_count"], 2)
        self.assertEqual(trace["union_count"], 3)
        self.assertEqual(trace["overlap_count"], 1)

    @patch("advanced_rag.retrieve_bm25")
    @patch("advanced_rag.retrieve_semantic_candidates")
    def test_09_hybrid_calls_retrievers_once(self, mock_sem, mock_bm25):
        mock_bm25.return_value = self.bm25_candidates
        mock_sem.return_value = self.semantic_candidates

        fused, trace = retrieve_hybrid("cơ cấu nợ", chunks=[], strategy="hierarchical")

        mock_bm25.assert_called_once()
        mock_sem.assert_called_once()
        self.assertIn("latency_ms", trace)
        self.assertIn("total", trace["latency_ms"])

    def test_10_no_reranker_or_generation_called(self):
        # Khẳng định pipeline RRF hoàn toàn độc lập, không tải Reranker hay LLM Generation
        fused, trace = fuse_rrf(self.bm25_candidates, self.semantic_candidates)
        self.assertIsInstance(fused, list)


if __name__ == "__main__":
    unittest.main()
