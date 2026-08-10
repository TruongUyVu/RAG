"""
=============================================================================
UNIT TESTS FOR ANSWER PIPELINE & CITATIONS (BUỔI 08 - BƯỚC 08)
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

from advanced_rag import (
    advanced_query,
    compare_retrieval_modes,
)


class TestAnswerPipelineAndCitations(unittest.TestCase):

    def setUp(self):
        self.mock_gemini_client = MagicMock()
        self.sample_candidates = [
            {
                "chunk_id": "chk_001",
                "text": "Điều 7. Quy định về cơ cấu nợ.",
                "source": "TT_02.pdf",
                "page_start": 1,
                "page_end": 1,
                "bm25_rank": 1,
                "bm25_score": 4.5,
                "semantic_rank": 1,
                "semantic_distance": 0.12,
                "rrf_score": 0.03,
                "fused_rank": 1,
                "rerank_raw_score": 2.0,
                "rerank_score": 0.88,
                "rerank_rank": 1,
                "rank_change": 0,
            },
            {
                "chunk_id": "chk_002",
                "text": "Khoản 2. Nghĩa vụ tài sản.",
                "source": "TT_02.pdf",
                "page_start": 2,
                "page_end": 2,
                "bm25_rank": 2,
                "bm25_score": 2.0,
                "semantic_rank": 2,
                "semantic_distance": 0.80,  # Unaccepted (> 0.45)
                "rrf_score": 0.01,
                "fused_rank": 2,
                "rerank_raw_score": -2.0,
                "rerank_score": 0.11,  # Unaccepted (< 0.50)
                "rerank_rank": 2,
                "rank_change": 0,
            },
        ]

    @patch("advanced_rag.retrieve_hybrid")
    @patch("advanced_rag.rerank_candidates")
    def test_01_full_answer_schema_and_citation(self, mock_rerank, mock_hybrid):
        mock_hybrid.return_value = (self.sample_candidates, {
            "bm25_candidate_count": 2, "semantic_candidate_count": 2,
            "union_count": 2, "overlap_count": 2, "fused_count": 2,
            "latency_ms": {"bm25": 1.0, "semantic": 10.0, "fusion": 1.0, "total": 12.0}
        })
        mock_rerank.return_value = self.sample_candidates

        # Mock LLM generation output with [E1] citation label
        mock_gen_res = MagicMock()
        mock_gen_res.text = "Quy định cơ cấu nợ tại Điều 7 [E1]."
        self.mock_gemini_client.models.generate_content.return_value = mock_gen_res

        res = advanced_query(
            question="Cơ cấu nợ quy định ở đâu?",
            mode="hybrid_rerank",
            strategy="hierarchical",
            client_helper=self.mock_gemini_client,
            reranker_fn=lambda q, t: [2.0, -2.0],
        )

        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["mode"], "hybrid_rerank")
        self.assertIn("trace", res)
        self.assertIn("latency_ms", res["trace"])
        self.assertTrue(res["trace"]["generation_called"])
        
        # Check evidence schema & gating
        self.assertEqual(len(res["evidence"]), 2)
        self.assertTrue(res["evidence"][0]["accepted"])
        self.assertFalse(res["evidence"][1]["accepted"])

        # Check citation mapping
        self.assertEqual(len(res["citations"]), 1)
        self.assertIn("[Nguồn: TT_02.pdf, tr. 1, chunk: chk_001]", res["answer"])

    @patch("advanced_rag.retrieve_hybrid")
    def test_02_rejected_evidence_not_in_prompt(self, mock_hybrid):
        mock_hybrid.return_value = (self.sample_candidates, {
            "bm25_candidate_count": 2, "semantic_candidate_count": 2,
            "union_count": 2, "overlap_count": 2, "fused_count": 2,
            "latency_ms": {"bm25": 1.0, "semantic": 10.0, "fusion": 1.0, "total": 12.0}
        })

        mock_gen_res = MagicMock()
        mock_gen_res.text = "Trả lời dựa trên [E1]."
        self.mock_gemini_client.models.generate_content.return_value = mock_gen_res

        advanced_query(
            question="Hỏi",
            mode="hybrid_rerank",
            strategy="hierarchical",
            client_helper=self.mock_gemini_client,
            reranker_fn=lambda q, t: [2.0, -2.0], # E1 accepted (0.88), E2 rejected (0.11)
        )

        # Check prompt content sent to LLM
        prompt_sent = self.mock_gemini_client.models.generate_content.call_args[1]["contents"]
        self.assertIn("[E1]:", prompt_sent)
        self.assertNotIn("[E2]:", prompt_sent)

    @patch("advanced_rag.retrieve_hybrid")
    def test_03_insufficient_evidence_does_not_call_generation(self, mock_hybrid):
        # All evidence rejected
        mock_hybrid.return_value = ([self.sample_candidates[1]], {
            "bm25_candidate_count": 1, "semantic_candidate_count": 1,
            "union_count": 1, "overlap_count": 1, "fused_count": 1,
            "latency_ms": {"bm25": 1.0, "semantic": 10.0, "fusion": 1.0, "total": 12.0}
        })

        res = advanced_query(
            question="Hỏi",
            mode="hybrid_rerank",
            strategy="hierarchical",
            client_helper=self.mock_gemini_client,
            reranker_fn=lambda q, t: [-2.0], # rerank_score = 0.11 < 0.50 (rejected)
        )

        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertFalse(res["trace"]["generation_called"])
        self.mock_gemini_client.models.generate_content.assert_not_called()

    @patch("advanced_rag.retrieve_hybrid")
    @patch("advanced_rag.rerank_candidates")
    def test_04_reranker_unavailable_status(self, mock_rerank, mock_hybrid):
        mock_hybrid.return_value = (self.sample_candidates, {
            "bm25_candidate_count": 2, "semantic_candidate_count": 2,
            "union_count": 2, "overlap_count": 2, "fused_count": 2,
            "latency_ms": {"bm25": 1.0, "semantic": 10.0, "fusion": 1.0, "total": 12.0}
        })
        mock_rerank.side_effect = RuntimeError("reranker_unavailable: Model not found")

        res = advanced_query(
            question="Hỏi",
            mode="hybrid_rerank",
            strategy="hierarchical",
            client_helper=self.mock_gemini_client,
        )

        self.assertEqual(res["status"], "reranker_unavailable")
        self.assertIn("reranker_unavailable", res["warnings"][0])

    @patch("advanced_rag.retrieve_bm25")
    @patch("advanced_rag.retrieve_semantic_candidates")
    def test_05_compare_does_not_call_generation(self, mock_sem, mock_bm25):
        mock_bm25.return_value = [self.sample_candidates[0]]
        mock_sem.return_value = [self.sample_candidates[0]]

        cmp_res = compare_retrieval_modes(
            question="So sánh",
            strategy="hierarchical",
            client_helper=self.mock_gemini_client,
            reranker_fn=lambda q, t: [2.0],
        )

        self.assertIn("latencies_ms", cmp_res)
        self.assertIn("comparison_table", cmp_res)
        self.assertEqual(len(cmp_res["results_by_mode"]), 4)
        self.mock_gemini_client.models.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
