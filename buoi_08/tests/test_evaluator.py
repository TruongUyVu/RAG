"""
=============================================================================
UNIT TESTS FOR EVALUATOR METRICS & BENCHMARK REPORT (BUỔI 08 - BƯỚC 10)
=============================================================================
"""

import sys
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluate import (
    calculate_recall_at_k,
    calculate_mrr_at_k,
    calculate_ndcg_at_k,
    calculate_latency_stats,
    run_evaluation,
)


class TestEvaluatorMetrics(unittest.TestCase):

    def test_01_recall_hand_calculated(self):
        retrieved = ["A", "B", "C", "D", "E"]
        relevant = ["B", "D"]
        # Top-5 contains both B and D => 2/2 = 1.0
        self.assertEqual(calculate_recall_at_k(retrieved, relevant, k=5), 1.0)

        # Top-1 only contains A => 0/2 = 0.0
        self.assertEqual(calculate_recall_at_k(retrieved, relevant, k=1), 0.0)

    def test_02_mrr_hand_calculated(self):
        retrieved = ["A", "B", "C", "D", "E"]
        relevant = ["B", "D"]
        # First match is B at rank 2 => 1/2 = 0.5
        self.assertEqual(calculate_mrr_at_k(retrieved, relevant, k=5), 0.5)

        retrieved_first = ["B", "A", "C"]
        # First match is B at rank 1 => 1/1 = 1.0
        self.assertEqual(calculate_mrr_at_k(retrieved_first, relevant, k=5), 1.0)

    def test_03_ndcg_hand_calculated(self):
        retrieved = ["A", "B", "C", "D", "E"]
        relevant = ["B", "D"]
        # DCG@5 = 1/log2(3) + 1/log2(5) = 0.6309297 + 0.4306765 = 1.0616062
        # IDCG@5 = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309297 = 1.6309297
        # nDCG@5 = 1.0616062 / 1.6309297 = 0.6509
        ndcg_val = calculate_ndcg_at_k(retrieved, relevant, k=5)
        self.assertAlmostEqual(ndcg_val, 0.6509, places=3)

    def test_04_latency_stats_mean_and_p50(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = calculate_latency_stats(latencies)
        self.assertEqual(stats["mean_ms"], 30.0)
        self.assertEqual(stats["p50_ms"], 30.0)

    def test_05_run_evaluation_offline_mock(self):
        mock_questions = [
            {
                "query_id": "Q01",
                "question": "Điều 7",
                "relevant_chunk_ids": ["chk_001"],
                "scope": "in_scope",
                "needs_human_review": True
            }
        ]

        # Run offline evaluation with BM25 mode only
        report = run_evaluation(
            eval_questions=mock_questions,
            strategy="hierarchical",
            k=5,
            modes=["bm25"],
        )

        self.assertEqual(report["questions_count"], 1)
        self.assertTrue(report["has_human_review_flag"])
        self.assertIn("bm25", report["mode_summaries"])
        self.assertTrue(len(report["warnings"]) > 0)
        self.assertIn("NEEDS_HUMAN_REVIEW", report["warnings"][0].upper())


if __name__ == "__main__":
    unittest.main()
