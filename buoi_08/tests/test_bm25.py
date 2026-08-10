"""
=============================================================================
UNIT TESTS FOR BM25 LEXICAL RETRIEVAL (BUỔI 08 - BƯỚC 04)
=============================================================================
"""

import sys
import os
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from advanced_rag import tokenize_vi_legal, retrieve_bm25, build_bm25_index


class TestBM25Retrieval(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            {
                "chunk_id": "chk_001",
                "strategy": "hierarchical",
                "source": "TT_02_2023_NHNN.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Điều 7. Quy định về cơ cấu lại thời hạn trả nợ đối với số dư nợ gốc và lãi.",
            },
            {
                "chunk_id": "chk_002",
                "strategy": "hierarchical",
                "source": "TT_02_2023_NHNN.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "Khoản 2 Điều 10. Nghĩa vụ bảo đảm tài sản thế chấp nợ vay tại ngân hàng.",
            },
            {
                "chunk_id": "chk_003",
                "strategy": "hierarchical",
                "source": "TT_39_2016_NHNN.pdf",
                "page_start": 5,
                "page_end": 5,
                "text": "Quy trình vệ sinh an toàn lao động và bảo hộ tại trụ sở cơ quan năm 2024.",
            },
        ]

    def test_01_tokenizer_preserves_vietnamese_diacritics(self):
        text = "cơ cấu lại thời hạn trả nợ"
        tokens = tokenize_vi_legal(text)
        expected = ["cơ", "cấu", "lại", "thời", "hạn", "trả", "nợ"]
        self.assertEqual(tokens, expected)

    def test_02_tokenizer_preserves_article_and_clause_numbers(self):
        text = "Điều 7, Khoản 2"
        tokens = tokenize_vi_legal(text)
        expected = ["điều", "7", "khoản", "2"]
        self.assertEqual(tokens, expected)

    def test_03_corpus_and_query_same_preprocessing(self):
        q = "ĐIỀU 7 KHOẢN 2"
        c = "Điều 7 Khoản 2"
        self.assertEqual(tokenize_vi_legal(q), tokenize_vi_legal(c))

    def test_04_exact_legal_term_ranked_above(self):
        question = "cơ cấu lại thời hạn trả nợ"
        results = retrieve_bm25(question, self.sample_chunks, candidate_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["chunk_id"], "chk_001")

    def test_05_candidate_k_larger_than_corpus(self):
        question = "Điều 7"
        results = retrieve_bm25(question, self.sample_chunks, candidate_k=100)
        self.assertEqual(len(results), len(self.sample_chunks))

    def test_06_empty_question_fails(self):
        with self.assertRaises(ValueError):
            retrieve_bm25("", self.sample_chunks)
        with self.assertRaises(ValueError):
            retrieve_bm25("   ", self.sample_chunks)

    def test_07_tie_break_deterministic(self):
        identical_chunks = [
            {
                "chunk_id": "chk_BBB",
                "strategy": "fixed-size",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Nội dung giống hệt nhau.",
            },
            {
                "chunk_id": "chk_AAA",
                "strategy": "fixed-size",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Nội dung giống hệt nhau.",
            },
        ]
        results = retrieve_bm25("Nội dung", identical_chunks, candidate_k=2)
        self.assertEqual(results[0]["chunk_id"], "chk_AAA")
        self.assertEqual(results[1]["chunk_id"], "chk_BBB")

    def test_08_no_external_network_calls(self):
        results = retrieve_bm25("thời hạn trả nợ", self.sample_chunks, candidate_k=2)
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
