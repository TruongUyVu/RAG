"""
=============================================================================
UNIT TESTS FOR SEMANTIC CANDIDATE RETRIEVAL (BUỔI 08 - BƯỚC 05)
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

import chromadb
from advanced_rag import (
    retrieve_semantic_candidates,
    get_advanced_status,
    get_advanced_config,
)


class TestSemanticCandidateRetrieval(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_client = chromadb.PersistentClient(path=self.temp_dir)
        self.mock_gemini_client = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_semantic_retrieval_order_and_metadata(self):
        coll_name = "nhnn-hierarchical-768-fec747"
        coll = self.chroma_client.create_collection(
            name=coll_name,
            metadata={"strategy": "hierarchical", "embedding_model": "gemini-embedding-2", "embedding_dim": 768},
        )
        
        vec1 = [0.1] * 768
        vec2 = [0.9] * 768
        coll.add(
            ids=["chk_001", "chk_002"],
            embeddings=[vec1, vec2],
            documents=["Nội dung chunk 1", "Nội dung chunk 2"],
            metadatas=[
                {"chunk_id": "chk_001", "source": "TT_02.pdf", "page_start": 1, "page_end": 1},
                {"chunk_id": "chk_002", "source": "TT_02.pdf", "page_start": 2, "page_end": 2},
            ]
        )

        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = [0.1] * 768
        mock_res = MagicMock()
        mock_res.embeddings = [mock_embedding_obj]
        self.mock_gemini_client.models.embed_content.return_value = mock_res

        os.environ["GEMINI_API_KEY"] = "test_mock_key"

        results = retrieve_semantic_candidates(
            question="Cơ cấu nợ",
            strategy="hierarchical",
            candidate_k=2,
            client_helper=self.mock_gemini_client,
            chroma_client=self.chroma_client,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "chk_001")
        self.assertEqual(results[0]["semantic_rank"], 1)
        self.assertIn("semantic_distance", results[0])
        self.assertIn("source", results[0])

    def test_02_status_does_not_create_collection(self):
        cols_before = len(self.chroma_client.list_collections())
        st = get_advanced_status(strategy="hierarchical", chroma_client=self.chroma_client)
        cols_after = len(self.chroma_client.list_collections())
        self.assertEqual(cols_before, cols_after)
        self.assertFalse(st["collection_exists"])

    def test_03_missing_api_key_fails(self):
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

        with self.assertRaises(ValueError):
            retrieve_semantic_candidates(
                question="Cơ cấu nợ",
                strategy="hierarchical",
                candidate_k=2,
                chroma_client=self.chroma_client,
            )

    def test_04_no_generation_called(self):
        os.environ["GEMINI_API_KEY"] = "test_mock_key"
        coll_name = "nhnn-hierarchical-768-fec747"
        coll = self.chroma_client.create_collection(
            name=coll_name,
            metadata={"strategy": "hierarchical", "embedding_model": "gemini-embedding-2", "embedding_dim": 768},
        )
        coll.add(
            ids=["chk_001"],
            embeddings=[[0.1] * 768],
            documents=["Text"],
            metadatas=[{"chunk_id": "chk_001", "source": "TT.pdf", "page_start": 1, "page_end": 1}]
        )

        mock_emb_obj = MagicMock()
        mock_emb_obj.values = [0.1] * 768
        mock_res = MagicMock()
        mock_res.embeddings = [mock_emb_obj]
        self.mock_gemini_client.models.embed_content.return_value = mock_res

        retrieve_semantic_candidates(
            question="Hỏi đáp",
            strategy="hierarchical",
            candidate_k=1,
            client_helper=self.mock_gemini_client,
            chroma_client=self.chroma_client,
        )

        self.mock_gemini_client.models.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
