"""
=============================================================================
UNIT TESTS FOR LOADER & VALIDATOR (BUỔI 07 - BƯỚC 08)
=============================================================================
Test cases 1-9, 38:
1. Loader đọc JSON list.
2. Loader đọc object có field chunks.
3. Chỉ lấy đúng strategy.
4. Thiếu field bắt buộc phải fail.
5. Field sai kiểu phải fail.
6. Boolean không được chấp nhận làm page number.
7. page_start > page_end phải fail.
8. Text rỗng bị bỏ qua và thống kê đúng.
9. Duplicate chunk_id phải fail.
38. Loader chặn record không phải JSON object.
=============================================================================
"""

import json
import tempfile
import unittest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rag import load_chunks, validate_chunk


class TestLoaderAndValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_json_file(self, filename, data):
        file_path = self.dir_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return file_path

    # Test Case 1: Loader đọc JSON list
    def test_01_loader_reads_json_list(self):
        data = [
            {
                "chunk_id": "c1",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Nội dung 1",
            }
        ]
        self._create_json_file("data1.json", data)
        chunks, stats = load_chunks(input_dir=self.dir_path, strategy="hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "c1")
        self.assertEqual(stats["valid_chunks"], 1)

    # Test Case 2: Loader đọc object có field chunks
    def test_02_loader_reads_object_with_chunks_field(self):
        data = {
            "chunks": [
                {
                    "chunk_id": "c2",
                    "strategy": "hierarchical",
                    "source": "doc1.pdf",
                    "page_start": 1,
                    "page_end": 2,
                    "text": "Nội dung 2",
                }
            ]
        }
        self._create_json_file("data2.json", data)
        chunks, stats = load_chunks(input_dir=self.dir_path, strategy="hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "c2")

    # Test Case 3: Chỉ lấy đúng strategy
    def test_03_filter_exact_strategy(self):
        data = [
            {
                "chunk_id": "c1",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Text 1",
            },
            {
                "chunk_id": "c2",
                "strategy": "semantic",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Text 2",
            },
        ]
        self._create_json_file("data3.json", data)

        chunks_h, stats_h = load_chunks(input_dir=self.dir_path, strategy="hierarchical")
        self.assertEqual(len(chunks_h), 1)
        self.assertEqual(chunks_h[0]["chunk_id"], "c1")

        chunks_s, stats_s = load_chunks(input_dir=self.dir_path, strategy="semantic")
        self.assertEqual(len(chunks_s), 1)
        self.assertEqual(chunks_s[0]["chunk_id"], "c2")

    # Test Case 4: Thiếu field bắt buộc phải fail
    def test_04_missing_required_field_fails(self):
        data = [
            {
                "chunk_id": "c1",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 1,
                # Thiếu page_end và text
            }
        ]
        self._create_json_file("data4.json", data)
        with self.assertRaises(ValueError) as ctx:
            load_chunks(input_dir=self.dir_path, strategy="hierarchical")
        self.assertIn("Thiếu các trường bắt buộc", str(ctx.exception))

    # Test Case 5: Field sai kiểu phải fail
    def test_05_field_wrong_type_fails(self):
        data = [
            {
                "chunk_id": 12345,  # int thay vì str
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Text",
            }
        ]
        self._create_json_file("data5.json", data)
        with self.assertRaises(ValueError) as ctx:
            load_chunks(input_dir=self.dir_path, strategy="hierarchical")
        self.assertIn("phải là chuỗi", str(ctx.exception))

    # Test Case 6: Boolean không được chấp nhận làm page number
    def test_06_boolean_page_number_fails(self):
        data = [
            {
                "chunk_id": "c1",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": True,  # bool
                "page_end": 1,
                "text": "Text",
            }
        ]
        self._create_json_file("data6.json", data)
        with self.assertRaises(ValueError) as ctx:
            load_chunks(input_dir=self.dir_path, strategy="hierarchical")
        self.assertIn("page_start", str(ctx.exception))

    # Test Case 7: page_start > page_end phải fail
    def test_07_page_start_greater_than_page_end_fails(self):
        data = [
            {
                "chunk_id": "c1",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 5,
                "page_end": 2,
                "text": "Text",
            }
        ]
        self._create_json_file("data7.json", data)
        with self.assertRaises(ValueError) as ctx:
            load_chunks(input_dir=self.dir_path, strategy="hierarchical")
        self.assertIn("page_start", str(ctx.exception))

    # Test Case 8: Text rỗng bị bỏ qua và thống kê đúng
    def test_08_empty_text_skipped_and_counted(self):
        data = [
            {
                "chunk_id": "c1",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "   ",  # Rỗng sau strip
            },
            {
                "chunk_id": "c2",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Text hợp lệ",
            },
        ]
        self._create_json_file("data8.json", data)
        chunks, stats = load_chunks(input_dir=self.dir_path, strategy="hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(stats["empty_text_skipped"], 1)
        self.assertEqual(stats["valid_chunks"], 1)

    # Test Case 9: Duplicate chunk_id phải fail
    def test_09_duplicate_chunk_id_fails(self):
        data = [
            {
                "chunk_id": "dup_id",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Text 1",
            },
            {
                "chunk_id": "dup_id",
                "strategy": "hierarchical",
                "source": "doc2.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "Text 2",
            },
        ]
        self._create_json_file("data9.json", data)
        with self.assertRaises(ValueError) as ctx:
            load_chunks(input_dir=self.dir_path, strategy="hierarchical")
        self.assertIn("Phát hiện trùng 'chunk_id'", str(ctx.exception))

    # Test Case 38: Loader chặn record không phải JSON object
    def test_38_loader_blocks_non_dict_records(self):
        data = ["string_record_not_dict"]
        self._create_json_file("data38.json", data)
        with self.assertRaises(ValueError) as ctx:
            load_chunks(input_dir=self.dir_path, strategy="hierarchical")
        self.assertIn("phần tử record phải là json object", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
