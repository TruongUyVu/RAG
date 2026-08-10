# SPECIFICATION BUỔI 08: ADVANCED HYBRID RAG PIPELINE

## 1. Workspace & Security Contract
- **Workspace Isolation**: Tất cả tài nguyên (mã nguồn, cấu hình, dữ liệu kiểm thử, storage vector) của Buổi 08 nằm hoàn toàn độc lập trong thư mục `rag_foundation/buoi_08/`.
- **Không truy cập trực tiếp Runtime khác**: Buổi 08 tuyệt đối không import runtime hay đọc `.env`/`storage` trực tiếp từ thư mục `buoi_07` hay `buoi_05`.
- **Security & Secret Management**: 
  - Khóa API (`GEMINI_API_KEY`) được quản lý qua biến môi trường trong file `.env` (không commit lên Git, tuân theo `.env.example`).
  - Không ghi log hoặc hiển thị giá trị secret ra giao diện, console hay trace output.

---

## 2. Relationship with Buổi 05 & Buổi 07
- **Buổi 05 (Chunking Foundation)**: Đóng vai trò là data provider cung cấp cấu trúc Chunks chuẩn (`fixed-size`, `hierarchical`, `semantic`).
- **Buổi 07 (Semantic RAG Baseline)**: Cung cấp mô hình Semantic RAG cơ bản (Gemini Embeddings + ChromaDB Cosine Distance) làm baseline đối chứng.
- **Buổi 08 (Advanced RAG)**: Nâng cấp kiến trúc lên **Advanced Hybrid RAG**, tích hợp thêm Lexical Search (BM25), Dung hợp kết quả (Reciprocal Rank Fusion - RRF) và Xếp hạng lại đa tầng (Cross-Encoder Re-ranking).

---

## 3. Data Contract
Các dữ liệu chunks nạp vào pipeline phải tuân thủ nghiêm ngặt JSON Schema tiêu chuẩn từ Buổi 07/08:

| Trường (Field) | Kiểu dữ liệu | Mô tả | Ràng buộc validation |
|---|---|---|---|
| `chunk_id` | `str` | Mã định danh duy nhất của chunk | Chuỗi NFC không rỗng, không trùng lặp |
| `strategy` | `str` | Chiến lược chunking | Thuộc `{"fixed-size", "hierarchical", "semantic"}` |
| `source` | `str` | Tên tài liệu văn bản gốc | Chuỗi NFC kết thúc bằng `.pdf` hoặc `.json` |
| `page_start` | `int` | Số trang bắt đầu | Số nguyên $\ge 1$ |
| `page_end` | `int` | Số trang kết thúc | Số nguyên $\ge 1$ và $\ge page\_start$ |
| `text` | `str` | Nội dung văn bản của chunk | Chuỗi NFC không rỗng sau khi `.strip()` |
| `metadata` | `dict` | Thông tin bổ sung | Dictionary chứa thuộc tính tùy chọn |

---

## 4. BM25 Tokenizer & Lexical Retrieval Contract
- **Thuật toán**: BM25 Okapi với thông số mặc định $k_1 = 1.5$, $b = 0.75$.
- **Tokenization quy chuẩn tiếng Việt**:
  - Chuyển toàn bộ ký tự về chữ thường (lowercasing).
  - Chuẩn hóa Unicode NFC (`unicodedata.normalize("NFC", text)`).
  - Loại bỏ ký tự đặc biệt, giữ lại từ ngữ tiếng Việt, số Điều/Khoản (ví dụ: `điều 5`, `khoản 2`, `nhnn`).
- **Tập ứng viên (Candidate Pool)**: Lấy ra $K_{\text{bm25}} = 20$ kết quả có điểm BM25 cao nhất.
- **Cấu trúc trả về**: `{"chunk_id": str, "bm25_score": float, "bm25_rank": int, "text": str, "source": str}`.

---

## 5. Dense Semantic Retrieval Candidate Contract
- **Model Embedding**: Gemini Embedding (`gemini-embedding-2`), kích thước vector $D = 768$ (hoặc 128 trong test environment).
- **Khoảng cách (Distance Metric)**: Cosine Distance ($d \in [0, 2]$). Điểm tương đồng được tính theo công thức $s_{\text{dense}} = 1 - d$.
- **Tập ứng viên (Candidate Pool)**: Lấy ra $K_{\text{dense}} = 20$ kết quả có khoảng cách ngắn nhất.
- **Cấu trúc trả về**: `{"chunk_id": str, "distance": float, "dense_score": float, "dense_rank": int, "text": str, "source": str}`.

---

## 6. Reciprocal Rank Fusion (RRF) Contract
- **Công thức dung hợp**:
  $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  Trong đó:
  - $M = \{\text{BM25}, \text{Dense}\}$ là tập hợp các danh sách tìm kiếm độc lập.
  - $r_m(d)$ là thứ tự xếp hạng (1-indexed rank) của tài liệu $d$ trong danh sách $m$. Nếu không xuất hiện, $r_m(d) = \infty \implies \text{thành phần} = 0$.
  - $k = 60$ (Hằng số bình ổn RRF tiêu chuẩn).
- **Đầu ra**: Tập ứng viên đã dung hợp gồm $K_{\text{fused}} = 20$ tài liệu có điểm RRF cao nhất.

---

## 7. Cross-Encoder Re-ranking Contract
- **Mô hình Re-ranker**: `BAAI/bge-reranker-v2-m3` (hoặc Cross-Encoder tương đương hỗ trợ đa ngôn ngữ tiếng Việt).
- **Đầu vào**: Cặp chuỗi `(Câu hỏi, Nội dung chunk)` cho từng ứng viên trong Top-$K_{\text{fused}}$.
- **Đầu ra**: Điểm số Logit / Probability đại diện cho mức độ liên quan ngữ nghĩa chi tiết.
- **Lọc Top-K cuối cùng**: Chọn $K_{\text{reranked}} = 5$ ứng viên có điểm Reranker cao nhất để chuyển cho LLM Generation.

---

## 8. Final Evidence & Citation Contract
- **Ngưỡng tin cậy (Confidence Gate)**: Lọc loại bỏ các ứng viên có điểm Reranker quá thấp hoặc khoảng cách vượt ngưỡng quy định.
- **Citation Format**: Trích dẫn bắt buộc ở định dạng tiêu chuẩn:
  `[Nguồn: <source>, tr. <page_start>-<page_end>, chunk: <chunk_id>]`
- **Validation Citation**: Loại bỏ hoàn toàn các trích dẫn giả do LLM tự bịa không nằm trong danh sách bằng chứng được cung cấp.

---

## 9. Pipeline Trace Contract
Để phục vụ mục đích kiểm thử và hiển thị trực quan UI, pipeline trả về đối tượng `trace` chi tiết gồm:
- `query`: Câu hỏi của người dùng.
- `bm25_top_k`: Danh sách Top-20 từ nhánh Lexical Search.
- `dense_top_k`: Danh sách Top-20 từ nhánh Dense Semantic Search.
- `rrf_top_k`: Danh sách Top-20 sau khi dung hợp RRF.
- `reranked_top_k`: Danh sách Top-5 sau khi chạy Cross-Encoder Reranker.
- `latency_ms`: Thời gian thực thi của từng công đoạn (BM25, Dense, RRF, Reranker, LLM).

---

## 10. Evaluation Metrics Contract
Bộ mô-đun đánh giá đinh lượng tính toán các chỉ số Retrieval Benchmark trên tập `eval/questions.json`:

1. **Hit Rate@K**:
   $$\text{Hit Rate}@K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( \text{Top-K}(q) \cap \text{Gold}(q) \neq \emptyset \right)$$
2. **Mean Reciprocal Rank (MRR@K)**:
   $$\text{MRR}@K = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q^{\text{first\_hit}}}$$
3. **Precision@K**:
   $$\text{Precision}@K = \frac{|\text{Top-K}(q) \cap \text{Gold}(q)|}{K}$$
4. **Recall@K**:
   $$\text{Recall}@K = \frac{|\text{Top-K}(q) \cap \text{Gold}(q)|}{|\text{Gold}(q)|}$$

---

## 11. Offline Testing Contract
- **Chạy Test Không Phụ Thuộc Network**: Tất cả các unit test trong `tests/` phải hỗ trợ chạy ở chế độ offline (sử dụng Fake/Mock Vector Embedding và Mock Reranker).
- **Mã lệnh thực thi**: `python -m unittest discover -s tests -p "test_*.py"` phải hoàn thành thành công 100% trong thời gian $< 15$ giây.

---

## 12. UI Comparison Contract
Giao diện ứng dụng Streamlit trong `app.py` được thiết kế dạng Dashboard so sánh song song:
- **Cột Trái (Left Panel)**: Baseline Semantic RAG (Buổi 07) - Chỉ dùng Dense Semantic Retrieval.
- **Cột Phải (Right Panel)**: Advanced Hybrid RAG (Buổi 08) - BM25 + Dense + RRF + Cross-Encoder Reranker.
- **Khu vực Trực quan hóa (Trace Panel)**: Bảng biểu so sánh thứ tự xếp hạng của từng chunk qua từng công đoạn (BM25 vs Dense vs RRF vs Reranker).
