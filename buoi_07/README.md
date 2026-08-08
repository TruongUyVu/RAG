# BUỔI 07: RAG FOUNDATION PRODUCTION-READY PROJECT

Hệ thống Retrieval-Augmented Generation (RAG) hoàn chỉnh, trực quan hóa quy trình tra cứu văn bản pháp quy Ngân hàng Nhà nước Việt Nam theo chuẩn Production-Ready Patterns.

---

## 🎯 1. Mục tiêu dự án
- Xây dựng Pipeline RAG chuẩn hóa bao gồm: Data Validation, Gemini Embedding, ChromaDB Persistence, Semantic Retrieval, Confidence Gate, LLM Generation và Citation Mapping.
- Trực quan hóa toàn bộ quy trình qua giao diện Streamlit UI và các câu lệnh CLI.
- Đảm bảo 100% kiểm thử tự động (Unit Tests) chạy offline không phụ thuộc kết nối mạng hoặc API key thật.

---

## 🔄 2. Quan hệ với Buổi 05 và Buổi 06
- **Buổi 05 (Black-Box Data Provider)**: Nguồn dữ liệu chunks đã qua xử lý OCR & NFC Unicode, lưu trữ tại `rag_foundation/buoi_05/output/chunks/`. Buổi 07 chỉ được phép **ĐỌC** dữ liệu từ Buổi 05 và dùng chung môi trường ảo `.venv`.
- **Buổi 06 (Reference Baseline)**: Dự án tham khảo ban đầu. Buổi 07 nâng cấp độc lập với quy chuẩn kỹ thuật nghiêm ngặt hơn (Data Contract, Index Identity, Confidence Gate, Citation Mapping).

---

## 📐 3. Sơ đồ Pipeline Architecture

```text
[Input JSON Chunks] ➔ [Loader & Validator] ➔ [Gemini Embedding API]
                                                   │
                                                   ▼
[Streamlit UI / CLI] ◄─ [Citation Mapping] ◄─ [Confidence Gate] ◄─ [ChromaDB Vector Store]
                                                   │
                                                   ▼
                                         [Gemini Generation LLM]
```

---

## 📁 4. Cấu trúc thư mục dự án

```text
rag_foundation/buoi_07/
├── SPEC_buoi_07.md             # Quy chuẩn kỹ thuật chi tiết cho AI Agent
├── buoi_07.md                  # Hướng dẫn lộ trình thực hiện các bước
├── rag.py                      # Core RAG Pipeline Module (Loader, Embedding, Index, Query)
├── app.py                      # Giao diện ứng dụng Streamlit UI
├── requirements.txt            # Danh sách thư viện quy định (streamlit, google-genai, chromadb, python-dotenv)
├── .env.example                # File mẫu biến môi trường
├── .env                        # File biến môi trường chứa API Key thực tế (đã gitignore)
├── .gitignore                  # Cấu hình bỏ qua các file bí mật và dữ liệu tạm
├── README.md                   # Tài liệu hướng dẫn sử dụng & nghiệm thu
├── tests/                      # Thư mục chứa 30 Unit Tests tự động (Offline)
│   ├── __init__.py
│   ├── test_loader.py          # Unit tests cho Loader & Validator
│   ├── test_index.py           # Unit tests cho Embedding & ChromaDB Persistent Index
│   ├── test_query.py           # Unit tests cho Retrieval, Confidence Gate & Citation
│   └── fixtures/
│       └── chunks_sample.json  # Dữ liệu mẫu kiểm thử
└── storage/
    ├── .gitkeep
    └── chroma/                 # Lưu trữ Vector Database lâu dài (Persistent Storage)
```

---

## 📌 5. Điều kiện đầu vào & Môi trường

1. Python **>= 3.11** (Sử dụng trực tiếp `.venv` của Buổi 05).
2. Thư mục dữ liệu: `rag_foundation/buoi_05/output/chunks/` chứa 3 file JSON (`chunks_fixed_size.json`, `chunks_semantic.json`, `chunks_hierarchical.json`).

---

## 💻 6. Hướng dẫn Lệnh Thực thi (CLI Commands)

> **Lưu ý:** Chạy lệnh từ thư mục gốc của repository (thư mục chứa `rag_foundation/`).

### Windows PowerShell

#### a) Sử dụng Virtual Environment Buổi 05
```powershell
$PYTHON = ".\rag_foundation\buoi_05\.venv\Scripts\python.exe"
```

#### b) Cài đặt Requirements (nếu cần)
```powershell
& $PYTHON -m pip install -r .\rag_foundation\buoi_07\requirements.txt
```

#### c) Kiểm tra & Tạo file `.env`
Sao chép `.env.example` thành `.env` và điền `GEMINI_API_KEY`:
```powershell
Copy-Item .\rag_foundation\buoi_07\.env.example .\rag_foundation\buoi_07\.env
```

#### d) Lệnh Validate Dữ liệu
```powershell
& $PYTHON .\rag_foundation\buoi_07\rag.py validate --strategy hierarchical
```

#### e) Lệnh Kiểm tra Trạng thái (Status Read-Only)
```powershell
& $PYTHON .\rag_foundation\buoi_07\rag.py status --strategy hierarchical
```

#### f) Lệnh Index Dữ liệu vào ChromaDB
```powershell
& $PYTHON .\rag_foundation\buoi_07\rag.py index --strategy hierarchical
```

#### g) Lệnh Reset Collection và Index lại
```powershell
& $PYTHON .\rag_foundation\buoi_07\rag.py index --strategy hierarchical --reset
```

#### h) Lệnh Truy vấn Hỏi đáp RAG CLI
```powershell
& $PYTHON .\rag_foundation\buoi_07\rag.py query --strategy hierarchical --top-k 5 --question "Cơ cấu lại thời hạn trả nợ được quy định như thế nào?"
```

#### i) Lệnh Chạy 30 Unit Tests Offline
```powershell
& $PYTHON -m unittest discover -s .\rag_foundation\buoi_07\tests -v
```

#### j) Lệnh Khởi chạy Giao diện Streamlit UI
```powershell
& $PYTHON -m streamlit run .\rag_foundation\buoi_07\app.py
```
*(Để dừng ứng dụng Streamlit, nhấn nút `Ctrl + C` trong cửa sổ Terminal).*

---

### Linux / macOS

```bash
PYTHON="./rag_foundation/buoi_05/.venv/bin/python"

# Validate
$PYTHON ./rag_foundation/buoi_07/rag.py validate --strategy hierarchical

# Status
$PYTHON ./rag_foundation/buoi_07/rag.py status --strategy hierarchical

# Index
$PYTHON ./rag_foundation/buoi_07/rag.py index --strategy hierarchical

# Index Reset
$PYTHON ./rag_foundation/buoi_07/rag.py index --strategy hierarchical --reset

# Query
$PYTHON ./rag_foundation/buoi_07/rag.py query --strategy hierarchical --top-k 5 --question "Cơ cấu lại thời hạn trả nợ được quy định như thế nào?"

# Unit Tests
$PYTHON -m unittest discover -s ./rag_foundation/buoi_07/tests -v

# Streamlit UI
$PYTHON -m streamlit run ./rag_foundation/buoi_07/app.py
```

---

## ⚙️ 7. Giải thích các Biến Môi Trường (.env)

- `GEMINI_API_KEY`: API Key kết nối dịch vụ Google Gemini (Bắt buộc để Index và Query).
- `GEMINI_EMBEDDING_MODEL`: Tên mô hình vector embedding (Mặc định: `gemini-embedding-2`).
- `GEMINI_EMBEDDING_DIM`: Số chiều của vector embedding (Mặc định: `768`, giá trị hợp lệ từ 128 đến 3072).
- `GEMINI_GENERATION_MODEL`: Tên mô hình LLM sinh câu trả lời (Mặc định: `gemini-3.5-flash-lite`).
- `DEFAULT_TOP_K`: Số lượng khối văn bản truy xuất mặc định (Mặc định: `5`, giá trị từ 1 đến 20).
- `RAG_MAX_DISTANCE`: Ngưỡng khoảng cách Cosine tối đa để lọc bằng chứng tin cậy (Mặc định: `0.45`).

---

## 📚 8. Giải thích Thuật ngữ Kỹ thuật

- **Strategy**: Phương pháp cắt nhỏ văn bản (`fixed-size`, `semantic`, `hierarchical`).
- **Embedding Dimension**: Độ dài chuỗi số thực biểu diễn ngữ nghĩa của đoạn văn (768 chiều).
- **Collection Identity**: Tên định danh duy nhất cho từng tập vector trong ChromaDB theo cú pháp `nhnn-<strategy>-<dimension>-<model_hash>`.
- **Top-K**: Số lượng đoạn văn có độ tương đồng cao nhất được lấy ra từ Vector DB.
- **Cosine Distance**: Điểm khoảng cách góc giữa 2 vector. Giá trị càng nhỏ càng thể hiện sự tương đồng ngữ nghĩa cao.
- **Confidence Gate**: Bộ lọc loại bỏ các bằng chứng có `distance > RAG_MAX_DISTANCE` để tránh đưa thông tin rác vào LLM.
- **Retrieval-Only**: Trạng thái hệ thống đã lấy được nguồn tài liệu nhưng quá trình sinh câu trả lời bằng LLM bị lỗi hoặc không tạo văn bản.
- **Citation**: Trích dẫn chính xác nguồn file, số trang và chunk ID được mã hóa bằng thuật toán Python thay vì tin tưởng văn bản LLM sinh ra.

---

## 🧪 9. Kế hoạch Kiểm thử Thủ công (Manual Test Plan)

Thực hiện truy vấn 3 câu hỏi sau trên giao diện Streamlit UI hoặc CLI:

### Câu A (Trong phạm vi tài liệu):
- **Câu hỏi**: `"Cơ cấu lại thời hạn trả nợ được quy định như thế nào?"`
- **Kỳ vọng**: Trạng thái `answered`, trích dẫn nguồn `TT_02_2023_NHNN.pdf`, câu trả lời được gắn nhãn `[Nguồn: ..., tr. ..., chunk: ...]`.

### Câu B (Trong phạm vi tài liệu):
- **Câu hỏi**: `"Việc phân loại nợ và trích lập dự phòng được thực hiện như thế nào?"`
- **Kỳ vọng**: Trạng thái `answered`, trích dẫn chính xác quy định từ các Thông tư NHNN.

### Câu C (Ngoài phạm vi tài liệu):
- **Câu hỏi**: `"Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?"`
- **Kỳ vọng mong muốn**: Bị chặn bởi Confidence Gate, trả về trạng thái `insufficient_evidence` với câu thông báo `"Không tìm thấy đủ thông tin liên quan trong tài liệu đã cung cấp."`, **không bịa đặt** tên ngân hàng hay lãi suất.
*(Lưu ý: Nếu câu hỏi C vẫn vượt qua threshold, đó được ghi nhận là False Positive của retrieval/gate do ngưỡng distance cần được fine-tune).*

---

## ❓ 10. Xử lý Lỗi Thường Gặp (Troubleshooting)

1. **Lỗi `ModuleNotFoundError: No module named 'streamlit'`**:
   Chưa kích hoạt đúng interpreter của Buổi 05. Kiểm tra đường dẫn `rag_foundation/buoi_05/.venv/Scripts/python.exe`.
2. **Lỗi `Thiếu GEMINI_API_KEY`**:
   Chưa cấu hình API Key. Hãy mở file `.env` tại thư mục `buoi_07/` và điền key hợp lệ.
3. **Lỗi `Collection tồn tại nhưng không tương thích`**:
   Do thay đổi `embedding_dim` hoặc `strategy`. Chạy lệnh index kèm tham số `--reset` để tạo lại collection.
4. **Lỗi `Collection chưa được khởi tạo hoặc rỗng`**:
   Cần chạy lệnh `index` trước khi gửi câu hỏi `query`.

---

## ⚠️ 11. Cảnh báo & Giới hạn của Demo
- **Giới hạn**: Demo phục vụ mục đích học tập và hướng dẫn xây dựng RAG Pipeline.
- **Cảnh báo Pháp lý**: Câu trả lời của ứng dụng **không phải tư vấn pháp lý chính thức**.
- **An toàn Dữ liệu**: Khi thực hiện Indexing và Query, nội dung văn bản sẽ được gửi tới dịch vụ Google Gemini Cloud API. Chỉ sử dụng dữ liệu được phép chia sẻ với bên thứ ba.
