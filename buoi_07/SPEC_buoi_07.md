# AGENT SPECIFICATION - BUỔI 07: RAG FOUNDATION PRODUCTION-READY

Tài liệu quy chuẩn chi tiết cho AI Agent thực hiện dự án RAG Foundation Buổi 07.

---

## 1. Workspace (Phạm vi thao tác)
- **Vùng được phép đọc:**
  - `rag_foundation/buoi_05/output/chunks/`
  - `rag_foundation/buoi_05/.venv/`
  - `rag_foundation/buoi_06/`
  - `rag_foundation/buoi_07/`
- **Vùng được phép ghi:**
  - `rag_foundation/buoi_07/` (duy nhất)
- **Ràng buộc tuyệt đối:** **KHÔNG** chỉnh sửa bất kỳ file nào thuộc Buổi 05 hoặc Buổi 06.

---

## 2. Python Environment
- Sử dụng đúng Python Interpreter từ môi trường ảo của Buổi 05:
  - Windows: `rag_foundation/buoi_05/.venv/Scripts/python.exe`
  - Linux/macOS: `rag_foundation/buoi_05/.venv/bin/python`
- **KHÔNG** tạo virtual environment (`.venv`) mới.

---

## 3. Input Data (Dữ liệu đầu vào)
- Đọc các file JSON chứa chunks đã xử lý sẵn từ `rag_foundation/buoi_05/output/chunks/`.
- Buổi 05 là nguồn dữ liệu chuẩn bị sẵn. **KHÔNG** chạy lại OCR, parse PDF hay chunk lại văn bản.

---

## 4. Packages (Thư viện quy định)
Chỉ được phép sử dụng các package khai báo trong `requirements.txt`:
- `streamlit>=1.61,<2`
- `google-genai>=2.16,<3`
- `chromadb>=1.5,<2`
- `python-dotenv>=1.2,<2`

---

## 5. Pipeline Architecture (Luồng xử lý RAG)
Hệ thống RAG Buổi 07 bao gồm các giai đoạn chuẩn hóa:
1. **Validate**: Kiểm tra cấu trúc dữ liệu đầu vào.
2. **Embedding**: Tạo vector biểu diễn sử dụng Gemini API.
3. **Chroma Persistent**: Lưu trữ và quản lý index vector bằng ChromaDB.
4. **Retrieval**: Tìm kiếm khối văn bản tương đồng cao nhất.
5. **Confidence Gate**: Lọc khoảng cách vector (distance threshold) để đảm bảo độ tin cậy.
6. **Generation**: Sinh câu trả lời dựa trên bằng chứng bằng Gemini LLM.
7. **Citation**: Trích dẫn nguồn và số trang từ metadata chính xác.
8. **Streamlit UI**: Giao diện tương tác người dùng.
9. **Unittest Offline**: Kiểm thử tự động không cần kết nối mạng.

---

## 6. Data Contract (Hợp đồng dữ liệu Chunk)
Mỗi đối tượng Chunk bắt buộc phải chứa đủ các trường:
- `chunk_id`: Mã định danh duy nhất của chunk
- `strategy`: Chiến lược chunking (`fixed-size`, `semantic`, `hierarchical`)
- `source`: Tên file tài liệu nguồn
- `page_start`: Trang bắt đầu (1-indexed)
- `page_end`: Trang kết thúc (1-indexed)
- `text`: Nội dung văn bản của chunk

---

## 7. Index Contract (Quy định Indexing Vector)
- Mỗi chiến lược chunking (`strategy`) được quản lý trong một Collection riêng biệt trong ChromaDB.
- Model và số chiều (dimension) của embedding khi index và query phải khớp tuyệt đối.
- Dùng embedding thật từ API, **không** dùng vector giả lập (mock vectors).
- **Chặn dữ liệu lỗi**: Không chấp nhận `NaN`, `Infinity`, `boolean` hoặc zero vector.
- Cấu hình ChromaDB dùng khoảng cách Cosine (`metadata={"hnsw:space": "cosine"}`) và `embedding_function=None`.
- Đảm bảo tính **idempotent** (chạy lại nhiều lần không nhân bản dữ liệu).
- Hàm kiểm tra trạng thái (`status`) chỉ thực hiện đọc (read-only).
- Phải validate toàn bộ embedding xong trước khi tiến hành reset/upsert vào database.

---

## 8. Retrieval Contract (Quy định Tìm kiếm)
- Trả về danh sách bằng chứng (`evidence`) thực tế từ database.
- Bắt buộc đi kèm điểm số khoảng cách (`distance`).
- Chỉ những bằng chứng đạt ngưỡng tin cậy (`distance <= RAG_MAX_DISTANCE`) mới được đưa vào ngữ cảnh sinh câu trả lời (`generation`).
- Nếu bằng chứng yếu hoặc không đạt threshold: **KHÔNG** gọi mô hình sinh câu trả lời LLM.

---

## 9. Citation Contract (Quy định Trích dẫn)
- Trích dẫn (`citation`) bắt buộc lấy từ metadata thực tế (`source`, `page_start`, `page_end`, `chunk_id`).
- **Không tin tưởng** nguồn/trang do LLM tự bịa ra trong câu trả lời text.
- Kết quả trả về chứa danh sách `citations` và `warnings`; code sẽ tự động khớp trích dẫn chuẩn.

---

## 10. Security (Bảo mật)
- **Tuyệt đối không lộ secret**: Không hardcode API Key, không log API Key hay password ra console/UI.

---

## 11. Testing (Quy định Kiểm thử)
- Sử dụng `unittest` của Python.
- Mock API và sử dụng storage tạm thời (temporary directory).
- Không yêu cầu kết nối Internet hoặc API key thật khi chạy unit tests.

---

## 12. Coding Style & Path Rule
- Mã nguồn viết tối giản: ít file, ít class, ít function, dễ đọc.
- Không áp dụng kiến trúc phức tạp (Repository, Service layer, DI,...).
- Mọi đường dẫn file trong code bắt buộc sử dụng `Path(__file__).resolve()`, **không hardcode** đường dẫn tuyệt đối theo máy cá nhân.
