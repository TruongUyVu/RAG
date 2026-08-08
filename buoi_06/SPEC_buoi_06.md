# AGENT SPEC - BUỔI 06: RAG FOUNDATION WORKSHOP

Tài liệu hướng dẫn quy chuẩn cho AI Agent khi thực hiện công việc trong Buổi 06.

---

## 1. Quyền truy cập Workspace (Workspace Rules)

**Chỉ được phép đọc:**
- `RAG/rag_foundation/buoi_05/output/chunks/`
- `RAG/rag_foundation/buoi_05/.venv/`
- `RAG/rag_foundation/buoi_06/`

**Không được phép đọc:**
- Source code của Buổi 05 (hoặc các buổi trước)
- File `README.md` của các buổi trước
- File Notebooks (`.ipynb`)
- Git history (`git log`, `git diff` xem lịch sử cũ)
- Các thư mục khác ngoài phạm vi được phép

> **Lưu ý:** Buổi 5 được xem là **Black Box**. Không reverse engineering. Không phân tích cách Buổi 5 hoạt động.

---

## 2. Môi trường Python (Python Environment)
- Sử dụng đúng interpreter trong: `RAG/rag_foundation/buoi_05/.venv/`
- **Không** tạo virtual environment (`.venv`) mới.

---

## 3. Quản lý Thư viện (Package Management)
Chỉ sử dụng và cài đặt các package sau:
- `streamlit`
- `google-genai`
- `chromadb`
- `psycopg`
- `python-dotenv`

> **Lưu ý:** Không cài đặt thêm framework nào khác ngoài danh sách trên.

---

## 4. Phong cách Lập trình (Coding Style)
- **Ưu tiên:** Ít file, ít class, ít function, code ngắn gọn và dễ đọc.
- **Không tạo các kiến trúc phức tạp:** Repository pattern, Service layer, Dependency injection, Factory, Plugin pattern.

---

## 5. Phạm vi Công việc (Scope)
- **Chỉ tập trung vào:** Indexing (nạp dữ liệu), Retrieval (tìm kiếm), Answer generation (sinh câu trả lời), giao diện Streamlit.
- Không phát triển các tính năng ngoài yêu cầu.

---

## 6. Xử lý Lỗi (Error Handling)
- Chỉ áp dụng `try/except` tối thiểu tại các điểm dễ phát sinh lỗi (kết nối DB, gọi API).
- **Không cài đặt:** Retry logic phức tạp, Logging framework, Monitoring tool.

---

## 7. Bảo mật (Security)
- Không in ra console hay hiển thị trên UI: API Key, Password, Secret token.

---

## 8. Giới hạn Dòng Code (Code Size Limit)
- Mục tiêu tổng dung lượng code khoảng **300 – 500 dòng Python**.
- Nếu vượt quá **700 dòng**, hãy đơn giản hóa thiết kế kiến trúc.

---

## 9. Hướng dẫn thiết lập Môi trường (Environment Setup Prompt)

Dưới đây là prompt chuẩn bị môi trường dành cho AI Agent khi bắt đầu Bước 3:

```text
[CONTEXT] Đọc agent spec tại file SPEC_buoi_06.md 

[GOAL] 
Chuẩn bị toàn bộ môi trường để chạy project RAG. 
Đây là workshop dành cho người mới. 
Ưu tiên tự động hóa tối đa, giảm thao tác thủ công. 

--- 
[WORKSPACE] 
Chỉ được phép thao tác trong: 
- RAG/rag_foundation/buoi_06/ 
- RAG/rag_foundation/buoi_05/.venv/ 
- RAG/rag_foundation/buoi_05/output/chunks/ 
Không đọc source code của các buổi trước. 

--- 
[PYTHON] 
Sử dụng đúng Python interpreter trong: 
RAG/rag_foundation/buoi_05/.venv/ 
Không tạo virtual environment mới. 

--- 
[.ENV] 
Nếu chưa có `.env`: 
- Tạo từ `.env.example` 
Nếu thiếu các biến sau thì tự động thêm: 
GEMINI_API_KEY= 
POSTGRES_HOST=localhost 
POSTGRES_PORT=5432 
POSTGRES_DB=rag_db 
POSTGRES_USER=postgres 
POSTGRES_PASSWORD= 
Không ghi đè giá trị đã tồn tại. 

--- 
[PACKAGE] 
Kiểm tra và cài nếu còn thiếu: 
- streamlit 
- google-genai 
- chromadb 
- psycopg 
- python-dotenv 

Sau khi cài đặt: 
- Import thử từng package 
- Báo PASS hoặc FAIL 

--- 
[CHROMADB] 
Ưu tiên: 
1. Nếu phát hiện Chroma Server đang chạy thì sử dụng. 
2. Nếu không có Chroma Server: 
   Tự động sử dụng Embedded Persistent Client. 
Lưu dữ liệu tại: 
storage/chroma/ 
Không yêu cầu người dùng cài đặt ChromaDB Server. 

--- 
[SECURITY] 
- API Key 
- Password 
- Secret 
Không hardcode thông tin nhạy cảm. 

--- 
[OUTPUT] 
Hiển thị: 
- Danh sách package đã cài 
- Kết quả import từng package 
- Python interpreter đang sử dụng 
- Trạng thái ChromaDB 
  - Server 
  hoặc 
  - Embedded Local 
- Trạng thái PostgreSQL 
- Trạng thái database rag_db 
- Những việc người dùng cần thực hiện (nếu có) 

Không tạo code RAG ở bước này. Không in API Key/Password/Secret.
```
