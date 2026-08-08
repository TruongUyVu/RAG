# TÀI LIỆU HƯỚNG DẪN THỰC HÀNH BUỔI 07: RAG FOUNDATION PRODUCTION-READY

Tài liệu này định hướng thứ tự các bước thực hiện xây dựng hệ thống RAG hoàn chỉnh cho Buổi 07.

---

## 📌 Liên kết quy chuẩn kỹ thuật
Vui lòng đọc kỹ toàn bộ quy chuẩn tại [SPEC_buoi_07.md](SPEC_buoi_07.md) trước khi triển khai code.

---

## 🗺️ Thứ tự các bước thực hiện

1. **Bước 01 — Kiểm tra Workspace & Dữ liệu đầu vào**: Kiểm tra môi trường `.venv` và dữ liệu chunks Buổi 05. *(Đã hoàn thành)*
2. **Bước 02 — Khởi tạo Cấu trúc Dự án & Agent Specification**: Tạo các file khung và tài liệu quy chuẩn `SPEC_buoi_07.md`. *(Đang thực hiện)*
3. **Bước 03 — Cài đặt Môi trường & Khai báo Biến**: Kiểm tra các gói thư viện và chuẩn bị file `.env`.
4. **Bước 04 — Xây dựng Core RAG Pipeline (`rag.py`)**: Triển khai logic Validation, Indexing, Vector Retrieval, Confidence Threshold, LLM Generation và Citation Tracking.
5. **Bước 05 — Xây dựng Giao diện Streamlit (`app.py`)**: Thiết kế UI trực quan hóa quy trình tra cứu RAG.
6. **Bước 06 — Kiểm thử Tự động (`tests/`)**: Viết Unit Test offline với Mock API và Fixture data.
