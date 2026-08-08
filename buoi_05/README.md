# Buổi 05: Hướng Dẫn Thực Hành RAG (Retrieval-Augmented Generation) Với Văn Bản Tiếng Việt

Chào mừng bạn đến với **Buổi 05** trong chuỗi bài học RAG Cơ Bản (RAG Foundation)! 
Trong buổi học này, chúng ta sẽ cùng xây dựng một hệ thống RAG cơ bản để xử lý và hỏi đáp trên các văn bản quy định / thông tư ngân hàng (PDF tiếng Việt) công khai.

---

## 🎯 Mục Tiêu Bài Học
1. **Hiểu luồng hoạt động chính của RAG:**
   - **Document Loading:** Đọc file PDF tiếng Việt từ thư mục `datademo/`.
   - **Text Chunking:** Phân đoạn văn bản thành các đoạn nhỏ (chunks) tối ưu cho Embedding.
   - **Vector Embedding & Indexing:** Chuyển đổi các đoạn văn bản thành vector và lưu trữ vào Vector Store (ví dụ: ChromaDB / FAISS).
   - **Retrieval & Generation:** Tìm kiếm các đoạn tài liệu liên quan nhất với câu hỏi và đưa vào Prompt tạo câu trả lời.
2. **Thực hành độc lập:** Toàn bộ mã nguồn và dữ liệu thực hành nằm gọn trong thư mục `buoi_05/`.

---

## 📁 Cấu Trúc Thư Mục Buổi 05

```
RAG/
└── rag_foundation/
    └── buoi_05/
        ├── datademo/
        │   ├── TT_02_2023_NHNN.pdf
        │   ├── TT_06_2023_NHNN.pdf
        │   └── TT_39_2016_NHNN.pdf
        ├── rag_demo.py
        ├── requirements.txt
        └── README.md
```

---

## 🚀 Các Bước Thực Hành

### Bước 1: Cài đặt thư viện cần thiết
Mở terminal tại thư mục bài học và chạy lệnh:
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy chương trình demo RAG
Chạy file script `rag_demo.py`:
```bash
python rag_demo.py
```

---

## 📌 Ghi Chú Dữ Liệu
Các file PDF trong thư mục `datademo/` là các **Thông tư chính thức của Ngân hàng Nhà nước Việt Nam (NHNN)** công khai công bố trên trang tin điện tử chính phủ và NHNN:
- `TT_02_2023_NHNN.pdf`: Thông tư quy định về việc TCTD, chi nhánh ngân hàng nước ngoài cơ cấu lại thời hạn trả nợ và giữ nguyên nhóm nợ.
- `TT_06_2023_NHNN.pdf`: Thông tư sửa đổi, bổ sung một số điều của Thông tư 39/2016/TT-NHNN về hoạt động cho vay.
- `TT_39_2016_NHNN.pdf`: Thông tư quy định về hoạt động cho vay của TCTD, chi nhánh ngân hàng nước ngoài đối với khách hàng.

*Lưu ý: Dữ liệu hoàn toàn công khai, phục vụ mục đích học tập và mô phỏng hệ thống RAG xử lý tài liệu pháp lý.*
