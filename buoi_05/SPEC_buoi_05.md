# AGENT SPEC - BUỔI 05: OCR VĂN BẢN TIẾNG VIỆT & CHUNKING STRATEGIES

Tài liệu hướng dẫn và quy chuẩn kỹ thuật dành cho AI Agent trong phạm vi bài học Buổi 05.

---

## 1. Mục tiêu & Phạm vi (Goal & Scope)
- **Mục tiêu:** Thực hiện xử lý trích xuất văn bản (OCR) từ các tài liệu PDF tiếng Việt và áp dụng, so sánh hiệu quả của 3 chiến lược cắt nhỏ văn bản (Chunking).
- **Phạm vi:** Code ở mức **Demo đơn giản**, dễ hiểu, tập trung vào bản chất luồng dữ liệu, không phức tạp hóa kiến trúc.

---

## 2. Đầu vào (Input)
- Các file tài liệu PDF tiếng Việt được lưu trữ tại thư mục:
  `RAG/rag_foundation/buoi_05/datademo/`
  *(Ví dụ: `TT_02_2023_NHNN.pdf`, `TT_06_2023_NHNN.pdf`, `TT_39_2016_NHNN.pdf`)*

---

## 3. Đầu ra (Output Requirements)

### 3.1. Dữ liệu Văn bản (Extracted Text)
- Văn bản sau khi OCR/trích xuất phải được chuẩn hóa về định dạng **Unicode NFC** (`unicodedata.normalize('NFC', text)`).

### 3.2. Cấu trúc Metadata (Chunk Metadata)
Mỗi chunk văn bản xuất ra bắt buộc chứa đầy đủ các trường thông tin sau:
- `source`: Tên file PDF nguồn (ví dụ: `TT_02_2023_NHNN.pdf`)
- `page`: Số trang tương ứng trong file PDF (1-indexed)
- `ocr_used`: Công cụ trích xuất được sử dụng (ví dụ: `PyMuPDF` hoặc `LlamaParse`)
- `language`: Ngôn ngữ của tài liệu (`vi`)

### 3.3. Báo cáo Bảng so sánh (Comparison Report)
- Báo cáo kết quả so sánh giữa 3 chiến lược chunking bao gồm: tổng số chunk tạo ra, độ dài trung bình của chunk, và mức độ giữ vẹn nguyên cấu trúc câu/điều khoản.

---

## 4. Ba Chiến lược Chunking cần So sánh (Chunking Strategies)

1. **Fixed-size Chunking (Cắt theo kích thước cố định):**
   - Cắt văn bản theo số lượng ký tự hoặc token cố định (ví dụ: chunk_size = 500 characters).
   - Có sử dụng vùng chồng lấp giữa các chunk kế tiếp (chunk_overlap = 50 characters).

2. **Semantic Chunking (Cắt theo ngữ nghĩa / ranh giới đoạn):**
   - Ưu tiên cắt dựa trên ranh giới tự nhiên của văn bản như xuống dòng (`\n\n`), ngắt đoạn, kết đoạn hoặc khoảng trắng phân chia các khối nội dung.

3. **Hierarchical Chunking (Cắt phân cấp theo cấu trúc):**
   - Phân chia dựa theo cấu trúc văn bản pháp quy Việt Nam.
   - Mỗi mốc phân mục cấp bậc: **Chương** → **Mục** → **Điều/Khoản** → **Điểm** sẽ trở thành mốc bắt đầu của 1 chunk mới.

---

## 5. Quy tắc Bảo mật & Quản lý `.env`
- Các thông tin cấu hình và API Key (nếu có) được đọc từ file `.env` đặt trong thư mục dự án hoặc `src/`.
- **Ràng buộc bảo mật:** Được phép đọc key để phục vụ kết nối dịch vụ nhưng **TUYỆT ĐỐI KHÔNG** in (print/log/show UI) giá trị secret/API Key dưới bất kỳ hình thức nào.

---

## 6. Ràng buộc Tối quan trọng (Strict Constraints)

> [!IMPORTANT]
> Trong Buổi 05, hệ thống **CHỈ** dừng lại ở bước OCR và Chunking.

- ❌ **KHÔNG** tạo Embedding vectors.
- ❌ **KHÔNG** lưu dữ liệu vào Vector Database (như ChromaDB, FAISS, Pinecone,...).
- ❌ **KHÔNG** gọi mô hình ngôn ngữ lớn (LLM) để sinh câu trả lời.
- ✅ Code thiết kế tối giản, dễ đọc, viết dưới dạng demo phục vụ workshop giảng dạy, không bỏ sót các yêu cầu trên.
