"""
=============================================================================
BUỔI 05: HƯỚNG DẪN XÂY DỰNG RAG CƠ BẢN VỚI TÀI LIỆU TIẾNG VIỆT (PDF)
=============================================================================
Bài học này giúp người mới tiếp cận RAG nắm vững các bước cơ bản:
1. Đọc văn bản PDF tiếng Việt (Document Loading)
2. Chia nhỏ văn bản (Text Chunking)
3. Tạo Embedding & lưu trữ vào Vector Store (Vector Indexing)
4. Truy vấn câu hỏi & Tìm kiếm tài liệu liên quan (Retrieval)
=============================================================================
"""

import os
import sys
import io
from pathlib import Path

# Đảm bảo in tiếng Việt không bị lỗi font trên Console Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Thư mục chứa dữ liệu demo của bài học Buổi 05
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datademo"

def buoc_1_kiem_tra_du_lieu():
    """Bước 1: Kiểm tra danh sách file PDF tiếng Việt công khai."""
    print("--- [BƯỚC 1] KIỂM TRA TÀI LIỆU PDF TRONG DATADEMO ---")
    if not DATA_DIR.exists():
        print(f"❌ Không tìm thấy thư mục: {DATA_DIR}")
        return []
    
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    print(f"📁 Tìm thấy {len(pdf_files)} file PDF công khai:")
    for file in pdf_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   - {file.name} ({size_mb:.2f} MB)")
    return pdf_files

def buoc_2_doc_file_pdf(pdf_path):
    """Bước 2: Trích xuất nội dung văn bản từ PDF (Document Loading)."""
    print(f"\n--- [BƯỚC 2] ĐỌC FILE PDF: {pdf_path.name} ---")
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        print(f"📄 Tổng số trang: {len(reader.pages)}")
        
        # Đọc 2 trang đầu tiên minh họa
        text = ""
        for i, page in enumerate(reader.pages[:2]):
            extracted = page.extract_text() or ""
            text += f"\n--- Trang {i+1} ---\n" + extracted
            
        print("📝 Trích đoạn nội dung (2 trang đầu):")
        clean_preview = text[:500].replace('\n', ' ')
        print(clean_preview + "...\n")
        return text
    except ImportError:
        print("⚠️ Chưa cài đặt thư viện 'pypdf'. Đang dùng trình đọc văn bản giả lập để minh họa luồng RAG.")
        print("💡 Gợi ý: Hãy cài đặt thư viện bằng lệnh: pip install pypdf")
        return f"Mô phỏng nội dung trích xuất từ file {pdf_path.name}: Thông tư quy định về hoạt động cho vay, cơ cấu lại thời hạn trả nợ và giữ nguyên nhóm nợ nhằm hỗ trợ khách hàng gặp khó khăn..."

def buoc_3_cat_nho_van_ban(text, chunk_size=300, chunk_overlap=50):
    """Bước 3: Chia nhỏ văn bản thành các đoạn (Text Chunking)."""
    print("\n--- [BƯỚC 3] CHIA NHỎ VĂN BẢN (TEXT CHUNKING) ---")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    
    print(f"✂️ Đã chia văn bản thành {len(chunks)} đoạn (chunks).")
    if chunks:
        print("🔍 Đọc thử Chunk #1:")
        clean_c1 = chunks[0][:150].replace('\n', ' ')
        print(f"   \"{clean_c1}...\"")
    return chunks

def buoc_4_mo_phong_retrieval(query, chunks):
    """Bước 4: Mô phỏng tìm kiếm đoạn văn bản chứa từ khóa (Keyword-based / Semantic Retrieval)."""
    print(f"\n--- [BƯỚC 4] TRUY VẤN CÂU HỎI & TÌM KIẾM (RETRIEVAL) ---")
    print(f"❓ Câu hỏi: \"{query}\"")
    
    # Tìm kiếm đoạn văn có chứa các từ khóa trong câu hỏi
    keywords = [kw.lower() for kw in query.split() if len(kw) > 2]
    matched_chunks = []
    
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(1 for kw in keywords if kw in chunk_lower)
        if score > 0:
            matched_chunks.append((score, idx, chunk))
            
    matched_chunks.sort(key=lambda x: x[0], reverse=True)
    
    if matched_chunks:
        print(f"🎯 Tìm thấy {len(matched_chunks)} đoạn phù hợp nhất:")
        for score, idx, chunk in matched_chunks[:2]:
            clean_chunk = chunk.replace('\n', ' ')
            print(f"\n   [Chunk #{idx+1} | Điểm khớp: {score}]")
            print(f"   \"{clean_chunk[:200]}...\"")
    else:
        print("ℹ️ Không tìm thấy chunk phù hợp trực tiếp với từ khóa, hiển thị Chunk 1 làm ví dụ:")
        clean_chunk = chunks[0].replace('\n', ' ')
        print(f"   \"{clean_chunk[:200]}...\"")

def main():
    print("=" * 65)
    print("  CHƯƠNG TRÌNH DEMO RAG CƠ BẢN (RAG FOUNDATION - BUỔI 05)")
    print("=" * 65)
    
    pdf_files = buoc_1_kiem_tra_du_lieu()
    if not pdf_files:
        return
        
    # Lấy file thông tư đầu tiên để minh họa
    sample_pdf = pdf_files[0]
    raw_text = buoc_2_doc_file_pdf(sample_pdf)
    
    if raw_text:
        chunks = buoc_3_cat_nho_van_ban(raw_text)
        buoc_4_mo_phong_retrieval("cơ cấu lại thời hạn trả nợ", chunks)
        
    print("\n" + "=" * 65)
    print("✅ Hoàn thành minh họa luồng RAG cơ bản cho bài học Buổi 05!")
    print("=" * 65)

if __name__ == "__main__":
    main()
