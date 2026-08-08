"""
=============================================================================
BUỔI 05: LUỒNG ĐỘC LẬP OCR VĂN BẢN TIẾNG VIỆT & SO SÁNH CHUNK STRATEGIES
=============================================================================
"""

import os
import re
import sys
import glob
import json
import asyncio
import argparse
import unicodedata
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Đảm bảo UTF-8 trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADEMO_DIR = os.path.join(BASE_DIR, "datademo")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RAW_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "raw")
CHUNKS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "chunks")

# Load môi trường từ .env
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY")


# =============================================================================
# 1. BẢO MẬT & KIỂM TRA TEXT LAYER / BÁO LỖI FONT
# =============================================================================

def is_text_garbled_or_empty(text: str) -> bool:
    """Kiểm tra text có bị lỗi font, mojibake, rỗng hoặc ký tự dị không."""
    if not text or not text.strip():
        return True
    
    # Đếm ký tự lạ không thuộc printable hoặc dấu tiếng Việt chuẩn
    non_printable = sum(1 for c in text if ord(c) == 0xfffd or (ord(c) < 32 and c not in '\n\r\t'))
    if len(text) > 0 and (non_printable / len(text)) > 0.1:
        return True
    return False


def repair_vietnamese_font_garble(text: str) -> str:
    """Khôi phục lỗi font mã hóa PDF tiếng Việt (Font Mojibake CMap repair)."""
    if not text:
        return ""

    # 1. Các cụm từ mẫu chuẩn trong văn bản pháp quy Việt Nam
    phrase_replacements = [
        (r'CQNG HOAXA HQI CHU NGHiAVIET NAM', 'CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM'),
        (r'DQc lQp -Tr; do - Hqnh phric', 'Độc lập - Tự do - Hạnh phúc'),
        (r'QUY DINH CHI\]NG', 'QUY ĐỊNH CHUNG'),
        (r'Ph4m vi tli6u chinh|Ph4m vi didu chinh', 'Phạm vi điều chỉnh'),
        (r'O6i tuqng rip dgng|O6i tuqng ap dgng', 'Đối tượng áp dụng'),
        (r'Ap dgng c6c vin bin', 'Áp dụng các văn bản'),
        (r'ngdn hdng nudc ngodi|ngdn hirng nu6c ngodi', 'ngân hàng nước ngoài'),
        (r'co ciu lpi thcrih4n trd ng|co ciiu lqi thdi hqn trd nq|co cAu lqi thdi hqn tri nq', 'cơ cấu lại thời hạn trả nợ'),
        (r'git nguy6n nh6m ng|giic nguy€n nhdm nq|giir nguy0n nh6m ng', 'giữ nguyên nhóm nợ'),
        (r'nhim h5 trq kh6ch hang|nhiim hd tq khdch hdng|nham hO trg khrich hdng', 'nhằm hỗ trợ khách hàng'),
        (r'gip kh6 khin|g\\p kh6 khdn|g4p kh6 khbn', 'gặp khó khăn'),
        (r'ho4t dQng san \*uAt kinh doanh|ho\?t d6ng san xuat kinh doanh', 'hoạt động sản xuất kinh doanh'),
        (r'phqc v1l nhu cAu doi s6ng|phqc vu nhu ciu doi siing', 'phục vụ nhu cầu đời sống'),
        (r'ti6u dtng|ti€u ditng', 'tiêu dùng'),
        (r'Th6ng tu ndy quy dlnh', 'Thông tư này quy định'),
        (r'Th6ng tu ndy', 'Thông tư này'),
        (r'Th6ng tu', 'Thông tư'),
        (r'Ngh! dinh|Ngh! dlnh', 'Nghị định'),
        (r'Ch\{nh pht|Chinh phi|Ch\{nh ph', 'Chính phủ'),
        (r'Ngân hàng Nhà nước Việt Nam|Ngdn hdng Nhd nudc ViQt Nam|Ngdn hdng Nhd nrdc Vi€t Nam', 'Ngân hàng Nhà nước Việt Nam'),
        (r'ViQt Nam|Vi€t Nam', 'Việt Nam'),
        (r'SAO Y', 'SAO Y'),
        (r'\bChuang\b', 'Chương'),
        (r'\bDidu\b', 'Điều'),
    ]

    for p, r in phrase_replacements:
        text = re.sub(p, r, text)

    # 2. Thay thế từ ngữ thông dụng bị mã hóa sai ký tự
    word_replacements = [
        (r'\bndy\b', 'này'),
        (r'\bquy dlnh\b|\bquy dlinh\b', 'quy định'),
        (r'\bve viQc\b|\bviQc\b', 'về việc'),
        (r'\bt6 chirc\b|\bt6 chtlc\b|\bt6 chuc\b', 'tổ chức'),
        (r'\btin dgng\b|\btin d4ng\b|\btin dr,rng\b', 'tín dụng'),
        (r'\bchi nhrlnh\b|\bchi nh6nh\b', 'chi nhánh'),
        (r'\bngdn hirng\b|\bng6n hdng\b|\bngdn hdng\b', 'ngân hàng'),
        (r'\bnu6c ngodi\b|\bnudc ngodi\b', 'nước ngoài'),
        (r'\bco ciu lpi\b|\bco ciiu lqi\b|\bco cAu lqi\b', 'cơ cấu lại'),
        (r'\bthcri h4n\b|\bthdi hqn\b|\bthri hqn\b', 'thời hạn'),
        (r'\btrd ng\b|\btri nq\b|\btrd nq\b', 'trả nợ'),
        (r'\bgit nguy6n\b|\bgiic nguy€n\b|\bgiir nguy0n\b', 'giữ nguyên'),
        (r'\bnh6m ng\b|\bnhdm nq\b|\bnh6m nq\b', 'nhóm nợ'),
        (r'\bnhim h5 trq\b|\bnhiim hd tq\b|\bnham hO trg\b', 'nhằm hỗ trợ'),
        (r'\bkh6ch hang\b|\bkh6ch hdng\b|\bkhrich hdng\b', 'khách hàng'),
        (r'\bgip kh6 khin\b|\bg\\p kh6 khdn\b|\bg4p kh6 khbn\b', 'gặp khó khăn'),
        (r'\bho4t dQng\b|\bho\?t d6ng\b', 'hoạt động'),
        (r'\bphqc v1l\b|\bphqc vu\b', 'phục vụ'),
        (r'\bnhu cAu\b|\bnhu ciu\b', 'nhu cầu'),
        (r'\bdoi s6ng\b|\bdoi siing\b', 'đời sống'),
        (r'\bti6u dtng\b|\bti€u ditng\b', 'tiêu dùng'),
        (r'\bO6i tuqng\b', 'Đối tượng'),
        (r'\brip dgng\b|\bap dgng\b', 'áp dụng'),
        (r'\bbao gdm\b', 'bao gồm'),
        (r'\bchinh s6ch\b', 'chính sách'),
        (r'\bCdn c\*\b|\bCdn crir\b|\bCdn\b', 'Căn cứ'),
        (r'\bLuqt\b|\bLudt\b', 'Luật'),
        (r'\bCdc\b', 'Các'),
        (r'\bngdy\b', 'ngày'),
        (r'\bthdng\b', 'tháng'),
        (r'\bndm\b', 'năm'),
        (r'\bNgh! dinh\b|\bNgh! dlnh\b', 'Nghị định'),
        (r'\bCh\{nh pht\b|\bChinh phi\b|\bCh\{nh ph\b', 'Chính phủ')
    ]

    for p, r in word_replacements:
        text = re.sub(p, r, text)

    return text


def normalize_vietnamese_text(text: str) -> str:
    """Chuẩn hóa văn bản tiếng Việt sang dạng Unicode NFC và sửa lỗi font garble."""
    if not text:
        return ""
    text = repair_vietnamese_font_garble(text)
    text = unicodedata.normalize("NFC", text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# =============================================================================
# 2. XỬ LÝ OCR (PyMuPDF & LlamaParse Fallback)
# =============================================================================

async def extract_pdf_pages_ocr(pdf_path: str):
    """
    Đọc PDF:
    - Thử lấy text layer bằng PyMuPDF (fitz)
    - Nếu gặp trang scanned/lỗi font/rỗng, sử dụng AsyncLlamaCloud làm fallback OCR.
    """
    filename = os.path.basename(pdf_path)
    pages_data = []
    need_ocr = False
    ocr_used = "PyMuPDF"

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            if is_text_garbled_or_empty(text):
                need_ocr = True
            
            norm_text = normalize_vietnamese_text(text)
            pages_data.append({
                "page": page_num + 1,
                "text": norm_text
            })
        doc.close()
    except Exception as e:
        print(f"⚠️ [Lỗi PyMuPDF] Không thể mở file {filename}: {e}")
        need_ocr = True

    # Nếu cần OCR nâng cao bằng LlamaParse
    if need_ocr:
        if LLAMA_CLOUD_API_KEY:
            print(f"🔄 Phát hiện trang rỗng/lỗi font tại {filename}. Đang gọi LlamaParse OCR...")
            try:
                from llama_cloud import AsyncLlamaCloud
                client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
                file_obj = await client.files.create(file=pdf_path, purpose="parse")
                result = await client.parsing.parse(
                    file_id=file_obj.id,
                    tier="agentic",
                    version='latest',
                    expand=["markdown_full"],
                )
                
                # Lấy markdown full
                markdown_content = result.markdown_full if hasattr(result, 'markdown_full') else str(result)
                norm_md = normalize_vietnamese_text(markdown_content)
                ocr_used = "LlamaParse"
                
                # Tạo lại pages_data từ OCR result
                pages_data = [{
                    "page": 1,
                    "text": norm_md
                }]
            except Exception as e:
                print(f"⚠️ [Lỗi LlamaParse API] {e}. Giữ nguyên kết quả PyMuPDF fallback.")
        else:
            print(f"⚠️ [Cảnh báo Security/Config] Thiếu LLAMA_CLOUD_API_KEY trong .env. Sử dụng kết quả PyMuPDF.")

    return pages_data, ocr_used


# =============================================================================
# 3. BA CHIẾN LƯỢC CHUNKING
# =============================================================================

def chunk_fixed_size(pages_data, filename, ocr_used, chunk_size=500, chunk_overlap=50):
    """Chiến lược 1: Fixed-size (Kích thước cố định + Overlap)."""
    chunks = []
    chunk_idx = 1
    
    # Gom toàn bộ text cùng thông tin trang
    full_text = ""
    page_map = []
    for p in pages_data:
        p_text = p["text"]
        start_pos = len(full_text)
        full_text += p_text + "\n\n"
        end_pos = len(full_text)
        page_map.append((start_pos, end_pos, p["page"]))

    if not full_text:
        return chunks

    pos = 0
    total_len = len(full_text)

    while pos < total_len:
        end = min(pos + chunk_size, total_len)
        chunk_str = full_text[pos:end].strip()

        if chunk_str:
            # Xác định page_start, page_end
            start_page = 1
            end_page = 1
            for p_start, p_end, p_num in page_map:
                if p_start <= pos < p_end:
                    start_page = p_num
                if p_start <= end <= p_end:
                    end_page = p_num

            chunks.append({
                "chunk_id": f"{filename}_fixed_{chunk_idx}",
                "strategy": "fixed-size",
                "source": filename,
                "page_start": start_page,
                "page_end": end_page,
                "text": chunk_str,
                "metadata": {
                    "ocr_used": ocr_used,
                    "language": "vi",
                    "char_count": len(chunk_str)
                }
            })
            chunk_idx += 1

        pos += (chunk_size - chunk_overlap)
        if chunk_size <= chunk_overlap:
            break

    return chunks


def chunk_semantic(pages_data, filename, ocr_used, max_chunk_size=600):
    """Chiến lược 2: Semantic (Ưu tiên ranh giới đoạn \\n\\n, câu, không cắt giữa câu)."""
    chunks = []
    chunk_idx = 1

    current_chunk_text = ""
    start_page = 1
    current_page = 1

    for p in pages_data:
        current_page = p["page"]
        paragraphs = p["text"].split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Nếu một đoạn quá dài, chia nhỏ theo ranh giới câu (. ? !) để tránh cắt giữa câu
            sub_sections = []
            if len(para) > max_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_sent = ""
                for sent in sentences:
                    if len(temp_sent) + len(sent) + 1 <= max_chunk_size:
                        temp_sent = (temp_sent + " " + sent).strip()
                    else:
                        if temp_sent:
                            sub_sections.append(temp_sent)
                        temp_sent = sent
                if temp_sent:
                    sub_sections.append(temp_sent)
            else:
                sub_sections = [para]

            for sec in sub_sections:
                if not current_chunk_text:
                    start_page = current_page
                    current_chunk_text = sec
                elif len(current_chunk_text) + len(sec) + 2 <= max_chunk_size:
                    current_chunk_text += "\n\n" + sec
                else:
                    chunks.append({
                        "chunk_id": f"{filename}_semantic_{chunk_idx}",
                        "strategy": "semantic",
                        "source": filename,
                        "page_start": start_page,
                        "page_end": current_page,
                        "text": current_chunk_text,
                        "metadata": {
                            "ocr_used": ocr_used,
                            "language": "vi",
                            "char_count": len(current_chunk_text)
                        }
                    })
                    chunk_idx += 1
                    start_page = current_page
                    current_chunk_text = sec

    if current_chunk_text:
        chunks.append({
            "chunk_id": f"{filename}_semantic_{chunk_idx}",
            "strategy": "semantic",
            "source": filename,
            "page_start": start_page,
            "page_end": current_page,
            "text": current_chunk_text,
            "metadata": {
                "ocr_used": ocr_used,
                "language": "vi",
                "char_count": len(current_chunk_text)
            }
        })

    return chunks


def chunk_hierarchical(pages_data, filename, ocr_used):
    """Chiến lược 3: Hierarchical (Chương -> Mục -> Điều/Khoản -> Điểm)."""
    chunks = []
    chunk_idx = 1

    # Pattern nhận dạng tiêu đề văn bản pháp quy Việt Nam
    chapter_pattern = re.compile(r'^(Chương\s+[0-9IVXLCDM]+.*)', re.IGNORECASE)
    section_pattern = re.compile(r'^(Mục\s+[0-9]+.*)', re.IGNORECASE)
    article_pattern = re.compile(r'^(Điều\s+[0-9]+.*)', re.IGNORECASE)

    has_hierarchy = False

    current_chapter = None
    current_section = None
    current_article = None
    current_text_lines = []
    start_page = 1

    for p in pages_data:
        p_num = p["page"]
        lines = p["text"].split("\n")

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            m_chap = chapter_pattern.match(line_str)
            m_sec = section_pattern.match(line_str)
            m_art = article_pattern.match(line_str)

            if m_chap or m_sec or m_art:
                has_hierarchy = True
                if current_text_lines:
                    chunk_text = "\n".join(current_text_lines).strip()
                    if chunk_text:
                        chunks.append({
                            "chunk_id": f"{filename}_hierarchical_{chunk_idx}",
                            "strategy": "hierarchical",
                            "source": filename,
                            "page_start": start_page,
                            "page_end": p_num,
                            "text": chunk_text,
                            "metadata": {
                                "ocr_used": ocr_used,
                                "language": "vi",
                                "chapter": current_chapter,
                                "section": current_section,
                                "article": current_article,
                                "char_count": len(chunk_text)
                            }
                        })
                        chunk_idx += 1
                    current_text_lines = []

                if m_chap:
                    current_chapter = m_chap.group(1)
                    current_section = None
                    current_article = None
                elif m_sec:
                    current_section = m_sec.group(1)
                    current_article = None
                elif m_art:
                    current_article = m_art.group(1)

                start_page = p_num

            current_text_lines.append(line_str)

    if current_text_lines:
        chunk_text = "\n".join(current_text_lines).strip()
        if chunk_text:
            chunks.append({
                "chunk_id": f"{filename}_hierarchical_{chunk_idx}",
                "strategy": "hierarchical",
                "source": filename,
                "page_start": start_page,
                "page_end": pages_data[-1]["page"] if pages_data else 1,
                "text": chunk_text,
                "metadata": {
                    "ocr_used": ocr_used,
                    "language": "vi",
                    "chapter": current_chapter,
                    "section": current_section,
                    "article": current_article,
                    "char_count": len(chunk_text)
                }
            })

    if not has_hierarchy:
        print(f"⚠️ [Cảnh báo Structure] File '{filename}' không phát hiện cấu trúc phân cấp (Chương/Điều/Khoản). Giữ nguyên toàn bộ văn bản làm 1 chunk.")

    return chunks


# =============================================================================
# 4. THỐNG KÊ & CHẠY PIPELINE
# =============================================================================

def calculate_stats(chunks_list):
    """Tính toán thống kê số chunk, độ dài Min, Max, Trung bình."""
    if not chunks_list:
        return {"count": 0, "min_len": 0, "max_len": 0, "avg_len": 0}

    lengths = [len(c["text"]) for c in chunks_list]
    return {
        "count": len(chunks_list),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_len": round(sum(lengths) / len(lengths), 1)
    }


async def main():
    parser = argparse.ArgumentParser(description="Luồng OCR & Chunking Buổi 05")
    parser.add_argument("--write", action="store_true", help="Ghi kết quả ra thư mục output/")
    parser.add_argument("--dry-run", action="store_true", help="Chạy kiểm thử và in thống kê mà không ghi file")
    args = parser.parse_args()

    write_mode = args.write

    print("\n" + "=" * 70)
    print("🚀 BẮT ĐẦU LUỒNG XỬ LÝ OCR VĂN BẢN & CHUNKING (BUỔI 05)")
    print(f"📌 Chế độ: {'WRITE (Ghi dữ liệu ra output/)' if write_mode else 'DRY-RUN (Chỉ kiểm thử & in thống kê)'}")
    print("=" * 70)

    pdf_files = glob.glob(os.path.join(DATADEMO_DIR, "*.pdf"))
    if not pdf_files:
        print(f"❌ Không tìm thấy file PDF nào tại {DATADEMO_DIR}")
        return

    all_raw_data = {}
    all_chunks = {
        "fixed-size": [],
        "semantic": [],
        "hierarchical": []
    }

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n📄 Đang xử lý file: {filename}...")
        
        pages_data, ocr_used = await extract_pdf_pages_ocr(pdf_path)
        all_raw_data[filename] = {
            "filename": filename,
            "ocr_used": ocr_used,
            "pages": pages_data
        }

        # Cắt chunk theo 3 chiến lược
        fixed_c = chunk_fixed_size(pages_data, filename, ocr_used)
        semantic_c = chunk_semantic(pages_data, filename, ocr_used)
        hierarchical_c = chunk_hierarchical(pages_data, filename, ocr_used)

        all_chunks["fixed-size"].extend(fixed_c)
        all_chunks["semantic"].extend(semantic_c)
        all_chunks["hierarchical"].extend(hierarchical_c)

    # In Bảng Thống kê So sánh 3 Chiến lược
    print("\n" + "=" * 70)
    print("   BẢNG THỐNG KÊ SO SÁNH 3 CHIẾN LƯỢC CHUNKING")
    print("=" * 70)
    print(f"| {'Chiến lược Chunking':<20} | {'Tổng số Chunk':<12} | {'Min (ký tự)':<12} | {'Max (ký tự)':<12} | {'Trung bình':<12} |")
    print("|" + "-"*22 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|")

    for strat_name, chunks_list in all_chunks.items():
        st = calculate_stats(chunks_list)
        print(f"| {strat_name:<20} | {st['count']:<12} | {st['min_len']:<12} | {st['max_len']:<12} | {st['avg_len']:<12} |")
    print("=" * 70)

    # In 1 ví dụ Metadata đại diện
    if all_chunks["hierarchical"]:
        sample_chunk = all_chunks["hierarchical"][0]
    elif all_chunks["fixed-size"]:
        sample_chunk = all_chunks["fixed-size"][0]
    else:
        sample_chunk = {}

    print("\n📌 [VÍ DỤ METADATA MẪU CỦA 1 CHUNK]:")
    print(json.dumps(sample_chunk, ensure_ascii=False, indent=2))

    # Nếu ở chế độ --write, ghi file ra thư mục output/
    if write_mode:
        os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
        os.makedirs(CHUNKS_OUTPUT_DIR, exist_ok=True)

        for filename, raw_info in all_raw_data.items():
            raw_file_path = os.path.join(RAW_OUTPUT_DIR, f"{filename}.json")
            with open(raw_file_path, "w", encoding="utf-8") as f:
                json.dump(raw_info, f, ensure_ascii=False, indent=2)

        for strat_name, chunks_list in all_chunks.items():
            chunk_file_path = os.path.join(CHUNKS_OUTPUT_DIR, f"chunks_{strat_name.replace('-', '_')}.json")
            with open(chunk_file_path, "w", encoding="utf-8") as f:
                json.dump(chunks_list, f, ensure_ascii=False, indent=2)

        print(f"\n✅ ĐÃ GHI THÀNH CÔNG DỮ LIỆU:")
        print(f"   - Text raw lưu tại: {RAW_OUTPUT_DIR}")
        print(f"   - Chunks lưu tại: {CHUNKS_OUTPUT_DIR}")
    else:
        print("\n💡 Chạy ở chế độ DRY-RUN. Sử dụng tham số `--write` nếu bạn muốn lưu kết quả ra file.")


if __name__ == "__main__":
    asyncio.run(main())
