"""
=============================================================================
BUỔI 05: STREAMLIT UI VISUALIZE CHUNKING & OCR DATA
=============================================================================
"""

import os
import json
import glob
import streamlit as st

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Buổi 05 - RAG Chunking Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .chunk-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 4px;
        color: white;
        margin-right: 5px;
    }
    .badge-fixed { background-color: #42A5F5; }
    .badge-semantic { background-color: #66BB6A; }
    .badge-hierarchical { background-color: #AB47BC; }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")


@st.cache_data
def load_data():
    """Load toàn bộ dữ liệu chunks từ thư mục output."""
    chunks_by_strategy = {}
    strategy_files = {
        "Fixed-size": os.path.join(CHUNKS_DIR, "chunks_fixed_size.json"),
        "Semantic": os.path.join(CHUNKS_DIR, "chunks_semantic.json"),
        "Hierarchical": os.path.join(CHUNKS_DIR, "chunks_hierarchical.json"),
    }

    for strat, path in strategy_files.items():
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chunks_by_strategy[strat] = json.load(f)
            except Exception as e:
                chunks_by_strategy[strat] = []
        else:
            chunks_by_strategy[strat] = []

    return chunks_by_strategy


def main():
    st.markdown('<div class="main-header">🔍 Visualizer Cắt Nhỏ Văn Bản (RAG Chunking - Buổi 05)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Trực quan hóa và so sánh 3 chiến lược: Fixed-size, Semantic và Hierarchical</div>', unsafe_allow_html=True)

    chunks_data = load_data()
    all_chunks = []
    for strat, items in chunks_data.items():
        all_chunks.extend(items)

    if not all_chunks:
        st.warning("⚠️ Chưa phát hiện dữ liệu chunks tại `output/chunks/`. Vui lòng chạy lệnh `--write` trong `src/rag_buoi_05.py` trước khi xem UI!")
        st.code('& "d:/OneDrive - Dai Nam University/Google DataAnalyst/Agribank/RAG/rag_foundation/buoi_05/.venv/Scripts/python.exe" RAG/rag_foundation/buoi_05/src/rag_buoi_05.py --write', language="powershell")
        return

    # Lấy danh sách các file nguồn (sources)
    sources = sorted(list(set(c.get("source", "Unknown") for c in all_chunks)))

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⚙️ Bộ Lọc Dữ Liệu")
    
    selected_source = st.sidebar.selectbox("📄 Chọn Tài Liệu Nguồn:", ["Tất cả"] + sources)
    
    selected_strategy = st.sidebar.multiselect(
        "✂️ Chọn Chiến Lược Chunking:",
        ["Fixed-size", "Semantic", "Hierarchical"],
        default=["Fixed-size", "Semantic", "Hierarchical"]
    )
    
    search_keyword = st.sidebar.text_input("🔎 Tìm kiếm từ khóa trong Chunk:", "")
    
    min_len, max_len = st.sidebar.slider(
        "📏 Kích thước Chunk (Ký tự):",
        min_value=0,
        max_value=max(len(c.get("text", "")) for c in all_chunks) if all_chunks else 2000,
        value=(0, 25000)
    )

    # --- FILTERING LOGIC ---
    filtered_chunks = []
    for c in all_chunks:
        strat_key = c.get("strategy", "").capitalize()
        if strat_key == "Fixed-size" and "Fixed-size" not in selected_strategy:
            continue
        if strat_key == "Semantic" and "Semantic" not in selected_strategy:
            continue
        if strat_key == "Hierarchical" and "Hierarchical" not in selected_strategy:
            continue
        if selected_source != "Tất cả" and c.get("source") != selected_source:
            continue
        
        chunk_text = c.get("text", "")
        if search_keyword and search_keyword.lower() not in chunk_text.lower():
            continue
            
        if not (min_len <= len(chunk_text) <= max_len):
            continue

        filtered_chunks.append(c)

    # --- METRICS DASHBOARD ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng Số Chunks Hợp Lệ", len(filtered_chunks))
    
    avg_len = round(sum(len(c["text"]) for c in filtered_chunks) / len(filtered_chunks), 1) if filtered_chunks else 0
    m2.metric("Độ Dài Trung Bình", f"{avg_len} ký tự")
    
    min_c = min((len(c["text"]) for c in filtered_chunks), default=0)
    m3.metric("Kích Thước Nhỏ Nhất", f"{min_c} ký tự")
    
    max_c = max((len(c["text"]) for c in filtered_chunks), default=0)
    m4.metric("Kích Thước Lớn Nhất", f"{max_c} ký tự")

    st.markdown("---")

    # --- TAB NAVIGATION ---
    tab_list, tab_compare, tab_raw = st.tabs(["📜 Danh Sách Chunks", "📊 So Sánh 3 Chiến Lược", "📁 Dữ Liệu Raw Text"])

    with tab_list:
        st.subheader(f"Hiển thị {len(filtered_chunks)} Chunks phù hợp")
        
        for idx, chunk in enumerate(filtered_chunks, 1):
            strat = chunk.get("strategy", "fixed-size")
            badge_class = f"badge-{strat.lower()}"
            
            with st.expander(f"Chunk #{idx} | [{strat.upper()}] - {chunk.get('chunk_id')} (Trang {chunk.get('page_start')}-{chunk.get('page_end')})"):
                col_info, col_meta = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"**Tài liệu nguồn:** `{chunk.get('source')}`")
                    st.markdown(f"**Nội dung Chunk ({len(chunk.get('text', ''))} ký tự):**")
                    st.text_area("", value=chunk.get("text", ""), height=150, key=f"txt_{idx}")

                with col_meta:
                    st.markdown("**Metadata:**")
                    st.json(chunk.get("metadata", {}))

    with tab_compare:
        st.subheader("📊 Bảng So Sánh Số Lượng & Độ Dài Theo Chiến Lược")
        
        comp_data = []
        for strat in ["Fixed-size", "Semantic", "Hierarchical"]:
            sub = [c for c in filtered_chunks if c.get("strategy", "").lower() == strat.lower()]
            if sub:
                lens = [len(c["text"]) for c in sub]
                comp_data.append({
                    "Chiến Lược": strat,
                    "Số Chunk": len(sub),
                    "Độ Dài Nhỏ Nhất": min(lens),
                    "Độ Dài Lớn Nhất": max(lens),
                    "Trung Bình (Ký Tự)": round(sum(lens) / len(lens), 1)
                })
        
        if comp_data:
            st.dataframe(comp_data, use_container_width=True)
            
            chart_data = {d["Chiến Lược"]: d["Số Chunk"] for d in comp_data}
            st.bar_chart(chart_data)
        else:
            st.info("Không có dữ liệu phù hợp với bộ lọc hiện tại.")

    with tab_raw:
        st.subheader("📁 Dữ Liệu Raw OCR Trích Xuất (Unicode NFC)")
        raw_files = glob.glob(os.path.join(RAW_DIR, "*.json"))
        if raw_files:
            selected_raw = st.selectbox("Chọn File Raw JSON:", [os.path.basename(f) for f in raw_files])
            file_path = os.path.join(RAW_DIR, selected_raw)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_json = json.load(f)
                st.json(raw_json)
        else:
            st.info("Chưa có dữ liệu raw tại `output/raw/`.")


if __name__ == "__main__":
    main()
