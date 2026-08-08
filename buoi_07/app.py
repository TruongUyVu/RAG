"""
=============================================================================
BUỔI 07: STREAMLIT UI APPLICATION FOR RAG FOUNDATION
=============================================================================
Giao diện ứng dụng Streamlit trực quan hóa toàn bộ luồng RAG:
- Sidebar: Cấu hình hệ thống, Trạng thái Collection, Bộ chọn Strategy & Top-K.
- Indexing Area: Thực hiện Index dữ liệu với tùy chọn Reset.
- Question & Answering Area: Hỏi đáp RAG, hiển thị Answer, Citation và Evidence.
=============================================================================
"""

import sys
import streamlit as st
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Import các hàm cốt lõi từ rag.py
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import rag

st.set_page_config(
    page_title="Buổi 07 - RAG Foundation",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Buổi 07 — RAG Foundation Application")
st.caption("Hệ thống Retrieval-Augmented Generation chuẩn Production-Ready với Gemini & ChromaDB")

# Initial Session State
if "last_index_res" not in st.session_state:
    st.session_state["last_index_res"] = None

if "last_query_res" not in st.session_state:
    st.session_state["last_query_res"] = None

# =============================================================================
# SIDEBAR: SYSTEM STATUS & CONFIGURATION
# =============================================================================
st.sidebar.header("⚙️ Cấu hình & Trạng thái")

strategy_option = st.sidebar.selectbox(
    "Chọn chiến lược Chunking (Strategy):",
    options=["hierarchical", "semantic", "fixed-size"],
    index=0,
)

top_k_option = st.sidebar.slider(
    "Chọn số lượng Top-K:",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)

st.sidebar.divider()

# Gọi hàm get_status read-only khi đổi strategy
try:
    status_info = rag.get_status(strategy=strategy_option)

    st.sidebar.markdown("### 📊 Trạng thái Hệ thống")
    st.sidebar.text(f"GEMINI_API_KEY      : {status_info['api_key_status']}")
    st.sidebar.text(f"Embedding Model     : {status_info['embedding_model']}")
    st.sidebar.text(f"Dimension           : {status_info['embedding_dim']}")
    st.sidebar.text(f"Generation Model    : {status_info['generation_model']}")
    st.sidebar.text(f"Strategy Đang Chọn  : {status_info['strategy']}")
    st.sidebar.text(f"Collection Name     : {status_info['collection_name']}")
    st.sidebar.text(f"Collection Tồn Tại  : {'Có ✅' if status_info['collection_exists'] else 'Chưa ❌'}")
    st.sidebar.text(f"Số lượng Vector     : {status_info['record_count']}")
    st.sidebar.text(f"Max Distance Gate   : {status_info['max_distance']}")

    if not status_info["api_key_status"] == "Có":
        st.sidebar.warning("⚠️ Thiếu GEMINI_API_KEY trong file .env! Vui lòng bổ sung key vào file .env.")
except Exception as err:
    st.sidebar.error(f"❌ Không thể tải trạng thái: {err}")

# =============================================================================
# MAIN TAB 1: INDEXING AREA
# =============================================================================
st.subheader("📥 Indexing Dữ liệu Chunk")

col_idx1, col_idx2 = st.columns([2, 1])

with col_idx1:
    reset_db_option = st.checkbox("Reset collection trước khi index (Xóa vector cũ của strategy này)")

with col_idx2:
    btn_index = st.button("🚀 Index dữ liệu ngay", use_container_width=True)

if btn_index:
    if not status_info["api_key_status"] == "Có":
        st.error("❌ Không thể thực hiện Index do thiếu GEMINI_API_KEY trong file `.env`. Vui lòng điền API Key vào `.env` rồi thử lại.")
    else:
        with st.spinner(f"Đang tạo Gemini Embeddings & Indexing cho strategy '{strategy_option}'..."):
            try:
                res_idx = rag.index_chunks(strategy=strategy_option, reset_db=reset_db_option)
                st.session_state["last_index_res"] = res_idx
                st.success(f"🎉 Index thành công! Đã lưu {res_idx['chunks_indexed']} chunks vào collection `{res_idx['collection_name']}`.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình Indexing: {e}")

if st.session_state["last_index_res"]:
    with st.expander("ℹ️ Chi tiết kết quả Index gần nhất", expanded=False):
        st.json(st.session_state["last_index_res"])

st.divider()

# =============================================================================
# MAIN TAB 2: QUESTION & ANSWERING AREA
# =============================================================================
st.subheader("💬 Hỏi Đáp RAG Thông Minh")

user_question = st.text_area("Nhập câu hỏi của bạn:", height=100, placeholder="Ví dụ: Thông tư 02/2023/TT-NHNN quy định về đối tượng nào được cơ cấu lại thời hạn trả nợ?")

btn_ask = st.button("🔍 Gửi câu hỏi", type="primary")

if btn_ask:
    if not user_question.strip():
        st.warning("⚠️ Vui lòng nhập câu hỏi trước khi gửi.")
    elif not status_info["api_key_status"] == "Có":
        st.error("❌ Thiếu GEMINI_API_KEY trong file `.env`. Vui lòng điền API Key để thực hiện truy vấn.")
    elif not status_info["collection_exists"] or status_info["record_count"] == 0:
        st.warning("⚠️ Collection chưa được khởi tạo hoặc chưa có dữ liệu vector. Vui lòng bấm nút **'Index dữ liệu ngay'** ở trên trước.")
    else:
        with st.spinner("Đang truy xuất thông tin & tạo câu trả lời tổng hợp..."):
            try:
                q_result = rag.query(
                    question=user_question,
                    strategy=strategy_option,
                    top_k=top_k_option,
                )
                st.session_state["last_query_res"] = q_result
            except Exception as e:
                st.error(f"❌ Lỗi khi thực hiện truy vấn RAG: {e}")

# Render Query Results
q_res = st.session_state["last_query_res"]
if q_res:
    st.markdown("---")
    st.subheader("💡 Kết Quả Trả Lời")

    st_status = q_res.get("status")
    if st_status == "answered":
        st.success("✅ Trạng thái: Answered (Đã tạo câu trả lời tổng hợp thành công)")
    elif st_status == "insufficient_evidence":
        st.warning("⚠️ Trạng thái: Insufficient Evidence (Không tìm thấy đủ thông tin liên quan trong tài liệu đã cung cấp)")
    elif st_status == "retrieval_only":
        st.info("ℹ️ Trạng thái: Retrieval Only (Đã truy xuất được nguồn nhưng chưa thể tạo câu trả lời tổng hợp)")

    # Display Answer
    st.markdown("#### 📝 Câu trả lời:")
    st.markdown(q_res["answer"])

    # Display Warnings if any
    if q_res.get("warnings"):
        for w in q_res["warnings"]:
            st.warning(f"⚠️ Cảnh báo: {w}")

    # Display Citations if any
    if q_res.get("citations"):
        st.markdown("#### 📌 Danh sách Trích dẫn (Citations):")
        for cit in q_res["citations"]:
            st.markdown(f"- `{cit['display']}`")

    st.markdown("---")

    # Display Evidence List
    st.subheader("📚 Nguồn Tham Khảo (Evidence List)")

    evidences = q_res.get("evidence", [])
    if not evidences:
        st.info("Chưa có evidence nào được truy xuất.")
    else:
        st.caption("🔍 Ghi chú: Distance thấp hơn thể hiện mức độ tương đồng ngữ cảnh cao hơn.")
        for ev in evidences:
            eid = ev["evidence_id"]
            src = ev["source"]
            p_start = ev["page_start"]
            p_end = ev["page_end"]
            page_str = f"tr. {p_start}" if p_start == p_end else f"tr. {p_start}-{p_end}"
            cid = ev["chunk_id"]
            dist = ev["distance"]
            acc = ev["accepted"]

            acc_label = "✅ ĐẠT GATE" if acc else "❌ BỎ QUA (KHOẢNG CÁCH LỚN)"
            header_str = f"[{eid}] {src} – {page_str} – {cid} | (Distance: {dist} | {acc_label})"

            with st.expander(header_str, expanded=acc):
                st.write(f"**Evidence ID**: `{eid}`")
                st.write(f"**Source**: `{src}` | **Page**: `{page_str}` | **Chunk ID**: `{cid}`")
                st.write(f"**Distance**: `{dist}` | **Confidence Gate Status**: `{acc_label}`")
                st.text_area("Nội dung Chunk:", value=ev["text"], height=150, key=f"txt_{eid}_{cid}")
