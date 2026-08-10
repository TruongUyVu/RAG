"""
=============================================================================
BUỔI 08: STREAMLIT ADVANCED RAG COMPARISON DASHBOARD
=============================================================================
Giao diện trực quan so sánh song song:
1. Baseline Semantic RAG (Buổi 07)
2. Advanced Hybrid RAG (BM25 + Dense + RRF + Cross-Encoder Reranker)
3. Pipeline Trace & Thống kê độ trễ Latency
=============================================================================
"""

import sys
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from advanced_rag import (
    get_advanced_config,
    get_advanced_status,
    advanced_query,
    compare_retrieval_modes,
)

st.set_page_config(
    page_title="Advanced RAG Comparison Dashboard - Buổi 08",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Advanced RAG System: Hybrid Search & Re-ranking Comparison")
st.caption("Buổi 08: So sánh trực diện Semantic Baseline (Buổi 07) vs Advanced Hybrid RAG (BM25 + Dense + RRF + Cross-Encoder)")

# Sidebar Settings
st.sidebar.header("⚙️ Cấu hình Pipeline")
strategy = st.sidebar.selectbox("Chiến lược Chunking", ["hierarchical", "fixed-size", "semantic"], index=0)
top_k = st.sidebar.slider("Top-K Final Candidates", min_value=1, max_value=10, value=5)

# Status Overview
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Trạng thái Hệ thống")
try:
    st_info = get_advanced_status(strategy=strategy)
    st.sidebar.success(f"Corpus: {st_info['corpus_size']} chunks")
    st.sidebar.info(f"Embedding: {st_info['embedding_model']} ({st_info['embedding_dim']})")
    st.sidebar.info(f"Reranker: {st_info['reranker_model']}")
    st.sidebar.write(f"API Key Status: **{st_info['api_key_status']}**")
except Exception as e:
    st.sidebar.warning(f"Chưa thể đọc status: {e}")

# Query Input
question = st.text_input("nhập câu hỏi tìm kiếm / pháp lý tiếng Việt:", "Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?")

col_btn1, col_btn2 = st.columns([1, 1])
btn_compare = col_btn1.button("🔍 So sánh Trực diện 4 Modes (Retrieval Only)", use_container_width=True)
btn_query = col_btn2.button("🚀 Chạy Advanced RAG Answer (Hybrid + Rerank + LLM)", use_container_width=True)

if btn_compare and question:
    st.markdown("---")
    st.subheader("📊 Bảng So sánh Thứ hạng Candidate qua các Modes")
    with st.spinner("Đang truy xuất và rerank ứng viên..."):
        try:
            cmp_res = compare_retrieval_modes(question=question, strategy=strategy)
            
            # Show Latencies
            lats = cmp_res["latencies_ms"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("BM25 Latency", f"{lats['bm25']} ms")
            c2.metric("Semantic Latency", f"{lats['semantic']} ms")
            c3.metric("Hybrid RRF Latency", f"{lats['hybrid']} ms")
            c4.metric("Rerank Latency", f"{lats['hybrid_rerank']} ms")

            # Table Comparison
            st.dataframe(cmp_res["comparison_table"], use_container_width=True)

        except Exception as e:
            st.error(f"Lỗi so sánh: {e}")

if btn_query and question:
    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🔵 Baseline Semantic RAG (Buổi 07)")
        with st.spinner("Đang chạy Semantic Baseline..."):
            try:
                res_base = advanced_query(question=question, mode="semantic", strategy=strategy, top_k=top_k)
                st.write(f"**Status:** `{res_base['status']}`")
                st.write(f"**Trả lời:**\n{res_base['answer']}")
                st.markdown("**Nguồn trích dẫn:**")
                for cit in res_base["citations"]:
                    st.caption(cit["display"])
            except Exception as e:
                st.error(f"Lỗi Baseline: {e}")

    with col_right:
        st.subheader("🟢 Advanced Hybrid RAG (Buổi 08)")
        with st.spinner("Đang chạy Advanced Hybrid RAG..."):
            try:
                res_adv = advanced_query(question=question, mode="hybrid_rerank", strategy=strategy, top_k=top_k)
                st.write(f"**Status:** `{res_adv['status']}`")
                st.write(f"**Trả lời:**\n{res_adv['answer']}")
                st.markdown("**Nguồn trích dẫn:**")
                for cit in res_adv["citations"]:
                    st.caption(cit["display"])

                # Pipeline Trace Detail
                with st.expander("🔍 Xem Pipeline Trace Chi Tiết"):
                    st.json(res_adv["trace"])

            except Exception as e:
                st.error(f"Lỗi Advanced RAG: {e}")
