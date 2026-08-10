"""
=============================================================================
BUỔI 08: RAG EVALUATION & BENCHMARK MODULE
=============================================================================
Mô-đun đánh giá hiệu năng định lượng của RAG Retrieval Pipeline:
1. Metrics: Recall@K, MRR@K, nDCG@K (Binary Relevance), Latency Mean & P50.
2. So sánh 4 Retrieval Modes: BM25, Semantic, Hybrid, Hybrid_Rerank.
3. Xuất Báo cáo JSON chi tiết trong thư mục reports/.
=============================================================================
"""

import sys
import os
import math
import time
import json
import argparse
import datetime
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
ENV_EXAMPLE = BASE_DIR / ".env.example"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
elif ENV_EXAMPLE.exists():
    load_dotenv(ENV_EXAMPLE)

try:
    from advanced_rag import (
        get_advanced_config,
        load_chunks,
        retrieve_bm25,
        retrieve_semantic_candidates,
        fuse_rrf,
        rerank_candidates,
    )
except ImportError:
    from .advanced_rag import (
        get_advanced_config,
        load_chunks,
        retrieve_bm25,
        retrieve_semantic_candidates,
        fuse_rrf,
        rerank_candidates,
    )


# =============================================================================
# 1. METRIC CALCULATION FUNCTIONS (HAND-CALCULABLE & UNIT-TESTABLE)
# =============================================================================

def calculate_recall_at_k(retrieved_chunk_ids: list, relevant_chunk_ids: list, k: int = 5) -> float:
    """Tính chỉ số Recall@K cho danh sách chunk IDs được truy xuất."""
    if not relevant_chunk_ids:
        return 1.0 if not retrieved_chunk_ids else 0.0

    top_k_ids = set(retrieved_chunk_ids[:k])
    rel_set = set(relevant_chunk_ids)
    hits = len(top_k_ids.intersection(rel_set))
    return round(hits / len(rel_set), 4)


def calculate_mrr_at_k(retrieved_chunk_ids: list, relevant_chunk_ids: list, k: int = 5) -> float:
    """Tính chỉ số Mean Reciprocal Rank (MRR@K)."""
    if not relevant_chunk_ids:
        return 1.0 if not retrieved_chunk_ids else 0.0

    top_k_ids = retrieved_chunk_ids[:k]
    rel_set = set(relevant_chunk_ids)

    for rank_idx, cid in enumerate(top_k_ids, 1):
        if cid in rel_set:
            return round(1.0 / rank_idx, 4)

    return 0.0


def calculate_ndcg_at_k(retrieved_chunk_ids: list, relevant_chunk_ids: list, k: int = 5) -> float:
    """
    Tính chỉ số Normalized Discounted Cumulative Gain (nDCG@K) với binary relevance.
    DCG@K = sum(rel_i / log2(i + 1)) cho i từ 1 đến K.
    IDCG@K = sum(1 / log2(i + 1)) cho i từ 1 đến min(K, |relevant|).
    """
    if not relevant_chunk_ids:
        return 1.0 if not retrieved_chunk_ids else 0.0

    top_k_ids = retrieved_chunk_ids[:k]
    rel_set = set(relevant_chunk_ids)

    dcg = 0.0
    for i, cid in enumerate(top_k_ids, 1):
        rel = 1.0 if cid in rel_set else 0.0
        dcg += rel / math.log2(i + 1)

    idcg = 0.0
    num_ideal = min(k, len(rel_set))
    for i in range(1, num_ideal + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return round(dcg / idcg, 4)


def calculate_latency_stats(latencies: list) -> dict:
    """Tính Latency Mean và Median P50."""
    if not latencies:
        return {"mean_ms": 0.0, "p50_ms": 0.0}

    sorted_l = sorted(latencies)
    n = len(sorted_l)
    mean_val = sum(sorted_l) / n

    if n % 2 == 1:
        p50_val = sorted_l[n // 2]
    else:
        p50_val = (sorted_l[n // 2 - 1] + sorted_l[n // 2]) / 2.0

    return {
        "mean_ms": round(mean_val, 2),
        "p50_ms": round(p50_val, 2),
    }


# =============================================================================
# 2. EVALUATION PIPELINE
# =============================================================================

def load_eval_questions(eval_file_path=None) -> list:
    """Nạp bộ câu hỏi kiểm thử đánh giá từ file JSON."""
    path = Path(eval_file_path) if eval_file_path else BASE_DIR / "eval" / "questions.json"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi đánh giá tại: '{path}'")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"File '{path}' chứa dữ liệu không hợp lệ (Phải là danh sách câu hỏi).")
    return data


def run_evaluation(
    eval_questions: list = None,
    strategy: str = "hierarchical",
    k: int = 5,
    modes: list = None,
    client_helper=None,
    chroma_client=None,
    reranker_fn=None,
) -> dict:
    """
    Thực thi đánh giá benchmark định lượng cho các retrieval modes mà KHÔNG gọi LLM Generation.
    """
    cfg = get_advanced_config()
    questions = eval_questions if eval_questions is not None else load_eval_questions()
    target_modes = modes or ["bm25", "semantic", "hybrid", "hybrid_rerank"]

    chunks, _ = load_chunks(strategy=strategy)

    # Check if human review is needed
    has_human_review_flag = any(q.get("needs_human_review", False) for q in questions)

    results_by_mode = {m: {"recalls": [], "mrrs": [], "ndcgs": [], "latencies": []} for m in target_modes}
    query_details = []

    for q_item in questions:
        qid = q_item.get("query_id", "Q_UNKNOWN")
        q_text = q_item.get("question", "").strip()
        rel_chunk_ids = q_item.get("relevant_chunk_ids", [])
        scope = q_item.get("scope", "in_scope")

        q_detail = {
            "query_id": qid,
            "question": q_text,
            "scope": scope,
            "relevant_chunk_ids": rel_chunk_ids,
            "modes": {},
        }

        # 1. BM25 Mode Retrieval
        if "bm25" in target_modes:
            t0 = time.perf_counter()
            bm25_cands = retrieve_bm25(q_text, chunks, candidate_k=cfg["bm25_candidates"])
            t1 = time.perf_counter()
            lat = round((t1 - t0) * 1000, 2)
            c_ids = [c["chunk_id"] for c in bm25_cands[:k]]

            rec = calculate_recall_at_k(c_ids, rel_chunk_ids, k=k)
            mrr = calculate_mrr_at_k(c_ids, rel_chunk_ids, k=k)
            ndcg = calculate_ndcg_at_k(c_ids, rel_chunk_ids, k=k)

            results_by_mode["bm25"]["recalls"].append(rec)
            results_by_mode["bm25"]["mrrs"].append(mrr)
            results_by_mode["bm25"]["ndcgs"].append(ndcg)
            results_by_mode["bm25"]["latencies"].append(lat)

            q_detail["modes"]["bm25"] = {
                "top_k_chunk_ids": c_ids,
                "recall_at_k": rec,
                "mrr_at_k": mrr,
                "ndcg_at_k": ndcg,
                "latency_ms": lat,
            }

        # 2. Semantic Mode Retrieval
        sem_cands = []
        if "semantic" in target_modes or "hybrid" in target_modes or "hybrid_rerank" in target_modes:
            t0 = time.perf_counter()
            try:
                sem_cands = retrieve_semantic_candidates(
                    q_text,
                    strategy=strategy,
                    candidate_k=cfg["semantic_candidates"],
                    client_helper=client_helper,
                    chroma_client=chroma_client,
                )
                t1 = time.perf_counter()
                lat = round((t1 - t0) * 1000, 2)
                sem_error = None
            except Exception as e:
                t1 = time.perf_counter()
                lat = round((t1 - t0) * 1000, 2)
                sem_cands = []
                sem_error = str(e)

            if "semantic" in target_modes:
                c_ids = [c["chunk_id"] for c in sem_cands[:k]]
                rec = calculate_recall_at_k(c_ids, rel_chunk_ids, k=k)
                mrr = calculate_mrr_at_k(c_ids, rel_chunk_ids, k=k)
                ndcg = calculate_ndcg_at_k(c_ids, rel_chunk_ids, k=k)

                results_by_mode["semantic"]["recalls"].append(rec)
                results_by_mode["semantic"]["mrrs"].append(mrr)
                results_by_mode["semantic"]["ndcgs"].append(ndcg)
                results_by_mode["semantic"]["latencies"].append(lat)

                q_detail["modes"]["semantic"] = {
                    "top_k_chunk_ids": c_ids,
                    "recall_at_k": rec,
                    "mrr_at_k": mrr,
                    "ndcg_at_k": ndcg,
                    "latency_ms": lat,
                    "error": sem_error,
                }

        # 3. Hybrid Mode (RRF)
        hyb_cands = []
        if "hybrid" in target_modes or "hybrid_rerank" in target_modes:
            bm25_for_hyb = retrieve_bm25(q_text, chunks, candidate_k=cfg["bm25_candidates"])
            t0 = time.perf_counter()
            hyb_cands, _ = fuse_rrf(
                bm25_candidates=bm25_for_hyb,
                semantic_candidates=sem_cands,
                rrf_k=cfg["rrf_k"],
                bm25_weight=cfg["rrf_bm25_weight"],
                semantic_weight=cfg["rrf_semantic_weight"],
                top_k=cfg["rerank_candidates"],
            )
            t1 = time.perf_counter()
            lat = round((t1 - t0) * 1000, 2)

            if "hybrid" in target_modes:
                c_ids = [c["chunk_id"] for c in hyb_cands[:k]]
                rec = calculate_recall_at_k(c_ids, rel_chunk_ids, k=k)
                mrr = calculate_mrr_at_k(c_ids, rel_chunk_ids, k=k)
                ndcg = calculate_ndcg_at_k(c_ids, rel_chunk_ids, k=k)

                results_by_mode["hybrid"]["recalls"].append(rec)
                results_by_mode["hybrid"]["mrrs"].append(mrr)
                results_by_mode["hybrid"]["ndcgs"].append(ndcg)
                results_by_mode["hybrid"]["latencies"].append(lat)

                q_detail["modes"]["hybrid"] = {
                    "top_k_chunk_ids": c_ids,
                    "recall_at_k": rec,
                    "mrr_at_k": mrr,
                    "ndcg_at_k": ndcg,
                    "latency_ms": lat,
                }

        # 4. Hybrid Rerank Mode
        if "hybrid_rerank" in target_modes:
            t0 = time.perf_counter()
            try:
                rr_cands = rerank_candidates(
                    q_text,
                    candidates=hyb_cands,
                    top_k=k,
                    rerank_candidates_limit=cfg["rerank_candidates"],
                    model_name=cfg["reranker_model"],
                    reranker_fn=reranker_fn,
                )
                t1 = time.perf_counter()
                lat = round((t1 - t0) * 1000, 2)
                rr_error = None
            except Exception as e:
                t1 = time.perf_counter()
                lat = round((t1 - t0) * 1000, 2)
                rr_cands = []
                rr_error = str(e)

            c_ids = [c["chunk_id"] for c in rr_cands[:k]]
            rec = calculate_recall_at_k(c_ids, rel_chunk_ids, k=k)
            mrr = calculate_mrr_at_k(c_ids, rel_chunk_ids, k=k)
            ndcg = calculate_ndcg_at_k(c_ids, rel_chunk_ids, k=k)

            results_by_mode["hybrid_rerank"]["recalls"].append(rec)
            results_by_mode["hybrid_rerank"]["mrrs"].append(mrr)
            results_by_mode["hybrid_rerank"]["ndcgs"].append(ndcg)
            results_by_mode["hybrid_rerank"]["latencies"].append(lat)

            q_detail["modes"]["hybrid_rerank"] = {
                "top_k_chunk_ids": c_ids,
                "recall_at_k": rec,
                "mrr_at_k": mrr,
                "ndcg_at_k": ndcg,
                "latency_ms": lat,
                "error": rr_error,
            }

        query_details.append(q_detail)

    # Calculate overall summary per mode
    mode_summaries = {}
    for m in target_modes:
        m_data = results_by_mode[m]
        n_q = len(questions)
        mean_rec = round(sum(m_data["recalls"]) / n_q, 4) if n_q else 0.0
        mean_mrr = round(sum(m_data["mrrs"]) / n_q, 4) if n_q else 0.0
        mean_ndcg = round(sum(m_data["ndcgs"]) / n_q, 4) if n_q else 0.0
        lat_stats = calculate_latency_stats(m_data["latencies"])

        mode_summaries[m] = {
            f"recall_at_{k}": mean_rec,
            f"mrr_at_{k}": mean_mrr,
            f"ndcg_at_{k}": mean_ndcg,
            "latency": lat_stats,
        }

    warnings = []
    if has_human_review_flag:
        warnings.append(
            "CẢNH BÁO: Bộ dữ liệu đánh giá chứa các câu hỏi ghi nhận 'needs_human_review=true'. "
            "Đây là bộ đánh giá sơ bộ (Starter Set), chưa được chuyên gia pháp lý thẩm định chính thức. "
            "Báo cáo này KHÔNG tuyên bố mode chiến thắng chính thức cho hệ thống sản xuất."
        )

    now_iso = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "strategy": strategy,
        "eval_k": k,
        "questions_count": len(questions),
        "has_human_review_flag": has_human_review_flag,
        "config_snapshot": {
            "embedding_model": cfg["embedding_model"],
            "embedding_dim": cfg["embedding_dim"],
            "reranker_model": cfg["reranker_model"],
            "rrf_k": cfg["rrf_k"],
        },
        "mode_summaries": mode_summaries,
        "warnings": warnings,
        "query_details": query_details,
    }

    # Save JSON report into reports/
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"eval_report_{strategy}_k{k}_{now_iso}.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report["saved_file"] = str(report_file)
    return report


# =============================================================================
# 3. CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Buổi 08 RAG Offline Benchmark Evaluator")
    parser.add_argument("--strategy", default="hierarchical", choices=["fixed-size", "semantic", "hierarchical"])
    parser.add_argument("--k", type=int, default=5, help="Số lượng k đánh giá Top-K (Mặc định: 5)")
    parser.add_argument("--eval-file", default=None, help="Đường dẫn file câu hỏi đánh giá JSON")

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(f"📊 BẮT ĐẦU ĐÁNH GIÁ BENCHMARK RAG (Strategy: {args.strategy}, K={args.k})")
    print("=" * 65)

    try:
        questions = load_eval_questions(args.eval_file)
        report = run_evaluation(
            eval_questions=questions,
            strategy=args.strategy,
            k=args.k,
        )

        print(f"\n✅ ĐÃ HOÀN THÀNH ĐÁNH GIÁ CHO {report['questions_count']} CÂU HỎI")
        print(f"📂 Báo cáo đã lưu tại: {report['saved_file']}\n")

        print("TỔNG HỢP CHỈ SỐ THEO MÔ HÌNH (RETRIEVAL MODES):")
        print(f"{'Mode':<15} | {'Recall@'+str(args.k):<10} | {'MRR@'+str(args.k):<8} | {'nDCG@'+str(args.k):<8} | {'Latency Mean (ms)':<18}")
        print("-" * 65)
        for mode_name, sum_data in report["mode_summaries"].items():
            r_val = sum_data[f"recall_at_{args.k}"]
            m_val = sum_data[f"mrr_at_{args.k}"]
            n_val = sum_data[f"ndcg_at_{args.k}"]
            l_val = sum_data["latency"]["mean_ms"]
            print(f"{mode_name:<15} | {r_val:<10.4f} | {m_val:<8.4f} | {n_val:<8.4f} | {l_val:<18.2f}")

        if report["warnings"]:
            print("\n⚠️ " + report["warnings"][0])
        print("=" * 65)

    except Exception as e:
        print(f"\n❌ LỖI EVALUATION: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
