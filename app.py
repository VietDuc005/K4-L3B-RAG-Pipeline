"""Streamlit modern chat interface & interactive demo for the Toyota policy RAG pipeline."""

import json
import os
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search
from src.task9_retrieval_pipeline import retrieve, SCORE_THRESHOLD
from src.task10_generation import (
    generate_with_citation,
    generate_from_sources,
    format_context,
    reorder_for_llm,
    SYSTEM_PROMPT,
)

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Toyota Policy Assistant · Hybrid RAG",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Light Enterprise & Toyota Corporate Red Aesthetics)
CUSTOM_CSS = """
<style>
/* Global App Canvas & Background */
.stApp {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 8px rgba(0, 0, 0, 0.02) !important;
}

/* Header Container */
.toyota-header {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-top: 4px solid #eb0a1e;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03);
}
.toyota-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: -0.02em;
}
.toyota-title span {
    color: #eb0a1e;
}
.toyota-subtitle {
    font-size: 0.95rem;
    color: #475569;
    margin-top: 6px;
    line-height: 1.5;
    font-weight: 400;
}

/* Quick KPI Cards */
.kpi-container {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 9px 15px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.85rem;
    color: #334155;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}
.kpi-card strong {
    color: #0f172a;
    font-size: 0.92rem;
}

/* Citation Highlighting Pills */
.cite-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(235, 10, 30, 0.08);
    border: 1px solid rgba(235, 10, 30, 0.35);
    color: #c50014 !important;
    font-weight: 700;
    font-size: 0.82rem;
    padding: 1px 7px;
    border-radius: 6px;
    margin: 0 4px;
    text-decoration: none !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(235, 10, 30, 0.08);
}
.cite-badge:hover {
    background: #eb0a1e;
    color: #ffffff !important;
    transform: translateY(-1px) scale(1.06);
    box-shadow: 0 4px 10px rgba(235, 10, 30, 0.25);
}

/* Interactive Source Cards */
.source-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    margin-top: 14px;
    margin-bottom: 14px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
    scroll-margin-top: 80px;
}
.source-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 8px 20px -3px rgba(15, 23, 42, 0.08);
}
.source-card:target,
.source-card.is-active {
    border: 2px solid #eb0a1e !important;
    background: #fffdfd !important;
    box-shadow: 0 0 0 4px rgba(235, 10, 30, 0.1), 0 8px 25px rgba(235, 10, 30, 0.12) !important;
    transform: translateY(-2px);
}

.source-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.source-index {
    background: #eb0a1e;
    color: #ffffff;
    font-weight: 800;
    font-size: 0.8rem;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
}
.source-title {
    font-weight: 700;
    font-size: 0.98rem;
    color: #0f172a;
    margin-left: 8px;
}
.source-meta-tag {
    font-size: 0.76rem;
    padding: 3px 9px;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
.tag-legal {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}
.tag-news {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}
.tag-score {
    background: #f5f3ff;
    color: #6d28d9;
    border: 1px solid #ddd6fe;
}

.source-body {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 16px;
    color: #1e293b;
    font-size: 0.88rem;
    line-height: 1.65;
    margin-top: 10px;
    white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* Pipeline Step Cards */
.step-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
}
.step-box h4 {
    color: #0f172a;
    margin-top: 0;
    margin-bottom: 8px;
    font-size: 1.05rem;
    font-weight: 700;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    padding: 16px 20px !important;
    margin-bottom: 14px !important;
}

/* Streamlit Tabs Customization */
button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    padding: 10px 18px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0f172a !important;
    border-bottom: 2px solid #eb0a1e !important;
}

/* Expander Styling */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# State initialization
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_pipeline_trace", None)
st.session_state.setdefault("selected_prompt", "")


def get_active_provider() -> tuple[str, str, bool]:
    """Check provider status and API key availability."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    key_name = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider)
    key_val = os.getenv(key_name, "").strip() if key_name else ""
    return provider, key_name or "API_KEY", bool(key_val)


def render_header():
    """Render modern banner with key metrics."""
    provider, _, is_ready = get_active_provider()
    status_icon = "🟢" if is_ready else "🔴"
    status_text = f"{provider.upper()} ({'Sẵn sàng' if is_ready else 'Thiếu Key'})"

    st.markdown(
        f"""
        <div class="toyota-header">
            <div class="toyota-title">
                🚗 <span>Toyota AI</span> Policy Assistant
            </div>
            <div class="toyota-subtitle">
                Hệ thống RAG Pipeline Đa phương thức · Hybrid Retrieval (MiniLM-L6-v2 + BM25) · Reciprocal Rank Fusion (RRF) · Verifiable Citation Highlighting
            </div>
            <div class="kpi-container">
                <div class="kpi-card">📁 <strong>10 Tài liệu</strong> (5 Chính sách, 5 Bài báo)</div>
                <div class="kpi-card">🧩 <strong>628 Chunks</strong> (Recursive 500c/50ov)</div>
                <div class="kpi-card">⚡ <strong>Hybrid RRF</strong> + Cosine Fallback</div>
                <div class="kpi-card">{status_icon} Model: <strong>{status_text}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_citations_in_html(answer: str, message_id: int) -> str:
    """Format [1], [2] citations into clickable interactive anchor badges."""
    def replace_cite(match):
        num = match.group(1)
        return (
            f'<a href="#source-card-{message_id}-{num}" class="cite-badge" '
            f'title="Nhấn để cuộn tới nguồn [{num}]">[{num}]</a>'
        )

    return re.sub(r"\[(\d+)\]", replace_cite, answer)


def show_sources(message: dict) -> None:
    """
    Renders cited sources conforming strictly to acceptance test contracts
    while offering interactive visual cards and citation jumps.
    """
    sources = message.get("sources", [])
    if not sources:
        return

    msg_id = message.get("id", int(time.time()))
    retrieval_src = message.get("retrieval_source", "hybrid")

    st.caption(f"Truy xuất: {retrieval_src} · {len(sources)} đoạn tài liệu nguồn")

    # Contract requirement: expander with exact markdown format for tests
    with st.expander("Xem nguồn và điểm truy xuất", expanded=True):
        # Interactive jump indicator
        st.markdown(
            "<small style='color:#64748b;'>💡 <em>Gợi ý: Nhấp vào số [1], [2] trong câu trả lời hoặc danh sách bên dưới để đối chiếu nội dung:</em></small>",
            unsafe_allow_html=True,
        )

        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            title = metadata.get("title") or metadata.get("source") or "Tài liệu"
            doc_type = metadata.get("doc_type", "legal")
            method = source.get("retrieval_method", "hybrid")
            score = source.get("score", 0.0)
            chunk_idx = metadata.get("chunk_index", 0)
            source_file = metadata.get("source", "")
            url = metadata.get("url", "")

            # Exact contract markdown requirement (tested in test_generation_evaluation.py)
            st.markdown(f"**[{index}] {title}**")
            st.caption(
                f"{source_file} · {method} · score {score:.4f} · chunk {chunk_idx}"
            )

            # High-tech card layout with HTML target anchor for bonus citation highlighting
            tag_class = "tag-legal" if doc_type == "legal" else "tag-news"
            tag_name = "Chính sách pháp lý" if doc_type == "legal" else "Bản tin truyền thông"

            st.markdown(
                f"""
                <div id="source-card-{msg_id}-{index}" class="source-card">
                    <div class="source-header">
                        <div>
                            <span class="source-index">[{index}]</span>
                            <span class="source-title">{title}</span>
                        </div>
                        <div>
                            <span class="source-meta-tag {tag_class}">{tag_name}</span>
                            <span class="source-meta-tag tag-score">{method.upper()} · Score: {score:.4f}</span>
                        </div>
                    </div>
                    <div style="font-size:0.82rem; color:#64748b; margin-top:4px;">
                        📄 File: <code>{source_file}</code> | Chunk Index: <strong>#{chunk_idx}</strong>
                    </div>
                    <div class="source-body">{source.get("content", "").strip()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if url:
                st.link_button("🌐 Mở liên kết tài liệu gốc", url, key=f"src-link-{msg_id}-{index}")


def execute_pipeline(query: str, top_k: int, score_threshold: float, retrieval_mode: str) -> dict:
    """Execute full RAG pipeline and record deep inspection trace."""
    start_time = time.perf_counter()

    # 1. Dense retrieval
    t0 = time.perf_counter()
    dense_candidates = semantic_search(query, top_k=top_k * 2)
    t_dense = (time.perf_counter() - t0) * 1000

    # 2. Lexical retrieval
    t1 = time.perf_counter()
    sparse_candidates = lexical_search(query, top_k=top_k * 2)
    t_bm25 = (time.perf_counter() - t1) * 1000

    # 3. Reranking / Fusion
    best_dense_score = float(dense_candidates[0]["score"]) if dense_candidates else 0.0
    fallback_triggered = False

    if retrieval_mode == "Dense Only":
        final_chunks = dense_candidates[:top_k]
        retrieval_method_tag = "dense"
    elif retrieval_mode == "Lexical Only":
        final_chunks = sparse_candidates[:top_k]
        retrieval_method_tag = "bm25"
    else:  # Hybrid RRF
        hybrid_candidates = rerank_rrf([dense_candidates, sparse_candidates], top_k=top_k)
        if best_dense_score < score_threshold:
            fallback_triggered = True
            try:
                fallback_chunks = pageindex_search(query, top_k=top_k)
                if fallback_chunks:
                    final_chunks = fallback_chunks[:top_k]
                    retrieval_method_tag = "pageindex"
                else:
                    final_chunks = hybrid_candidates[:top_k]
                    retrieval_method_tag = "hybrid"
            except Exception:
                final_chunks = hybrid_candidates[:top_k]
                retrieval_method_tag = "hybrid"
        else:
            final_chunks = hybrid_candidates[:top_k]
            retrieval_method_tag = "hybrid"

    t_retrieval = (time.perf_counter() - start_time) * 1000

    # 4. Generation
    t_gen_start = time.perf_counter()
    gen_result = generate_from_sources(query, final_chunks)
    t_generation = (time.perf_counter() - t_gen_start) * 1000
    t_total = (time.perf_counter() - start_time) * 1000

    # Store trace for Pipeline Inspector Tab
    st.session_state.last_pipeline_trace = {
        "query": query,
        "dense_candidates": dense_candidates[:top_k],
        "sparse_candidates": sparse_candidates[:top_k],
        "final_chunks": final_chunks,
        "best_dense_score": best_dense_score,
        "score_threshold": score_threshold,
        "fallback_triggered": fallback_triggered,
        "retrieval_method_tag": retrieval_method_tag,
        "t_dense": t_dense,
        "t_bm25": t_bm25,
        "t_retrieval": t_retrieval,
        "t_generation": t_generation,
        "t_total": t_total,
        "prompt_context": format_context(reorder_for_llm(final_chunks)) if final_chunks else "",
    }

    return {
        "id": len(st.session_state.messages) + 1,
        "role": "assistant",
        "content": gen_result["answer"],
        "sources": gen_result["sources"],
        "retrieval_source": gen_result["retrieval_source"],
        "latency_ms": t_total,
    }


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/e/e7/Toyota.svg",
        width=130,
    )
    st.header("⚙️ Cấu hình Pipeline")

    provider, key_name, is_ready = get_active_provider()
    provider_options = ["gemini", "openai", "anthropic"]
    current_idx = provider_options.index(provider) if provider in provider_options else 0
    selected_provider = st.selectbox("LLM Provider", provider_options, index=current_idx)

    if selected_provider != provider:
        os.environ["LLM_PROVIDER"] = selected_provider

    if is_ready:
        st.success(f"Khóa {key_name} hợp lệ.", icon="✅")
    else:
        st.warning(f"Chưa có {key_name} trong .env.", icon="⚠️")

    st.divider()

    st.subheader("Tham số Tìm kiếm")
    top_k = st.slider("Số đoạn tài liệu (Top-K)", min_value=3, max_value=10, value=5)

    retrieval_mode = st.radio(
        "Chiến lược Retrieval",
        ["Hybrid (Dense + BM25 + RRF)", "Dense Only", "Lexical Only"],
        index=0,
    )

    threshold_val = float(os.getenv("SCORE_THRESHOLD", "0.3") or "0.3")
    score_threshold = st.slider(
        "Ngưỡng Cosine Fallback",
        min_value=0.0,
        max_value=1.0,
        value=threshold_val,
        step=0.05,
        help="Nếu điểm cosine của Dense Search thấp hơn ngưỡng này, hệ thống sẽ kích hoạt Fallback.",
    )

    st.divider()
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_pipeline_trace = None
        st.rerun()

    st.caption("Khóa K4-L3B · RAG Pipeline Project")


# ==========================================
# MAIN INTERFACE TABS
# ==========================================
render_header()

tab_chat, tab_inspector, tab_corpus, tab_eval = st.tabs(
    [
        "💬 Trợ lý Tra cứu (Chat & Citation)",
        "🔍 RAG Pipeline Inspector",
        "📚 Kho Tri thức (Corpus)",
        "📊 Đánh giá A/B & Benchmark",
    ]
)

# ------------------------------------------
# TAB 1: CHAT & CITATION HIGHLIGHTING
# ------------------------------------------
with tab_chat:
    st.subheader("💡 Gợi ý câu hỏi kiểm tra nhanh (1-Click Demo)")
    quick_prompts = [
        ("🎯 ABAC: Facilitation payments có được phép?", "Are facilitation payments permitted under Toyota's anti-bribery policy?"),
        ("📢 Speak Up: Ai được liên hệ đường dây nóng?", "Who can contact Toyota's Global Speak Up Line?"),
        ("⚖️ Nhân quyền: Xử lý khi chuẩn mực xung đột?", "What does Toyota do when national and international human rights standards differ?"),
        ("🛡️ Bảo mật: Trách nhiệm quản lý mật khẩu?", "What are the rules regarding password management in Toyota information security?"),
        ("🚫 Out-of-Domain: Cách nướng bánh quy? (Safe Refusal)", "How to bake delicious chocolate chip cookies?"),
    ]

    cols = st.columns(len(quick_prompts))
    for idx, (label, prompt_text) in enumerate(quick_prompts):
        if cols[idx].button(label, key=f"chip-{idx}", use_container_width=True):
            st.session_state.selected_prompt = prompt_text

    st.divider()

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message)

    # Chat Input Handling
    query_input = st.chat_input("Nhập câu hỏi về chính sách hoặc tin tức Toyota...")
    active_query = query_input or st.session_state.selected_prompt

    if active_query and active_query.strip():
        # Clear selected prompt from state
        st.session_state.selected_prompt = ""

        # Display user message
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Assistant thinking & retrieval
        with st.chat_message("assistant"):
            with st.spinner("Đang thực hiện Hybrid Search (Dense + BM25) & RRF Rerank..."):
                response = execute_pipeline(
                    active_query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    retrieval_mode=retrieval_mode,
                )

            st.markdown(response["content"])

            if response.get("latency_ms"):
                st.caption(f"⚡ Thời gian xử lý: **{response['latency_ms']:.1f} ms**")

            show_sources(response)

        st.session_state.messages.append(response)


# ------------------------------------------
# TAB 2: RAG PIPELINE INSPECTOR
# ------------------------------------------
with tab_inspector:
    st.subheader("🔍 Trực quan hóa Pipeline Tra cứu & Fusion")
    trace = st.session_state.last_pipeline_trace

    if not trace:
        st.info("Chưa có lượt tra cứu nào. Hãy gửi một câu hỏi ở Tab 1 để kiểm tra chi tiết các bước xử lý!")
    else:
        st.markdown(f"**Câu hỏi đang kiểm tra:** *\"{trace['query']}\"*")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Thời gian Semantic (Dense)", f"{trace['t_dense']:.1f} ms")
        m_col2.metric("Thời gian Lexical (BM25)", f"{trace['t_bm25']:.1f} ms")
        m_col3.metric("Tổng thời gian Retrieval", f"{trace['t_retrieval']:.1f} ms")
        m_col4.metric("Thời gian sinh câu trả lời", f"{trace['t_generation']:.1f} ms")

        st.divider()

        # Step 1: Semantic Search
        with st.container():
            st.markdown("### Bước 1: Dense Semantic Search (BAAI/MiniLM-L6-v2)")
            st.caption("Tìm kiếm vector embedding dựa trên không gian cosine similarity trong ChromaDB:")
            dense_rows = []
            for r_idx, doc in enumerate(trace["dense_candidates"], 1):
                dense_rows.append({
                    "Hạng": f"#{r_idx}",
                    "Score Cosine": f"{doc['score']:.4f}",
                    "Chunk ID": doc["id"],
                    "Tài liệu": doc["metadata"].get("title", ""),
                    "Trích đoạn": doc["content"][:100] + "...",
                })
            if dense_rows:
                st.dataframe(dense_rows, use_container_width=True)
            else:
                st.write("Không tìm thấy ứng viên Dense.")

        # Step 2: Lexical Search
        with st.container():
            st.markdown("### Bước 2: Lexical Search (BM25 Okapi)")
            st.caption("Tìm kiếm từ khóa chính xác dựa trên tần suất từ (TF-IDF cải tiến):")
            bm25_rows = []
            for r_idx, doc in enumerate(trace["sparse_candidates"], 1):
                bm25_rows.append({
                    "Hạng": f"#{r_idx}",
                    "Score BM25": f"{doc['score']:.4f}",
                    "Chunk ID": doc["id"],
                    "Tài liệu": doc["metadata"].get("title", ""),
                    "Trích đoạn": doc["content"][:100] + "...",
                })
            if bm25_rows:
                st.dataframe(bm25_rows, use_container_width=True)
            else:
                st.write("Không tìm thấy ứng viên BM25.")

        # Step 3: RRF Fusion
        with st.container():
            st.markdown("### Bước 3: Reciprocal Rank Fusion (RRF)")
            st.caption("Hợp nhất hai bảng xếp hạng theo công thức: $RRF(d) = \\sum \\frac{1}{60 + rank(d)}$")
            rrf_rows = []
            for r_idx, doc in enumerate(trace["final_chunks"], 1):
                rrf_rows.append({
                    "Hạng RRF": f"#{r_idx}",
                    "Score Fusion": f"{doc['score']:.5f}",
                    "Phương thức": doc.get("retrieval_method", "hybrid"),
                    "Chunk ID": doc["id"],
                    "Tài liệu": doc["metadata"].get("title", ""),
                    "Nội dung": doc["content"][:120] + "...",
                })
            st.dataframe(rrf_rows, use_container_width=True)

        # Step 4: Fallback Threshold Check
        with st.container():
            st.markdown("### Bước 4: Fallback & Threshold Check")
            b_score = trace["best_dense_score"]
            thresh = trace["score_threshold"]
            is_fallback = trace["fallback_triggered"]

            f_col1, f_col2 = st.columns([1, 2])
            f_col1.metric("Best Dense Cosine", f"{b_score:.4f}", delta=f"{b_score - thresh:.4f} vs Ngưỡng")
            with f_col2:
                if is_fallback:
                    st.warning(
                        f"⚠️ Điểm dense ({b_score:.4f}) < Ngưỡng ({thresh:.4f})! Hệ thống kích hoạt **PageIndex Vectorless Fallback**.",
                        icon="⚠️",
                    )
                else:
                    st.success(
                        f"✅ Điểm dense ({b_score:.4f}) >= Ngưỡng ({thresh:.4f}). Đạt độ tin cậy in-domain, sử dụng kết quả Hybrid RRF.",
                        icon="✅",
                    )

        # Step 5: Formatted Prompt payload
        with st.expander("📝 Xem Payload Context đưa vào LLM (Edge-reordered)"):
            st.code(trace["prompt_context"], language="markdown")


# ------------------------------------------
# TAB 3: CORPUS EXPLORER
# ------------------------------------------
with tab_corpus:
    st.subheader("📚 Kho Tri thức Chuẩn hóa của Nhóm")
    st.markdown("Bộ dữ liệu bao gồm **5 văn bản chính sách pháp lý** và **5 bài báo truyền thông** chính thức từ Toyota:")

    data_dir = Path("data/standardized")
    legal_files = sorted(list((data_dir / "legal").glob("*.md")))
    news_files = sorted(list((data_dir / "news").glob("*.md")))

    c_col1, c_col2 = st.columns([1, 2])

    with c_col1:
        st.markdown("#### Danh sách tài liệu")
        all_options = {}
        for p in legal_files:
            all_options[f"📜 [Legal] {p.name}"] = p
        for p in news_files:
            all_options[f"📰 [News] {p.name}"] = p

        selected_label = st.selectbox("Chọn tài liệu để xem:", list(all_options.keys()))
        selected_path = all_options[selected_label]

        if selected_path.exists():
            content = selected_path.read_text(encoding="utf-8")
            st.info(f"**Độ dài:** {len(content):,} ký tự (~{len(content)//500 + 1} chunks)")
            st.caption(f"Đường dẫn: `{selected_path}`")

    with c_col2:
        st.markdown(f"#### Nội dung: `{selected_path.name}`")
        with st.container(height=500):
            st.markdown(content)


# ------------------------------------------
# TAB 4: A/B BENCHMARK & EVALUATION
# ------------------------------------------
with tab_eval:
    st.subheader("📊 Báo cáo Đánh giá A/B & Bộ Golden Dataset")
    st.markdown(
        """
        So sánh thực nghiệm giữa hai chiến lược retrieval trên bộ **18 câu hỏi Golden Dataset**:
        - **Cấu hình A (Dense-Only):** Tìm kiếm vector ngữ nghĩa với ChromaDB & MiniLM-L6-v2.
        - **Cấu hình B (Hybrid + RRF):** Kết hợp Dense Top-10 + BM25 Top-10 qua Reciprocal Rank Fusion.
        """
    )

    ev_col1, ev_col2, ev_col3 = st.columns(3)
    ev_col1.metric("Tỷ lệ bao phủ Gold Span", "15/18 (83.3%)", delta="+2 câu (+11.1%) vs Dense")
    ev_col2.metric("Tài liệu chính xác Top-5", "18/18 (100%)", delta="Cả 2 cấu hình")
    ev_col3.metric("Độ trễ trung bình", "0.150 s", delta="+0.077 s (Chi phí hợp lý)")

    st.divider()

    st.markdown("#### Bảng so sánh chi tiết từ `group_project/evaluation/RESULT.md`")
    eval_table = [
        {"Tiêu chí đánh giá": "Tài liệu mong đợi nằm trong Top-5", "Config A (Dense)": "18/18 (100%)", "Config B (Hybrid+RRF)": "18/18 (100%)", "Độ chênh lệch": "0"},
        {"Tiêu chí đánh giá": "Đoạn gold span chuẩn xác nằm trong Top-5", "Config A (Dense)": "13/18 (72.2%)", "Config B (Hybrid+RRF)": "15/18 (83.3%)", "Độ chênh lệch": "+2 câu (+11.1%)"},
        {"Tiêu chí đánh giá": "Thời gian truy xuất trung bình", "Config A (Dense)": "0.072 s", "Config B (Hybrid+RRF)": "0.150 s", "Độ chênh lệch": "+0.077 s"},
        {"Tiêu chí đánh giá": "Kích thước context trung bình", "Config A (Dense)": "2,054 ký tự", "Config B (Hybrid+RRF)": "2,101 ký tự", "Độ chênh lệch": "+47 ký tự"},
    ]
    st.table(eval_table)

    st.divider()

    # Golden Dataset explorer
    st.markdown("#### Duyệt toàn bộ 18 câu hỏi trong Golden Dataset")
    golden_path = Path("group_project/evaluation/golden_dataset.json")
    if golden_path.exists():
        golden_data = json.loads(golden_path.read_text(encoding="utf-8"))
        for item in golden_data:
            with st.expander(f"🔹 [{item['id']}] {item['question']}"):
                st.markdown(f"**Câu trả lời chuẩn (Ground Truth):**\n> {item['expected_answer']}")
                st.markdown(f"**Ngữ cảnh trích xuất mong đợi:**\n```text\n{item['expected_context']}\n```")
                st.caption(f"Nguồn: `{item['source']}`")
