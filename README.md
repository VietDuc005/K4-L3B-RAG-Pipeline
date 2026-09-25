# Toyota Policy Assistant — Hybrid RAG Pipeline

> **Đồ án Day 8 — Khóa học RAG Pipeline (Nhóm K4-L3B / AIGAN)**  
> Chatbot tra cứu chính sách và tin tức Toyota sử dụng kiến trúc **Hybrid Retrieval (Dense Semantic + BM25)**, **Reciprocal Rank Fusion (RRF)**, **Vectorless Fallback**, và **Verifiable Citation Highlighting**.

---

## 👥 Thành viên nhóm & Phân công

Chi tiết thông tin thành viên và phân công trong [TEAMMATES.md](TEAMMATES.md):

| STT | Họ và tên | Mã học viên | Vai trò phụ trách | Báo cáo cá nhân |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **Dương Văn Thành** *(Trưởng nhóm)* | 2A202602368 | Data Engineering (`task1`, `task2`, `task3`) | [Báo cáo](reports/K4-L3B-2A202602368-DuongVanThanh.md) |
| 2 | **Mai Văn Trường** | 2A202602983 | Indexing & Dual Search (`task4`, `task5`, `task6`) | [Báo cáo](reports/K4-L3B-2A202602983-Truong.md) |
| 3 | **Nguyễn Viết Đức** | 2A202602732 | Fusion & Fallback Pipeline (`task7`, `task8`, `task9`) | [Báo cáo](reports/K4-L3B-2A202602732-NguyenVietDuc.md) |
| 4 | **Hồ Ngọc Mai** | 2A202602509 | Generation, UI & Evaluation (`task10`, `app.py`, `evaluation`) | [Báo cáo](reports/K4-L3B-2A202602509-HoNgocMai.md) |

---

## 🏛️ Kiến trúc RAG Pipeline

```mermaid
flowchart TD
    subgraph Data_Pipeline [1. Thu thập & Chuẩn hóa Data]
        A[Toyota Official PDFs] -->|task1: requests| B[data/landing/legal/]
        C[Toyota Newsroom URLs] -->|task2: Crawl4AI| D[data/landing/news/]
        B & D -->|task3: MarkItDown| E[data/standardized/ - 10 Docs]
    end

    subgraph Indexing [2. Chunking & Indexing]
        E -->|task4: Recursive 500c/50ov| F[628 Chunks]
        F -->|all-MiniLM-L6-v2| G[(ChromaDB 384-dim)]
        F -->|Custom BM25Okapi| H[(BM25 Inverted Index)]
    end

    subgraph Retrieval [3. Hybrid Retrieval & Fusion]
        Q[User Query] -->|task5| I[Dense Top-10 Cosine]
        Q -->|task6| J[BM25 Top-10 Lexical]
        I & J -->|task7: RRF k=60| K[Fused Hybrid Ranking Top-5]
        I -->|task9: Best Cosine Score vs Threshold| L{Score < 0.30?}
        L -->|Có: Low Confidence| M[task8: PageIndex Fallback]
        L -->|Không: High Confidence| K
        M -.->|Nếu lỗi/rỗng| K
    end

    subgraph Generation [4. Generation & UI]
        K -->|task10: reorder_for_llm| N[Context Edge-Reordered]
        N & Q -->|LLM: Gemini / OpenAI| O[Answer có Citation [1], [2] hoặc Safe Refusal]
        O -->|app.py| P[Streamlit UI: Interactive Citation Highlighting]
    end
```

---

## 📁 Bộ dữ liệu chuẩn hóa (`data/standardized/`)

Toàn bộ tài liệu chính thức từ Toyota Global được chuẩn hóa sang định dạng Markdown UTF-8:
1. **5 Văn bản chính sách pháp lý (`legal/`):**
   - `toyota_global_abac_policy.md` (Chính sách chống hối lộ & tham nhũng toàn cầu)
   - `toyota_global_speakup_policy.md` (Chính sách đường dây tố giác Speak Up toàn cầu)
   - `toyota_human_rights_policy.md` (Chính sách nhân quyền)
   - `toyota_information_security_policy.md` (Chính sách an toàn thông tin)
   - `toyota_tax_policy.md` (Chính sách thuế Toyota)
2. **5 Bài báo truyền thông chính thức (`news/`):**
   - `article_01.md` đến `article_05.md` (Thông cáo báo chí & sự kiện từ Toyota Newsroom).
3. **Tổng số chunks:** **628 chunks** (kích thước 500 ký tự, overlap 50 ký tự, bảo toàn metadata `source`, `title`, `chunk_index`).

---

## 📊 Kết quả đánh giá A/B & Benchmark

Báo cáo đầy đủ được lưu tại [group_project/evaluation/RESULT.md](group_project/evaluation/RESULT.md) và [reports/RESULT.md](reports/RESULT.md) dựa trên bộ **18 câu hỏi Golden Dataset** ([golden_dataset.json](group_project/evaluation/golden_dataset.json)):

| Chỉ số đánh giá | Config A (Dense-Only) | Config B (Hybrid + RRF) | Độ chênh lệch (B − A) |
|---|:---:|:---:|:---:|
| **Tài liệu đúng nằm trong Top-5** | 18/18 (100%) | 18/18 (100%) | 0 |
| **Đoạn Gold Span chuẩn xác trong Top-5** | 13/18 (72.2%) | **15/18 (83.3%)** | **+2 ca (+11.1%)** |
| **Thời gian truy xuất trung bình** | 0.072 s | 0.150 s | +0.077 s |
| **Kích thước context trung bình** | 2,054 ký tự | 2,101 ký tự | +47 ký tự |

> **Nhận định cốt lõi:** Kết hợp Dense Semantic với BM25 thông qua Reciprocal Rank Fusion (RRF) giúp bổ sung các từ khóa chính xác (exact keywords), nâng tỷ lệ tìm thấy đoạn bằng chứng vàng thêm **+11.1%** trên tập dữ liệu chính sách Toyota.

---

## 🌟 Điểm thưởng Bonus (+2đ): Interactive Citation Highlighting

Giao diện [app.py](app.py) đã triển khai trọn vẹn tiêu chí Bonus:
- **Trích dẫn tương tác:** Các ký hiệu `[1]`, `[2]` trong câu trả lời được gắn thẻ liên kết anchor badge.
- **Cuộn mượt & Highlight nguồn:** Khi nhấp chuột vào `[1]`, trang tự động cuộn đến thẻ tài liệu nguồn tương ứng, kích hoạt hiệu ứng viền phát sáng màu đỏ Toyota (`:target` glow pulse animation).
- **Trực quan hóa RAG Pipeline Inspector:** Tab riêng biệt cho phép theo dõi chi tiết điểm số Dense Cosine, điểm BM25, thứ hạng sau RRF và kiểm tra ngưỡng Fallback theo thời gian thực.
- **Giao diện Light Enterprise Theme:** Tông màu sáng dịu mắt, thẻ nội dung trắng viền xám bạc sang trọng, chuẩn nhận diện Toyota.

---

## 🚀 Hướng dẫn cài đặt và Khởi chạy

### 1. Cài đặt môi trường

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
source .venv/bin/activate       # Trên Windows: .venv\Scripts\activate

# Cài đặt dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium

# Cấu hình biến môi trường
cp .env.example .env
```

Điền API key cần dùng trong `.env` (`GEMINI_API_KEY`, `OPENAI_API_KEY`, hoặc `ANTHROPIC_API_KEY`). Mặc định dự án sử dụng `LLM_PROVIDER=gemini` với model `gemini-3.5-flash-lite`.

### 2. Chạy pipeline thu thập & index (khi cần tái lập từ đầu)

```bash
# Thu thập và chuẩn hóa dữ liệu
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# Phân đoạn và tạo vector database
python -m src.task4_chunking_indexing
```

### 3. Khởi chạy ứng dụng Chatbot Demo

```powershell
# Chạy Streamlit UI với Python của dự án
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Truy cập trên trình duyệt tại: **`http://localhost:8501`** (hoặc `http://localhost:8502`).

### 4. Chạy A/B Evaluation Benchmark

```bash
# Chạy A/B evaluation bằng Gemini
python -m group_project.evaluation.run_ab_evaluation --generator-provider gemini --evaluator-provider gemini

# Chạy retrieval diagnostics cục bộ
python -m group_project.evaluation.run_retrieval_diagnostics
```

---

## ✅ Trạng thái Kiểm thử Kỹ thuật

Toàn bộ **28 bài kiểm thử tự động** trong repo đạt **100% Pass**:

```bash
# Chạy toàn bộ test suite
pytest -q
# Kết quả: 28 passed in 12.04s

# Chạy contract tests
pytest tests/test_contracts.py -q

# Chạy acceptance tests
pytest tests/test_acceptance.py -q

# Chạy generation & evaluation tests
pytest tests/test_generation_evaluation.py -q
```
