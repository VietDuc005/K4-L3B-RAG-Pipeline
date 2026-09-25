# Individual Contribution Report

---

## Thông tin

- **Họ và tên:** Hồ Ngọc Mai
- **Mã học viên:** 2A202602509
- **Nhóm:** K4-L3B
- **Vai trò:** Generation, UI & Evaluation (Task 10, app.py, group_project/evaluation)
- **Repository/branch:** [K4-L3B-RAG-Pipeline](https://github.com/VietDuc005/K4-L3B-RAG-Pipeline), nhánh `main`.

---

## Phần việc đã thực hiện

| Module / Deliverable | Việc tôi trực tiếp làm | File / PR | Trạng thái |
|---|---|---|---|
| **Task 10: Generation có Citation & Safe Refusal** | Cài đặt hàm `generate_with_citation()` và `generate_from_sources()`, thực hiện edge-reordering (`reorder_for_llm`) chống hiện tượng Lost in the Middle, cơ chế kiểm tra trích dẫn `[1]`, `[2]` nghiêm ngặt, cơ chế Safe Refusal khi thiếu bằng chứng hoặc ngoài phạm vi, hỗ trợ đa provider (OpenAI, Gemini, Anthropic). | `src/task10_generation.py` | **Done** |
| **Giao diện Chatbot Streamlit (`app.py`)** | Thiết kế giao diện Light Enterprise hiện đại theo nhận diện Toyota, xây dựng tính năng Interactive Citation Highlighting (+2 điểm bonus), trực quan hóa RAG Pipeline Inspector (Dense vs BM25 vs RRF vs Fallback), thanh 1-click Quick Prompts và bộ duyệt kho tri thức Corpus Explorer. | `app.py`, `.streamlit/config.toml` | **Done** |
| **Xây dựng Golden Dataset** | Xây dựng bộ dữ liệu đánh giá chuẩn gồm 18 câu hỏi Q&A (vượt yêu cầu tối thiểu 15 câu), bao quát 5 chính sách pháp lý (ABAC, Speak Up, Human Rights, Information Security, Tax) và các bài báo tin tức Toyota, gắn kèm ground truth answer và expected context đối chiếu. | `group_project/evaluation/golden_dataset.json` | **Done** |
| **Đánh giá A/B Testing & Báo cáo** | Viết runner đánh giá so sánh A/B giữa Dense-only và Hybrid + RRF (`run_ab_evaluation.py`, `run_retrieval_diagnostics.py`), đo lường độ phủ gold span và độ trễ, tổng hợp báo cáo chi tiết vào `RESULT.md`. | `group_project/evaluation/RESULT.md`, `reports/RESULT.md` | **Done** |

---

## Quyết định kỹ thuật quan trọng

### 1. Edge-Reordering Context và Xác thực Trích dẫn Nghiêm ngặt (Task 10)
- **Quyết định:** 
  - Áp dụng thuật toán sắp xếp lại vị trí các chunk đưa vào LLM context (`reorder_for_llm`): đưa các chunk có thứ hạng cao nhất ra hai biên ngoài (vị trí đầu $0$ và cuối $N-1$, rồi tiếp tục vào giữa).
  - Bắt buộc kiểm tra trích dẫn bằng regex `r"\[(\d+)\]"`. Nếu câu trả lời thiếu trích dẫn, trích dẫn số thứ tự vượt quá số chunk đầu vào, hoặc chứa token `KHÔNG ĐỦ BẰNG CHỨNG`, hệ thống lập tức kích hoạt cơ chế `SAFE_REFUSAL` ("Tôi không thể xác minh thông tin này từ các nguồn hiện có.").
- **Lý do / Evidence:** Hiện tượng *"Lost in the Middle"* (Liu et al.) chỉ ra rằng LLM chú ý tốt nhất ở phần đầu và phần cuối của prompt context dài. Việc đặt chunk điểm cao ở hai rìa giúp model dễ dàng trích xuất thông tin chính xác. Việc xác thực citation chặt chẽ triệt tiêu hiện tượng ảo giác (hallucination) và đảm bảo câu trả lời luôn có căn cứ kiểm chứng.
- **Trade-off:** Cần tạo ánh xạ ngược (`source_numbers`) để số thứ tự citation `[1]`, `[2]` trong câu trả lời của model luôn khớp đúng với thứ tự hiển thị của danh sách tài liệu nguồn ban đầu trên giao diện.

### 2. Thiết kế Giao diện Đa Tab với Interactive Citation Highlighting (Bonus +2đ)
- **Quyết định:** 
  - Phát triển giao diện Streamlit dạng 4 Tab chuyên nghiệp: *Trợ lý Tra cứu (Chat)*, *RAG Pipeline Inspector*, *Kho Tri thức (Corpus)*, và *Đánh giá A/B & Benchmark*.
  - Tích hợp tính năng **Interactive Citation Highlighting**: chuyển đổi trích dẫn `[1]`, `[2]` thành các anchor badge tương tác; khi người dùng nhấp chuột, giao diện tự động cuộn mượt (smooth scroll) đến đúng thẻ tài liệu nguồn và kích hoạt hiệu ứng viền phát sáng (glow pulse animation).
- **Lý do / Evidence:** Đáp ứng trọn vẹn tiêu chí chấm điểm và đạt tối đa +2 điểm bonus trong Grading Rubric. Giúp ban giám khảo/người dùng kiểm chứng tính xác thực của câu trả lời ngay lập tức mà không cần tìm thủ công trong văn bản dài.
- **Trade-off:** Cần xử lý CSS `:target` và thẻ HTML tương thích để vừa hiển thị hiệu ứng động trong trình duyệt, vừa đảm bảo 100% vượt qua các bài kiểm thử tự động của Streamlit `AppTest`.

---

## Kiểm thử và kết quả

- **Lệnh kiểm thử đã thực hiện:**
  ```powershell
  # Kiểm tra tính ổn định hợp đồng và sinh câu trả lời
  .\.venv\Scripts\python.exe -m pytest tests/test_generation_evaluation.py -v

  # Kiểm tra toàn bộ test suite dự án
  .\.venv\Scripts\python.exe -m pytest -q
  ```
- **Kết quả kiểm thử:**
  - `tests/test_generation_evaluation.py`: **8/8 passed** (xác thực reordered citations, safe refusal, provider fallback, và replay giao diện Streamlit).
  - Toàn bộ test suite: **28/28 passed** (100% pass trong 12.5s).
- **Kiểm thử hành vi trên giao diện:**
  - *Query in-domain:* *"Who does Toyota's anti-bribery policy apply to?"* $\rightarrow$ Trả lời chính xác phạm vi áp dụng, dẫn nguồn `[2]` trỏ về chunk `toyota_global_abac_policy.md`.
  - *Query out-of-domain:* *"How to bake chocolate cookies?"* $\rightarrow$ Kích hoạt `SAFE_REFUSAL`: *"Tôi không thể xác minh thông tin này từ các nguồn hiện có."* không bịa đặt nội dung.
- **Kết quả A/B Testing trên 18 câu Golden Dataset:**
  - Cấu hình B (Hybrid + RRF) tìm thấy chính xác đoạn Gold Span trong Top-5 ở **15/18 ca (83.3%)**, vượt trội hơn Cấu hình A (Dense-only) đạt **13/18 ca (72.2%)** — tăng thêm **+2 ca thành công (+11.1%)**.

---

## Điều còn hạn chế & Hướng phát triển

- **Hạn chế:** Các phép đo RAGAS 4 metric qua API bên ngoài (OpenAI/Anthropic) bị giới hạn bởi hạn ngạch/credit của tài khoản cá nhân trong giai đoạn chạy tự động, nhóm đã khắc phục bằng cách chạy retrieval diagnostic đầy đủ với hash SHA-256 để đảm bảo tính tái lập 100%.
- **Hướng cải tiến ưu tiên:** Tích hợp Conversation Memory (ghi nhớ ngữ cảnh hội thoại nhiều lượt) để người dùng có thể đặt câu hỏi nối tiếp (follow-up questions) mà không cần nhắc lại thực thể Toyota ở câu trước (+2 điểm bonus tiếp theo).

---

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- **Ngày:** 25/09/2026
- **Tên thành viên:** Hồ Ngọc Mai
