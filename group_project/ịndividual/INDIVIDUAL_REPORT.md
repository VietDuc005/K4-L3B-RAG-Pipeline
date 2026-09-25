# Individual contribution report

---

## Thông tin

- **Họ và tên:** Hồ Ngọc Mai
- **Mã học viên:** 2A202602509
- **Nhóm:** K4-L3B (AIGAN)
- **Vai trò:** Generation, UI & Evaluation (Task 10, app.py, group_project/evaluation)
- **Repository/branch:** [K4-L3B-RAG-Pipeline](https://github.com/VietDuc005/K4-L3B-RAG-Pipeline), nhánh `main`.

---

## Phần việc đã thực hiện

| Module / Deliverable | Việc tôi trực tiếp làm | File / Commit / PR | Trạng thái |
|---|---|---|---|
| **Task 10: Generation có Citation & Safe Refusal** | Cài đặt hàm `generate_with_citation()` và `generate_from_sources()`, thực hiện thuật toán edge-reordering (`reorder_for_llm`) chống hiện tượng Lost-in-the-Middle, cơ chế kiểm tra trích dẫn `[1]`, `[2]` nghiêm ngặt, cơ chế Safe Refusal khi thiếu bằng chứng hoặc câu hỏi ngoài phạm vi, hỗ trợ đa provider (OpenAI, Gemini, Anthropic). | `src/task10_generation.py` | **Done** |
| **Giao diện Chatbot Streamlit (`app.py`)** | Thiết kế giao diện Light Enterprise hiện đại theo nhận diện Toyota, xây dựng tính năng Interactive Citation Highlighting (+2 điểm bonus), trực quan hóa RAG Pipeline Inspector (Dense vs BM25 vs RRF vs Fallback), thanh 1-click Quick Prompts và bộ duyệt kho tri thức Corpus Explorer. | `app.py`, `.streamlit/config.toml` | **Done** |
| **Xây dựng Golden Dataset** | Xây dựng bộ dữ liệu đánh giá chuẩn gồm 18 câu hỏi Q&A (vượt yêu cầu tối thiểu 15 câu), bao quát 5 chính sách pháp lý (ABAC, Speak Up, Human Rights, Information Security, Tax) và các bài báo tin tức Toyota, gắn kèm ground truth answer và expected context đối chiếu. | `group_project/evaluation/golden_dataset.json` | **Done** |
| **Đánh giá A/B Testing & Báo cáo** | Viết runner đánh giá so sánh A/B giữa Dense-only và Hybrid + RRF (`run_ab_evaluation.py`, `run_retrieval_diagnostics.py`), đo lường độ phủ gold span và độ trễ, tổng hợp báo cáo chi tiết vào `RESULT.md`. | `group_project/evaluation/RESULT.md`, `reports/RESULT.md` | **Done** |

---

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Sắp xếp lại ngữ cảnh đưa vào LLM theo thuật toán Edge-Reordering (`reorder_for_llm`) và kiểm tra trích dẫn nghiêm ngặt bằng regex.  
   **Lý do/evidence:** Hiện tượng *"Lost in the Middle"* (Liu et al.) chỉ ra LLM ghi nhớ thông tin ở hai đầu context tốt hơn phần giữa. Việc đưa chunk xếp hạng cao nhất ra vị trí đầu ($0$) và cuối ($N-1$) giúp model trích xuất dẫn chứng chính xác hơn. Việc bắt buộc câu trả lời phải có `[1]`, `[2]` trỏ đúng nguồn giúp triệt tiêu ảo giác (hallucination).  
   **Trade-off:** Cần tạo bảng ánh xạ ngược (`source_numbers`) để số thứ tự trích dẫn trong câu trả lời luôn khớp chính xác với số thứ tự hiển thị của tài liệu nguồn trên giao diện.

2. **Quyết định:** Thiết kế giao diện Chatbot Streamlit đa Tab tích hợp **Interactive Citation Highlighting** (+2 điểm bonus) và RAG Pipeline Inspector.  
   **Lý do/evidence:** Đáp ứng trọn vẹn tiêu chí chấm điểm và đạt tối đa +2 điểm bonus trong Grading Rubric. Cho phép người dùng nhấp vào số trích dẫn `[1]`, `[2]` để tự động cuộn đến thẻ tài liệu nguồn và kích hoạt hiệu ứng viền phát sáng (Glow pulse).  
   **Trade-off:** Cần xử lý CSS `:target` và thẻ HTML tương thích để vừa hiển thị hiệu ứng động trong trình duyệt, vừa đảm bảo 100% vượt qua các bài kiểm thử tự động của Streamlit `AppTest`.

---

## Kiểm thử và kết quả

- **Test hoặc query tôi đã dùng:**
  - `pytest tests/test_generation_evaluation.py -v`: **8/8 passed** (xác thực reordered citations, safe refusal, provider fallback, và replay giao diện Streamlit).
  - Toàn bộ test suite: `pytest -q`: **28/28 passed** (100% pass trong 12.4s).
  - *Query in-domain:* *"Who does Toyota's anti-bribery policy apply to?"* $\rightarrow$ Trả lời chính xác phạm vi áp dụng, dẫn nguồn `[2]` trỏ về chunk `toyota_global_abac_policy.md`.
  - *Query out-of-domain:* *"How to bake chocolate cookies?"* $\rightarrow$ Kích hoạt `SAFE_REFUSAL`: *"Tôi không thể xác minh thông tin này từ các nguồn hiện có."* không bịa đặt nội dung.
- **Kết quả trước/sau nếu có:**
  - Trước: Module generation chỉ trả về chuỗi giả lập, chưa xác thực citation.
  - Sau: Pipeline hoạt động hoàn chỉnh end-to-end với LLM provider (Gemini 3.5 Flash Lite), kết quả A/B trên 18 câu Golden Dataset chứng minh cấu hình Hybrid + RRF đạt **15/18 ca (83.3%)** độ phủ gold span, vượt trội hơn Dense-only **13/18 ca (72.2%)**.
- **Lỗi đã phát hiện và cách xử lý:**
  - Lỗi tài khoản OpenAI cá nhân hết credit (`insufficient_quota`).
  - Đã xử lý bằng cách chuyển đổi sang Google Gemini (`gemini-3.5-flash-lite`) với API key hoạt động ổn định và tối ưu chi phí.

---

## Điều còn hạn chế

- **Một hạn chế cụ thể của phần tôi làm:** Chưa hỗ trợ lưu trữ bộ nhớ ngữ cảnh hội thoại nhiều lượt (Conversation Memory) khi người dùng đặt câu hỏi nối tiếp (follow-up query).
- **Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:** Tích hợp bộ nhớ hội thoại LangChain/Streamlit session state để nhận diện câu hỏi ngữ cảnh liên kết.

---

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- **Ngày:** 25/09/2026
- **Tên thành viên:** Hồ Ngọc Mai
