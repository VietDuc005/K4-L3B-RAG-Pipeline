# Individual contribution report

## Thông tin

- Họ và tên: Dương Văn Thành
- Mã học viên: 2A202602368
- Nhóm: AIGAN
- Vai trò: Data — thu thập và chuẩn hóa dữ liệu đầu vào cho RAG.
- Repository/branch: [K4-L3B-RAG-Pipeline](https://github.com/VietDuc005/K4-L3B-RAG-Pipeline), nhánh `main`.

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 1 — Tài liệu chính sách | Khai báo 5 URL chính thức của Toyota; tải PDF bằng requests, kiểm tra HTTP và bỏ qua file đã tồn tại. Thu thập chính sách ABAC, Speak Up, thuế, bảo mật thông tin và nhân quyền. | `src/task1_collect_legal_docs.py`; commit `49566ac`; `data/landing/legal/` | Done |
| Task 2 — Bài viết | Khai báo 5 URL Toyota Newsroom; crawl bằng Crawl4AI, kiểm tra kết quả và lưu JSON với URL, tiêu đề, thời điểm crawl UTC và nội dung Markdown. | `src/task2_crawl_news.py`; commit `49566ac`; `data/landing/news/` | Done |
| Task 3 — Chuẩn hóa | Chuyển PDF bằng MarkItDown; chuyển JSON thành Markdown có metadata đầu bài; kiểm tra trường bắt buộc và nội dung rỗng, dùng tên output cố định khi chạy lại. | `src/task3_convert_markdown.py`; commit `b1db69a`; `data/standardized/` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Thu thập từ website chính thức Toyota, tách tài liệu chính sách (`legal`) và bài viết (`news`).  
   **Lý do/evidence:** Giữ nguồn có thể đối chiếu, đáp ứng yêu cầu tối thiểu 3 tài liệu và 5 bài viết; thực tế có 5 PDF và 5 JSON.  
   **Trade-off:** Phạm vi dữ liệu chỉ tập trung vào Toyota và phụ thuộc khả năng truy cập website.

2. **Quyết định:** Giữ dữ liệu gốc trong `landing`, xuất Markdown UTF-8 sang `standardized` theo tên file nguồn; giữ URL và thời điểm crawl ở đầu bài viết.  
   **Lý do/evidence:** Có thể đối chiếu dữ liệu trước/sau chuyển đổi và chạy lại với cùng tên output để không tạo thêm bản trùng.  
   **Trade-off:** Chạy lại ghi đè Markdown hiện tại, chưa quản lý phiên bản nội dung; PDF chưa được bổ sung URL nguồn vào Markdown.

## Kiểm thử và kết quả

- Lệnh kiểm tra tại thư mục project: `.\.venv\Scripts\python.exe -m pytest tests/test_acceptance.py -k "corpus_has_required or standardized_output" -q`.
- Kết quả kiểm tra ngày 25/09/2026: **3 passed, 2 deselected**. Có 5 PDF, 5 JSON đủ metadata và 10 Markdown (5 legal, 5 news); các Markdown đều đạt ngưỡng tối thiểu 200 ký tự của test.
- Trước triển khai, Task 1–3 còn TODO/`NotImplementedError`; hiện đã có code và bộ dữ liệu đầu ra để bàn giao cho bước chunking/indexing. Các test trên xác nhận số lượng, metadata và độ dài; chưa đánh giá chất lượng retrieval/generation.
- Lỗi môi trường ghi nhận: lệnh `python` trỏ tới Python 3.9 và báo thiếu `markitdown` khi chạy Task 3. Dùng Python trong `.venv` của project để kiểm thử; khi demo cần sử dụng đúng môi trường đã cài dependencies.

## Điều còn hạn chế

- Một hạn chế cụ thể: Markdown bài viết còn chứa menu điều hướng của website (quan sát trong `data/standardized/news/article_01.md`), có thể tạo chunk nhiễu và làm giảm chất lượng tìm kiếm.
- Cải tiến ưu tiên: chỉ trích xuất vùng nội dung bài viết, loại menu/footer và kiểm tra lại nội dung với trang nguồn trước khi bàn giao để index.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: Dương Văn Thành
