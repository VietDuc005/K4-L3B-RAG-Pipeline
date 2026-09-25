# Individual Contribution Report

---

## Thông tin

- **Họ và tên:** Nguyễn Viết Đức
- **Mã học viên:** 2A202602732
- **Nhóm:** K4-L3B
- **Vai trò:** Fusion & Fallback Pipeline (Task 7, 8, 9)
- **Repository/branch:** main

---

## Phần việc đã thực hiện

| Module / Deliverable | Việc tôi trực tiếp làm | File / PR | Trạng thái |
|---|---|---|---|
| **Task 7: RRF Reranking** | Cài đặt hàm `rerank_rrf()` theo công thức Reciprocal Rank Fusion $\sum \frac{1}{k + \text{rank}}$, gộp kết quả theo ID, copy object tránh mutation danh sách gốc, chuẩn hóa `score` và gán `retrieval_method="hybrid"`. | [task7_reranking.py](file:///d:/Duc_Ky_6/VinUni/K4-L3B-RAG-Pipeline/src/task7_reranking.py) | **Done** |
| **Task 8: PageIndex Fallback** | Cài đặt cơ chế vectorless fallback: `upload_documents()`, lưu cache `pageindex_doc_ids.json`, cấu hình timeout và wrap try/except trong `pageindex_search()` để chống crash pipeline khi provider lỗi. | [task8_pageindex_vectorless.py](file:///d:/Duc_Ky_6/VinUni/K4-L3B-RAG-Pipeline/src/task8_pageindex_vectorless.py) | **Done** |
| **Task 9: Retrieval Pipeline** | Cài đặt hàm `retrieve()` kết nối dense và BM25, fuse RRF đúng 1 lần, so sánh `best_dense_score` với `SCORE_THRESHOLD`, kích hoạt fallback nếu không đủ tự tin và bảo toàn hybrid kết quả nếu fallback lỗi. | [task9_retrieval_pipeline.py](file:///d:/Duc_Ky_6/VinUni/K4-L3B-RAG-Pipeline/src/task9_retrieval_pipeline.py) | **Done** |

---

## Quyết định kỹ thuật quan trọng

### 1. Hợp nhất thứ hạng RRF và bảo toàn dữ liệu đầu vào (Task 7)
- **Quyết định:** Áp dụng chuẩn công thức RRF: $RRF(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$ với hằng số $k = 60$, rank 1-indexed. Thực hiện copy nông/sâu bản ghi gốc (`dict(item)`) trước khi ghi đè `score` và `retrieval_method = "hybrid"`.
- **Lý do / Evidence:** Dense similarity (cosine) và BM25 score nằm ở hai thang đo hoàn toàn khác nhau (cosine $\in [0, 1]$, BM25 $\in [0, +\infty)$), không thể cộng trực tiếp. RRF tận dụng vị trí xếp hạng để đưa chunk có độ liên quan cao ở cả 2 phương pháp lên đầu. Việc copy item giúp tránh side-effect làm biến đổi danh sách đầu vào, giữ nguyên `score` cosine gốc phục vụ việc đánh giá fallback ở Task 9.
- **Trade-off:** RRF score chỉ phản ánh thứ hạng tương đối ($\approx 0.01 - 0.03$), mất đi ý nghĩa khoảng cách xác suất/ngữ nghĩa tuyệt đối của cosine similarity.

### 2. Tách biệt Score quyết định Fallback và RRF Score (Task 9)
- **Quyết định:** Nhánh quyết định fallback dựa hoàn toàn vào `best_dense_score` (cosine similarity cao nhất trong danh sách `semantic_search`), tuyệt đối không dùng RRF score.
- **Lý do / Evidence:** RRF score là đại lượng tích lũy thứ hạng, giá trị phụ thuộc vào số lượng danh sách fuse và vị trí rank, không phản ánh mức độ tự tin ngữ nghĩa đối với câu hỏi. Cosine similarity của Dense model là chỉ số đo lường chính xác nhất về độ tin cậy của tài liệu đối với câu hỏi trong/ngoài miền.
- **Trade-off:** Cần lưu trữ lại `best_dense_score` từ danh sách dense trước khi fuse hoặc trích xuất từ phần tử đầu tiên của dense search.

### 3. Hiệu chỉnh ngưỡng `SCORE_THRESHOLD` (Calibration)
- **Quyết định:** Lựa chọn `SCORE_THRESHOLD = 0.35` (hoặc cấu hình động qua biến môi trường).
- **Thực nghiệm hiệu chỉnh:**
  - **In-domain query:** *"What is Toyota's anti-bribery policy?"*  
    $\rightarrow$ Best Dense Cosine Score = **0.8001** (Độ tin cậy rất cao, vượt ngưỡng $\rightarrow$ Sử dụng hybrid RRF results: chunk `legal/toyota_global_abac_policy.md::chunk-54`).
  - **Out-of-domain query:** *"How to bake chocolate cookies?"*  
    $\rightarrow$ Best Dense Cosine Score = **0.1173** (Độ tin cậy rất thấp, dưới ngưỡng $\rightarrow$ Kích hoạt nhánh fallback sang PageIndex).
- **Quan sát & Nhận định:** Không tồn tại một ngưỡng cố định cho mọi corpus; giá trị `0.35` phân tách rõ rệt giữa query đúng domain và query ngoài domain trên tập tài liệu của Toyota (chính sách ESG/ABAC và tin tức).

### 4. Thiết kế phòng vệ chịu lỗi (Fault-Tolerance) cho Fallback (Task 8 & 9)
- **Quyết định:** Toàn bộ lệnh gọi đến PageIndex API được bọc trong block `try...except`, thiết lập timeout và kiểm tra khóa API. Khi PageIndex gặp lỗi kết nối hoặc quota limit, pipeline không ném lỗi ra ngoài mà trả về hybrid result sẵn có hoặc danh sách rỗng để tầng generation sinh safe refusal.
- **Lý do / Evidence:** Dịch vụ ngoài tiềm ẩn rủi ro mạng và hạn ngạch (rate limit). Hệ thống không được dừng hoặc crash giao diện Streamlit khi dịch vụ phụ trợ gặp sự cố.

---

## Kiểm thử và kết quả

- **Lệnh kiểm thử đã chạy:**
  ```powershell
  .venv\Scripts\python -m src.task7_reranking
  .venv\Scripts\python -m src.task9_retrieval_pipeline
  .venv\Scripts\python -m pytest tests/test_contracts.py -v
  ```
- **Kết quả kiểm thử:**
  - `tests/test_contracts.py::test_rrf_uses_rank_deduplicates_and_marks_hybrid` $\rightarrow$ **PASSED**
  - `tests/test_contracts.py::test_retrieve_uses_dense_score_for_fallback` $\rightarrow$ **PASSED**
  - `tests/test_contracts.py::test_retrieve_fuses_once_when_dense_is_confident` $\rightarrow$ **PASSED**
  - `tests/test_contracts.py::test_retrieve_survives_fallback_provider_error` $\rightarrow$ **PASSED**
  - Contract validation xác nhận các chunk xuất hiện ở cả 2 danh sách nhận đúng tổng điểm RRF, kết quả không trùng ID, sắp xếp giảm dần và đúng định dạng `SearchResult`.

---

## Điều còn hạn chế

- **Hạn chế:** Khi PageIndex chưa cấu hình API key, fallback trả về hybrid result hiện có ngay cả khi điểm tự tin thấp. Nếu truy vấn hoàn toàn vô nghĩa, generation vẫn cần một tầng lọc bằng chứng (evidence check) để đưa ra câu trả lời từ chối an toàn (safe refusal).
- **Cải tiến nếu có thêm thời gian:** Tích hợp thêm Cross-Encoder reranker làm tầng rerank thứ hai sau RRF để so sánh hiệu năng scoring giữa RRF baseline và Cross-Encoder học sâu.

---

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- **Ngày:** 25/09/2026
- **Tên thành viên:** Nguyễn Viết Đức
