"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

SOURCES = {
    "toyota_global_abac_policy.pdf": (
        "https://global.toyota/pages/global_toyota/company/vision-and-philosophy/"
        "en_Global_ABAC_Policy.pdf"
    ),
    "toyota_global_speakup_policy.pdf": (
        "https://global.toyota/pages/global_toyota/company/vision-and-philosophy/"
        "en_Toyota_Global_Speakup_Policy.pdf"
    ),
    "toyota_tax_policy.pdf": (
        "https://global.toyota/pages/global_toyota/sustainability/esg/tax-policy_en.pdf"
    ),
    "toyota_information_security_policy.pdf": (
        "https://global.toyota/pages/global_toyota/sustainability/esg/"
        "information-security-policy_en.pdf"
    ),
    "toyota_human_rights_policy.pdf": (
        "https://global.toyota/pages/global_toyota/sustainability/esg/social/"
        "human_rights_policy_en.pdf"
    ),
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    for filename, url in SOURCES.items():
        destination = DATA_DIR / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Skipped (already exists): {destination}")
            continue

        response = requests.get(
            url,
            headers={"User-Agent": "K4-L3B-RAG-Pipeline/1.0"},
            timeout=30,
            stream=True,
        )
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
        print(f"Saved: {destination}")

if __name__ == "__main__":
    setup_directory()
    download_documents()