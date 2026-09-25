"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

import json
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".doc", ".docx"}


def write_markdown(path: Path, content: str) -> None:
    """Write non-empty Markdown to its deterministic destination path."""
    cleaned_content = content.strip()
    if not cleaned_content:
        raise ValueError(f"Refusing to write empty Markdown: {path.name}")

    path.write_text(f"{cleaned_content}\n", encoding="utf-8")
    print(f"Saved: {path}")


def convert_legal_docs() -> None:
    """Convert every landing PDF/DOCX document to standardized Markdown."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()

    for path in sorted(legal_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in LEGAL_EXTENSIONS:
            continue

        result = converter.convert(str(path))
        write_markdown(output_dir / f"{path.stem}.md", result.text_content)


def convert_news_articles() -> None:
    """Convert crawled article JSON files to Markdown with source metadata."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    required_fields = ("url", "title", "date_crawled", "content_markdown")
    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"News record must be a JSON object: {path.name}")

        missing_fields = [
            field
            for field in required_fields
            if not isinstance(data.get(field), str) or not data[field].strip()
        ]
        if missing_fields:
            raise ValueError(f"{path.name} is missing: {', '.join(missing_fields)}")

        header = (
            f"# {data['title'].strip()}\n\n"
            f"**Source:** {data['url'].strip()}\n\n"
            f"**Crawled:** {data['date_crawled'].strip()}\n\n---\n\n"
        )
        write_markdown(
            output_dir / f"{path.stem}.md",
            header + data["content_markdown"],
        )


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
