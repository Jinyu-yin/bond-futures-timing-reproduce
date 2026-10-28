from __future__ import annotations

from pathlib import Path


def render_pdf_to_pngs(pdf_path: Path, out_dir: Path, dpi: int = 220) -> list[Path]:
    """
    Render each PDF page to a PNG.

    Notes:
    - The PDF is not reliably searchable as plain text for figures.
    - Output file names: page_01.png, page_02.png, ...
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: pymupdf (import fitz). Run: py -m pip install pymupdf") from e

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    out_paths: list[Path] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"page_{i + 1:02d}.png"
        pix.save(out_path.as_posix())
        out_paths.append(out_path)
        print("saved", out_path)
    return out_paths


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pdf = project_root / "just_reference.pdf"
    out_dir = project_root / "reproduce" / "pdf_pages"
    render_pdf_to_pngs(pdf, out_dir, dpi=220)


if __name__ == "__main__":
    main()
