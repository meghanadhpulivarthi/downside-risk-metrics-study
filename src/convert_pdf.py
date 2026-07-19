"""
Convert the paper Markdown to PDF with no system dependencies.
Path: markdown -> HTML (tables) -> PDF via xhtml2pdf (pure Python), using the
DejaVuSans Unicode font so ρ, †, ≤, — render correctly.

Run: uv run python -u src/convert_pdf.py
"""

import os
from pathlib import Path

import markdown
from xhtml2pdf import pisa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_MD = PROJECT_ROOT / "writeup" / "volatility_persistence_paper.md"
OUT_PDF = PROJECT_ROOT / "writeup" / "volatility_persistence_paper.pdf"

FONT_REG = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf")


def build_css():
    faces = [f"@font-face {{ font-family: 'DejaVu'; src: url('{FONT_REG}'); }}"]
    if FONT_BOLD.exists():
        faces.append(f"@font-face {{ font-family: 'DejaVu'; src: url('{FONT_BOLD}'); font-weight: bold; }}")
    return "\n".join(faces) + """
    @page { size: A4; margin: 1.6cm; }
    body { font-family: 'DejaVu'; font-size: 9.5pt; line-height: 1.35; color: #111; }
    h1 { font-size: 15pt; margin: 0 0 6pt 0; }
    h2 { font-size: 12pt; margin: 12pt 0 4pt 0; border-bottom: 0.5pt solid #999; }
    h3 { font-size: 10.5pt; margin: 8pt 0 3pt 0; }
    p, li { margin: 0 0 4pt 0; }
    code { font-size: 8.5pt; }
    table { border-collapse: collapse; width: 100%; margin: 4pt 0; }
    th, td { border: 0.5pt solid #666; padding: 2pt 3pt; font-size: 7.6pt; text-align: left; }
    th { background: #eee; }
    em { color: #333; }
    img { max-width: 100%; display: block; margin: 4pt auto; }
    """


def link_callback(uri, rel):
    """Resolve <img>/font resources for xhtml2pdf. Relative paths (e.g. figures/*.png)
    are resolved against the paper's directory so the run works from any CWD."""
    if os.path.isabs(uri) and os.path.exists(uri):
        return uri
    resolved = (SRC_MD.parent / uri).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"PDF resource not found: {uri} -> {resolved}")
    return str(resolved)


def main():
    md_text = SRC_MD.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><style>{build_css()}</style></head><body>{html_body}</body></html>"
    with open(OUT_PDF, "wb") as out:
        result = pisa.CreatePDF(html, dest=out, encoding="utf-8", link_callback=link_callback)
    if result.err:
        raise RuntimeError(f"PDF conversion reported {result.err} error(s)")
    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"Wrote {OUT_PDF} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
