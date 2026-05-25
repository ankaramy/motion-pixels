"""
build_sandbox_exit_pdf.py
-------------------------
Render docs/sandbox_exit_brief.md to docs/sandbox_exit_brief.pdf.

Handles the subset of markdown the brief actually uses:
  - ATX headings (#, ##, ###)
  - Paragraphs
  - Blockquotes (> ...)
  - Horizontal rules (---)
  - Unordered lists (-, *) including bold-prefixed items
  - Ordered lists (1. 2. 3.)
  - Pipe tables with --- separator row
  - Inline bold (**...**) and inline code (`...`)
  - Fenced code blocks (```...```)

Usage:
    python docs/build_sandbox_exit_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
SRC = HERE / "sandbox_exit_brief.md"
OUT = HERE / "sandbox_exit_brief.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontName="Helvetica-Bold", fontSize=20, leading=24,
            spaceAfter=10, textColor=colors.HexColor("#222"),
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontName="Helvetica-Bold", fontSize=14, leading=18,
            spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#222"),
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"],
            fontName="Helvetica-Bold", fontSize=11.5, leading=15,
            spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#333"),
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"],
            fontName="Helvetica", fontSize=9.5, leading=13,
            alignment=TA_LEFT, spaceAfter=5,
        ),
        "quote": ParagraphStyle(
            "quote", parent=base["BodyText"],
            fontName="Helvetica-Oblique", fontSize=10.5, leading=14,
            leftIndent=14, rightIndent=14, spaceBefore=4, spaceAfter=8,
            textColor=colors.HexColor("#444"),
            borderColor=colors.HexColor("#bbb"),
            borderPadding=6, borderWidth=0,
        ),
        "list": ParagraphStyle(
            "list", parent=base["BodyText"],
            fontName="Helvetica", fontSize=9.5, leading=13,
            leftIndent=14, bulletIndent=2, spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "code", parent=base["BodyText"],
            fontName="Courier", fontSize=8.5, leading=11,
            leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=8,
            backColor=colors.HexColor("#f4f4f4"),
            borderColor=colors.HexColor("#ddd"), borderWidth=0.5, borderPadding=6,
        ),
        "table_cell": ParagraphStyle(
            "table_cell", parent=base["BodyText"],
            fontName="Helvetica", fontSize=8.5, leading=11,
            alignment=TA_LEFT,
        ),
        "table_head": ParagraphStyle(
            "table_head", parent=base["BodyText"],
            fontName="Helvetica-Bold", fontSize=8.5, leading=11,
            alignment=TA_LEFT, textColor=colors.white,
        ),
    }
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Inline markup
# ─────────────────────────────────────────────────────────────────────────────
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+?)`")


def inline(text: str) -> str:
    """Convert the subset of inline markdown used in the brief to ReportLab
    paragraph markup."""
    out = text
    # Escape XML first so we don't break ReportLab's parser
    out = out.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = _BOLD.sub(r"<b>\1</b>", out)
    out = _CODE.sub(
        r'<font face="Courier" size="9" backColor="#f4f4f4">\1</font>',
        out,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Block parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_table(lines, start):
    """Parse a pipe table starting at `lines[start]`. Returns (rows, end_index)
    where rows is a list of list-of-strings (header first) and end_index is the
    index past the last consumed line."""
    rows = []
    i = start
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1
    # Drop the separator row (---|---|---) if present
    rows = [r for r in rows if not all(set(c) <= set("-: ") and c for c in r)]
    return rows, i


def build_table_flowable(rows, styles, page_width):
    if not rows:
        return None
    head = [Paragraph(inline(c), styles["table_head"]) for c in rows[0]]
    body = [
        [Paragraph(inline(c), styles["table_cell"]) for c in row]
        for row in rows[1:]
    ]
    data = [head] + body

    n_cols = len(rows[0])
    col_width = page_width / n_cols
    t = Table(data, colWidths=[col_width] * n_cols, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#444")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#fafafa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def parse_markdown(md: str, styles, page_width):
    """Walk through markdown line-by-line and emit a list of ReportLab
    flowables."""
    lines = md.splitlines()
    flow = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # blank line
        if not stripped:
            i += 1
            continue

        # fenced code block
        if stripped.startswith("```"):
            j = i + 1
            buf = []
            while j < n and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            code = "\n".join(buf).replace("&", "&amp;") \
                                  .replace("<", "&lt;").replace(">", "&gt;")
            code = code.replace("\n", "<br/>")
            flow.append(Paragraph(code, styles["code"]))
            i = j + 1
            continue

        # horizontal rule
        if stripped == "---":
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(
                width="100%", thickness=0.6, color=colors.HexColor("#999"),
                spaceBefore=2, spaceAfter=6,
            ))
            i += 1
            continue

        # headings
        if stripped.startswith("# "):
            flow.append(Paragraph(inline(stripped[2:]), styles["title"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flow.append(Paragraph(inline(stripped[3:]), styles["h2"]))
            i += 1
            continue
        if stripped.startswith("### "):
            flow.append(Paragraph(inline(stripped[4:]), styles["h3"]))
            i += 1
            continue

        # blockquote
        if stripped.startswith("> "):
            buf = []
            while i < n and lines[i].strip().startswith("> "):
                buf.append(lines[i].strip()[2:])
                i += 1
            flow.append(Paragraph(inline(" ".join(buf)), styles["quote"]))
            continue

        # pipe table (header row followed by | --- |)
        if (stripped.startswith("|")
                and i + 1 < n
                and set(lines[i + 1].strip().strip("|").replace("|", "")
                        .replace(" ", "")) <= set("-:")):
            rows, j = parse_table(lines, i)
            t = build_table_flowable(rows, styles, page_width)
            if t is not None:
                flow.append(Spacer(1, 2))
                flow.append(t)
                flow.append(Spacer(1, 4))
            i = j
            continue

        # unordered list
        if stripped.startswith(("- ", "* ")):
            while i < n and lines[i].strip().startswith(("- ", "* ")):
                item = lines[i].strip()[2:]
                flow.append(Paragraph(
                    f"• {inline(item)}", styles["list"]))
                i += 1
            flow.append(Spacer(1, 2))
            continue

        # ordered list
        if re.match(r"^\d+\.\s", stripped):
            while i < n and re.match(r"^\d+\.\s", lines[i].strip()):
                m = re.match(r"^(\d+)\.\s(.*)$", lines[i].strip())
                num, rest = m.group(1), m.group(2)
                flow.append(Paragraph(
                    f"{num}. {inline(rest)}", styles["list"]))
                i += 1
            flow.append(Spacer(1, 2))
            continue

        # paragraph — accumulate consecutive non-blank, non-special lines
        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (not nxt
                or nxt.startswith(("#", "- ", "* ", ">", "```", "|"))
                or nxt == "---"
                or re.match(r"^\d+\.\s", nxt)):
                break
            buf.append(nxt)
            i += 1
        flow.append(Paragraph(inline(" ".join(buf)), styles["body"]))

    return flow


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    md = SRC.read_text(encoding="utf-8")
    styles = make_styles()

    margin = 1.6 * cm
    page_width = A4[0] - 2 * margin

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title="Motion Pixels — Sandbox Exit Brief",
        author="Motion Pixels project",
    )

    flow = parse_markdown(md, styles, page_width)
    doc.build(flow)
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
