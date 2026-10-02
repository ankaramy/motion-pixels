"""Assemble the Anthropic-draft chapters into one master Markdown and typeset a clean A4 PDF
(black text on white), placing real figures from the repo and THESIS/figures/booklet inline.

Read-only over research data. Does not train, infer, or alter any dataset or checkpoint.
Run:  python THESIS/research/build_thesis_pdf.py
"""
from pathlib import Path
from html import escape
import re

from markdown_it import MarkdownIt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak,
    LongTable, TableStyle, KeepTogether, Image, Table, HRFlowable,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / 'THESIS'
BOOKLET = THESIS / 'figures' / 'booklet'
MASTER = THESIS / 'MASTER_THESIS.md'
OUTDIR = ROOT / 'output' / 'pdf'
OUTPUT = OUTDIR / 'MOTION_PIXELS_CONTENT_FINAL.pdf'
OUTDIR.mkdir(parents=True, exist_ok=True)

MP_LOGO = BOOKLET / 'mp_logo.png'          # white artwork on transparent
MP_BLACK = BOOKLET / 'mp_logo_black.png'   # black version for white pages
IAAC_LOGO = BOOKLET / 'iaac_logo.png'      # already black

# make a black version of the (white) Motion Pixels logo so it reads on white pages
if MP_LOGO.exists() and not MP_BLACK.exists():
    try:
        from PIL import Image as PImage
        im = PImage.open(MP_LOGO).convert('RGBA')
        im.putdata([(17, 17, 20, a) for r, g, b, a in im.getdata()])
        im.save(MP_BLACK)
    except Exception:
        pass
MP_COVER = MP_BLACK if MP_BLACK.exists() else MP_LOGO

# ---- 1. Assemble master in reading order -------------------------------------
ORDER = [
    'front_matter/00_title_page.md',
    'front_matter/01_abstract.md',
    'front_matter/02_preface.md',
    '__INDEX__',
    'chapters/01_the_flaw_in_the_plan.md',
    'chapters/02_tools_and_instruments_for_behavioral_analysis.md',
    'chapters/03_motion_pixels_learning_movement_from_barcelona.md',
    'chapters/04_what_comes_next.md',
    'chapters/05_conclusion.md',
    'back_matter/bibliography.md',
]

def strip_audit_footer(text):
    text = re.sub(r'\n---\n\n\*Target[^\n]*\n?', '\n', text)
    text = re.sub(r'\n\*Target[^\n]*\n?', '\n', text)
    return text.rstrip()

chunks = []
for rel in ORDER:
    if rel == '__INDEX__':
        chunks.append('# Index\n')
        continue
    chunks.append(strip_audit_footer((THESIS / rel).read_text(encoding='utf8')))
master = '\n\n'.join(chunks) + '\n'
MASTER.write_text(master, encoding='utf8')

SEP = '\x1f'
FIG_RE = re.compile(
    r'\[FIGURE\s+([\d.]+)\s+HERE\]\s*\nSource file:\s*(.*?)\s*\nProposed caption:\s*(.*?)(?=\n\n|\n#|\Z)',
    re.S)

def fig_repl(m):
    figno, src, cap = m.group(1), m.group(2).strip(), ' '.join(m.group(3).split())
    return f"\n\n@@FIG@@{figno}{SEP}{src}{SEP}{cap}@@\n\n"

master_pdf = FIG_RE.sub(fig_repl, master)

# ---- 2. Fonts + palette (light) ----------------------------------------------
fonts = Path('C:/Windows/Fonts')
for name, filename in [
    ('Georgia', 'georgia.ttf'), ('GeorgiaBold', 'georgiab.ttf'),
    ('GeorgiaItalic', 'georgiai.ttf'), ('GeorgiaBoldItalic', 'georgiaz.ttf'),
    ('Calibri', 'calibri.ttf'), ('CalibriBold', 'calibrib.ttf'),
    ('CalibriItalic', 'calibrii.ttf'), ('CalibriBoldItalic', 'calibriz.ttf'),
]:
    pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
pdfmetrics.registerFontFamily('Georgia', normal='Georgia', bold='GeorgiaBold',
                              italic='GeorgiaItalic', boldItalic='GeorgiaBoldItalic')
pdfmetrics.registerFontFamily('Calibri', normal='Calibri', bold='CalibriBold',
                              italic='CalibriItalic', boldItalic='CalibriBoldItalic')

WIDTH, HEIGHT = A4
MARGIN = 24 * mm
CONTENT_WIDTH = WIDTH - 2 * MARGIN
PAPER = colors.HexColor('#ffffff')
INK = colors.HexColor('#1d1d1f')
BLACK = colors.HexColor('#111114')
MUTE = colors.HexColor('#5a5a55')
FAINT = colors.HexColor('#9a9a93')
ACCENT = colors.HexColor('#5b4bc4')       # deep violet, ties to the animation palette
RULE = colors.HexColor('#d9d7cc')
RULE2 = colors.HexColor('#c7c5b9')
MAX_IMG_W = CONTENT_WIDTH
MAX_IMG_H = 152 * mm

styles = {
    'body': ParagraphStyle('body', fontName='Georgia', fontSize=10.7, leading=15.6,
                           spaceAfter=9, textColor=INK, splitLongWords=1),
    'cover': ParagraphStyle('cover', fontName='CalibriBold', fontSize=41, leading=45,
                            spaceBefore=96, spaceAfter=12, textColor=BLACK),
    'coversub': ParagraphStyle('coversub', fontName='CalibriBold', fontSize=15, leading=20,
                               spaceBefore=2, spaceAfter=14, textColor=ACCENT),
    'h1': ParagraphStyle('h1', fontName='CalibriBold', fontSize=24, leading=28,
                         spaceBefore=8, spaceAfter=5, textColor=BLACK, keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName='CalibriBold', fontSize=14.5, leading=19,
                         spaceBefore=17, spaceAfter=8, textColor=BLACK, keepWithNext=True),
    'h3': ParagraphStyle('h3', fontName='CalibriBoldItalic', fontSize=11.8, leading=15.5,
                         spaceBefore=12, spaceAfter=6, textColor=colors.HexColor('#3a3a42'),
                         keepWithNext=True),
    'note': ParagraphStyle('note', fontName='Calibri', fontSize=9.2, leading=12.8,
                           spaceBefore=4, spaceAfter=13, backColor=colors.HexColor('#f2f0e8'),
                           borderColor=colors.HexColor('#ddd7bf'), borderWidth=0.6,
                           borderPadding=8, textColor=colors.HexColor('#8a5a00')),
    'placeholder': ParagraphStyle('placeholder', fontName='CalibriBold', fontSize=9.5, leading=13,
                                  spaceBefore=8, spaceAfter=2, textColor=ACCENT,
                                  backColor=colors.HexColor('#f1effb'),
                                  borderColor=colors.HexColor('#d5cef2'), borderWidth=0.6, borderPadding=9),
    'caption': ParagraphStyle('caption', fontName='Calibri', fontSize=8.9, leading=12,
                              spaceBefore=6, spaceAfter=15, textColor=MUTE, alignment=0),
    'cell': ParagraphStyle('cell', fontName='Calibri', fontSize=9, leading=11.6, textColor=INK),
    'cellhead': ParagraphStyle('cellhead', fontName='CalibriBold', fontSize=9, leading=11.6, textColor=BLACK),
    'bullet': ParagraphStyle('bullet', fontName='Georgia', fontSize=10.7, leading=15.6,
                             spaceAfter=5, leftIndent=14, firstLineIndent=-10, textColor=INK),
    'keywords': ParagraphStyle('keywords', fontName='GeorgiaItalic', fontSize=10.5, leading=15,
                               spaceBefore=6, spaceAfter=9, textColor=colors.HexColor('#3a3a42')),
    'toc': ParagraphStyle('toc', fontName='CalibriBold', fontSize=12, leading=19,
                          leftIndent=0, rightIndent=22, firstLineIndent=0, textColor=BLACK),
    'toc2': ParagraphStyle('toc2', fontName='Calibri', fontSize=10.3, leading=16,
                           leftIndent=16, rightIndent=22, firstLineIndent=0, textColor=colors.HexColor('#4a4a52')),
    'flag': ParagraphStyle('flag', fontName='Calibri', fontSize=9.2, leading=12.8,
                           spaceBefore=4, spaceAfter=13, backColor=colors.HexColor('#fdecea'),
                           borderColor=colors.HexColor('#eeb4ad'), borderWidth=0.6, borderPadding=8,
                           textColor=colors.HexColor('#c81e1e')),
}

# every author-input / citation marker is numbered document-wide (reading order) and shown in red
FLAG_RE = re.compile(r'\[(?:AUTHOR INPUT REQUIRED|CITATION REQUIRED)[^\]]*\]')
FLAGNO = [0]

def flagify(html):
    def _r(m):
        FLAGNO[0] += 1
        return '<font color="#c81e1e"><b>[' + str(FLAGNO[0]) + ']</b> ' + m.group(0)[1:-1] + '</font>'
    return FLAG_RE.sub(_r, html)

def titlecase(s):
    return ' '.join((w[:1].upper() + w[1:]) if w else w for w in s.split(' '))

def normalize(s):
    return (s.replace('—', ', ').replace('–', '-').replace('‑', '-')
             .replace('’', "'").replace('‘', "'")
             .replace('“', '"').replace('”', '"'))

def inline(token):
    out = []
    for c in token.children or []:
        t = c.type
        if t == 'text':
            out.append(escape(normalize(c.content)))
        elif t == 'softbreak':
            out.append(' ')
        elif t == 'hardbreak':
            out.append('<br/>')
        elif t == 'strong_open':
            out.append('<b>')
        elif t == 'strong_close':
            out.append('</b>')
        elif t == 'em_open':
            out.append('<i>')
        elif t == 'em_close':
            out.append('</i>')
        elif t == 'code_inline':
            out.append('<font name="Calibri" size="9">' + escape(normalize(c.content)) + '</font>')
        elif t == 'link_open':
            href = c.attrGet('href') or ''
            out.append('<link href="' + escape(href, quote=True) + '" color="#3b5bdb">')
        elif t == 'link_close':
            out.append('</link>')
        else:
            out.append(escape(normalize(c.content)))
    return ''.join(out)

def figure_flow(figno, src, cap):
    # Source may list several images separated by ' | ' -> one centred row, no frames.
    src = src.strip()
    parts = [Spacer(1, 7)]
    if not src.startswith('['):
        paths = [ROOT / p.strip() for p in src.split('|')]
        paths = [p for p in paths if p.exists()]
        if paths:
            n = len(paths)
            avail = (MAX_IMG_W - 8 * (n - 1)) / n
            row_h = MAX_IMG_H if n == 1 else 108 * mm
            cells = []
            for p in paths:
                pw, ph = ImageReader(str(p)).getSize()
                scale = min(avail / pw, row_h / ph)
                cells.append(Image(str(p), width=pw * scale, height=ph * scale))
            row = Table([cells], colWidths=[MAX_IMG_W / n] * n)
            row.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ]))
            row.hAlign = 'CENTER'
            parts.append(row)
        else:
            FLAGNO[0] += 1
            parts.append(Paragraph('<font color="#c81e1e"><b>[' + str(FLAGNO[0]) + ']</b> Figure ' + escape(figno)
                                   + ': image not found (' + escape(src) + ')</font>', styles['flag']))
    else:
        note = src[1:-1] if src.startswith('[') and src.endswith(']') else src
        FLAGNO[0] += 1
        parts.append(Paragraph('<font color="#c81e1e"><b>[' + str(FLAGNO[0]) + ']</b> Figure ' + escape(figno)
                               + ': image to be supplied. ' + escape(normalize(note)) + '</font>', styles['flag']))
    cap_html = '<font color="#5b4bc4"><b>Figure ' + escape(figno) + '.</b></font> ' + flagify(escape(normalize(cap)))
    parts.append(Paragraph(cap_html, styles['caption']))
    return KeepTogether(parts)

class Doc(BaseDocTemplate):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.addPageTemplates(PageTemplate(id='main', frames=[Frame(
            MARGIN, 20 * mm, CONTENT_WIDTH, HEIGHT - 42 * mm, id='t',
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
            onPage=self.decorate))

    def afterFlowable(self, f):
        if hasattr(f, 'toc_title'):
            lvl = getattr(f, 'toc_level', 0)
            self.canv.bookmarkPage(f.toc_key)
            self.canv.addOutlineEntry(f.toc_title, f.toc_key, level=lvl, closed=False)
            if f.toc_title not in ('Motion Pixels', 'Index'):
                self.notify('TOCEntry', (lvl, f.toc_title, self.page, f.toc_key))

    def _cover(self, canvas):
        if MP_COVER.exists():
            pw, ph = ImageReader(str(MP_COVER)).getSize()
            w = 104 * mm
            h = w * ph / pw
            canvas.drawImage(str(MP_COVER), (WIDTH - w) / 2, HEIGHT - 32 * mm - h,
                             width=w, height=h, mask='auto', preserveAspectRatio=True)
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(1.6)
        canvas.line(MARGIN, 44 * mm, MARGIN + 30 * mm, 44 * mm)
        if IAAC_LOGO.exists():
            pw, ph = ImageReader(str(IAAC_LOGO)).getSize()
            w = 22 * mm
            h = w * ph / pw
            canvas.drawImage(str(IAAC_LOGO), MARGIN, 20 * mm, width=w, height=h,
                             mask='auto', preserveAspectRatio=True)

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(PAPER)
        canvas.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
        if doc.page == 1:
            self._cover(canvas)
            canvas.restoreState()
            return
        canvas.setFont('Calibri', 8)
        canvas.setFillColor(FAINT)
        canvas.drawString(MARGIN, HEIGHT - 14 * mm, 'MOTION PIXELS')
        canvas.drawRightString(WIDTH - MARGIN, HEIGHT - 14 * mm, 'THESIS DRAFT')
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, HEIGHT - 16 * mm, WIDTH - MARGIN, HEIGHT - 16 * mm)
        canvas.setFillColor(MUTE)
        canvas.drawCentredString(WIDTH / 2, 11 * mm, str(doc.page))
        canvas.restoreState()

md = MarkdownIt('commonmark').enable('table')
tokens = md.parse(master_pdf)
story = []
i = 0
heading_count = 0
list_depth = 0
skip_index = False
sub_count = [0]
while i < len(tokens):
    t = tokens[i]
    if t.type == 'heading_open':
        level = int(t.tag[1])
        following = tokens[i + 1]
        title = normalize(following.content)
        if level == 1:
            if story:
                story.append(PageBreak())
            heading_count += 1
            skip_index = (title == 'Index')
            disp = titlecase(normalize(following.content))
            if heading_count == 1:
                anchor = Spacer(1, 172)   # cover masthead is the logo, drawn on the canvas
                anchor.toc_title = disp
                anchor.toc_key = 'sec1'
                story.append(anchor)
                i += 3
                continue
            p = Paragraph(escape(disp), styles['h1'])
            p.toc_title = disp
            p.toc_key = f'sec{heading_count}'
            p.toc_level = 0
            story.append(p)
            if not skip_index:
                story.append(HRFlowable(width='100%', thickness=1.2, color=ACCENT,
                                        spaceBefore=1, spaceAfter=15, lineCap='round'))
            if skip_index:
                toc = TableOfContents()
                toc.levelStyles = [styles['toc'], styles['toc2']]
                story += [Spacer(1, 10), toc]
            i += 3
            continue
        elif not skip_index:
            disp = titlecase(normalize(following.content))
            key = 'coversub' if heading_count == 1 else f'h{min(level, 3)}'
            p = Paragraph(escape(disp), styles[key])
            if level == 2 and heading_count > 1:
                sub_count[0] += 1
                p.toc_title = disp
                p.toc_key = f'sub{sub_count[0]}'
                p.toc_level = 1
            story.append(p)
        i += 3
        continue
    if skip_index:
        i += 1
        continue
    if t.type == 'table_open':
        rows = []
        header = True
        i += 1
        while tokens[i].type != 'table_close':
            tt = tokens[i]
            if tt.type == 'thead_close':
                header = False
            if tt.type == 'tr_open':
                row = []
            if tt.type == 'inline':
                row.append(Paragraph(inline(tt), styles['cellhead' if header else 'cell']))
            if tt.type == 'tr_close':
                rows.append(row)
            i += 1
        n = len(rows[0])
        weights = [0.5, 0.25, 0.25] if n == 3 else [1.0 / n] * n
        tbl = LongTable(rows, colWidths=[CONTENT_WIDTH * w for w in weights], repeatRows=1, hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eae7dd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f6f0')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, 0), (-1, 0), 0.9, ACCENT),
            ('LINEBELOW', (0, 1), (-1, -2), 0.3, RULE),
        ]))
        story += [Spacer(1, 4), KeepTogether([tbl]), Spacer(1, 13)]
        i += 1
        continue
    if t.type in ('bullet_list_open', 'ordered_list_open'):
        list_depth += 1
    elif t.type in ('bullet_list_close', 'ordered_list_close'):
        list_depth -= 1
    elif t.type == 'paragraph_open':
        ft = tokens[i + 1]
        raw = ft.content
        if raw.strip().startswith('@@FIG@@'):
            m = re.match(r'@@FIG@@(.*?)' + SEP + r'(.*?)' + SEP + r'(.*)@@$', raw.strip(), re.S)
            if m:
                story.append(figure_flow(m.group(1), m.group(2), m.group(3)))
                i += 3
                continue
        content = flagify(inline(ft))
        style = 'body'
        if raw.lstrip().startswith('[AUTHOR INPUT REQUIRED') or raw.lstrip().startswith('[CITATION REQUIRED'):
            style = 'flag'
        elif raw.startswith('**Keywords'):
            style = 'keywords'
        if list_depth:
            style = 'bullet'
            content = '&#8226;&nbsp; ' + content
        story.append(Paragraph(content, styles[style]))
        i += 3
        continue
    i += 1

doc = Doc(str(OUTPUT), pagesize=A4, title='Motion Pixels, Thesis Draft', author='Ramy Anka')
doc.multiBuild(story)

words = len(re.sub(r'\[[^\]]*\]', '', master).split())
print('PDF:', OUTPUT, '| bytes:', OUTPUT.stat().st_size if OUTPUT.exists() else 0)
print('figures:', len(FIG_RE.findall(master)), '| approx words:', words)
