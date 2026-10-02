"""Typeset the existing Markdown manuscript as a simple, linked review PDF."""
from pathlib import Path
from html import escape
import json
import re

from markdown_it import MarkdownIt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak,
    LongTable, TableStyle, KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'THESIS/MASTER_THESIS.md'
OUTPUT = ROOT / 'output/pdf/Motion_Pixels_Thesis_Review.pdf'
QA = ROOT / 'tmp/pdfs/thesis_review'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

fonts = Path('C:/Windows/Fonts')
for name, filename in [
    ('Georgia','georgia.ttf'),('GeorgiaBold','georgiab.ttf'),
    ('GeorgiaItalic','georgiai.ttf'),('GeorgiaBoldItalic','georgiaz.ttf'),
    ('Calibri','calibri.ttf'),('CalibriBold','calibrib.ttf'),
    ('CalibriItalic','calibrii.ttf'),('CalibriBoldItalic','calibriz.ttf'),
]:
    pdfmetrics.registerFont(TTFont(name, str(fonts/filename)))
pdfmetrics.registerFontFamily('Georgia',normal='Georgia',bold='GeorgiaBold',
                            italic='GeorgiaItalic',boldItalic='GeorgiaBoldItalic')
pdfmetrics.registerFontFamily('Calibri',normal='Calibri',bold='CalibriBold',
                            italic='CalibriItalic',boldItalic='CalibriBoldItalic')

WIDTH, HEIGHT = A4
MARGIN = 23*mm
CONTENT_WIDTH = WIDTH - 2*MARGIN
styles = {
    'body': ParagraphStyle('body',fontName='Georgia',fontSize=10.5,leading=14.5,
                           spaceAfter=8,allowWidows=0,allowOrphans=0,
                           textColor=colors.HexColor('#222222'),splitLongWords=1),
    'h1': ParagraphStyle('h1',fontName='CalibriBold',fontSize=22,leading=26,
                         spaceAfter=22,keepWithNext=True),
    'cover': ParagraphStyle('cover',fontName='CalibriBold',fontSize=38,leading=43,
                            spaceBefore=45,spaceAfter=18,keepWithNext=True),
    'h2': ParagraphStyle('h2',fontName='CalibriBold',fontSize=14,leading=18,
                         spaceBefore=15,spaceAfter=10,keepWithNext=True),
    'h3': ParagraphStyle('h3',fontName='CalibriBold',fontSize=11.7,leading=15,
                         spaceBefore=12,spaceAfter=8,keepWithNext=True),
    'note': ParagraphStyle('note',fontName='Calibri',fontSize=9.3,leading=12.5,
                           spaceBefore=3,spaceAfter=12,backColor=colors.HexColor('#F3F3F1'),
                           borderPadding=7,textColor=colors.HexColor('#444444')),
    'cell': ParagraphStyle('cell',fontName='Calibri',fontSize=8.5,leading=10.8,
                           spaceAfter=0,splitLongWords=1),
    'cellhead': ParagraphStyle('cellhead',fontName='CalibriBold',fontSize=8.5,leading=10.8,
                               spaceAfter=0),
    'bullet': ParagraphStyle('bullet',fontName='Georgia',fontSize=10.5,leading=15.2,
                             spaceAfter=6,leftIndent=13,firstLineIndent=-10),
    'bib': ParagraphStyle('bib',fontName='Georgia',fontSize=9.4,leading=12.5,
                          spaceAfter=8,allowWidows=0,allowOrphans=0,splitLongWords=1),
    'closing': ParagraphStyle('closing',fontName='Georgia',fontSize=10.3,leading=13.2,
                              spaceAfter=6,allowWidows=0,allowOrphans=0),
}

def normalize(s):
    return s.replace('\u2014','-').replace('\u2013','-').replace('\u2011','-')

def inline(token):
    result=[]
    for child in token.children or []:
        typ=child.type
        if typ=='text': result.append(escape(normalize(child.content)))
        elif typ in ('softbreak','hardbreak'): result.append('<br/>' if typ=='hardbreak' else ' ')
        elif typ=='strong_open': result.append('<b>')
        elif typ=='strong_close': result.append('</b>')
        elif typ=='em_open': result.append('<i>')
        elif typ=='em_close': result.append('</i>')
        elif typ=='code_inline': result.append('<font name="Calibri">'+escape(normalize(child.content))+'</font>')
        elif typ=='link_open':
            href=child.attrGet('href')
            if not href.startswith(('http:','https:','file:','#')):
                p=Path(href) if re.match(r'^[A-Za-z]:/',href) else SOURCE.parent/href
                href=p.resolve().as_uri()
            result.append('<link href="'+escape(href,quote=True)+'" color="#31546B">')
        elif typ=='link_close': result.append('</link>')
        elif typ=='html_inline': result.append(escape(child.content))
        else: result.append(escape(normalize(child.content)))
    return ''.join(result)

class ReviewDoc(BaseDocTemplate):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.current_section='Motion Pixels'
        self.addPageTemplates(PageTemplate(id='main',frames=[Frame(
            MARGIN,22*mm,CONTENT_WIDTH,HEIGHT-45*mm,id='text',
            leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],
            onPage=self.decorate))
    def beforeDocument(self):
        self.current_section='Motion Pixels'
    def afterFlowable(self,f):
        if hasattr(f,'section_title'):
            title=f.section_title; key=f.bookmark_key
            self.current_section=title
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title,key,level=0,closed=False)
            if title not in ('Motion Pixels','Index'):
                self.notify('TOCEntry',(0,title,self.page,key))
    def decorate(self,canvas,doc):
        canvas.saveState()
        canvas.setFont('Calibri',8)
        canvas.setFillColor(colors.HexColor('#727272'))
        if doc.page>1:
            canvas.drawString(MARGIN,HEIGHT-13*mm,'MOTION PIXELS  |  REVIEW DRAFT')
            canvas.setStrokeColor(colors.HexColor('#D8D8D8'))
            canvas.setLineWidth(.4)
            canvas.line(MARGIN,HEIGHT-16*mm,WIDTH-MARGIN,HEIGHT-16*mm)
        canvas.drawString(MARGIN,12*mm,'Ramy  |  IAAC')
        canvas.drawRightString(WIDTH-MARGIN,12*mm,str(doc.page))
        canvas.restoreState()

md=MarkdownIt('commonmark').enable('table')
tokens=md.parse(SOURCE.read_text(encoding='utf8'))
story=[];i=0;section='';list_depth=0;skip_index=False;heading_count=0
while i<len(tokens):
    t=tokens[i]
    if t.type=='heading_open':
        level=int(t.tag[1]); following=tokens[i+1]; title=normalize(following.content)
        if level==1:
            if story: story.append(PageBreak())
            section=title;skip_index=title=='Index';heading_count+=1
            p=Paragraph(inline(following),styles['cover' if heading_count==1 else 'h1'])
            p.section_title=title;p.bookmark_key=f'section_{heading_count}'
            story.append(p)
            if skip_index:
                toc=TableOfContents()
                toc.levelStyles=[ParagraphStyle('toc',fontName='Calibri',fontSize=12,
                                               leading=18,spaceBefore=8,leftIndent=0,
                                               rightIndent=20,firstLineIndent=0)]
                story += [toc,Spacer(1,20),Paragraph(
                    'Project source labels P01-P12 resolve in the bibliography. '
                    'The original graph links are available in the companion figure register. '
                    'Shaded author-input notes are retained for review.',styles['note'])]
        elif not skip_index:
            story.append(Paragraph(inline(following),styles[f'h{min(level,3)}']))
        i+=3;continue
    if skip_index: i+=1;continue
    if t.type=='table_open':
        rows=[];row=[];header=True;i+=1
        while tokens[i].type!='table_close':
            tt=tokens[i]
            if tt.type=='thead_close':header=False
            if tt.type=='tr_open':row=[]
            if tt.type=='inline':row.append(Paragraph(inline(tt),styles['cellhead' if header else 'cell']))
            if tt.type=='tr_close':rows.append(row)
            i+=1
        n=len(rows[0])
        weights=([.56,.22,.22] if n==3 else [.12,.13,.13,.28,.18,.16] if n==6 else [1/n]*n)
        table=LongTable(rows,colWidths=[CONTENT_WIDTH*x for x in weights],repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAECEA')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F8F8F7')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#A4AAA6')),
            ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#C9CCCA')),
        ]))
        story += [Spacer(1,5),KeepTogether([table]),Spacer(1,14)];i+=1;continue
    if t.type in ('bullet_list_open','ordered_list_open'):list_depth+=1
    elif t.type in ('bullet_list_close','ordered_list_close'):list_depth-=1
    elif t.type=='paragraph_open':
        ft=tokens[i+1];content=inline(ft)
        style='bib' if section=='Bibliography' else 'body'
        if section=='Conclusion + Closing':style='closing'
        if ft.content.startswith('[AUTHOR INPUT REQUIRED:'):style='note'
        if list_depth:style='bullet';content='&#8226; '+content
        story.append(Paragraph(content,styles[style]));i+=3;continue
    elif t.type in ('fence','code_block'):
        story.append(Paragraph(escape(t.content).replace('\n','<br/>'),styles['note']))
    i+=1

doc=ReviewDoc(str(OUTPUT),pagesize=A4,title='Motion Pixels - Thesis Review',
              author='Ramy',subject='Complete thesis manuscript for review',
              pageCompression=1,allowSplitting=1)
doc.multiBuild(story)

pdf=pymupdf.open(OUTPUT)
checks=[]
for idx,page in enumerate(pdf):
    page.get_pixmap(matrix=pymupdf.Matrix(1,1)).save(QA/f'page_{idx+1:03}.png')
    blocks=page.get_text('dict')['blocks']
    spans=[s for b in blocks if 'lines' in b for line in b['lines'] for s in line['spans']]
    outliers=[s['text'] for s in spans if s['bbox'][0]<0 or s['bbox'][2]>WIDTH+1 or s['bbox'][1]<0 or s['bbox'][3]>HEIGHT+1]
    checks.append({'page':idx+1,'words':len(page.get_text().split()),'out_of_page':outliers})
text='\n'.join(p.get_text() for p in pdf)
report={'file':str(OUTPUT),'pages':len(pdf),'bytes':OUTPUT.stat().st_size,
        'author_markers':text.count('AUTHOR INPUT REQUIRED'),
        'replacement_characters':text.count('\ufffd'),'checks':checks}
(QA/'qa_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
