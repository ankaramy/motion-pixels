# -*- coding: utf-8 -*-
"""Build MOTION PIXELS final publication: HTML/CSS -> A4 PDF via headless Chromium, then QA rasters."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def u(rel):
    p = (ROOT / rel).resolve()
    return 'file:///' + str(p).replace('\\', '/')

# ---- image optimiser (deferred): emits a token resolved at render time per PROFILE ----
# PROFILE 'print'  -> keep native resolution, near-lossless JPEG (no visible compression)
# PROFILE 'digital'-> downscale + moderate JPEG for a light file
from PIL import Image
_OPT = HERE / '_opt'; _OPT.mkdir(exist_ok=True)
_optcache = {}
PROFILE = 'digital'

def opt(rel, maxpx=2000, q=90):
    """Return a deferred token; the real file is produced at render time for the active PROFILE."""
    return "@@IMG|%s|%d|%d@@" % (rel, maxpx, q)

def _realopt(rel, maxpx, q):
    rp = Path(rel)
    src = rp.resolve() if rp.is_absolute() else (ROOT / rel).resolve()
    if PROFILE == 'print':
        eff_max = max(maxpx, 6000)   # effectively no downscale for our sources
        eff_q = 96
    else:
        eff_max = min(maxpx, 1700)
        eff_q = 84
    key = (str(src), PROFILE, eff_max, eff_q)
    if key in _optcache:
        return _optcache[key]
    im = Image.open(src)
    if im.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', im.size, (255, 255, 255))
        im = im.convert('RGBA'); bg.paste(im, mask=im.split()[-1]); im = bg
    else:
        im = im.convert('RGB')
    w, h = im.size
    if max(w, h) > eff_max:
        s = eff_max / max(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    stem = src.stem.replace(' ', '_')
    dst = _OPT / ('%s_%s_%d.jpg' % (stem, PROFILE, min(eff_max, max(w, h))))
    im.save(dst, 'JPEG', quality=eff_q, optimize=True, progressive=True)
    uri = 'file:///' + str(dst).replace('\\', '/')
    _optcache[key] = uri
    return uri

BK  = 'THESIS/figures/booklet'
DS  = 'THESIS/figures/booklet/dataset'
AST = 'publication_final/assets'
FONT = 'publication_rebrand/fonts'

# ---------------------------------------------------------------- page registry
PAGES = []
def add(body, chapter='', dark=False, folio=True, header=True, opener=False, fullbleed=False, cls='', vcenter=True):
    PAGES.append(dict(body=body, chapter=chapter, dark=dark, folio=folio,
                      header=header, opener=opener, fullbleed=fullbleed, cls=cls, vcenter=vcenter))

def void_page():
    add('', header=False, folio=False, cls='voidpage')

def ensure_left():
    # next page index should be even (left). cover is index 1.
    if (len(PAGES) + 1) % 2 != 0:
        void_page()

# ---------------------------------------------------------------- helpers
def img(src, cls='w100', style='', maxpx=2000, q=90):
    st = (' style=\'%s\'' % style) if style else ''
    return "<img class='%s' src='%s'%s>" % (cls, opt(src, maxpx, q), st)

def cap(num, text, style=''):
    st = (' style=\'%s\'' % style) if style else ''
    return "<div class='cap'%s><b>Fig %s</b>&nbsp; %s</div>" % (st, num, text)

def tlabel(text, style='', cls='tlabel'):
    st = (' style=\'%s\'' % style) if style else ''
    return "<div class='%s'%s>%s</div>" % (cls, st, text)

def opener(num, descriptor, title_html):
    # descriptor kept in signature but no longer rendered (author correction #6)
    return (
        "<div class='otick'><span class='tick'></span></div>"
        "<div class='cnum'>%s</div>"
        "<div class='ctitle'>%s</div>"
    ) % (num, title_html)

# ================================================================ CSS / HEAD
CSS = """
:root{
  --ink:#141416; --mag:#8E1C74; --magL:#C4238C; --grey:#6b6b6b; --grey2:#8c8c8c;
  --hair:#d0d0d0; --dark:#000000; --paper:#ffffff;
  --outer:17mm; --inner:21mm; --top:15mm; --bot:16mm;
}
@font-face{font-family:'Inter';src:url('%(inter)s');font-weight:100 900;font-style:normal;}
@font-face{font-family:'Serif';src:url('%(serif)s');font-weight:200 900;font-style:normal;}
@font-face{font-family:'Gugi';src:url('%(gugi)s');font-weight:400;}
@page{size:A4;margin:0;}
*{margin:0;padding:0;box-sizing:border-box;}
html,body{background:#8a8a8a;}
.page{position:relative;width:210mm;height:297mm;overflow:hidden;background:var(--paper);page-break-after:always;}
.page:last-child{page-break-after:auto;}
.frame{position:absolute;inset:0;padding:var(--top) var(--outer) var(--bot) var(--inner);}
.right .frame{padding-left:var(--inner);padding-right:var(--outer);}
.left  .frame{padding-left:var(--outer);padding-right:var(--inner);}
.fullbleed{position:absolute;inset:0;width:100%%;height:100%%;object-fit:cover;display:block;}

/* running header (no rule) + folio */
.rh{position:absolute;top:9mm;font-family:'Inter';font-size:7.5pt;letter-spacing:.14em;color:var(--grey);text-transform:uppercase;}
.rh-l{left:var(--outer);font-weight:700;color:var(--ink);}
.rh-r{right:var(--outer);text-align:right;font-weight:400;color:var(--grey);}
.folio{position:absolute;bottom:9mm;font-family:'Inter';font-size:8pt;color:var(--grey);}
.left .folio{left:var(--outer);} .right .folio{right:var(--outer);}

/* type */
.serif{font-family:'Serif';color:var(--ink);}
.body{font-family:'Serif';font-size:9.8pt;line-height:15.7pt;color:var(--ink);text-align:justify;hyphens:auto;-webkit-hyphens:auto;}
.body p{margin-bottom:7pt;} .body p:last-child{margin-bottom:0;}
.body.cols{columns:2;column-gap:9mm;}
.lead{font-family:'Serif';font-size:13pt;line-height:19pt;color:var(--ink);text-align:justify;hyphens:auto;margin-bottom:9pt;}
.pull{font-family:'Serif';font-size:19pt;line-height:26pt;color:var(--ink);}
.pull .mag{color:var(--mag);}
.secnum{font-family:'Inter';font-weight:600;font-size:11pt;color:var(--mag);letter-spacing:.02em;}
.sectitle{font-family:'Inter';font-weight:600;font-size:18pt;line-height:22pt;color:var(--ink);margin:2pt 0 11pt;}
.subhead{font-family:'Inter';font-weight:600;font-size:10.5pt;color:var(--ink);letter-spacing:.01em;margin:0 0 5pt;}
.cap{font-family:'Inter';font-size:7.8pt;line-height:11pt;color:var(--grey);}
.cap b{color:var(--mag);font-weight:700;}
.tlabel{font-family:'Inter';font-weight:500;font-size:7pt;letter-spacing:.16em;text-transform:uppercase;color:var(--grey);}
.tlabel.ink{color:var(--ink);} .tlabel.mag{color:var(--mag);}
.tick{width:6pt;height:6pt;background:var(--mag);display:inline-block;}
.mrule{display:none;}

/* chapter opener */
.opener .cnum{position:absolute;top:36mm;left:var(--outer);font-family:'Inter';font-weight:700;font-size:210pt;line-height:.85;color:var(--ink);letter-spacing:-.03em;}
.opener .ctitle{position:absolute;bottom:40mm;left:var(--outer);right:var(--inner);font-family:'Inter';font-weight:600;font-size:46pt;line-height:48pt;color:var(--ink);letter-spacing:-.01em;}
.opener .otick{position:absolute;top:29mm;left:var(--outer);}

/* images */
img.w100{display:block;width:100%%;height:auto;}
.imgcover{overflow:hidden;} .imgcover img{width:100%%;height:100%%;object-fit:cover;display:block;}
.center{margin-left:auto;margin-right:auto;}
.fh{height:78mm;display:flex;align-items:flex-start;justify-content:center;}
.fhb{display:flex;align-items:flex-end;justify-content:center;}
img.fit{max-width:100%%;max-height:100%%;width:auto;height:auto;object-fit:contain;display:block;}

/* grids */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:4mm;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:3.5mm;}
.cellcap{font-family:'Inter';font-weight:500;font-size:7pt;letter-spacing:.14em;text-transform:uppercase;color:var(--grey);margin-top:3pt;}

/* legend bar */
.legrow{display:flex;align-items:center;gap:6pt;}
.legrow span{font-family:'Inter';font-size:7pt;color:var(--grey);}
.legbar{flex:1;height:4pt;}

/* table */
.tbl{width:100%%;border-collapse:collapse;font-family:'Inter';font-size:8pt;color:var(--ink);}
.tbl th{font-weight:700;text-align:left;padding:5pt 6pt;border-bottom:.8pt solid var(--ink);font-size:7.4pt;letter-spacing:.06em;text-transform:uppercase;}
.tbl td{padding:4.6pt 6pt;border-bottom:.4pt solid var(--hair);}
.tbl td.n,.tbl th.n{text-align:right;font-variant-numeric:tabular-nums;}

/* dark */
.dark{background:var(--dark);}
.dark .body,.dark .lead,.dark .pull,.dark .sectitle,.dark .subhead{color:#efeeec;}
.dark .rh,.dark .folio{color:#8a8a8a;} .dark .rh-l{color:#e8e8e8;}
.dark .tlabel{color:#9a9a9a;} .dark .tlabel.mag{color:var(--magL);}
.dark .cap{color:#b7b7b7;} .dark .cap b{color:var(--magL);}
.dark .secnum{color:var(--magL);}
.voidpage{}
""" % {'inter': u(FONT+'/Inter.ttf'), 'serif': u(FONT+'/SourceSerif4.ttf'), 'gugi': u(FONT+'/Gugi.ttf')}

# external BOOKLET_images (author-updated covers live here; force reload by clearing _opt cache)
BKX = r'C:\Users\OWNER\Desktop\BOOKLET_images'

# ================================================================ FRONT MATTER
# P1 — front cover (standalone; reloaded from BOOKLET_images)
add("<img class='fullbleed' src='%s'>" % opt(BKX + r'\front_cover.png', 2200, 94),
    fullbleed=True, header=False, folio=False)

# P2 — title page (moved from p3)
add(
    "<div style='position:absolute;left:var(--inner);right:var(--outer);top:34mm;'>"
    "  <div style='font-family:Inter;font-weight:700;font-size:40pt;line-height:40pt;letter-spacing:-.02em;color:var(--ink);'>Motion<br>Pixels</div>"
    "  <div style='font-family:Inter;font-weight:500;font-size:12pt;letter-spacing:.02em;color:var(--mag);margin-top:7mm;'>Mapping Out Spatial Intelligence</div>"
    "  <div class='mrule' style='width:34mm;margin-top:8mm;'></div>"
    "</div>"
    "<div style='position:absolute;left:var(--inner);bottom:var(--bot);font-family:Inter;font-size:9pt;line-height:16pt;color:var(--ink);'>"
    "  <div style='font-weight:700;'>Ramy Anka</div>"
    "  <div style='color:var(--grey);'>Advisor &nbsp;Professor Wassim Jabi</div>"
    "  <div style='height:5mm;'></div>"
    "  <div style='color:var(--grey);'>Institute for Advanced Architecture of Catalonia</div>"
    "  <div style='color:var(--grey);'>MaAI, Master in AI for Architecture and the Built Environment</div>"
    "  <div style='color:var(--grey);'>Spatial Intelligence &middot; AI for Perception, Movement, Typology and Urban Performance</div>"
    "  <div style='height:5mm;'></div>"
    "  <div style='color:var(--grey);'>2025&ndash;2026 &nbsp;&middot;&nbsp; Barcelona, June 2026</div>"
    "</div>",
    header=False, folio=False)

# P3 — acknowledgments (moved from p8)
add(
    "<span class='secnum'>—</span>"
    "<div class='sectitle'>Acknowledgments</div>"
    "<div class='body' style='width:82%;margin-top:2mm;'>"
    "<p>This thesis was done over the course of a single year, and it has been an adventure filled with learning and self progression. Professor Wassim Jabi was my thesis advisor, and I learned a great deal from him.</p>"
    "<p>I am grateful to Angelos Chronis and Areti Markopoulou for initiating the programme and for offering me a full scholarship, and to Eleni Karafylli, the programme coordinator, for her constant support since the first year.</p>"
    "<p>I thank my parents, Milad and Marie, and my sister Zeina, for helping me finance my studies. And I thank Elias and Evangelo for their unwavering support and their moral and technical help during the thesis.</p>"
    "</div>",
    chapter='Acknowledgments')

# P4 — abstract I
add(
    "<span class='secnum'>00</span>"
    "<div class='sectitle'>Abstract</div>"
    "<div class='lead serif' style='width:92%;'>An architectural drawing describes how a space is meant to be used. The people who use it move in ways the drawing never records, and once a space is occupied that behaviour is usually lost. Motion Pixels asks whether it can be kept.</div>"
    "<div class='body' style='width:92%;'>"
    "<p>The thesis investigates whether pedestrian trajectories can be predicted from the relationship between human behaviour and architectural space, and whether the result can become a layer of evidence an architect can read alongside the plan.</p>"
    "<p>The method turns ordinary video into measured movement. Pedestrians are detected and tracked, the footage is calibrated to the architectural plan by homography so that movement is expressed in real metres, and each trajectory is encoded in terms of both its motion and its spatial situation, its distance to obstacles and boundaries. A recurrent model, a Long Short-Term Memory network chosen for its stability over long rollouts, then predicts movement forward from a short window of observed motion. The work was developed first on a single sandbox site, the esplanade in front of MACBA in Barcelona, and then on a dataset of five calibrated recordings across the city containing 3,534 tracked trajectories.</p>"
    "<p>An early failure shaped the research. The first models flattened the curved movement the sandbox was chosen to capture. A controlled capacity test showed that the architecture could represent angular movement once it had enough examples, which identified the problem as primarily a shortage of data rather than a flawed design. Prediction was then evaluated across several horizons, from roughly one to twenty metres of travel.</p>"
    "</div>")

# P5 — abstract II + keywords
add(
    "<div class='body' style='width:92%;'>"
    "<p>The results are bounded and consistent. Short-range prediction is reliable, with an average displacement error near half a metre at the shortest horizon and endpoint placement inside a one metre tolerance about eighty-three percent of the time. Accuracy falls as the horizon grows, and predictions tend to straighten and fall short of the real path. Later experiments that changed the training objective recovered a large part of the missing shape, which confirmed that the limits were as much about how the model learned as about how much data it had. The models were evaluated on unseen people within the same sites, not on entirely unseen spaces, and no claim of transfer beyond the studied sites is made.</p>"
    "<p>The contribution is less a model than a way of working. Prediction is used as a test of whether movement carries usable spatial structure, and it does, most clearly at the scale where an architect reasons about a threshold, an edge, or a crossing. Observed and predicted movement, together with behavioural maps of flow, speed and density, form an additional layer of architectural information that returns lived behaviour to the design process rather than leaving it in memory.</p>"
    "</div>"
    "<div style='position:absolute;left:var(--inner);right:var(--outer);bottom:var(--bot);'>"
    "  <div class='mrule' style='width:28mm;margin-bottom:5pt;'></div>"
    + tlabel("Keywords") +
    "  <div style='font-family:Serif;font-size:9.5pt;line-height:15pt;color:var(--ink);margin-top:3pt;'>pedestrian movement &middot; trajectory prediction &middot; behavioural mapping &middot; architectural analysis &middot; computer vision</div>"
    "</div>",
    chapter='Abstract')

# P6 — contents
def toc_row(n, t, secs):
    return (
        "<div style='display:flex;align-items:baseline;gap:6mm;margin-bottom:7mm;'>"
        "  <div style='font-family:Inter;font-weight:700;font-size:20pt;color:var(--mag);width:12mm;'>%s</div>"
        "  <div>"
        "    <div style='font-family:Inter;font-weight:600;font-size:13pt;color:var(--ink);'>%s</div>"
        "    <div style='font-family:Inter;font-size:8pt;letter-spacing:.04em;color:var(--grey);margin-top:2pt;'>%s</div>"
        "  </div>"
        "</div>") % (n, t, secs)
add(
    "<span class='secnum'>—</span>"
    "<div class='sectitle'>Contents</div>"
    "<div style='margin-top:6mm;width:96%;'>"
    + toc_row('1', 'The Flaw in the Plan', 'Introduction &middot; Early references &middot; Research question &middot; Hypothesis')
    + toc_row('2', 'Tools and Instruments', 'Space Syntax &middot; Literature review &middot; Practical review &middot; Gaps and positioning &middot; Computational pipeline &middot; The ethical question')
    + toc_row('3', 'Learning Movement from Barcelona', 'The MACBA sandbox &middot; Building the dataset &middot; Behavioural maps &middot; Horizon rollouts &middot; The prototype')
    + toc_row('4', 'What Comes Next', 'What the model learned &middot; Limitations &middot; The meaning of it all &middot; Future directions')
    + toc_row('5', 'Conclusion', 'Movement as spatial information &middot; from observation to design')
    + "<div style='display:flex;align-items:baseline;gap:6mm;margin-top:2mm;'>"
      "<div style='font-family:Inter;font-weight:700;font-size:13pt;color:var(--mag);width:12mm;'>&lowast;</div>"
      "<div style='font-family:Inter;font-weight:600;font-size:12pt;color:var(--ink);'>List of Figures &nbsp;+&nbsp; Bibliography</div></div>"
    "</div>",
    chapter='Contents')

# P7 — preface + AI declaration
add(
    "<span class='secnum'>—</span>"
    "<div class='sectitle'>Preface</div>"
    "<div class='body' style='width:88%;'>"
    "<p>This book follows the research in the order it happened. The first chapter sets out the problem, the gap between how a space is designed and how it is used. The second reviews the work Motion Pixels builds on and describes the pipeline that turns video into data. The third is the longest, and it moves from a single test site to a dataset across Barcelona, through the experiments, the maps, the horizon predictions and the prototype. The fourth interprets the results and looks ahead. A reader who wants the argument without the machinery can read the first and last chapters and the conclusion. A reader who wants the evidence will find it in the third.</p>"
    "<p>The intention throughout has been to keep claims proportionate to what the work actually shows. Where a result is strong it is stated plainly, and where it is weak or unfinished it is marked as such rather than smoothed over.</p>"
    "</div>"
    "<div style='width:82%;margin-top:9mm;'>"
    + tlabel("Declaration of AI Use") +
    "  <div class='body' style='margin-top:4pt;'><p>Agentic coding tools were used in the creation of the code for the computational pipeline. Large language models were used as spell-checking tools for the writing of this thesis and in the creation of the PDF for this booklet.</p></div>"
    "</div>",
    chapter='Preface')

# ================================================================ CHAPTER 1
CH1 = 'The Flaw in the Plan'
ensure_left()
add(opener('1', 'Observation', 'The Flaw<br>in the Plan'), opener=True, header=False, folio=False)

# P11 — §1.1 introduction
add(
    "<div style='display:grid;grid-template-columns:14mm 1fr;gap:4mm;'>"
    "  <div><span class='secnum'>1.1</span></div>"
    "  <div style='max-width:150mm;'>"
    "    <div class='sectitle' style='margin-top:-2pt;'>Introduction</div>"
    "    <div class='lead serif'>An architectural drawing is a description of intention. It says where a wall should stand, how wide a passage should be, where a person is expected to enter and where they are meant to arrive. The drawing is confident about all of this. What it cannot describe is what people actually do once the building or the plaza is finished and occupied.</div>"
    "    <div class='body' style='width:88%;'><p>I keep returning to a simple observation. No matter how carefully we design for a client, and no matter how well we think we understand the people who will use a space, one instinct always survives the drawing. A person will take the path they want, not the path we drew for them. The designed route and the desired route are rarely the same line.</p></div>"
    "  </div>"
    "</div>",
    chapter=CH1)

# P12 — body cont + supporting photo
add(
    "<div class='body' style='width:74%;'>"
    "<p>This is not a failure of design. It is the normal condition of architecture. A plan proposes an order, and the people who move through it answer with their own. The relationship between the two is a kind of negotiation that goes on quietly, every day, in every public space. A doorway invites, a corner slows people down, a shortcut appears across a lawn that was never meant to be crossed. The building sets terms and the crowd renegotiates them.</p>"
    "<p>The evidence of this negotiation is everywhere, but it is rarely collected. A worn strip of grass records a shortcut more honestly than any survey. A cluster of people always forming at one end of a plaza says something the plan did not predict. These are readings of a space that only exist once it is in use, and they tend to disappear from the architectural record precisely because they arrive after the drawing is finished. The plan is archived. The behaviour is forgotten.</p>"
    "</div>"
    "<div class='imgcover' style='position:absolute;left:var(--outer);bottom:var(--bot);width:58mm;height:42mm;'>" + img(BK+'/beginning_1.jpg') + "</div>"
    "<div class='cap' style='position:absolute;left:calc(var(--outer) + 62mm);bottom:calc(var(--bot) + 3mm);width:44mm;'>A paved route, and beside it the line people actually walk.</div>",
    chapter=CH1)

# P13 — desire lines figure (flow layout)
add(
    img(BK+'/beginning_4.jpg') +
    "<div class='grid2' style='margin-top:4mm;align-items:start;'>"
    "  <div>" + img(BK+'/beginning_3.jpg') + "</div>"
    "  <div style='align-self:end;'>"
    + cap('1.1', 'Desire lines worn into planted ground, where people ignore the paved routes laid out for them and cut the path they actually want. The designed path and the desired path rarely coincide, and the gap between them is what this thesis tries to read. Stock photographs.') +
    "    <div style='margin-top:8mm;'>" + tlabel('Designed intention &nbsp;/&nbsp; lived movement') + "</div>"
    "  </div>"
    "</div>",
    chapter=CH1)

# P14 — reading (two-col)
add(
    "<div class='body cols' style='width:100%;'>"
    "<p>Architects have always known this relationship exists. The problem is that we have mostly known it as intuition. We talk about how a space wants to be used, about desire lines, about places that feel alive and places that feel dead. The vocabulary is rich and the observation is real, but it stays largely qualitative. The link between a spatial configuration and the behaviour it produces has been described and theorised for decades. It has been harder to hold it as evidence, in numbers, tied to a specific place.</p>"
    "<p>There have been serious attempts to close that gap, and I return to them in the next chapter. What most of them share is a starting point in the space itself, in the geometry of the plan, rather than in the movement of the people. My interest runs the other way. I wanted to begin with the behaviour, with the actual traces people leave as they cross a real site, and then ask what that behaviour reveals about the space.</p>"
    "<p>Motion Pixels grew out of that need. The idea is direct. Ordinary video of a public space already contains a detailed record of how people move through it. If that record can be pulled out of the footage, placed back onto the architectural plan in real measurements, and described in terms an architect recognises, then movement stops being an anecdote. It becomes a layer of spatial information that sits next to the plan rather than vanishing once the building is occupied.</p>"
    "<p>Video is the right medium for this because it is ordinary. Most public space is already filmed, and a single camera looking at a plaza holds more behavioural detail than a team of observers could record by hand. What has been missing is not the footage but a reliable way to convert it into spatial measurement. Progress in detection and tracking has made that conversion possible at a scale that was not practical when earlier researchers were counting people by eye.</p>"
    "</div>",
    chapter=CH1)

# P15 — early references + Whyte image
add(
    "<span class='secnum'>1.2</span>"
    "<div class='sectitle'>Early References</div>"
    "<div class='body' style='width:96%;'>"
    "<p>Two bodies of work stand behind this starting point, and they approach it from opposite ends. The first is Space Syntax, developed by Bill Hillier and his colleagues at University College London. By describing a plan as a network of connected spaces and measuring properties such as how integrated or how segregated each part is, Space Syntax showed that certain configurational measures correlate with observed patterns of movement and co-presence (Hillier and Hanson, 1984; Hillier, 1996).</p>"
    "<p>The second reference is William H. Whyte. Where Space Syntax reasons from the plan, Whyte reasoned from the pavement. In The Social Life of Small Urban Spaces he and his team filmed New York plazas over long periods and simply watched what people did (Whyte, 1980). The most popular spaces were not the grand ones but the ones that offered small, ordinary comforts at a human scale.</p>"
    "</div>"
    "<div style='position:absolute;left:var(--inner);right:var(--outer);bottom:calc(var(--bot) + 14mm);'>" + img(BK+'/whyte.jpg') + "</div>"
    "<div class='cap' style='position:absolute;left:var(--inner);right:var(--outer);bottom:var(--bot);'><b>Fig 1.2</b>&nbsp; William H. Whyte filming and observing New York plazas for The Social Life of Small Urban Spaces. He treated patient observation of real behaviour as architectural evidence, and used film to capture it over time. Images courtesy of the Project for Public Spaces.</div>",
    chapter=CH1)

# P16 — Whyte method / between references
add(
    "<div class='body' style='width:78%;'>"
    "<p>What I take from Whyte is method as much as message. He treated observation as a legitimate form of architectural evidence, and he used film to do it, because film captures behaviour over time in a way a survey cannot. Motion Pixels is an attempt to give that patient watching a contemporary set of instruments. The camera is still doing the observing. What has changed is that the movement in the footage can now be extracted, measured, and placed back onto the plan automatically, at a scale of thousands of trajectories rather than a clipboard of counts.</p>"
    "<p>Between these two references sits the position this thesis takes. Space Syntax reads behaviour from the space. Whyte reads the space from behaviour. Motion Pixels leans toward Whyte&rsquo;s direction, starts from the observed movement, but tries to bring the result back into the measured, plan based world that Space Syntax works in.</p>"
    "</div>"
    "<div style='position:absolute;left:var(--outer);bottom:var(--bot);'>" + tlabel('Space reads behaviour &nbsp;/&nbsp; behaviour reads space') + "</div>",
    chapter=CH1)

# P17 — research question + hypothesis lead
add(
    "<span class='secnum'>1.3</span>"
    "<div class='sectitle'>Research Question</div>"
    "<div class='body' style='width:88%;margin-bottom:9mm;'>"
    "<p>If configuration and behaviour are genuinely coupled, then that coupling should leave a trace, and a trace can in principle be learned. When the coupling is weak, or when a design ignores it, the symptoms are familiar to anyone who has watched a space fail. People hesitating at a junction that made sense on paper. Crowds thickening into a bottleneck where two flows were never meant to meet.</p>"
    "</div>"
    "<div class='pull serif' style='width:94%;'>Can pedestrian trajectories be <span class='mag'>predicted</span> from the relationship between human behaviour and architectural and urban space?</div>"
    "<div class='body' style='width:88%;margin-top:9mm;'>"
    "<p>The word predicted is deliberate, and it needs a qualification. Prediction here is a way of testing whether the coupling carries real, usable information, not a promise that future movement can be known. If a model can look at a short stretch of someone&rsquo;s motion in a particular place and anticipate where they go next, then the relationship between behaviour and space contains structure a machine can pick up. If it cannot, that is also a finding. The question is a probe into how much of movement is legible, and at what range.</p>"
    "</div>",
    chapter=CH1)

# P18 — DARK research question diagram
add(
    "<div style='position:absolute;left:var(--outer);top:var(--top);'>" + tlabel('Observation &nbsp;/&nbsp; the coupling', cls='tlabel mag') + "</div>"
    "<div style='position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);padding:0 12mm;'>" + img(AST+'/research_question_crop.png') + "</div>"
    "<div class='cap' style='position:absolute;left:var(--outer);right:var(--inner);bottom:12mm;'><b>Fig 1.3</b>&nbsp; The research question, framed as a test of whether the link between architectural and urban characteristics and spatial user behaviour carries enough structure to anticipate movement, and where its absence surfaces as misaligned cues, wayfinding failures and bottlenecks.</div>",
    chapter=CH1, dark=True, header=False, folio=True)

# P19 — DARK hypothesis diagram + text
add(
    "<div style='position:absolute;left:var(--inner);top:var(--top);'>" + tlabel('Hypothesis', cls='tlabel mag') + "</div>"
    "<div class='body' style='position:absolute;left:var(--inner);right:var(--outer);top:26mm;width:82%;'>"
    "<p>Space shapes how people behave in it. If that behaviour can be captured, it becomes data. And if the data holds enough of the underlying structure, some of that behaviour can be anticipated rather than only recorded.</p>"
    "</div>"
    "<div style='position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);padding:0 10mm;'>" + img(AST+'/hypothesis_crop.png') + "</div>"
    "<div class='cap' style='position:absolute;left:var(--inner);right:var(--outer);bottom:12mm;'><b>Fig 1.4</b>&nbsp; The hypothesis as a chain of claims the research must earn in turn, from capturing movement to anticipating it: space, behaviour, capture, data, prediction.</div>",
    chapter=CH1, dark=True, header=False, folio=True)

# ================================================================ CHAPTER 2
CH2 = 'Tools and Instruments'
ensure_left()
add(opener('2', 'Instrument', 'Tools and<br>Instruments'), opener=True, header=False, folio=False)

# P21 — chapter intro + Space Syntax lead
add(
    "<div class='body' style='width:88%;'>"
    "<p>The previous chapter argued that observed movement can be treated as architectural evidence, and it framed that argument as a question about prediction. Before any of that can be tested, the research has to stand on existing work. Other people have studied how space shapes movement, how movement can be modelled, and how software already tries to simulate crowds. This chapter sets Motion Pixels against that background, and then describes the pipeline that turns a video into data. It ends with the question that any project handling footage of real people has to answer.</p>"
    "</div>"
    "<div style='margin-top:12mm;'>"
    "<span class='secnum'>2.1</span>"
    "<div class='sectitle'>Space Syntax</div>"
    "<div class='lead serif' style='width:86%;'>Space Syntax is the clearest existing attempt to make the relationship between configuration and behaviour measurable. It deserves a closer look.</div>"
    "</div>",
    chapter=CH2)

# P22 — Space Syntax body (two-col)
add(
    "<div class='body cols'>"
    "<p>The core move in Hillier&rsquo;s work is to stop treating a plan as a picture and start treating it as a network (Hillier and Hanson, 1984). A space is broken into its component parts, the lines of sight or movement that connect them are recorded, and the resulting graph is analysed. From that graph come measures such as integration, which describes how easily one part of the system can be reached from all the others.</p>"
    "<p>The finding that gave the method its weight is that these measures correlate with real patterns of use. More integrated streets tend to carry more movement, and they tend to do so whether or not there is an obvious destination on them. Hillier called this natural movement and treated it as evidence that the grid itself, not only its attractors, organises where people go (Hillier, 1996).</p>"
    "<p>The analysis is not limited to axial lines. Related techniques describe what can be seen from a given point, the isovist, and build visibility graphs that measure how much of a space is exposed to each location. These give a reading of how open or enclosed a position feels, which is close to the kind of spatial context this project later encodes for each pedestrian. The vocabulary is different, but the instinct is shared.</p>"
    "<p>This is where Motion Pixels takes a different position rather than a better one. Space Syntax starts from the plan and derives likely movement. This project starts from the observed movement and works back toward the space. The two are complementary. One gives a configurational expectation, the other gives a behavioural record, and the interesting ground is where they can be compared.</p>"
    "</div>",
    chapter=CH2)

# P23 — Space Syntax image (Fig 2.1)
add(
    "<div style='width:112mm;' class='center'>" + img(BK+'/space_syntax.jpg') +
    cap('2.1', 'A Space Syntax reading of an urban grid, where the configuration of the network is used to estimate where movement should concentrate. It reasons from the drawing toward behaviour, the opposite direction to the observed record Motion Pixels builds. After B. Hillier, Space is the Machine.', 'margin-top:5pt;') +
    "</div>",
    chapter=CH2)

# P24 — Literature review body (two-col)
add(
    "<span class='secnum'>2.2</span>"
    "<div class='sectitle'>Literature Review</div>"
    "<div class='body cols'>"
    "<p>The second body of work sits in computer vision and machine learning, where pedestrian trajectory prediction has been an active problem for years. The models matter to this thesis less as engineering and more as a map of what has already been tried, and on what kind of data.</p>"
    "<p>Early learned approaches treated a trajectory as a sequence and used recurrent networks to continue it. A recurrent model reads a person&rsquo;s recent positions one step at a time, keeps an internal memory of the motion so far, and uses it to predict the next step. The idea that reshaped the field was to let people influence one another. Alahi and colleagues introduced Social-LSTM, which gave each pedestrian their own recurrent network and then pooled the hidden states of nearby people into a shared social tensor, so that a prediction accounts for the neighbours crowding a person&rsquo;s path and not only their own history (Alahi et al., 2016).</p>"
    "<p>The weakness of a single predicted line is that people rarely have one available future. At a junction a person might go left or right with almost equal reason, and a model that averages those options produces a path down the middle that no one would actually take. Generative models were introduced to represent that spread. Gupta and colleagues built Social GAN, which pairs a recurrent generator with a discriminator and samples several socially plausible futures instead of committing to one (Gupta et al., 2018). Sadeghian and colleagues extended this with SoPhie, which adds attention over the physical scene and over the surrounding agents (Sadeghian et al., 2019).</p>"
    "</div>",
    chapter=CH2)

# P25 — Literature review cont (two-col)
add(
    "<div class='body cols'>"
    "<p>Attention then became the organising idea of the field. Giuliari and colleagues showed that a plain transformer, with no social pooling at all, could match or beat the more elaborate recurrent models on the standard benchmarks simply by attending over a person&rsquo;s own past positions (Giuliari et al., 2020). Later transformer work put the interaction back in a more principled way. Yuan and colleagues built AgentFormer, which attends jointly over time and over agents (Yuan et al., 2021).</p>"
    "<p>Not every useful idea is recurrent or attention based. Bai and colleagues argued that temporal convolutional networks often match or exceed recurrent models on sequence tasks while being easier to train and more stable over long outputs (Bai et al., 2018). That property matters here, because stability over a long rolled-out prediction, rather than accuracy on a single next step, is exactly what this project cares about.</p>"
    "<p>Two things stand out when this body of work is read together. The first is the data. Almost all of it is trained and evaluated on the same small set of public benchmarks, chiefly the ETH and UCY pedestrian videos and the Stanford Drone Dataset. They are also not architectural. The scene, when it is used at all, enters as a background image rather than as a calibrated drawing.</p>"
    "<p>The second is the question being asked. Almost none of this work is framed the way an architect would frame it. The goal is a lower error on the shared benchmark, and the space is treated as context for the people rather than as the object of study. Motion Pixels borrows this machinery, and settles on a recurrent model in the end, but it points the machinery at a different target. The people are the instrument. The space is what the research is trying to read.</p>"
    "</div>",
    chapter=CH2)

# P26 — Practical review body (two-col)
add(
    "<span class='secnum'>2.3</span>"
    "<div class='sectitle'>Practical Review</div>"
    "<div class='body cols'>"
    "<p>Alongside the research literature there is a mature software industry aimed at the same broad problem, and it already does parts of this well. Bentley&rsquo;s tools, including OpenPaths and the LEGION product line, are used to model crowd movement in complex environments such as stations, stadiums and airports. Autodesk&rsquo;s InfraWorks sits at a larger scale, modelling transport networks and urban context so that planners can evaluate mobility and circulation across a site or district (Bentley Systems, n.d.; Autodesk, n.d.).</p>"
    "<p>Most of these tools rely on some form of agent based simulation, often built on social force ideas, where each simulated pedestrian is pushed and pulled by goals, obstacles and other agents. The behaviour that emerges can be calibrated against observed counts and tuned until the flow looks realistic. This works well for the questions the tools are built for, such as how a concourse clears in an evacuation, or whether a stadium exit meets a capacity standard.</p>"
    "<p>These platforms are powerful, and Motion Pixels is not trying to replace them. The distinction is in what feeds them. Agent based simulation generates movement from assumed rules. It does not begin from how people actually moved through one real space, and it is not designed to. That gap, between simulated behaviour and observed behaviour, is part of what this project is trying to occupy. Motion Pixels does not ask what a crowd would do under a rule set. It asks what a real crowd did, and whether that record predicts itself.</p>"
    "</div>",
    chapter=CH2)

# P27 — commercial tools (Fig 2.2)
add(
    "<div style='margin-top:2mm;'>" + img(BK+'/bentley_openpaths_practicalreview.png') + "</div>"
    "<div style='margin-top:5mm;'>" + img(BK+'/infraworks_practicalreview.png') + "</div>"
    + cap('2.2', 'Two commercial approaches to the same problem. Bentley&rsquo;s OpenPaths and LEGION (top) model crowd movement through agent based simulation; Autodesk InfraWorks (bottom) models mobility across an urban network. Both generate movement from assumed rules rather than from a record of observed behaviour. Screenshots courtesy of Bentley Systems and Autodesk.', 'margin-top:6pt;width:92%;'),
    chapter=CH2)

# P28 — gaps & positioning body (two-col)
add(
    "<span class='secnum'>2.4</span>"
    "<div class='sectitle'>The Gaps and Positioning</div>"
    "<div class='body cols'>"
    "<p>Read together, these three bodies of work leave a clear opening. Space Syntax explains movement. It provides a configurational account of why some parts of a layout carry more life than others, and it does so from the plan. What it does not provide is a direct record of what people actually did once the space was built and occupied.</p>"
    "<p>The prediction literature measures movement. It provides models that can continue a trajectory and account for the people around it. What it tends to lack is any grip on a specific architectural setting. The models are trained on a handful of generic scenes, and the output is judged by an error figure rather than by what it reveals about a place.</p>"
    "<p>The simulation software simulates movement. It provides planning, crowd modelling and decision support at the scale of a real project. What it lacks is a foundation in observed behaviour. Its pedestrians move according to assumed rules, so the answer it gives is always conditional on those rules being right.</p>"
    "<p>Motion Pixels sits in the space these three leave between them. It starts from observed movement in a real, calibrated site, describes that movement in both behavioural and spatial terms, and then tests whether the description carries enough structure to predict. The aim is not to win a benchmark or to out simulate a crowd engine. It is to turn a specific space into evidence, and then to ask what that evidence can anticipate, and for how far.</p>"
    "</div>",
    chapter=CH2)

# P29 — positioning diagram (Fig 2.3)
add(
    tlabel('Positioning', cls='tlabel mag') +
    "<div style='margin-top:6mm;'>" + img(AST+'/gaps_tight.png') + "</div>"
    + cap('2.3', 'Space Syntax explains, the prediction literature measures, simulation software simulates. Motion Pixels occupies the predictive gap between them, grounded in observed movement.', 'margin-top:9pt;width:80%;'),
    chapter=CH2)

# P30 — computational pipeline body (two-col)
add(
    "<span class='secnum'>2.5</span>"
    "<div class='sectitle'>Computational Pipeline</div>"
    "<div class='lead serif' style='width:86%;'>The rest of the thesis depends on one practical thing. Ordinary video has to become measured movement on an architectural plan.</div>"
    "<div class='body cols'>"
    "<p>The pipeline that does this is a sequence of steps, each of which was chosen because it earns its place in the argument. It begins with detection and tracking. Each frame of a recording is passed through an object detector, YOLOv8, which finds the people in it. A tracker, ByteTrack, then links those detections across frames so that each person keeps a stable identity for as long as they stay in view. Several recordings were filmed sideways, so each frame is rotated upright before detection and run at high resolution with a low confidence threshold and a high recall tracking setting.</p>"
    "<p>Detection gives movement in image pixels. Architecture needs it in metres, on the plan. The second step is calibration by homography. For each recording, points that can be identified both in the footage and on the plan are matched by hand, and from those correspondences a transformation maps any point in the image to a position on the plan. After calibration, a trajectory that was a line of pixels becomes a path in real coordinates.</p>"
    "<p>With movement placed in space, each trajectory is described in two registers at once. The first is behavioural: speed, direction, the distances covered, local density, stops and dwell time. The second is spatial: using a manually prepared mask of each site, the encoding records where a person is relative to the walkable area, how close they are to obstacles and boundaries, and their affinity to entrances. That description is what the model learns from.</p>"
    "<p>The model is a recurrent network, an LSTM, chosen after a comparison described in the next chapter. It reads a short window of a person&rsquo;s recent movement and predicts their next displacement, and by repeating that step it rolls a trajectory forward. It is asked to look different distances into the future, the horizons, written as H20, H60, H100, H200 and H400.</p>"
    "</div>",
    chapter=CH2)

# P31 — pipeline diagram PRIMARY (Fig 2.4)
def stage(n, t, d):
    return ("<div><div style='font-family:Inter;font-weight:700;font-size:8pt;color:var(--mag);'>%s</div>"
            "<div style='font-family:Inter;font-weight:600;font-size:8pt;color:var(--ink);margin-top:2pt;'>%s</div>"
            "<div style='font-family:Inter;font-size:7pt;line-height:9.5pt;color:var(--grey);margin-top:1pt;'>%s</div></div>") % (n, t, d)
add(
    tlabel('The Instrument', cls='tlabel mag') +
    "<div class='body serif' style='width:78%;font-size:9.6pt;line-height:14.8pt;margin-top:3mm;'>Read left to right, the pipeline turns a video and a plan into a dataset the model can learn from, then into predictions and the maps that make them legible. Each stage earns its place in the argument.</div>"
    "<div style='margin-top:8mm;'>" + img(AST+'/pipeline_crop.png') + "</div>"
    "<div class='mrule' style='width:100%;margin-top:9mm;margin-bottom:6pt;'></div>"
    "<div style='display:grid;grid-template-columns:repeat(6,1fr);gap:4mm;'>"
    + stage('01', 'Detection &amp; tracking', 'YOLOv8 and ByteTrack')
    + stage('02', 'Homography', 'Pixels to metres, on the plan')
    + stage('03', 'Spatial encoding', 'The ten-feature schema')
    + stage('04', 'LSTM', 'Autoregressive rollout')
    + stage('05', 'Horizons', 'H20 to H400, 1 to 20 m')
    + stage('06', 'Maps &amp; plots', 'Legible to a designer')
    + "</div>"
    + cap('2.4', 'The full pipeline, from video and plan through detection, tracking, homography and spatial encoding to the LSTM, its horizon predictions, and the plots and behavioural maps that make the result legible to a designer.', 'margin-top:8mm;width:92%;'),
    chapter=CH2)

# P32 — the ethical question
add(
    "<span class='secnum'>2.6</span>"
    "<div class='sectitle'>The Ethical Question</div>"
    "<div class='body' style='width:86%;'>"
    "<p>None of this is possible without recording people in public space, and that raises a real question rather than a rhetorical one. Is it acceptable to capture and store footage of pedestrians for academic research?</p>"
    "<p>The relevant frame is the General Data Protection Regulation. Article 89(1) allows personal data to be processed for scientific research purposes when appropriate safeguards are in place to protect the rights of individuals, including measures such as data minimisation (European Union, 2016). The design of Motion Pixels fits the spirit of that provision. The system does not identify anyone. It extracts anonymous trajectories and behavioural patterns, and it works at the level of movement across a space rather than the level of a recognisable person.</p>"
    "<p>Part of the argument is technical rather than legal. The pipeline is built so that identity is discarded early. Detection and tracking assign each person a temporary number that lasts only as long as they are in frame, and the data that survives downstream is a set of coordinates and derived quantities, not images of faces. This is data minimisation in practice. What the research keeps is the movement, not the person who made it.</p>"
    "</div>",
    chapter=CH2)

# P33 — ethical cont + transition (quiet)
add(
    "<div class='body' style='width:80%;'>"
    "<p>This is easy to overstate, and the source material for this project puts the point more categorically than I would. Article 89 is a safeguards and derogations provision. It sets conditions under which research processing is permitted, but it does not by itself grant blanket permission to record identifiable people. Lawful collection also depends on the controller, the legal basis, and the concrete arrangements around retention, access and anonymisation. Those operational matters were addressed for the recordings used in this work, in keeping with the research safeguards the regulation asks for.</p>"
    "</div>"
    "<div style='position:absolute;left:var(--outer);right:40%;bottom:var(--bot);'>"
    "  <div class='mrule' style='width:24mm;margin-bottom:6pt;'></div>"
    "  <div class='serif' style='font-size:11pt;line-height:16pt;color:var(--grey);'>A method can be coherent on paper and fall apart on the first crowded frame. The next chapter puts the tools to work.</div>"
    "</div>",
    chapter=CH2)

# ================================================================ CHAPTER 3
CH3 = 'Learning Movement from Barcelona'
ensure_left()
add(opener('3', 'Evidence', 'Learning Movement<br>from Barcelona'), opener=True, header=False, folio=False)

# P35 — chapter intro
add(
    "<div class='lead serif' style='width:90%;'>The previous chapter described the pipeline in principle. This one describes what happened when it was pointed at real footage of real streets.</div>"
    "<div class='body' style='width:86%;'>"
    "<p>The work did not arrive fully formed. It began with a single site used as a testing ground, ran into a clear failure, and only then grew into a dataset spread across Barcelona. I have kept that order here, because the mistakes are part of the argument. They are what turned a rough idea into a method.</p>"
    "</div>",
    chapter=CH3)

# P36 — MACBA the sandbox body
add(
    "<span class='secnum'>3.1</span>"
    "<div class='sectitle'>MACBA, the Sandbox</div>"
    "<div class='body' style='width:86%;'>"
    "<p>The first site was the esplanade in front of the Museu d&rsquo;Art Contemporani de Barcelona, MACBA. Anyone who has been there knows it less as a museum forecourt than as one of the best known skate spots in Europe. The wide, smooth plaza fills through the afternoon with skateboarders, and around them a normal crowd of pedestrians crosses, gathers and watches.</p>"
    "<p>I chose it on purpose. A pipeline meant to read movement needs movement worth reading, and skateboarding produces exactly the range that ordinary walking does not. Fast runs and slow rolls, sharp turns, sudden stops, curved lines that double back on themselves. If the tools could handle the variety of speeds and angles on that plaza, they could probably handle a calmer square. So one video shot from the museum toward the esplanade became the sandbox, the ground on which every part of the pipeline was tested for the first time.</p>"
    "</div>",
    chapter=CH3)

# P37 — MACBA context (Fig 3.1)
add(
    img(BK+'/MACBA_2011.jpg') +
    "<div class='grid2' style='margin-top:4mm;align-items:start;'>"
    "  <div>" + img(BK+'/source_bcncolours_macba011.jpg') + "</div>"
    "  <div style='align-self:end;'>" +
    cap('3.1', 'The MACBA esplanade in the Raval. The plaza was never designed as a skate park, but its smooth ground and low ledges made it one, the negotiation between design and use this thesis is about. Photographs courtesy of MACBA, Museu d&rsquo;Art Contemporani de Barcelona.') +
    "  </div>"
    "</div>",
    chapter=CH3)

# P38 — sandbox rationale + face blur
add(
    "<div class='body' style='width:80%;'>"
    "<p>MACBA sits in the Raval, and its plaza was never designed as a skate park. It became one. That fact is the thesis in miniature, a designed space repurposed by the behaviour it happened to afford, its smooth ground and low ledges reading as an invitation the architects never wrote. Filming there meant starting the research on exactly the kind of gap between intention and use that motivated it in the first place.</p>"
    "<p>Calling it a sandbox is deliberate. Nothing about this stage was meant to prove that the method generalised. It was meant to prove that the method ran at all, end to end, from raw footage to a trajectory placed on the plan and a prediction rolled forward from it. Everything that follows in this chapter is a response to what that first site showed.</p>"
    "</div>"
    "<div style='position:absolute;left:var(--outer);right:38%;bottom:var(--bot);'>"
    "  <div class='mrule' style='width:22mm;margin-bottom:6pt;'></div>"
    "  <div class='body' style='font-style:normal;'><p>As each frame is processed, detected faces are blurred before anything is stored, so the sandbox, and every recording after it, worked only with anonymous bodies and their movement, never with identifiable people.</p></div>"
    "</div>",
    chapter=CH3)

# P39 — sandbox tracking (Fig 3.2)
add(
    "<div class='grid2' style='align-items:start;'>"
    "  <div>" + img(BK+'/sandbox_experiment/skate_1_tracking_still_frame_01002.png') + "</div>"
    "  <div>" + img(BK+'/sandbox_experiment/skate_2_tracking_still_frame_00911.png') + "</div>"
    "</div>"
    + cap('3.2', 'Tracked pedestrians and skateboarders on the MACBA esplanade, the sandbox recording. The mix of fast rolls, sharp turns and ordinary walking gave the range of speeds and angles the pipeline was built to handle. Detected faces are blurred at capture.', 'margin-top:5pt;'),
    chapter=CH3)

# P40 — behavioural metrics (two-col)
add(
    "<span class='secnum'>3.2</span>"
    "<div class='sectitle'>Behavioural Metrics</div>"
    "<div class='body cols'>"
    "<p>Before anything could be predicted, the movement had to be described. Each test run over the sandbox produced a set of behavioural measures, and these are the quantities that make a trajectory legible as behaviour rather than as a line.</p>"
    "<p>The simplest are about presence and pace. How many people are active at once. How many are paused, standing still for a moment. How many are dwelling, staying in one area long enough to count as occupying it rather than passing through. Dwelling and lingering matter to an architect because they mark the places a space invites people to stop, which is often where a design succeeds or fails.</p>"
    "<p>Then there is direction. A trajectory is not only where a person went but how their heading changed along the way. Counting direction shifts gives a sense of how much a path bends, which separates a straight crossing from a wandering one. Lingering zones, the areas where dwelling concentrates, come out of the same data seen from above.</p>"
    "<p>Speed is described in three ways: the average speed of a person across their path, their maximum speed, and the number of stops they make. From these the movement is sorted into slow, medium and fast, with the boundaries set at 0.8 and 1.6 metres per second. None of these measures is exotic. What changes is that here they are computed for every tracked person at once and tied to a position on the plan, so the noticing becomes a record that can be compared between sites and across times of day.</p>"
    "</div>",
    chapter=CH3)

# P41 — early behaviour maps (Fig 3.3)
add(
    "<div class='grid3' style='align-items:start;'>"
    "  <div><div class='fh'>" + img(BK+'/sandbox_experiment/skate_1_bottleneck_heatmap_original.png', 'fit') + "</div><div class='cellcap'>Bottleneck heatmap</div></div>"
    "  <div><div class='fh'>" + img(BK+'/sandbox_experiment/skate_1_flow_field_quiver_original.png', 'fit') + "</div><div class='cellcap'>Flow field</div></div>"
    "  <div><div class='fh'>" + img(BK+'/sandbox_experiment/skate_1_linger_zones_plot_original.png', 'fit') + "</div><div class='cellcap'>Linger zones</div></div>"
    "</div>"
    + cap('3.3', 'Early behaviour maps from the MACBA sandbox: a bottleneck heatmap, a flow field and linger zones. These first rough versions confirmed that the tracking already carried readable spatial structure, before the method moved out to the other sites.', 'margin-top:6pt;'),
    chapter=CH3)

# P42 — prediction models (two-col)
add(
    "<span class='secnum'>3.3</span>"
    "<div class='sectitle'>Prediction Models</div>"
    "<div class='body cols'>"
    "<p>The sandbox was also where the choice of model was made. Rather than assume one architecture, I compared four families drawn from the literature: a Long Short-Term Memory network (LSTM), a Gated Recurrent Unit (GRU), a Temporal Convolutional Network (TCN), and a small Transformer.</p>"
    "<p>Each was trained on the sandbox trajectories under the same conditions, and their behaviour was compared not only on training loss but on how they held up when rolled forward step by step. That distinction turned out to matter more than the headline loss. A model that predicts a single next step well can still drift badly once its own predictions are fed back in to produce a longer path.</p>"
    "<p>The LSTM gave the most stable trajectory prediction, with the lowest endpoint drift. The GRU reached a slightly lower training loss but showed more drift once it was predicting autoregressively. The TCN reached the lowest loss overall and yet performed worst on prediction stability, which is the quality that actually matters when a path is rolled out over many steps. The small Transformer did not offer enough of an advantage on this scale of data to justify its added complexity.</p>"
    "<p>So the project settled on the LSTM. The goal here is not the lowest error on a single step. It is a predicted path that stays coherent as it extends, because that is what an architect would actually look at. The best model in the abstract is not always the best model for a small, specific dataset.</p>"
    "</div>",
    chapter=CH3)

# P43 — model comparison (Fig 3.4)
add(
    "<div style='margin-top:8mm;'>" + img('mp-data/outputs/prediction/experiments/phase-2b-final/phase2b_final_spatial_model_comparison.png') + "</div>"
    + cap('3.4', 'The model families compared on the sandbox. The LSTM holds a rolled out path together with the least endpoint drift, which is why it was chosen over the lower loss but less stable GRU and TCN.', 'margin-top:7pt;width:90%;'),
    chapter=CH3)

# P44 — overfit + ablation (two-col)
add(
    "<span class='secnum'>3.4</span>"
    "<div class='sectitle'>The Capacity Test</div>"
    "<div class='body cols'>"
    "<p>The sandbox did not only confirm what worked. It exposed the problem that shaped the rest of the research. When the trained model was asked to predict, its outputs collapsed toward straight lines. Faced with the turning, curving movement that made MACBA interesting, the model smoothed it away. A skater carving a long arc came back as a short straight stub. Was the architecture incapable of representing angular movement, or was it simply not seeing enough of it to learn from?</p>"
    "<p>To separate those two possibilities I ran a controlled capacity test. The original sandbox held 52 tracked trajectories, drawn from roughly eighteen thousand recorded positions. That is very little data. So the dataset was duplicated tenfold, exposing the same motion patterns to the model many times, to ask a narrower question. Given enough exposure to these exact patterns, can the architecture reproduce their angular structure at all?</p>"
    "<p>It could. On the duplicated data the best configuration recovered the turning behaviour that had been lost. Its average displacement error fell to about 0.05 metres, and its curvature correlation with the real paths rose to around 0.50. On the un-duplicated held-out data the same schema had shown a curvature correlation near zero. The architecture was capable of representing angular movement. It had simply lacked the data to learn it.</p>"
    "<p>The capacity result is a memorisation test, not a claim of real world accuracy. What it licenses is a narrower and still useful claim: the early failure was primarily a data problem, not a flaw in the model design, and the way forward was to grow the dataset rather than abandon the architecture.</p>"
    "</div>",
    chapter=CH3)

# P45 — capacity test collage (Fig 3.5)
add(
    "<div style='margin-top:14mm;'>" + img(BK+'/overfit10x_modelC_highlight_collage.png') + "</div>"
    + cap('3.5', 'The capacity test. With the sandbox data duplicated tenfold, Model C, highlighted, reproduces the curved paths that the model had flattened before. Axes are normalised, and this is memorisation evidence that the schema can fit angular motion, not a claim of predictive accuracy.', 'margin-top:8pt;width:88%;'),
    chapter=CH3)

# P46 — feature ablation (Fig 3.6)
add(
    "<div class='body' style='width:88%;margin-bottom:5mm;'>"
    "<p>The same experiment was used to choose the feature schema. Four schemas were compared, carrying six, eight, ten and eleven features. The configuration named Model C, with ten features combining ego motion, position within the site, and distance to obstacles and boundaries, gave the strongest capacity behaviour while staying tied to the architectural question, and it was kept for both reasons.</p>"
    "</div>"
    "<div class='grid3'>"
    "  <div>" + img(BK+'/feature_ablation/traj_00_id24004.png') + "</div>"
    "  <div>" + img(BK+'/feature_ablation/traj_02_id12007.png') + "</div>"
    "  <div>" + img(BK+'/feature_ablation/traj_03_id28007.png') + "</div>"
    "</div>"
    + cap('3.6', 'Feature ablation on the sandbox. Each panel overlays the schemas, from motion only up to the full Model C set, on the same ground truth path. The richer schemas recover the turning that the minimal one flattens, which is the capacity result that justified keeping the spatial features.', 'margin-top:6pt;'),
    chapter=CH3)

# P47 — dataset schema (Fig 3.7)
add(
    "<span class='secnum'>3.5</span>"
    "<div class='sectitle'>The Model C Schema</div>"
    "<div class='body' style='width:88%;margin-bottom:5mm;'>"
    "<p>Each step of a trajectory carries ten values. Two describe the immediate motion as a displacement in each direction. One is the speed. Two encode the heading as its sine and cosine, so that direction is continuous rather than jumping at north. One is the turn rate. Two give the normalised position within the calibrated site. The last two are the distances to the nearest obstacle and to the boundary of the walkable area. A person&rsquo;s next movement is treated as a function of how they were just moving, where they are in the space, and what lies immediately around them.</p>"
    "</div>"
    + img(BK+'/figure_3_7_model_c_feature_schema.png') +
    cap('3.7', 'The agreed Model C feature schema, ten values per step. Ego motion, position in the site, and proximity to obstacles and boundaries, the three registers the model reasons over.', 'margin-top:6pt;width:88%;'),
    chapter=CH3)

# P48 — building the dataset (two-col)
add(
    "<span class='secnum'>3.6</span>"
    "<div class='sectitle'>Building the Dataset</div>"
    "<div class='body cols'>"
    "<p>Seeing that the sandbox had worked, one thing became evident: it needed more data. I went site hunting across Barcelona, and the selection was not random. The end goal was variety, because a model that is trained on one kind of space learns only of that space and little else.</p>"
    "<p>MACBA and the esplanade of Espanya gave open plaza movement. The Montjuïc stairs gave vertical, constrained movement and the angular paths that stairs force on people. Open squares such as Plaça Montjuïc, Plaça Espanya and Plaça Catalunya gave large, loosely structured crossings. The Red Bridge on Passeig de Colom gave something the others did not, a curved structure that bends movement around it.</p>"
    "<p>The reason to spread the dataset across such different spaces is not only technical. It is the architectural claim that is being tested. If movement were purely a matter of individual intention, the space would not matter and one plaza would train the model as well as five. The bet behind Motion Pixels is the opposite, that the space affects movement, and the only way to see that mark is to hold many spaces side by side and ask whether the model reads them differently.</p>"
    "<p>Each site stresses the method differently. A staircase constrains where people can go, so the spatial features carry a lot of the signal. An open plaza does the opposite, giving people freedom and making the recent motion the stronger cue. A curved structure tests whether the model can follow a bend rather than cut across it.</p>"
    "</div>",
    chapter=CH3)

# P49 — site plans (Fig 3.8). Adjacent images aligned by their BOTTOM edges (author correction #13).
def plan_cell(name, label, h):
    return "<div><div class='fhb' style='height:%dmm;'>" % h + img(BK+'/dataset/sites_mass_plan/'+name, 'fit') + "</div><div class='cellcap'>" + label + "</div></div>"
add(
    "<div class='grid2' style='gap:5mm;align-items:end;'>"
    + plan_cell('esplanade-espanya.png', 'Esplanade Espanya', 40)
    + plan_cell('placa-catalunya.png', 'Plaça Catalunya', 40)
    + plan_cell('red-bridge-combined.png', 'Red Bridge', 60)
    + plan_cell('stairs-montjuic-1.png', 'Montjuïc Stairs', 60)
    + "</div>"
    "<div style='width:49%;margin-top:5mm;'>" + plan_cell('placa-montjuic.png', 'Plaça Montjuïc', 56) + "</div>"
    + cap('3.8', 'The site plans of the recordings, drawn to the same convention. From open plazas to a staircase and a curved bridge, the set was chosen so the difficulty is spread across the same spatial features the schema encodes.', 'margin-top:6pt;width:90%;'),
    chapter=CH3)

# P50 — dataset body + recording table
add(
    "<div class='body' style='width:88%;'>"
    "<p>The dataset that the final experiments were trained and evaluated on contains five recordings and 3,534 tracked trajectories, after two recordings were excluded for quality reasons. The seven thousand figure from the acquisition stage and the 3,534 that survived into training describe the two ends of that funnel, raw extraction on one side and quality controlled, calibrated trajectories on the other. Their sizes are uneven, and that reflects how busy each space was.</p>"
    "</div>"
    "<table class='tbl' style='margin-top:8mm;width:96%;'>"
    "<tr><th>Recording</th><th class='n'>Tracked positions</th><th class='n'>Trajectories</th></tr>"
    "<tr><td>esplanade_espanya_01</td><td class='n'>283,120</td><td class='n'>522</td></tr>"
    "<tr><td>placa_catalunya_01</td><td class='n'>355,583</td><td class='n'>1,255</td></tr>"
    "<tr><td>placa_espanya_01</td><td class='n'>212,224</td><td class='n'>824</td></tr>"
    "<tr><td>stairs_montjuic_01</td><td class='n'>127,605</td><td class='n'>334</td></tr>"
    "<tr><td>red_bridge_combined_01</td><td class='n'>85,847</td><td class='n'>599</td></tr>"
    "</table>"
    "<div class='body' style='width:88%;margin-top:7mm;'>"
    "<p>Plaça Catalunya alone contributes more than a third of the trajectories, and the Montjuïc stairs the fewest. A model trained on this mix sees far more open plaza movement than stair movement, and that shows up later in where the predictions are strongest. I did not rebalance the data, partly because the imbalance reflects how busy these spaces actually are.</p>"
    "</div>",
    chapter=CH3)

# P51 — tracking frames (Fig 3.9)
add(
    "<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:3mm;align-items:start;'>"
    "  <div>" + img(BK+'/dataset/esplanadeespanya_tracking.png') + "<div class='cellcap'>Esplanade</div></div>"
    "  <div>" + img(BK+'/dataset/placacatalunya_tracking.png') + "<div class='cellcap'>Catalunya</div></div>"
    "  <div>" + img(BK+'/dataset/placaespanya_tracking.png') + "<div class='cellcap'>Espanya</div></div>"
    "  <div>" + img(BK+'/dataset/stairsmontjuic1_tracking.png') + "<div class='cellcap'>Stairs</div></div>"
    "  <div>" + img(BK+'/dataset/redbridge_tracking.png') + "<div class='cellcap'>Red Bridge</div></div>"
    "</div>"
    + cap('3.9', 'A frozen tracking frame from each of the five recordings, with live trajectories and per-track metrics overlaid. The same pipeline meets a different kind of movement at each site, from the loose diagonals of the esplanade to the compressed lines of the stairs.', 'margin-top:6pt;'),
    chapter=CH3)

# P52 — culling / masks body
add(
    "<div class='body' style='width:84%;'>"
    "<p>Two further recordings, a second Montjuïc stairs clip and a Plaça Montjuïc clip, were prepared but left out of training because their tracking or calibration did not meet the standard the others held. A recording was kept if its tracking held identities cleanly enough and its calibration placed movement on the plan without obvious drift. The two that were dropped failed one of those tests badly enough that including them would have added noise dressed up as data.</p>"
    "<p>Preparing these sites was not automatic. Each recording needed its own calibration, matching points in the footage to points on the plan, and its own spatial mask marking the walkable area, the obstacles and the boundaries. The masks were drawn by hand for each site, which is slow, but it is what lets the spatial features mean something specific to each space rather than being generic.</p>"
    "<p>The final baseline model was evaluated with a track-held-out split, where whole trajectories are divided into training, validation and test sets, and every recording appears in all three. This tests whether the model generalises to unseen people within the same set of spaces. It does not test whether it generalises to an entirely unseen site, which is a harder question and a different experiment.</p>"
    "</div>",
    chapter=CH3)

# P53 — mask overlay (Fig 3.10)
add(
    "<div style='margin-top:16mm;'>" + img('mp-data/annotations/manual_masks_v3/placa_catalunya_01/overlay_manual.png') + "</div>"
    + cap('3.10', 'A hand drawn walkable and obstacle mask over Plaça Catalunya. Masks like this were made for every recording, and they let the model describe a pedestrian by their real relationship to the architecture around them.', 'margin-top:8pt;width:86%;'),
    chapter=CH3)

# ---- behavioural maps: intro | preview, then explanation | atlas pairs ----
FLOW = 'mp-visualization/behavior_maps/behaviormaps_final/flow_fields/'
def flow_cell(name, label):
    return "<div>" + img(FLOW + name) + "<div class='cellcap'>" + label + "</div></div>"
def atlas_cell(name, label):
    return "<div>" + img(name) + "<div class='cellcap'>" + label + "</div></div>"

# behavioural maps intro (left)
add(
    "<span class='secnum'>3.7</span>"
    "<div class='sectitle'>The Behavioural Maps</div>"
    "<div class='lead serif' style='width:88%;'>A plan shows a space as it was designed. A behavioural map shows the same space as it was used, drawn from the movement of everyone the camera saw.</div>"
    "<div class='body' style='width:86%;'>"
    "<p>Before the prediction results, the dataset produces images that are useful on their own. The behavioural maps turn thousands of trajectories into a single readable picture of how a space is used. Each one takes a different feature from the schema and gives it back to the eye. The maps are not predictions and they are not models. They are description, and description is already more than architecture usually keeps once a space is occupied.</p>"
    "</div>",
    chapter=CH3)

# preview of the three map types (right) — no titles naming the types
add(
    tlabel('Preview', cls='tlabel mag') +
    "<div class='grid3' style='margin-top:6mm;align-items:start;'>"
    "  <div><div class='fh' style='height:52mm;'>" + img(FLOW + 'placa_catalunya_01_flow_fields_still.png', 'fit') + "</div></div>"
    "  <div><div class='fh' style='height:52mm;'>" + img(DS+'/placacatalunya_speedmap.png', 'fit') + "</div></div>"
    "  <div><div class='fh' style='height:52mm;'>" + img(DS+'/placacatalunya_heatmap.png', 'fit') + "</div></div>"
    "</div>"
    "<div class='cap' style='margin-top:8pt;width:88%;'>Three readings of the same movement, each drawn from one feature of the dataset and laid over the same plan. The pages that follow take each in turn, an explanation beside its atlas across every recording.</div>",
    chapter=CH3)

# flow explanation (left)
add(
    "<div class='subhead'>Flow fields</div>"
    "<div class='body' style='width:86%;'>"
    "<p>The flow field renders the direction of movement across a site. Trajectories are drawn and coloured so that the dominant lines of travel become visible, the routes people actually take rather than the ones the plan suggests. It answers a plan reading question directly: where does this space channel movement, and where does it leave it diffuse.</p>"
    "<p>What the flow field cannot show is why a line forms where it does. That reading is left to the architect, who can see whether a dominant route follows an entrance, avoids an obstacle, or cuts a corner the design did not intend. The map narrows the question. It does not answer it.</p>"
    "</div>"
    "<div style='position:absolute;left:var(--outer);bottom:var(--bot);'>" + tlabel('Direction of travel &middot; the routes people take') + "</div>",
    chapter=CH3)

# FLOW FIELDS ATLAS (right) — Fig 3.11
add(
    "<div style='display:flex;justify-content:space-between;align-items:baseline;'>"
    "  <div class='sectitle' style='margin:0;'>Flow Fields Atlas</div>" + tlabel('Direction of travel across every recording', cls='tlabel mag') + "</div>"
    "<div class='grid2' style='gap:4mm;margin-top:5mm;'>"
    + flow_cell('placa_catalunya_01_flow_fields_still.png', 'Plaça Catalunya')
    + flow_cell('placa_espanya_01_flow_fields_still.png', 'Plaça Espanya')
    + flow_cell('esplanade_espanya_01_flow_fields_still.png', 'Esplanade Espanya')
    + flow_cell('placa_montjuic_01_flow_fields_still.png', 'Plaça Montjuïc')
    + flow_cell('red_bridge_combined_01_flow_fields_still.png', 'Red Bridge')
    + flow_cell('stairs_montjuic_01_flow_fields_still.png', 'Montjuïc Stairs I')
    + flow_cell('stairs_montjuic_02_flow_fields_still.png', 'Montjuïc Stairs II')
    + "</div>"
    + cap('3.11', 'Flow fields across every recording, the dominant lines of travel drawn from the accumulated trajectories. The open plazas fan movement along their main diagonals; the stairs and the bridge bend it along the paths the architecture allows. Each still keeps its own legend.', 'margin-top:6pt;'),
    chapter=CH3)

# speed explanation (left)
add(
    "<div class='subhead'>Speed</div>"
    "<div class='body' style='width:86%;'>"
    "<p>The speed map, built from the speed metric, shows one dot per pedestrian coloured by how fast they were moving, using the slow, medium and fast categories. Read across a whole site, it separates the parts of a space where people hurry from the parts where they slow down and settle. The two often correspond to something in the architecture, an edge, a threshold, a place with a reason to pause.</p>"
    "<p>The thresholds that sort the dots are fixed rather than relative, so a slow dot means the same pace on every site. That makes the maps comparable. A crowd that reads as mostly fast on the esplanade and mostly slow in a tighter square is telling you something about how the two spaces are used, not about how the colour scale was set. The Montjuïc stairs read as a field of slower dots, movement checked by the steps, while the open esplanade carries far more fast crossings.</p>"
    "</div>"
    "<div style='position:absolute;left:var(--outer);bottom:var(--bot);'>" + tlabel('One dot &middot; one pedestrian &middot; coloured by pace') + "</div>",
    chapter=CH3)

# SPEED ATLAS (right) — Fig 3.12
add(
    "<div style='display:flex;justify-content:space-between;align-items:baseline;'>"
    "  <div class='sectitle' style='margin:0;'>Speed Atlas</div>" + tlabel('Pace across every recording', cls='tlabel mag') + "</div>"
    "<div class='grid2' style='gap:4mm;margin-top:5mm;'>"
    + atlas_cell(DS+'/placacatalunya_speedmap.png', 'Plaça Catalunya')
    + atlas_cell(DS+'/placaespanya_speedmap.png', 'Plaça Espanya')
    + atlas_cell(DS+'/esplanadeespanya_speedmap.png', 'Esplanade Espanya')
    + atlas_cell(DS+'/placamontjuic_speedmap.png', 'Plaça Montjuïc')
    + atlas_cell(DS+'/redbridge_speedmap.png', 'Red Bridge')
    + atlas_cell(DS+'/stairsmontjuic1_speedmap.png', 'Montjuïc Stairs I')
    + atlas_cell(DS+'/stairsmontjuic2_speedmap.png', 'Montjuïc Stairs II')
    + "</div>"
    + cap('3.12', 'Speed maps across every recording, one point per pedestrian coloured by pace on a fixed slow / medium / fast scale. Plaça Catalunya, the busiest, shows the clearest separation between the fast lines people use to cross and the slow pockets where they gather; the stairs read as a slower field. Each map keeps its own legend.', 'margin-top:6pt;'),
    chapter=CH3)

# density explanation (left)
add(
    "<div class='subhead'>Density</div>"
    "<div class='body' style='width:86%;'>"
    "<p>The density map reads congestion. It scores areas by how heavily they are used and warms their colour accordingly, so that the pinch points and crowded cells stand out. This is the occupancy reading the project produces. The scores are precomputed from the trajectory data, and the map describes where use concentrated rather than diagnosing a given point as a circulation failure.</p>"
    "<p>Congestion is the behaviour most directly tied to how a space performs, and the one a plan is worst at predicting, since two corridors of equal width can behave completely differently once real flows meet in them. One caveat runs across all of the maps. They are built from tracked movement, and tracking is imperfect. At the scale of thousands of trajectories these errors mostly wash out, so the maps are best read as strong description rather than exact measurement.</p>"
    "</div>"
    "<div style='position:absolute;left:var(--outer);right:44%;bottom:var(--bot);'>"
    "  <div class='legrow'><span>Low</span><div class='legbar' style='background:linear-gradient(90deg,#ffe08a,#ff9d3c,#f0431f,#c0141c);'></div><span>High</span></div>"
    "  <div class='tlabel' style='margin-top:4pt;'>Bottleneck intensity</div>"
    "</div>",
    chapter=CH3)

# DENSITY ATLAS (right) — Fig 3.13
add(
    "<div style='display:flex;justify-content:space-between;align-items:baseline;'>"
    "  <div class='sectitle' style='margin:0;'>Density Atlas</div>" + tlabel('Bottlenecks across every recording', cls='tlabel mag') + "</div>"
    "<div class='grid2' style='gap:4mm;margin-top:5mm;'>"
    + atlas_cell(DS+'/placacatalunya_heatmap.png', 'Plaça Catalunya')
    + atlas_cell(DS+'/placaespanya_heatmap.png', 'Plaça Espanya')
    + atlas_cell(DS+'/esplanadeespanya_heatmap.png', 'Esplanade Espanya')
    + atlas_cell(DS+'/placamontjuic_heatmap.png', 'Plaça Montjuïc')
    + atlas_cell(DS+'/redbridge_heatmap.png', 'Red Bridge')
    + atlas_cell(DS+'/stairsmontjuic1_heatmap.png', 'Montjuïc Stairs I')
    + atlas_cell(DS+'/stairsmontjuic2_heatmap.png', 'Montjuïc Stairs II')
    + "</div>"
    + cap('3.13', 'Bottleneck density across every recording, drawn to a common intensity scale. Warmer cells mark where movement concentrated. In the open plazas the pressure spreads along the main crossing lines; on the Montjuïc stairs it compresses into the narrow band the steps allow.', 'margin-top:6pt;'),
    chapter=CH3)

# prediction rollout explanation (left)
add(
    "<div class='subhead'>Prediction rollouts</div>"
    "<div class='body' style='width:86%;'>"
    "<p>The last family, the prediction projections, shows the model&rsquo;s output rather than the observed data. A predicted path is drawn over the plan from a seed of real motion, so that the forecast can be read in the same architectural frame as everything else. At their best, over an open plaza, these projections lay a plausible near future onto the space, and they are the clearest expression of what the whole pipeline is for.</p>"
    "<p>Each map is really a view onto one column of the dataset. The flow field reads direction, the speed map reads the speed feature, the density map reads position and dwell. The same encoding that feeds the prediction model also feeds the images, so the maps and the forecasts are two readings of one description rather than two separate products. Taken together, the maps are the point where the dataset stops being a table and becomes something an architect can look at and argue with.</p>"
    "</div>",
    chapter=CH3)

# PREDICTION ATLAS (right) — Fig 3.15
add(
    "<div style='display:flex;justify-content:space-between;align-items:baseline;'>"
    "  <div class='sectitle' style='margin:0;'>Prediction Atlas</div>" + tlabel('Rollouts over the site plans', cls='tlabel mag') + "</div>"
    "<div style='margin-top:6mm;'>" + img(DS+'/placaespanya_predictionrollout.png') + "<div class='cellcap'>Plaça Espanya</div></div>"
    "<div style='margin-top:6mm;'>" + img(DS+'/esplanadeespanya_predictionrollout.png') + "<div class='cellcap'>Esplanade Espanya</div></div>"
    + cap('3.15', 'Prediction projections over the site plans. Each draws a long, illustrative rollout of the model forward from observed motion, laid onto the plan so the forecast can be read in the same frame as the design. At this length they should be read as tendency, not as an exact route. Plaça Catalunya is given the hero spread overleaf.', 'margin-top:7pt;'),
    chapter=CH3)

# P64/65 — DARK HERO: Plaça Catalunya prediction rollout (Fig 3.14). Real asset, aspect ratio preserved.
HERO = r'C:\Users\OWNER\Desktop\BOOKLET_images\dataset\placacatalunya_predictionrollout.png'  # 3200x1441 (2.22)
add(
    "<div style='position:absolute;inset:0;overflow:hidden;background:#000;'>"
    "<img src='%s' style='position:absolute;left:0;top:50%%;transform:translateY(-50%%);width:420mm;max-width:none;height:auto;'></div>"
    "<div style='position:absolute;left:var(--outer);top:var(--top);'><div class='tlabel' style='color:var(--magL);letter-spacing:.16em;'>Prediction &nbsp;&middot;&nbsp; H400 &nbsp;&middot;&nbsp; Plaça Catalunya rollout</div>"
    "<div style='margin-top:4pt;'><span class='tick'></span></div></div>" % opt(HERO,3600,92),
    chapter=CH3, dark=True, fullbleed=True, header=False, folio=False)
add(
    "<div style='position:absolute;inset:0;overflow:hidden;background:#000;'>"
    "<img src='%s' style='position:absolute;left:-210mm;top:50%%;transform:translateY(-50%%);width:420mm;max-width:none;height:auto;'></div>"
    "<div class='cap' style='position:absolute;right:var(--outer);bottom:12mm;width:96mm;text-align:right;color:#c2c2c2;'><b style='color:var(--magL);'>Fig 3.14</b>&nbsp; A prediction projection over Plaça Catalunya. Observed history and predicted continuation are drawn in the same architectural frame as the plan, which is the reading the whole pipeline is built to produce.</div>" % opt(HERO,3600,92),
    chapter=CH3, dark=True, fullbleed=True, header=False, folio=False)

# P66 — horizon rollouts body (two-col)
add(
    "<span class='secnum'>3.8</span>"
    "<div class='sectitle'>Predictive Layer: Horizon Rollouts</div>"
    "<div class='body cols'>"
    "<p>Once the dataset was stable, the real question was how far ahead the model could see. And that is exactly what a horizon is: how many steps into the future the model is asked to predict from a short slice of observed motion. I trained and evaluated five of them. In the notation I use through this chapter they are H20, H60, H100, H200 and H400, which correspond to roughly 1, 3, 5, 10 and 20 metres of travel. The numbers are frames, not minutes.</p>"
    "<p>The horizons were kept separated on purpose because each one is its own trained model with its own checkpoint. Asking a network to commit to twenty metres is a different problem from asking it to commit to one. Reporting them as a single number would have hidden where the method holds and where it starts to fail.</p>"
    "<p>The short horizons perform well. At H20 the average displacement error sits around 0.45 m, and the predicted endpoint lands inside a one metre tolerance about 83% of the time. For an architect that is a usable signal. It says the immediate intention of a pedestrian, the next step or two, is legible from recent motion alone.</p>"
    "<p>What matters more than the size of the error is its shape. The predictions do not only drift. They shorten and straighten. At H100 the median predicted path is about 1.9 m long while the recorded path over the same window is about 5.3 m. The model reaches less far than the person actually walked, and it smooths the turns out of the way.</p>"
    "</div>",
    chapter=CH3)

# P67 — horizon body + error table
add(
    "<div class='body' style='width:88%;'>"
    "<p>The error grows in a way that is easy to read. Later experiments that changed the training objective pushed back against the straightening. The first, a magnitude aware objective, corrected the tendency to under-reach and brought the predicted path length back close to the real one. The second, a curvature aware objective, went after the straightening itself and recovered much of the shape, at a modest cost to directional accuracy. This curvature aware model is the final model.</p>"
    "</div>"
    "<table class='tbl' style='margin-top:8mm;width:96%;'>"
    "<tr><th>Horizon</th><th class='n'>Approx. travel</th><th class='n'>ADE</th><th class='n'>FDE</th><th class='n'>Endpoint &lt; 1 m</th></tr>"
    "<tr><td>H20</td><td class='n'>1 m</td><td class='n'>0.45 m</td><td class='n'>0.7 m</td><td class='n'>83%</td></tr>"
    "<tr><td>H60</td><td class='n'>3 m</td><td class='n'>1.0 m</td><td class='n'>&mdash;</td><td class='n'>&mdash;</td></tr>"
    "<tr><td>H100</td><td class='n'>5 m</td><td class='n'>1.4 m</td><td class='n'>&mdash;</td><td class='n'>&mdash;</td></tr>"
    "<tr><td>H200</td><td class='n'>10 m</td><td class='n'>2.6 m</td><td class='n'>&mdash;</td><td class='n'>&mdash;</td></tr>"
    "<tr><td>H400</td><td class='n'>20 m</td><td class='n'>5.1 m</td><td class='n'>10 m</td><td class='n'>55%</td></tr>"
    "</table>"
    "<div class='body' style='width:88%;margin-top:7mm;'>"
    "<p>A short prediction describes intention. A long prediction describes tendency. At one to five metres the predicted path is close enough to the real one that it can be laid over a plan and trusted as a local reading of movement. Past ten metres it should be treated as a soft field of likely direction, not a line that someone will follow.</p>"
    "</div>",
    chapter=CH3)

def hcell(src, tag):
    return "<div>" + img(BK+'/horizon_rollouts/'+src) + "<div class='cellcap'>" + tag + "</div></div>"

# P68 — short-horizon sequence
add(
    "<div class='subhead'>Short horizons</div>"
    "<div class='body' style='width:86%;margin-bottom:5mm;'><p>Best and worst cases at each horizon, ranked by error, from the baseline model. Black is the observed history, the grey dotted line the recorded ground truth. Even the harder cases stay a plausible walk rather than collapsing.</p></div>"
    "<div class='grid2' style='gap:5mm;'>"
    + hcell('H20_best_track1261.png', 'H20 &middot; best')
    + hcell('H20_worst_track1647.png', 'H20 &middot; worst')
    + hcell('H60_best_track4394.png', 'H60 &middot; best')
    + hcell('H60_worst_track1822.png', 'H60 &middot; worst')
    + "</div>",
    chapter=CH3)

# P69 — H100 feature (Fig 3.16)
add(
    "<div class='subhead'>H100 &middot; roughly five metres</div>"
    "<div class='grid2' style='gap:5mm;margin-top:4mm;'>"
    + hcell('H100_best_track4703.png', 'best')
    + hcell('H100_worst_track3160.png', 'worst')
    + "</div>"
    + cap('3.16', 'Best and worst H100 rollouts, at roughly five metres. Purple marks the selected best case and green the selected worst; black is the observed history and the grey dotted line the recorded ground truth. Even the worst case stays a plausible walk rather than collapsing.', 'margin-top:7pt;'),
    chapter=CH3)

# P69b — H200 (completes the horizon sequence)
add(
    "<div class='subhead'>H200 &middot; roughly ten metres</div>"
    "<div class='grid2' style='gap:5mm;margin-top:4mm;'>"
    + hcell('H200_best_track1178.png', 'best')
    + hcell('H200_worst_track6268.png', 'worst')
    + "</div>"
    + cap('3.16b', 'Best and worst H200 rollouts, at roughly ten metres, the mid-point of the horizon sequence. The prediction still tracks the walk but begins to shorten and straighten, the drift the longer horizons make unmistakable.', 'margin-top:7pt;'),
    chapter=CH3)

# P70 — H400 feature (Fig 3.17) + stress test
add(
    "<div class='subhead'>H400 &middot; roughly twenty metres</div>"
    "<div class='grid2' style='gap:5mm;margin-top:4mm;'>"
    + hcell('H400_best_track112.png', 'best')
    + hcell('H400_worst_track540.png', 'worst')
    + "</div>"
    + cap('3.17', 'The same comparison at H400, roughly twenty metres, with purple best and green worst against the grey dotted ground truth. The best case still follows the shape of the walk; the worst drifts and straightens, the long-range limit the curvature aware objective was built to push back.', 'margin-top:7pt;') +
    "<div class='body' style='width:88%;margin-top:6mm;'><p>The stress test sits at the edge of what the data can support, pushing the model out to roughly 40, 50 and 60 metres. The point is not accuracy, since there is no reliable ground truth that far out. It is a check on how the model fails. A prediction that keeps moving like a person, even when it is wrong, is more useful to a designer than one that falls apart.</p></div>",
    chapter=CH3)

# ---- 3.9 Building the Prototype: begins on a new (left) spread, ends on the dashboard hero ----
ensure_left()  # 3.9 must begin cleanly on a left page / new spread
# prototype intro (left)
add(
    "<span class='secnum'>3.9</span>"
    "<div class='sectitle'>Building the Prototype</div>"
    "<div class='lead serif' style='width:88%;'>The last piece of the work is not an experiment. It is an attempt to make everything before it usable by someone who is not running Python scripts.</div>"
    "<div class='body' style='width:86%;'>"
    "<p>The prototype is a front end that wraps the pipeline into a single flow an architect could actually sit in front of. A user uploads footage of a space and its plan. They calibrate, matching points between the two so the movement can be placed in real coordinates. The system processes the footage, and the user inspects the result, both the observed movement, as trajectories and behavioural maps, and the predicted movement rolled forward by the model.</p>"
    "</div>",
    chapter=CH3)

# start + calibrate screenshots (Fig 3.18) with caption
add(
    "<div>" + img(BK+'/platform_screenshots/initiation.png') + "<div class='cellcap'>Initiation &middot; a new study</div></div>"
    "<div style='margin-top:5mm;'>" + img(BK+'/platform_screenshots/calibration.png') + "<div class='cellcap'>Calibration &middot; camera view and plan</div></div>"
    + cap('3.18', 'The opening and calibration screens. The landing page introduces Motion Pixels as a tool for mapping spatial intelligence and starts a new study. Calibration shows the camera view and the site plan side by side, where the user matches corresponding landmarks to establish the homography that links image to plan.', 'margin-top:6pt;'),
    chapter=CH3)

# prototype status text
add(
    "<div class='body' style='width:82%;'>"
    "<p>Its status matters, because a convincing interface can imply more than it delivers. The prototype is a demonstrator with mocked data. It shows the intended experience and the shape of the tool, but it is not connected to a live backend that runs detection, calibration and inference on demand. Presenting it as a working product would overstate where the research is.</p>"
    "<p>Turning the demonstrator into a working tool is mostly engineering, but not entirely. The mocked parts are the ones that are genuinely hard, automatic calibration, reliable tracking on unseen footage, and inference fast enough to feel interactive. Each of those is a real problem in its own right, and pretending the interface has solved them would repeat exactly the kind of overstatement this thesis has tried to avoid.</p>"
    "<p>An architect is not going to run a tracking model from a command line, and they should not have to. The prototype is a sketch of that form. It is included in the thesis because the research question was never only whether movement can be predicted, but whether the result can be made to matter to design.</p>"
    "</div>",
    chapter=CH3)

# export screenshot (Fig 3.20) + caption + lead-in to the dashboard finale
add(
    img(BK+'/platform_screenshots/save.png') +
    cap('3.20', 'Exporting the drawing. The dialog previews the current studio composition and offers a high-resolution PNG or an editable SVG of the layers, so the analysis can leave the tool as a drawing an architect can keep working on. The prototype runs on the bundled demonstration data rather than a live backend.', 'margin-top:6pt;width:92%;') +
    "<div class='body' style='width:86%;margin-top:7mm;'>"
    "<p>The inspection view is the part that matters most, because it is where the two halves of the research meet. Observed movement and predicted movement can be shown over the same plan, so a designer can compare what people did with what the model expects, and decide for themselves how far to trust the forecast. That decision, rather than a single accuracy figure, is what the tool is meant to support.</p>"
    "</div>",
    chapter=CH3)

# DASHBOARD HERO SPREAD (Fig 3.19) — final image of the prototype, immediately before Chapter 4
DASH = BK + '/platform_screenshots/dashboard.png'
add(
    "<div style='position:absolute;inset:0;overflow:hidden;background:#000;'>"
    "<img src='%s' style='position:absolute;left:0;top:50%%;transform:translateY(-50%%);width:420mm;max-width:none;height:auto;'></div>"
    "<div style='position:absolute;left:var(--outer);top:var(--top);'><div class='tlabel' style='color:var(--magL);letter-spacing:.16em;'>The Studio &nbsp;&middot;&nbsp; Plaça Espanya</div>"
    "<div style='margin-top:4pt;'><span class='tick'></span></div></div>" % opt(DASH, 3200, 92),
    chapter=CH3, dark=True, fullbleed=True, header=False, folio=False)
add(
    "<div style='position:absolute;inset:0;overflow:hidden;background:#000;'>"
    "<img src='%s' style='position:absolute;left:-210mm;top:50%%;transform:translateY(-50%%);width:420mm;max-width:none;height:auto;'></div>"
    "<div class='cap' style='position:absolute;right:var(--outer);bottom:12mm;width:104mm;text-align:right;color:#c2c2c2;'><b style='color:var(--magL);'>Fig 3.19</b>&nbsp; The studio dashboard for the Plaça Espanya demonstration. The space is drawn as layered spatial information, with speed, flow fields, bottlenecks and predictions switched on. The right panel holds saved studies, layer switches and export controls. This is the bundled precomputed demonstration, not a newly processed dataset.</div>" % opt(DASH, 3200, 92),
    chapter=CH3, dark=True, fullbleed=True, header=False, folio=False)

# ================================================================ CHAPTER 4
CH4 = 'What Comes Next'
ensure_left()
add(opener('4', 'Reflection', 'What<br>Comes Next'), opener=True, header=False, folio=False)

# P77 — what did the model learn
add(
    "<span class='secnum'>4.1</span>"
    "<div class='sectitle'>What Did the Model Learn</div>"
    "<div class='lead serif' style='width:88%;'>The clearest finding is also the simplest. Recent movement is the strongest predictor of short-term motion.</div>"
    "<div class='body' style='width:86%;'>"
    "<p>Across every experiment, a person&rsquo;s next step or two followed most reliably from how they were already moving, and this held before any spatial feature was added. If Motion Pixels shows one thing without qualification, it is that the immediate future of a pedestrian is written mostly in their recent past.</p>"
    "<p>The horizon results give that claim a scale. At the shortest range, a metre or so, the prediction is close enough to be trusted as a local reading. By about five metres it still holds together as a plausible path. Beyond ten it becomes a direction of travel rather than a route. So the model did not learn to see the future in general. It learned to extend the present a short way, and the useful part of that extension happens to fall at the scale where an architect reasons about a threshold or a turn.</p>"
    "</div>",
    chapter=CH4)

# P78 — body cont (two-col)
add(
    "<div class='body cols'>"
    "<p>Spatial context helps, but conditionally. The features describing where a person is and what lies around them improved prediction when there was enough data to learn from, and did little when there was not. On the small sandbox, adding spatial information barely moved the results. On the larger dataset it earned its place. The space does leave a mark on movement, but reading that mark takes more examples than reading motion alone.</p>"
    "<p>There was also a lesson that had nothing to do with prediction. The behavioural maps, built only from observed movement, already surfaced things a plan does not show, where a plaza slows people down, where crossings concentrate, where a space empties out. Before any forecast, describing the movement well turned out to be worth a great deal on its own. Not all of the value here is in the model.</p>"
    "<p>Direction and curvature are harder than position, and this ran through the entire project. A model could place a person roughly where they were going while getting the shape of the path wrong, because it flattened turns into straight lines. Landing near the right endpoint is not the same as tracing the right path, and a prediction can do the first while failing the second. The distinction matters to an architect, because the curve is often where the behaviour is, the swerve around an obstacle or the arc toward an entrance.</p>"
    "</div>",
    chapter=CH4)

# P79 — body cont (pull-quote treatment removed per author correction #10)
add(
    "<div class='body' style='width:86%;'>"
    "<p>The model also under-reached. Left to minimise error one step at a time, it predicted paths that were not only too straight but too short, stopping before the person did. Length and shape were two separate things to get wrong, and the later objective changes had to correct them one at a time rather than together.</p>"
    "<p>The last finding is the one that most shaped the research. The early failure at MACBA turned out to be primarily a data problem rather than a flaw in the model. The architecture could represent angular movement once it had enough examples. The later experiments that changed the training objective, not the data, also recovered a large part of the missing shape. Neither alone explains the collapse, and it would be too neat to blame only the dataset.</p>"
    "<p>Movement is partly legible. It is most legible at short range, more legible in motion than in geometry, and legible in space only with enough examples. That boundary, and where it falls, is the real result.</p>"
    "</div>",
    chapter=CH4)

# P80 — limitations body
add(
    "<span class='secnum'>4.2</span>"
    "<div class='sectitle'>Limitations of Research</div>"
    "<div class='body cols'>"
    "<p>The results come with real limits, and naming them precisely is more useful than a general disclaimer. The dataset is small and narrow. Three and a half thousand trajectories from five recordings is not a large or diverse sample of how people move through cities.</p>"
    "<p>The movement itself is imperfect. Detection and tracking fail in ordinary ways. People are occluded by others, identities are occasionally dropped or swapped, and a jittery track adds noise that looks like real motion. Some of the sharpest turns in the raw data are not people at all. They are the tracker changing its mind.</p>"
    "<p>There is also the matter of what the camera saw. A single viewpoint captures only part of a space, so a trajectory is only ever the portion of a journey that fell within view. Calibration and spatial encoding still depend on manual work, drawn by hand for each recording. This is slow, it does not scale, and it is one of the clearest gaps between the current method and a tool that anyone could pick up.</p>"
    "<p>Long predictions degrade, and they degrade in a particular way rather than at random. Beyond roughly five metres, a prediction should be read as a likely direction rather than a committed route. And every model here was tested on people within the same sites it was trained on, never on a genuinely unseen space. Whether the method transfers to a space it has never seen is a question this work sets up and does not answer.</p>"
    "</div>",
    chapter=CH4)

# P81 — limitations diagram LARGE (Fig 4.1)
add(
    tlabel('Reflection', cls='tlabel mag') +
    "<div style='margin-top:8mm;'>" + img(AST+'/limitations_tight.png') + "</div>"
    + cap('4.1', 'The limitations of the current method gathered in one view. Observed trajectories and spatial context both meet the same limiting factors, data scale, spatial variety and environmental complexity, which is why the method struggles to generalise beyond the spaces it was trained on.', 'margin-top:9pt;width:86%;'),
    chapter=CH4)

# P82 — the meaning of it all
add(
    "<span class='secnum'>4.3</span>"
    "<div class='sectitle'>The Meaning of It All</div>"
    "<div class='body' style='width:86%;'>"
    "<p>It would be easy to read this thesis as a project about prediction, and to judge it by how accurate the predictions are. That would miss the point. Prediction was never the final objective. It was the test. The question underneath was whether movement carries enough structure to be treated as spatial evidence, and prediction was the sharpest way to probe it.</p>"
    "<p>What the work is really proposing is that observed and predicted movement can become another layer of architectural information. A plan records intention. Movement records use. For most of architectural practice the second of these is lost once a building is occupied, surviving as memory and anecdote rather than as anything an architect can hold and compare. Motion Pixels is an argument that it does not have to be lost, that it can be captured, measured, placed back onto the plan, and read next to it.</p>"
    "<p>To make that concrete: an architect designing a new plaza could load footage of an existing one that works in a similar way, read how people actually move through it, and carry that reading into the new design as evidence rather than assumption. It replaces a guess with an observation, which is most of what evidence ever does in design.</p>"
    "</div>",
    chapter=CH4)

# P83 — meaning cont (pull-quote treatment removed per author correction #12)
add(
    "<div class='body' style='width:86%;'>"
    "<p>This is where the two references from the first chapter meet. Space Syntax reasons from configuration toward likely movement, and Whyte reasoned from observed behaviour back toward the space. This project sits between them, taking Whyte&rsquo;s starting point in the footage and trying to return the result to the measured, plan-based world Space Syntax works in. The contribution is less a new model than a way of moving between the two.</p>"
    "<p>A partial tool that is clear about its limits is more useful than a confident one that hides them. An architect can work with a reading that says trust me to five metres and treat the rest as tendency. The bounded, calibrated nature of the result is not a weakness to apologise for. It is part of what makes it usable.</p>"
    "<p>Observe behaviour, use it to understand a space more truthfully, and let that understanding feed design.</p>"
    "</div>",
    chapter=CH4)

# P84 — future research (two-col)
add(
    "<span class='secnum'>4.4</span>"
    "<div class='sectitle'>Future Research Directions</div>"
    "<div class='body cols'>"
    "<p>The most immediate is more data, across more sites and more kinds of space. The single clearest constraint on this work was the size and narrowness of the dataset, and almost every result would be firmer with a larger, more varied one. This is also what a real test of transfer to unseen sites requires, since that test only means something once there are enough sites to hold some back.</p>"
    "<p>Close behind is removing the manual bottleneck. Calibration and masking are the steps that stop this from scaling, and automating even part of them would do more for the method&rsquo;s reach than any change to the model. It is less glamorous than a new architecture and probably more important.</p>"
    "<p>The prediction itself has room to improve on the properties that proved hardest. Heading and curvature, and the behaviour of the model at long horizons, are where the objective-based experiments already pointed. The model also predicts each person on their own, as if they moved through an empty space. Adding pedestrian-to-pedestrian interaction would bring it closer to how movement works when a space is busy.</p>"
    "<p>The behavioural maps could also become richer. Maps that combine several features, or that surface more of the information the encoding already holds, would give a fuller picture of how a space is used, closer to a designer reading than a single metric.</p>"
    "</div>",
    chapter=CH4)

# P85 — future cont
add(
    "<div class='body' style='width:86%;'>"
    "<p>The larger step is dimensional. The current features reduce a space to distances to obstacles and boundaries in two dimensions, and the maps are drawn flat on the plan. A natural extension is to reconstruct the space itself in three dimensions from the same video, and to place the behaviour back into that model, so the heatmaps, flows and predicted paths can be read as volumes rather than as marks on a flat drawing.</p>"
    "<p>The furthest ambition is a loop rather than a pipeline. Everything here runs in one direction, from footage to prediction. The more interesting version would close that loop, so a designer could observe a space, predict how a change might affect movement, modify the design, and evaluate the result, then go round again. That is the destination that gives the work its point, a behaviour-informed way of designing rather than only a behaviour-reading one.</p>"
    "<p>Two quieter directions would strengthen the foundation under all of this. One is validation, checking the predicted and observed readings against the judgement of people who know a space well. The other is integration. Motion Pixels does not need to stand alone, and it would be stronger sitting alongside the tools architects already use, feeding observed movement into a Space Syntax reading or a building model rather than competing with them.</p>"
    "</div>",
    chapter=CH4)

# ================================================================ CHAPTER 5 — CONCLUSION
CH5 = 'Conclusion'
ensure_left()
add(opener('5', 'Synthesis', 'Conclusion'), opener=True, header=False, folio=False)

# conclusion — entire conclusion on ONE page (merged), composition centred (author corrections #24/#25)
add(
    "<div class='lead serif' style='width:86%;'>This thesis began with a gap that every architect knows and few can measure. A drawing describes how a space is meant to be used. The people who use it answer with their own movement, and that answer is usually lost once the building is occupied.</div>"
    "<div class='body' style='width:86%;'>"
    "<p>Motion Pixels was an attempt to hold onto it, to turn ordinary video of a public space into movement that can be measured, placed back onto the plan, and read as evidence. The path there was not straight. It started on one plaza in front of MACBA, where the first models failed in a way that turned out to be useful. A capacity test showed the architecture could represent angular movement once it had enough examples, which redirected the research from fixing the model to feeding it.</p>"
    "<p>What that judgement returned is a bounded claim, and I have tried to keep it bounded throughout. Short-range prediction works. At the scale of a step or two, movement follows reliably from recent motion, and spatial context adds to that once there is enough data to learn from. As the horizon grows the prediction weakens, until beyond about five metres it is better read as a direction than a route. None of the results transfer to a genuinely unseen space, and the thesis does not pretend they do.</p>"
    "<p>I could keep going on the prediction itself, on ways to shave the error down. But the larger actor in this thesis was never the model. It is spatial design. Motion Pixels is an attempt to read the feedback between behaviour and architecture, the quiet back and forth in which a space shapes how people move and their movement, in turn, reveals what the space is actually doing. Behaviour is not a by-product of architecture to be tidied away once a building opens. It is information, and it can feed back into design.</p>"
    "<p>The point will not be a better forecast. It will be a better understanding of the spaces we design, and a slow move toward something worth calling spatial intelligence.</p>"
    "</div>",
    chapter=CH5)

# ================================================================ BIBLIOGRAPHY
def bib(entries):
    rows = ''.join("<p style='margin-bottom:6pt;text-indent:-5mm;padding-left:5mm;'>%s</p>" % e for e in entries)
    return "<div class='body' style='columns:2;column-gap:9mm;text-align:left;font-size:8.4pt;line-height:12.4pt;'>%s</div>" % rows

BIB1 = [
 "Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., and Savarese, S. (2016). Social LSTM: Human Trajectory Prediction in Crowded Spaces. Proceedings of the IEEE CVPR.",
 "Autodesk (n.d.). InfraWorks: Mobility and Traffic Simulation documentation. Autodesk product documentation.",
 "Bai, S., Kolter, J. Z., and Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271.",
 "Bentley Systems (n.d.). LEGION Simulator and OpenPaths. Official product description.",
 "European Data Protection Supervisor (2020). A Preliminary Opinion on Data Protection and Scientific Research.",
 "European Union (2016). Regulation (EU) 2016/679, General Data Protection Regulation, especially Articles 5, 6 and 89 and Recital 26. Official Journal of the European Union, L119.",
 "Giuliari, F., Hasan, I., Cristani, M., and Galasso, F. (2020). Transformer Networks for Trajectory Forecasting. ICPR; arXiv:2003.08111.",
 "Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., and Alahi, A. (2018). Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks. Proceedings of CVPR.",
 "Hartley, R., and Zisserman, A. (2004). Multiple View Geometry in Computer Vision. 2nd edn. Cambridge University Press.",
 "Hillier, B. (1996). Space Is the Machine: A Configurational Theory of Architecture. Cambridge University Press; open edition in UCL Discovery.",
 "Hillier, B., and Hanson, J. (1984). The Social Logic of Space. Cambridge University Press.",
 "Hillier, B., Penn, A., Hanson, J., Grajewski, T., and Xu, J. (1993). Natural movement: or, configuration and attraction in urban pedestrian movement. Environment and Planning B, 20, 29-66.",
]
BIB2 = [
 "Lerner, A., Chrysanthou, Y., and Lischinski, D. (2007). Crowds by Example. Computer Graphics Forum, 26(3), 655-664. Source of the UCY benchmark.",
 "MACBA (n.d.). Architecture and Spaces. Museu d&rsquo;Art Contemporani de Barcelona.",
 "Milieu Consulting (2021, published 2022). Study on the appropriate safeguards under Article 89(1) GDPR for the processing of personal data for scientific research. EDPB.",
 "Pellegrini, S., Ess, A., Schindler, K., and van Gool, L. (2009). You&rsquo;ll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking. ICCV. Source of the ETH benchmark.",
 "Project for Public Spaces (n.d.). A Primer on Seating. Interpretation of Whyte&rsquo;s work used for the seating discussion.",
 "Robicquet, A., Sadeghian, A., Alahi, A., and Savarese, S. (2016). Learning Social Etiquette: Human Trajectory Understanding in Crowded Scenes. ECCV. Source of the Stanford Drone Dataset.",
 "Sadeghian, A., Kosaraju, V., Sadeghian, A., Hirose, N., Rezatofighi, H., and Savarese, S. (2019). SoPhie: An Attentive GAN for Predicting Paths Compliant to Social and Physical Constraints. CVPR.",
 "Ultralytics (n.d.). YOLOv8 documentation. Official model documentation for the detector used in the pipeline.",
 "Whyte, W. H. (1980). The Social Life of Small Urban Spaces. Washington, DC: The Conservation Foundation.",
 "Yuan, Y., Weng, X., Ou, Y., and Kitani, K. (2021). AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting. ICCV.",
 "Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., and Wang, X. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV. The tracker used in the pipeline.",
]
# (colophon page removed per author correction #25)

# ---- List of Figures (faces the bibliography) ----
FIG_TITLES = [
 ('1.1', 'Desire lines worn through planted ground'),
 ('1.2', 'William H. Whyte observing New York plazas'),
 ('1.3', 'The research question, as a diagram'),
 ('1.4', 'The hypothesis as a chain of claims'),
 ('2.1', 'A Space Syntax reading of an urban grid'),
 ('2.2', 'Commercial crowd-simulation tools'),
 ('2.3', 'Positioning: the predictive gap'),
 ('2.4', 'The computational pipeline'),
 ('3.1', 'The MACBA esplanade'),
 ('3.2', 'Tracked movement on the sandbox'),
 ('3.3', 'Early behaviour maps from the sandbox'),
 ('3.4', 'Prediction model families compared'),
 ('3.5', 'The capacity test'),
 ('3.6', 'Feature ablation'),
 ('3.7', 'The Model C feature schema'),
 ('3.8', 'The site plans of the recordings'),
 ('3.9', 'A tracking frame from each recording'),
 ('3.10', 'A hand-drawn walkable and obstacle mask'),
 ('3.11', 'The flow field for Plaça Catalunya'),
 ('3.12', 'Speed atlas, every recording'),
 ('3.13', 'Density atlas, every recording'),
 ('3.14', 'Prediction hero, Plaça Catalunya rollout'),
 ('3.15', 'Prediction projections over site plans'),
 ('3.16', 'Best and worst H100 rollouts'),
 ('3.17', 'Best and worst H400 rollouts'),
 ('3.18', 'The platform, start and calibration'),
 ('3.19', 'The studio dashboard'),
 ('3.20', 'Exporting the drawing'),
 ('4.1', 'Limitations of the current method'),
]
def _fig_page(num):
    tag = 'Fig %s</b>' % num
    for i, pg in enumerate(PAGES):
        if tag in pg['body']:
            return i + 1
    return None
def figures_body():
    rows = ''
    for num, title in FIG_TITLES:
        p = _fig_page(num)
        rows += ("<div style='display:flex;justify-content:space-between;align-items:baseline;gap:5mm;"
                 "font-family:Inter;font-size:8.4pt;line-height:11.5pt;'>"
                 "<div><span style='color:var(--mag);font-weight:700;'>Fig&nbsp;%s</span>"
                 "&nbsp;&nbsp;<span style='color:var(--ink);'>%s</span></div>"
                 "<div style='color:var(--grey);font-variant-numeric:tabular-nums;'>%s</div></div>"
                 ) % (num, title, (str(p) if p else ''))
    return rows

ensure_left()
# List of Figures — content distributed to fill the full page height (author correction #26)
add(
    "<div style='height:100%;display:flex;flex-direction:column;'>"
    "  <div><span class='secnum'>&mdash;</span><div class='sectitle'>List of Figures</div></div>"
    "  <div style='flex:1;display:flex;flex-direction:column;justify-content:space-between;padding:9mm 0 4mm;width:97%;'>"
    + figures_body() +
    "  </div>"
    "</div>",
    chapter='List of Figures', vcenter=False)

# Bibliography, single facing page — spaced to fill the page height (author correction #26)
def bib_one(entries):
    rows = ''.join("<p style='margin-bottom:9pt;text-indent:-4.5mm;padding-left:4.5mm;'>%s</p>" % e for e in entries)
    return "<div class='body' style='columns:2;column-gap:8mm;text-align:left;font-size:7.6pt;line-height:12.6pt;'>%s</div>" % rows
add(
    "<span class='secnum'>&mdash;</span>"
    "<div class='sectitle'>Bibliography</div>"
    "<div class='body' style='width:92%;margin:2mm 0 6mm;font-size:7.8pt;line-height:11pt;'><p>References follow the Harvard author-date style. Web resources were last checked in September 2026.</p></div>"
    + bib_one(BIB1 + BIB2),
    chapter='Bibliography', vcenter=False)

# back cover (standalone; reloaded from BOOKLET_images)
add("<img class='fullbleed' src='%s'>" % opt(BKX + r'\back_cover.png', 2200, 94),
    fullbleed=True, header=False, folio=False)

# ================================================================ TOP-ALIGN (corrections #6/#11)
def apply_top_align():
    marks = [
        'feature_ablation/traj_00', 'The Model C Schema', 'esplanade_espanya_01',
        'Endpoint &lt; 1 m', 'What Did the Model Learn', 'Spatial context helps, but conditionally',
        'Limitations of Research', 'limitations_tight', 'The Meaning of It All',
        'Future Research Directions', 'The larger step is dimensional',
    ]
    for pg in PAGES:
        if pg.get('chapter') == 'Conclusion' or any(m in pg['body'] for m in marks):
            pg['vcenter'] = False

# ================================================================ ASSEMBLE + RENDER
import re as _re
_IMG_TOKEN = _re.compile(r'@@IMG\|(.*?)\|(\d+)\|(\d+)@@', _re.DOTALL)
def _resolve_tokens(html):
    return _IMG_TOKEN.sub(lambda m: _realopt(m.group(1), int(m.group(2)), int(m.group(3))), html)

def render_html():
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><style>", CSS, "</style></head><body>"]
    for i, pg in enumerate(PAGES):
        idx = i + 1
        side = 'right' if idx == 1 else ('left' if idx % 2 == 0 else 'right')
        classes = ['page', side]
        if pg['dark']: classes.append('dark')
        if pg['opener']: classes.append('opener')
        if pg['cls']: classes.append(pg['cls'])
        chunk = ["<div class='%s'>" % ' '.join(classes)]
        show_furniture = not pg['fullbleed'] and not pg['opener']
        if pg['header'] and show_furniture:
            if side == 'left':
                chunk.append("<div class='rh rh-l'>MOTION PIXELS</div>")
            else:
                chunk.append("<div class='rh rh-r'>%s</div>" % pg['chapter'])
        if pg['fullbleed'] or pg['opener']:
            chunk.append(pg['body'])
        else:
            inner = pg['body']
            if pg.get('vcenter'):
                inner = "<div style='min-height:100%%;display:flex;flex-direction:column;justify-content:center;'>%s</div>" % inner
            chunk.append("<div class='frame'>%s</div>" % inner)
        if pg['folio'] and show_furniture:
            chunk.append("<div class='folio'>%d</div>" % idx)
        chunk.append("</div>")
        parts.append(''.join(chunk))
    parts.append("</body></html>")
    return _resolve_tokens('\n'.join(parts))

def make_spreads(src_pdf, out_pdf):
    """Reader-spread presentation (NO signature imposition). Front + back covers stand alone."""
    import fitz
    src = fitz.open(str(src_pdf))
    N = src.page_count
    w = src[0].rect.width; h = src[0].rect.height
    out = fitz.open()
    def sheet(left, right):
        pg = out.new_page(width=2*w, height=h)
        if left:  pg.show_pdf_page(fitz.Rect(0, 0, w, h),     src, left-1)
        if right: pg.show_pdf_page(fitz.Rect(w, 0, 2*w, h),   src, right-1)
    sheet(None, 1)                      # front cover alone (recto)
    i = 2
    while i <= N:
        if i == N:                      # back cover alone (last page)
            sheet(i, None); i += 1
        else:
            sheet(i, i+1); i += 2
    out.save(str(out_pdf)); out.close(); src.close()
    return out.page_count if False else None

if __name__ == '__main__':
    from playwright.sync_api import sync_playwright
    import fitz

    def build(profile, html_name, pdf_name, raster=False):
        global PROFILE
        PROFILE = profile
        html = render_html()
        hp = HERE / html_name
        hp.write_text(html, encoding='utf-8')
        pdf = HERE / pdf_name
        with sync_playwright() as p:
            b = p.chromium.launch(); pw = b.new_page()
            pw.goto(hp.as_uri()); pw.emulate_media(media='print'); pw.wait_for_timeout(1400)
            pw.pdf(path=str(pdf), width='210mm', height='297mm', print_background=True,
                   margin={'top': '0', 'bottom': '0', 'left': '0', 'right': '0'}, prefer_css_page_size=True)
            b.close()
        doc = fitz.open(str(pdf))
        print('%s: %d pages, %.1f MB' % (pdf_name, doc.page_count, pdf.stat().st_size/1e6))
        if raster:
            qc = HERE / 'qc'; qc.mkdir(exist_ok=True)
            for i, page in enumerate(doc):
                page.get_pixmap(matrix=fitz.Matrix(1.15, 1.15)).save(str(qc / ('p%03d.png' % (i+1))))
        doc.close()
        return pdf

    print('pages:', len(PAGES))
    # A. print-quality, individual sequential A4 pages (spiral binding)
    indiv = build('print', 'book_print.html', 'MOTION_PIXELS_FINAL_PRINT_INDIVIDUAL.pdf', raster=True)
    # B. print-quality facing spreads (covers standalone)
    make_spreads(indiv, HERE / 'MOTION_PIXELS_FINAL_PRINT_SPREADS.pdf')
    print('MOTION_PIXELS_FINAL_PRINT_SPREADS.pdf: %.1f MB' % ((HERE/'MOTION_PIXELS_FINAL_PRINT_SPREADS.pdf').stat().st_size/1e6))
    # C. digital, slightly compressed, facing spreads (covers standalone)
    dig = build('digital', 'book_digital.html', 'MOTION_PIXELS_FINAL_DIGITAL_SEQ.pdf')
    make_spreads(dig, HERE / 'MOTION_PIXELS_FINAL_DIGITAL_SPREADS.pdf')
    print('MOTION_PIXELS_FINAL_DIGITAL_SPREADS.pdf: %.1f MB' % ((HERE/'MOTION_PIXELS_FINAL_DIGITAL_SPREADS.pdf').stat().st_size/1e6))
