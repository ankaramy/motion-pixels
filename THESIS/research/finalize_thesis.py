"""Assemble completed chapter files and produce transparent manuscript QA."""
from pathlib import Path
import re, json, collections

ROOT = Path(__file__).resolve().parents[1]
targets = {
 '01_the_flaw_in_the_plan.md': (2000,[850,500,350,300]),
 '02_tools_and_instruments_for_behavioral_analysis.md': (3000,[450,700,350,350,800,350]),
 '03_motion_pixels_learning_movement_from_barcelona.md': (6000,[2000,1250,950,1000,800]),
 '04_what_comes_next.md': (2500,[650,650,600,600]),
 '05_conclusion_and_closing.md': (600,[]),
}

def clean(text):
    text=re.sub(r'\[(?:AUTHOR INPUT REQUIRED|CITATION REQUIRED)[\s\S]*?\]', '',text)
    text=re.sub(r'!?\[([^\]]+)\]\([^)]*\)',r'\1',text)
    return '\n'.join(l for l in text.splitlines() if not l.startswith(('#','|','>')))

def count(text):
    return len(clean(text).split())

chapters=sorted((ROOT/'chapters').glob('*.md'))
counts={}
for p in chapters:
    t=p.read_text(encoding='utf8');target,subtargets=targets[p.name]
    sections=[]
    for s,st in zip(re.split(r'(?m)^## ',t)[1:],subtargets):
        title,body=s.split('\n',1);n=count(body)
        sections.append({'section':title,'target':st,'actual':n,'within_5_percent':abs(n/st-1)<=.05})
    n=count(t)
    counts[p.name]={'target':target,'actual':n,'within_5_percent':abs(n/target-1)<=.05,'sections':sections}
abstract=ROOT/'front_matter/01_abstract.md'
counts['abstract']={'target':300,'actual':count(abstract.read_text(encoding='utf8'))}

index='''# Index

- Abstract
- Chapter 1: The flaw in the plan
- Chapter 2: Tools + Instruments for Behavioral Analysis
- Chapter 3: Motion Pixels, learning movement from Barcelona
- Chapter 4: What comes next?
- Conclusion + Closing
- Bibliography

The chapter files are the editable sources. Project source labels P01-P12 resolve in the bibliography. Figure provenance and original graph links are available in [FIGURE_REGISTER.md](FIGURE_REGISTER.md). Page numbers will depend on final book layout.
'''
(ROOT/'front_matter/02_index.md').write_text(index,encoding='utf8')
paths=[ROOT/'front_matter/00_cover_and_declaration.md',abstract,ROOT/'front_matter/02_index.md']+chapters+[ROOT/'BIBLIOGRAPHY.md']
parts=[p.read_text(encoding='utf8').strip() for p in paths]
master='\n\n---\n\n'.join(parts)+'\n'
(ROOT/'MASTER_THESIS.md').write_text(master,encoding='utf8')

patterns=[r'\bThis highlights\b',r'\bThis demonstrates\b',r'\bThis underscores\b',
 r'\bIt is important\b',r'\bIt is worth noting\b',r'\bFurthermore\b',r'\bMoreover\b',
 r'\bAdditionally\b',r'\bUltimately\b',r'\bBy leveraging\b',r'\bvaluable insights\b',
 r'\binnovative\b',r'\bsignificant advancement\b',r'\bnot only\b']
matches={p:re.findall(p,master,re.I) for p in patterns}
body='\n'.join(p.read_text(encoding='utf8') for p in chapters)
paragraphs=[p.strip() for p in body.split('\n\n') if p.strip() and not p.startswith(('#','|'))]
duplicates=[p for p,n in collections.Counter(paragraphs).items() if n>1]
openings=collections.Counter(' '.join(p.split()[:3]) for p in paragraphs)
issues=[]
for p in paths:
    for m in re.finditer(r'\[(?:AUTHOR INPUT REQUIRED|CITATION REQUIRED)[\s\S]*?\]',p.read_text(encoding='utf8')):
        issues.append({'file':p.relative_to(ROOT).as_posix(),'marker':m.group(0)})
report={'word_counts':counts,'prose_total':sum(v['actual'] for v in counts.values()),
        'em_dashes':master.count(chr(8212)),'style_pattern_matches':{k:v for k,v in matches.items() if v},
        'duplicate_paragraphs':duplicates,'common_paragraph_openings':openings.most_common(12),
        'placeholders':issues}
(ROOT/'research/manuscript_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps({'counts':counts,'prose_total':report['prose_total'],'em_dashes':report['em_dashes'],
 'style_pattern_matches':report['style_pattern_matches'],'duplicate_paragraphs':len(duplicates),
 'openings':report['common_paragraph_openings'],'placeholders':len(issues)},indent=2))
