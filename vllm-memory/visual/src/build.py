"""Builds ../index.html (vLLM Memory, Visualized) from the parts in this folder and the notebook one level up.
Run: python vllm-memory/visual/src/build.py"""
import json, re, html
import os
d=os.path.dirname(os.path.abspath(__file__))+'/'  # visual/src/
style=open(d+'_style.css').read(); helpers=open(d+'_helpers.js').read(); tail=open(d+'_tail.js').read(); defs=open(d+'_defs.svg').read()
body=open(d+'_body.html').read().replace('<div class="notation">','<div class="notation words">',1)
main=open(d+'_main.js').read(); sim=open(d+'simdata.json').read(); pycode=open(d+'notebook_core.py').read()
nb=json.load(open(d+'../../vllm_memory_explainer.ipynb'));c=nb['cells'];src=lambda i:''.join(c[i]['source'])
cells=json.load(open(d+'cells.json'))
code={
 'setup':cells['setup'],
 'weights':'\n'.join(src(5).split('\n')[:2]),
 'kv':cells['kv'],
 'pool':cells['pool'],
 'blocks':cells['blocks'],
 'request':cells['request'],
 'sim':src(27)[:src(27).index('log, pre, snaps')].strip(),
 'limits':'\n'.join(src(31).split('\n')[:2]),
 'speed':'\n'.join(src(38).split('\n')[:4]),
}
extra='''
.notation.words{display:block}
.notation.words dl{grid-template-columns:120px 1fr;gap:8px 14px}
.notation.words dt{font:700 14px var(--f-mono)}
.excards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.excard{background:var(--surface-2);border-radius:14px;padding:14px 16px;border-top:4px solid var(--k)}
.excard>b{font:800 16px var(--f-display);display:block;margin-bottom:8px}
.excard dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:5px 12px;font-size:13px}
.excard dt{color:var(--muted);font-family:var(--f-mono);font-size:11.5px;padding-top:2px}
.excard dd{margin:0;font-weight:600}
@media (max-width:900px){.excards{grid-template-columns:1fr}}
details.code{margin-top:12px;background:var(--surface);border:1px solid var(--line);border-radius:10px}
details.code summary{cursor:pointer;padding:8px 12px;font:600 12.5px var(--f-mono);color:var(--accent)}
details.code pre{margin:0;padding:10px 12px;font:12px/1.5 var(--f-mono);overflow:auto;max-height:360px;border-top:1px solid var(--line)}
pre.loop{font:12.5px/1.55 var(--f-mono);background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;white-space:pre-wrap;margin:10px 0}
pre.log{font:13px/1.6 var(--f-mono);background:#0d1220;color:#d6e2ff;border-radius:10px;padding:12px 14px;white-space:pre-wrap;overflow:auto;margin:12px 0 0}
#py-code{width:100%;min-height:380px;font:12.5px/1.5 var(--f-mono);background:var(--surface-2);color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:12px;resize:vertical;tab-size:4}
.lbl-h{font:600 11px var(--f-mono);letter-spacing:.08em;color:var(--muted);margin:12px 0 6px}
#sc-grid{width:100%;height:auto;display:block;border-radius:6px}
code{font-family:var(--f-mono);font-size:.9em;background:var(--surface-2);padding:.08em .35em;border-radius:4px}
'''
style=style.replace('footer{max-width',extra+'footer{max-width',1)
page=f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vLLM Memory, Visualized</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<style>{style}</style>
</head>
<body>
{defs}
{body}
<script>
const SIM={sim};
const PYCODE={json.dumps(pycode)};
const CODE={json.dumps(code)};
{helpers}
{main}
{tail}
</script>
</body>
</html>'''
open(d+'../index.html','w').write(page)
print(len(page))
