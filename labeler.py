#!/usr/bin/env python3
"""PiMorph Labeler v2: retro-IDE web app for AJ morphology analysis and labeling.

Usage:
    python labeler.py                          # localhost:5050
    python labeler.py --port 8080              # custom port
    python labeler.py --admin-key MY_SECRET    # set admin key for data export
"""
import os,sys,csv,json,hashlib,uuid,sqlite3,io,base64,glob,time
from pathlib import Path
from datetime import datetime
from functools import wraps
from flask import (Flask,render_template_string,request,jsonify,send_file,
                   redirect,url_for,session,flash,abort,Response)
import argparse

app=Flask(__name__)
app.secret_key=os.environ.get("FLASK_SECRET_KEY","pimorph-labeler-default-key-2026")
app.config["SESSION_COOKIE_SAMESITE"]="Lax"
app.config["SESSION_COOKIE_SECURE"]=False
app.config["SESSION_COOKIE_HTTPONLY"]=True
PROJ=Path(__file__).parent
DATA_DIR=Path(os.environ.get("PIMORPH_DATA_DIR",str(PROJ)))
DATA_DIR.mkdir(parents=True,exist_ok=True)
DB_PATH=DATA_DIR/"labeler_data.db"
CLASSES=["straight","reticular","fingers","thick","thick_to_reticular","other"]
CMAP={"straight":"#4CAF50","reticular":"#FF9800","fingers":"#E91E63","thick":"#2196F3","thick_to_reticular":"#00BCD4","other":"#795548"}
ABBR={"straight":"STR","reticular":"RET","fingers":"FIN","thick":"THK","thick_to_reticular":"T2R","other":"OTH"}
ADMIN_KEY=os.environ.get("PIMORPH_ADMIN_KEY","pimorph2026")

def get_db():
    db=sqlite3.connect(str(DB_PATH))
    db.row_factory=sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    return db

def init_db():
    db=get_db()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS sessions(
        id TEXT PRIMARY KEY,
        created TEXT,
        user_agent TEXT,
        ip TEXT,
        dataset TEXT,
        notes TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS labels(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        image_id TEXT,
        cell_i INTEGER,
        cell_j INTEGER,
        label TEXT,
        auto_label TEXT,
        timestamp TEXT,
        UNIQUE(session_id,image_id,cell_i,cell_j)
    );
    CREATE TABLE IF NOT EXISTS uploads(
        id TEXT PRIMARY KEY,
        session_id TEXT,
        filename TEXT,
        created TEXT,
        status TEXT DEFAULT 'pending',
        n_patches INTEGER DEFAULT 0
    );
    """)
    db.commit()
    db.close()

init_db()

def _resolve_run(run,*sub):
    p=DATA_DIR/"runs"/run/Path(*sub) if sub else DATA_DIR/"runs"/run
    if p.exists(): return p
    p=PROJ/"runs"/run/Path(*sub) if sub else PROJ/"runs"/run
    return p

def discover_datasets():
    runs=PROJ/"runs"
    found=[]
    seen=set()
    if runs.exists():
        for idx in sorted(runs.rglob("patch_index.csv")):
            rel=str(idx.relative_to(PROJ))
            name=str(idx.parent.parent.name)
            if name in seen: continue
            seen.add(name)
            n=sum(1 for _ in open(idx))-1
            found.append({"path":rel,"name":name,"count":n})
    return found

def load_demo_datasets():
    datasets=[]
    skip={"blur_robust_improvement","junction_mapper_comparison","cross_dataset_validation"}
    dirs=set()
    for runs in [PROJ/"runs",DATA_DIR/"runs"]:
        if not runs.exists(): continue
        for run_dir in sorted(runs.iterdir()):
            dirs.add(run_dir)
    for run_dir in sorted(dirs):
        if not run_dir.is_dir() or run_dir.name in skip: continue
        images=[]
        total_edges=0
        morph_agg={}
        for d in sorted(run_dir.iterdir()):
            if not d.is_dir() or d.name=="patches": continue
            edges_f=d/"edges.csv"
            if not edges_f.exists(): continue
            import pandas as pd
            try:
                edf=pd.read_csv(edges_f)
            except Exception:
                continue
            n=len(edf)
            if n==0: continue
            mc=next((c for c in ("aj_morph","AJ_morph_label") if c in edf.columns),None)
            md=edf[mc].value_counts().to_dict() if mc else {}
            total_edges+=n
            for k,v in md.items(): morph_agg[k]=morph_agg.get(k,0)+v
            images.append({"id":d.name,"run":run_dir.name,"n_edges":n,"morph_dist":md,
                          "has_qc_cells":(d/"qc_cells.png").exists(),
                          "has_qc_graph":(d/"qc_graph.png").exists(),
                          "has_graph":(d/"graph.json").exists()})
        if images:
            datasets.append({"name":run_dir.name,"images":images,"n_images":len(images),
                            "n_edges":total_edges,"morph_agg":morph_agg})
    return datasets

def load_patches(csv_path):
    rows=[]
    full=PROJ/csv_path
    if not full.exists(): return rows
    with open(full) as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            r["_abs"]=str(PROJ/r["patch_path"])
            rows.append(r)
    return rows

def get_or_create_session():
    sid=session.get("sid")
    if sid:
        db=get_db()
        row=db.execute("SELECT id FROM sessions WHERE id=?",(sid,)).fetchone()
        db.close()
        if row: return sid
    sid=str(uuid.uuid4())[:12]
    session["sid"]=sid
    db=get_db()
    db.execute("INSERT OR IGNORE INTO sessions(id,created,user_agent,ip,dataset) VALUES(?,?,?,?,?)",
               (sid,datetime.utcnow().isoformat(),str(request.headers.get("User-Agent",""))[:200],
                request.remote_addr,""))
    db.commit()
    db.close()
    return sid

def get_session_labels(sid,dataset=""):
    db=get_db()
    rows=db.execute("SELECT image_id,cell_i,cell_j,label FROM labels WHERE session_id=?",(sid,)).fetchall()
    db.close()
    out={}
    for r in rows:
        k=f"{r['image_id']}__{r['cell_i']}__{r['cell_j']}"
        out[k]=r["label"]
    return out

def save_label(sid,image_id,cell_i,cell_j,label,auto_label=""):
    db=get_db()
    db.execute("""INSERT OR REPLACE INTO labels(session_id,image_id,cell_i,cell_j,label,auto_label,timestamp)
                  VALUES(?,?,?,?,?,?,?)""",
               (sid,image_id,int(cell_i),int(cell_j),label,auto_label,datetime.utcnow().isoformat()))
    db.commit()
    db.close()

CSS=r"""
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#2b2b2b;--bg2:#3c3d3f;--bg3:#313335;--border:#515151;--text:#a9b7c6;--accent:#cc7832;--accent2:#ffc66d;--blue:#6897bb;--green:#6a8759;--red:#bc3f3c;--purple:#9876aa;--sel:#214283;--gutter:#606366}
body{background:var(--bg);color:var(--text);font-family:'JetBrains Mono',Consolas,'Courier New',monospace;font-size:13px;height:100vh;display:flex;flex-direction:column;overflow:hidden}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
.titlebar{background:#3c3d3f;border-bottom:1px solid var(--border);padding:2px 12px;display:flex;align-items:center;gap:12px;height:28px;font-size:11px;color:#999}
.titlebar .proj{color:var(--text);font-weight:bold}
.menubar{background:var(--bg3);border-bottom:1px solid var(--border);padding:0 8px;display:flex;align-items:center;height:26px;gap:0;font-size:12px;position:relative;z-index:100}
.menu-item{position:relative;display:inline-block}
.menu-item>span{padding:3px 10px;cursor:pointer;color:var(--text);display:block}
.menu-item:hover>span{background:var(--sel)}
.menu-drop{display:none;position:absolute;top:100%;left:0;background:var(--bg2);border:1px solid var(--border);min-width:200px;box-shadow:0 4px 12px rgba(0,0,0,0.5);z-index:200}
.menu-item:hover .menu-drop{display:block}
.menu-drop a,.menu-drop span{display:flex;justify-content:space-between;align-items:center;padding:5px 12px 5px 20px;color:var(--text);font-size:12px;cursor:pointer;white-space:nowrap;gap:20px}
.menu-drop a:hover,.menu-drop span:hover{background:var(--sel);text-decoration:none}
.menu-drop .sep{height:1px;background:var(--border);margin:2px 0;padding:0;cursor:default;display:block}
.menu-drop .shortcut{color:var(--gutter);font-size:11px;flex-shrink:0}
.toolbar{background:var(--bg3);border-bottom:1px solid var(--border);padding:4px 12px;display:flex;align-items:center;gap:6px;height:32px}
.tbtn{background:var(--bg2);border:1px solid var(--border);color:var(--text);padding:2px 10px;font-size:11px;cursor:pointer;font-family:inherit;border-radius:2px}
.tbtn:hover{background:var(--sel);border-color:var(--blue)}
.sep-v{width:1px;height:18px;background:var(--border);margin:0 4px}
.toolbar .info{color:var(--gutter);font-size:11px;margin-left:auto}
.main{flex:1;display:flex;overflow:hidden}
.sidebar{width:260px;background:var(--bg3);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;position:relative}
.stitle{padding:6px 10px;font-size:11px;color:var(--gutter);text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--border);background:var(--bg2);display:flex;justify-content:space-between;align-items:center}
.slist{flex:1;overflow-y:auto;font-size:11px}
.sitem{padding:4px 10px 4px 16px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;border-left:3px solid transparent;overflow:hidden}
.sitem>span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}
.sitem:hover{background:rgba(255,255,255,0.05)}
.sitem.current{background:var(--sel);border-left-color:var(--blue)}
.slbl{padding:1px 5px;border-radius:2px;font-size:9px;text-transform:uppercase;letter-spacing:0.5px}
.center{flex:1;display:flex;flex-direction:column;overflow:hidden}
.tabs{background:var(--bg2);border-bottom:1px solid var(--border);display:flex;height:28px;align-items:stretch;overflow-x:auto}
.tab{padding:0 14px;display:flex;align-items:center;font-size:12px;border-right:1px solid var(--border);cursor:pointer;color:var(--gutter);white-space:nowrap}
.tab.active{background:var(--bg);color:var(--text);border-bottom:2px solid var(--blue)}
.tab:hover:not(.active){background:rgba(255,255,255,0.03)}
.editor{flex:1;display:flex;align-items:center;justify-content:center;overflow:auto;position:relative;padding:16px}
.right-panel{width:220px;background:var(--bg3);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow-y:auto;position:relative}
.stats .row{display:flex;justify-content:space-between;padding:3px 12px;border-bottom:1px solid rgba(255,255,255,0.04);font-size:11px}
.stats .row .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
.stats .row .cnt{color:var(--blue);font-weight:bold}
.progress-bar{height:6px;background:var(--bg);border-radius:3px;overflow:hidden;margin:4px 12px}
.progress-bar .fill{height:100%;background:var(--green);transition:width 0.3s}
.ptext{font-size:10px;color:var(--gutter);padding:2px 12px 8px}
.statusbar{background:var(--sel);border-top:1px solid var(--border);padding:0 12px;height:22px;display:flex;align-items:center;justify-content:space-between;font-size:11px}
.statusbar .mode{background:var(--accent);color:#fff;padding:0 8px;font-size:10px;font-weight:bold;margin-right:8px}
.patch-view{display:flex;flex-direction:column;align-items:center;gap:12px;max-width:100%}
.patch-view img{image-rendering:pixelated;border:2px solid var(--border);background:#1e1e1e;max-width:min(45vw,420px);max-height:min(45vh,420px)}
.patch-info{text-align:center;color:var(--gutter);font-size:11px;line-height:1.7}
.class-bar{display:flex;gap:4px;flex-wrap:wrap;justify-content:center;max-width:600px}
.cbtn{padding:5px 12px;border:1px solid var(--border);background:var(--bg2);color:var(--text);cursor:pointer;font-family:inherit;font-size:11px;border-radius:3px;transition:all 0.1s;border-left:3px solid var(--border)}
.cbtn:hover{border-color:var(--cc);color:var(--cc);background:var(--bg)}
.cbtn.selected{border-color:var(--cc);color:var(--cc);background:var(--bg)}
.cbtn .key{color:var(--cc);font-weight:bold;margin-right:3px}
.nav-hint{position:absolute;bottom:8px;right:8px;font-size:10px;color:var(--gutter);background:rgba(0,0,0,0.5);padding:3px 8px;border-radius:3px}
.splash{display:flex;flex-direction:column;align-items:center;gap:12px;padding:40px;overflow-y:auto}
.splash h1{color:var(--accent);font-size:24px;font-weight:normal}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:16px;margin:8px 0;width:100%;max-width:500px}
.card h3{color:var(--accent2);font-size:13px;margin-bottom:8px;font-weight:normal}
.card p{font-size:11px;color:var(--gutter);line-height:1.5}
.run-link{display:block;padding:5px 12px;color:var(--blue);cursor:pointer;font-size:12px;margin:2px 0}
.run-link:hover{background:rgba(255,255,255,0.05);border-radius:2px}
.demo-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;padding:8px 10px}
.demo-card{background:var(--bg);border:1px solid var(--border);border-radius:3px;padding:8px;cursor:pointer;font-size:10px;overflow:hidden}
.demo-card:hover{border-color:var(--blue)}
.demo-card .name{color:var(--accent2);font-size:11px;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.demo-card .info{color:var(--gutter);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.qc-view{display:flex;flex-direction:column;gap:12px;align-items:center;padding:20px}
.qc-view img{max-width:90%;border:1px solid var(--border);border-radius:3px}
.filter-bar{padding:4px 10px;border-bottom:1px solid var(--border);display:flex;gap:3px;flex-wrap:wrap}
.fbtn{font-size:9px;padding:1px 5px;border:1px solid var(--border);background:var(--bg);color:var(--gutter);cursor:pointer;border-radius:2px;font-family:inherit}
.fbtn:hover,.fbtn.active{border-color:var(--cc);color:var(--cc)}
.upload-zone{position:relative;border:2px dashed var(--border);border-radius:8px;padding:40px 24px;text-align:center;cursor:pointer;transition:all 0.3s cubic-bezier(0.4,0,0.2,1);overflow:hidden}
.upload-zone:hover{border-color:var(--blue);background:rgba(104,151,187,0.05)}
.upload-zone.drag{border:2px dashed var(--accent);background:rgba(204,120,50,0.08)}
.upload-zone.drag-over{border:2px solid var(--accent);background:rgba(204,120,50,0.1)}
.upload-zone .drag-halo{position:absolute;width:200px;height:14px;background:var(--accent);filter:blur(28px);border-radius:7px;opacity:0;pointer-events:none;transform:translate(-50%,-50%);transition:opacity 0.3s cubic-bezier(0.4,0,0.2,1);z-index:2}
.upload-zone.drag-over .drag-halo{opacity:1}
.upload-zone .drop-icon{font-size:28px;margin-bottom:8px;opacity:0.4;transition:all 0.3s}
.upload-zone:hover .drop-icon,.upload-zone.drag .drop-icon{opacity:0.8}
.upload-zone.drag-over .drop-icon{opacity:1;transform:scale(1.15)}
.upload-zone .drop-text{color:var(--gutter);font-size:11px;transition:color 0.3s}
.upload-zone:hover .drop-text{color:var(--text)}
.upload-zone.drag-over .drop-text{color:var(--accent2)}
.upload-zone .drop-or{color:var(--border);font-size:10px;margin:8px 0}
.upload-zone .file-info{margin-top:8px;color:var(--green);font-size:11px;opacity:0;transform:translateY(6px);transition:all 0.2s}
.upload-zone .file-info.show{opacity:1;transform:translateY(0)}
.modal-bg{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.6);z-index:500;display:flex;align-items:center;justify-content:center}
.modal{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:20px;min-width:300px;max-width:500px}
.modal h3{color:var(--accent2);margin-bottom:12px;font-weight:normal;font-size:14px}
.modal input[type=text],.modal input[type=file]{background:var(--bg);border:1px solid var(--border);color:var(--text);padding:6px 10px;font-family:inherit;font-size:12px;width:100%;margin:4px 0 8px;border-radius:2px}
.modal input[type=file]::-webkit-file-upload-button{background:var(--bg2);color:var(--text);border:1px solid var(--border);padding:4px 12px;font-family:inherit;font-size:11px;cursor:pointer;border-radius:2px;margin-right:8px}
.modal input[type=file]::-webkit-file-upload-button:hover{background:var(--sel);border-color:var(--blue)}
.disabled{color:var(--border)!important;pointer-events:none}
.drag-handle{width:4px;cursor:col-resize;background:transparent;position:absolute;top:0;bottom:0;z-index:50;transition:background 0.15s}
.drag-handle:hover,.drag-handle.active{background:var(--blue)}
.modal .actions{display:flex;gap:8px;justify-content:flex-end;margin-top:12px}
"""

def render_page(page,**kw):
    kw.update(page=page,classes=CLASSES,cmap=CMAP,ABBR=ABBR,enumerate=enumerate,sid=session.get("sid",""))
    return render_template_string(TMPL,**kw)

TMPL=r'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PiMorph Labeler</title>
<style>''' + CSS + r'''</style></head><body>
<div class="titlebar">
<span class="proj">PiMorph Labeler</span>
<a href="https://github.com/okezue/EndoPiGraph-AJmorph" target="_blank" style="color:var(--text);text-decoration:none">EndoPiGraph-AJmorph</a>
<span style="margin-left:auto;color:var(--green)">session: {{ sid[:8] }}</span>
</div>
<div class="menubar">
<div class="menu-item"><span>File</span><div class="menu-drop">
<a href="/">Home<span class="shortcut">Ctrl+H</span></a>
<span onclick="document.getElementById('uploadModal').style.display='flex'">Upload Image...<span class="shortcut">Ctrl+U</span></span>
<div class="sep"></div>
<a href="/export_labels">Export Labels CSV<span class="shortcut">Ctrl+E</span></a>
<a href="/export_all">Export All Data<span class="shortcut">Ctrl+Shift+E</span></a>
</div></div>
<div class="menu-item"><span>Edit</span><div class="menu-drop">
<span {% if page!='label' %}class="disabled"{% endif %} onclick="undo()">Undo Last Label<span class="shortcut">Ctrl+Z</span></span>
<span {% if page!='label' %}class="disabled"{% endif %} onclick="assign('unknown')">Mark Unknown<span class="shortcut">U</span></span>
<div class="sep"></div>
<span {% if page!='label' %}class="disabled"{% endif %} onclick="clearLabel()">Clear Current Label<span class="shortcut">Del</span></span>
</div></div>
<div class="menu-item"><span>View</span><div class="menu-drop">
<a href="/demo">Demo Dataset (S-BIAD1540)<span class="shortcut">Ctrl+D</span></a>
<div class="sep"></div>
<span onclick="togglePanel('sidebar')">Toggle Sidebar<span class="shortcut">Ctrl+1</span></span>
<span onclick="togglePanel('right-panel')">Toggle Statistics<span class="shortcut">Ctrl+2</span></span>
<div class="sep"></div>
<span onclick="zoomIn()">Zoom In<span class="shortcut">+</span></span>
<span onclick="zoomOut()">Zoom Out<span class="shortcut">-</span></span>
</div></div>
<div class="menu-item"><span>Navigate</span><div class="menu-drop">
<span {% if page!='label' %}class="disabled"{% endif %} onclick="nav(-1)">Previous Patch<span class="shortcut">&larr;</span></span>
<span {% if page!='label' %}class="disabled"{% endif %} onclick="nav(1)">Next Patch<span class="shortcut">&rarr;</span></span>
<div class="sep"></div>
<span {% if page!='label' %}class="disabled"{% endif %} onclick="skip_labeled(-1)">Previous Unlabeled<span class="shortcut">Shift+&larr;</span></span>
<span {% if page!='label' %}class="disabled"{% endif %} onclick="skip_labeled(1)">Next Unlabeled<span class="shortcut">Shift+&rarr;</span></span>
<div class="sep"></div>
<span {% if page!='label' %}class="disabled"{% endif %} onclick="gotoIdx()">Go to Patch #...<span class="shortcut">Ctrl+G</span></span>
</div></div>
<div class="menu-item"><span>Run</span><div class="menu-drop">
<span onclick="document.getElementById('uploadModal').style.display='flex'">Run PiMorph on Image...<span class="shortcut">Ctrl+R</span></span>
<a href="/demo">View Demo Results</a>
</div></div>
<div class="menu-item"><span>Tools</span><div class="menu-drop">
<a href="/export_labels">Download Labels</a>
</div></div>
<div class="menu-item"><span>Help</span><div class="menu-drop">
<span onclick="showHelp()">Keyboard Shortcuts<span class="shortcut">?</span></span>
<div class="sep"></div>
<span style="color:var(--gutter)">PiMorph Labeler v2.0</span>
</div></div>
</div>
{% if page=='home' %}
<div class="toolbar"><span style="color:var(--gutter)">Welcome to PiMorph Labeler</span></div>
<div class="main"><div class="splash" style="width:100%">
<h1>// PiMorph Labeler</h1>
<p style="color:var(--gutter)">AJ morphology analysis and annotation</p>
<div class="card"><h3>Demo Dataset (S-BIAD1540)</h3><p>16 endothelial images, 3,520 patches pre-processed with PiMorph.</p><p style="margin-top:6px"><a href="/demo" class="tbtn" style="display:inline-block;text-decoration:none">View Demo &rarr;</a></p></div>
<div class="card"><h3>Label Patches</h3><p>Review PiMorph classifications and manually annotate AJ morphology.</p>
{% for ds in datasets %}<a class="run-link" href="/label_view?load={{ ds.path }}">{{ ds.name }} ({{ ds.count }} patches)</a>{% endfor %}
{% if not datasets %}<p style="color:var(--red);margin-top:4px">No datasets found.</p>{% endif %}
</div>
<div class="card"><h3>Upload &amp; Process</h3><p>Upload a TIFF and run the PiMorph pipeline.</p><p style="margin-top:6px"><button class="tbtn" onclick="document.getElementById('uploadModal').style.display='flex'">Upload Image...</button></p>
{% if active_uploads %}<div style="margin-top:8px;border-top:1px solid var(--border);padding-top:8px"><p style="font-size:10px;color:var(--accent);margin-bottom:4px">Processing:</p>
{% for u in active_uploads %}<a class="run-link" href="/upload_status?uid={{ u.id }}&run=upload_{{ u.id }}&filename={{ u.filename.rsplit('.',1)[0] }}" style="color:var(--accent)">{{ u.filename }} &mdash; processing...</a>{% endfor %}</div>{% endif %}
{% if recent_uploads %}<div style="margin-top:8px;border-top:1px solid var(--border);padding-top:8px"><p style="font-size:10px;color:var(--green);margin-bottom:4px">Recent:</p>
{% for u in recent_uploads %}<a class="run-link" href="/demo?run=upload_{{ u.id }}&img={{ u.filename.rsplit('.',1)[0] }}">{{ u.filename }} &mdash; done</a>{% endfor %}</div>{% endif %}
</div>
</div></div>
{% elif page=='demo' %}
<div class="toolbar"><a href="/" class="tbtn">Home</a><div class="sep-v"></div>
{% if run %}<a href="/demo" class="tbtn">All Datasets</a><div class="sep-v"></div><span style="color:var(--accent2)">{{ run }}</span><span style="color:var(--gutter)">&nbsp; {{ images|length }} images, {{ cur_ds.n_edges|default(0) }} interfaces</span>
{% else %}<span style="color:var(--gutter)">All PiMorph datasets</span>{% endif %}
</div>
<div class="main">
<div class="sidebar"><div class="stitle">{% if run %}{{ run }}{% else %}Datasets{% endif %}</div><div class="slist">
{% if not run %}
{% for ds in all_ds %}<div class="sitem" onclick="window.location='/demo?run={{ ds.name }}'"><span>{{ ds.name }}</span><span style="color:var(--blue);font-size:10px">{{ ds.n_images }} img</span></div>{% endfor %}
{% else %}
{% for img in images %}<div class="sitem {% if sel_img==img.id %}current{% endif %}" onclick="window.location='/demo?run={{ run }}&img={{ img.id }}'"><span style="font-size:10px">{{ img.id }}</span><span style="color:var(--blue);font-size:9px">{{ img.n_edges }}</span></div>{% endfor %}
{% endif %}
</div></div>
<div class="center"><div class="tabs">
{% if not run %}<div class="tab active">All Datasets</div>
{% elif not sel_img %}<div class="tab active">{{ run }}</div>
{% else %}<div class="tab" onclick="window.location='/demo?run={{ run }}'">{{ run }}</div>
<div class="tab active">{{ sel_img }}</div>
{% endif %}
</div><div class="editor">
{% if not run %}
<div style="text-align:center;max-width:800px"><p style="color:var(--accent2);font-size:16px;margin-bottom:16px">PiMorph Processed Datasets</p>
<div class="demo-grid">{% for ds in all_ds %}<div class="demo-card" onclick="window.location='/demo?run={{ ds.name }}'">
<div class="name">{{ ds.name }}</div><div class="info">{{ ds.n_images }} images, {{ ds.n_edges }} interfaces</div>
{% if ds.morph_agg %}<div class="info" style="margin-top:4px">{% for k,v in ds.morph_agg.items() %}{{ k[:3] }}:{{ v }} {% endfor %}</div>{% endif %}
</div>{% endfor %}</div></div>
{% elif not sel_img %}
<div style="text-align:center;max-width:700px"><p style="color:var(--accent2);font-size:14px;margin-bottom:8px">{{ run }}</p>
<p style="color:var(--gutter);font-size:11px;margin-bottom:16px">{{ images|length }} images, {{ cur_ds.n_edges }} total interfaces</p>
{% if cur_ds.morph_agg %}<div style="margin-bottom:16px">{% for k,v in cur_ds.morph_agg.items() %}<div class="stats"><div class="row"><span><span class="dot" style="background:{{ cmap.get(k,'#999') }}"></span>{{ k }}</span><span class="cnt">{{ v }}</span></div></div>{% endfor %}</div>{% endif %}
<div class="demo-grid">{% for img in images %}<div class="demo-card" onclick="window.location='/demo?run={{ run }}&img={{ img.id }}'"><div class="name">{{ img.id }}</div><div class="info">{{ img.n_edges }} interfaces</div></div>{% endfor %}</div></div>
{% elif sel_img %}
<div style="display:flex;flex-direction:column;align-items:center;width:100%;height:100%;overflow:auto;padding:8px">
<div style="display:flex;gap:10px;align-items:center;margin-bottom:4px;flex-wrap:wrap;font-size:11px">
<label style="color:var(--gutter)">View:</label>
<label style="cursor:pointer"><input type="checkbox" id="chkSeg" checked onchange="updateView()"> Segmentation</label>
<label style="cursor:pointer"><input type="checkbox" id="chkGraph" checked onchange="updateLayers()"> &pi;-graph</label>
<span style="color:var(--border)">|</span>
<label style="color:var(--gutter)">Edge:</label>
<input type="range" id="sliderEdge" min="0.5" max="6" value="2" step="0.5" style="width:60px" oninput="tweakWidths()">
<label style="color:var(--gutter)">Node:</label>
<input type="range" id="sliderNode" min="1" max="10" value="4" step="1" style="width:60px" oninput="tweakWidths()">
<span style="color:var(--border)">|</span>
<span style="color:var(--gutter)">Download:</span>
<button class="tbtn" onclick="downloadView()" style="font-size:10px;padding:1px 8px">View (SVG+img)</button>
<a class="tbtn" href="/api/cropped_bg?run={{ run }}&img={{ sel_img }}&mode=seg" download="{{ sel_img }}_segmentation.png" style="font-size:10px;padding:1px 8px;text-decoration:none">Segmentation</a>
<a class="tbtn" href="/api/cropped_bg?run={{ run }}&img={{ sel_img }}&mode=plain" download="{{ sel_img }}_plain.png" style="font-size:10px;padding:1px 8px;text-decoration:none">Plain</a>
<a class="tbtn" href="/download_run/{{ run }}" style="font-size:10px;padding:1px 8px;text-decoration:none">All (.zip)</a>
</div>
<div style="display:flex;gap:3px;align-items:center;margin-bottom:6px;flex-wrap:wrap;font-size:10px">
<span style="color:var(--gutter)">Filter:</span>
<button class="fbtn active" id="filtAll" onclick="toggleFilter('all')" style="font-weight:bold">All</button>
{% for c in classes %}<button class="fbtn active" id="filt_{{c}}" onclick="toggleFilter('{{c}}')" style="background:{{ cmap[c] }}25;border-color:{{ cmap[c] }};color:{{ cmap[c] }}">
<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{{ cmap[c] }};margin-right:3px"></span>{{c}}</button>{% endfor %}
</div>
<div id="mapWrap" style="position:relative;width:100%;flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;min-height:0">
<svg id="mapSvg" style="max-width:100%;max-height:100%" preserveAspectRatio="xMidYMid meet">
<image id="mapBg" href="/api/cropped_bg?run={{ run }}&img={{ sel_img }}" x="0" y="0" width="1024" height="1024"/>
</svg>
<div id="edgePopup" style="display:none;position:absolute;background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:10px;z-index:50;min-width:200px;font-size:11px;box-shadow:0 4px 12px rgba(0,0,0,0.5)">
<div id="popupTitle" style="color:var(--accent2);margin-bottom:6px"></div>
<div id="popupMorph" style="margin-bottom:8px"></div>
<div id="popupPatch" style="text-align:center;margin-bottom:8px"></div>
<div id="popupBtns" style="display:flex;flex-wrap:wrap;gap:3px"></div>
</div>
</div>
</div>
<script>
(function(){
var run='{{ run }}',imgName='{{ sel_img }}';
var CMAP={{ cmap|tojson }};
var CLASSES={{ classes|tojson }};
var mapImg=document.getElementById('mapBg');
var svg=document.getElementById('mapSvg');
var edges=[],cells=[],origW=1024,origH=1024,cropW=1024,cropH=1024;
var activeFilters=new Set(CLASSES);
var segUrl='/api/cropped_bg?run='+run+'&img='+imgName+'&mode=seg';
var plainUrl='/api/cropped_bg?run='+run+'&img='+imgName+'&mode=plain';

window.updateView=function(){
  var showSeg=document.getElementById('chkSeg').checked;
  mapImg.setAttributeNS('http://www.w3.org/1999/xlink','href',showSeg?segUrl:plainUrl);
  mapImg.setAttribute('href',showSeg?segUrl:plainUrl);
};

window.toggleFilter=function(f){
  if(f==='all'){
    if(activeFilters.size===CLASSES.length){activeFilters.clear();}
    else{CLASSES.forEach(function(c){activeFilters.add(c);});}
  } else {
    if(activeFilters.has(f))activeFilters.delete(f);else activeFilters.add(f);
  }
  syncFilterUI();
  updateLayers();
};

function syncFilterUI(){
  var allOn=activeFilters.size===CLASSES.length;
  var allBtn=document.getElementById('filtAll');
  if(allOn){allBtn.classList.add('active');allBtn.style.background='';allBtn.style.opacity='1';}
  else{allBtn.classList.remove('active');allBtn.style.opacity='0.5';}
  CLASSES.forEach(function(c){
    var btn=document.getElementById('filt_'+c);
    if(!btn)return;
    if(activeFilters.has(c)){
      btn.classList.add('active');btn.style.opacity='1';
    } else {
      btn.classList.remove('active');btn.style.opacity='0.3';
    }
  });
}

window.downloadView=function(){
  var svgEl=document.getElementById('mapSvg');
  var clone=svgEl.cloneNode(true);
  clone.querySelectorAll('.edge-hit').forEach(function(e){e.remove();});
  var s=new XMLSerializer().serializeToString(clone);
  var blob=new Blob([s],{type:'image/svg+xml'});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');
  a.href=url;a.download=imgName+'_graph_overlay.svg';
  a.click();URL.revokeObjectURL(url);
};
window.tweakWidths=function(){
  var ew=parseFloat(document.getElementById('sliderEdge').value);
  var nr=parseFloat(document.getElementById('sliderNode').value);
  svg.querySelectorAll('.edge-vis').forEach(function(l){l.setAttribute('stroke-width',ew);});
  svg.querySelectorAll('.node-dot').forEach(function(c){c.setAttribute('r',nr);});
  svg.querySelectorAll('.edge-hit').forEach(function(l){l.setAttribute('stroke-width',Math.max(12,ew*4));});
};

function init(){
  fetch('/api/map_data?run='+run+'&img='+imgName).then(function(r){return r.json()}).then(function(d){
    edges=d.edges;cells=d.cells;
    origW=d.img_w||1024;origH=d.img_h||1024;
    cropW=d.crop_w||1024;cropH=d.crop_h||1024;
    svg.setAttribute('viewBox','0 0 '+cropW+' '+cropH);
    var bgImg=document.getElementById('mapBg');
    bgImg.setAttribute('width',cropW);
    bgImg.setAttribute('height',cropH);
    updateLayers();
  });
}

function updateLayers(){
  var showSeg=document.getElementById('chkSeg').checked;
  var showGraph=document.getElementById('chkGraph').checked;
  var edgeW=parseFloat(document.getElementById('sliderEdge').value);
  var nodeR=parseFloat(document.getElementById('sliderNode').value);
  var sx=cropW/origW,sy=cropH/origH;
  var bgEl=document.getElementById('mapBg');
  while(svg.lastChild&&svg.lastChild!==bgEl)svg.removeChild(svg.lastChild);
  if(svg.firstChild!==bgEl)svg.insertBefore(bgEl,svg.firstChild);
  if(!showGraph){svg.style.pointerEvents='none';return;}
  svg.style.pointerEvents='all';
  var cellMap={};
  cells.forEach(function(c){cellMap[c.id]=c;});
  var visEdges=edges.filter(function(e){return activeFilters.has(e.morph);});
  // Draw visible edges + thick invisible hit lines on top
  visEdges.forEach(function(e){
    var ci=cellMap[e.i],cj=cellMap[e.j];
    if(!ci||!cj)return;
    var col=CMAP[e.morph]||'#888';
    var x1=ci.x*sx,y1=ci.y*sy,x2=cj.x*sx,y2=cj.y*sy;
    // Visible line
    var line=document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1',x1);line.setAttribute('y1',y1);
    line.setAttribute('x2',x2);line.setAttribute('y2',y2);
    line.setAttribute('stroke',col);line.setAttribute('stroke-width',edgeW);
    line.setAttribute('stroke-opacity','0.75');line.style.pointerEvents='none';
    line.setAttribute('class','edge-vis');line.setAttribute('data-ei',e.i);line.setAttribute('data-ej',e.j);
    svg.appendChild(line);
  });
  // Draw cell nodes
  var visCells=new Set();
  visEdges.forEach(function(e){visCells.add(e.i);visCells.add(e.j);});
  cells.forEach(function(c){
    if(!visCells.has(c.id))return;
    var dot=document.createElementNS('http://www.w3.org/2000/svg','circle');
    dot.setAttribute('cx',c.x*sx);dot.setAttribute('cy',c.y*sy);dot.setAttribute('r',nodeR);
    dot.setAttribute('fill','#6897bb');dot.setAttribute('fill-opacity','0.8');
    dot.setAttribute('class','node-dot');dot.style.pointerEvents='none';
    svg.appendChild(dot);
  });
  // Wide invisible hit lines on top for hover/click (every edge is hoverable)
  visEdges.forEach(function(e){
    var ci=cellMap[e.i],cj=cellMap[e.j];
    if(!ci||!cj)return;
    var col=CMAP[e.morph]||'#888';
    var x1=ci.x*sx,y1=ci.y*sy,x2=cj.x*sx,y2=cj.y*sy;
    var hit=document.createElementNS('http://www.w3.org/2000/svg','line');
    hit.setAttribute('x1',x1);hit.setAttribute('y1',y1);
    hit.setAttribute('x2',x2);hit.setAttribute('y2',y2);
    hit.setAttribute('stroke','transparent');hit.setAttribute('stroke-width',Math.max(12,edgeW*4));
    hit.setAttribute('class','edge-hit');hit.style.cursor='pointer';hit.style.pointerEvents='stroke';
    hit.addEventListener('mouseenter',function(){
      var w=parseFloat(document.getElementById('sliderEdge').value);
      svg.querySelectorAll('.edge-vis[data-ei="'+e.i+'"][data-ej="'+e.j+'"]').forEach(function(l){
        l.setAttribute('stroke-width',w*2.5);l.setAttribute('stroke-opacity','1');
      });
    });
    hit.addEventListener('mouseleave',function(){
      var w=parseFloat(document.getElementById('sliderEdge').value);
      svg.querySelectorAll('.edge-vis[data-ei="'+e.i+'"][data-ej="'+e.j+'"]').forEach(function(l){
        l.setAttribute('stroke-width',w);l.setAttribute('stroke-opacity','0.75');
      });
    });
    hit.addEventListener('click',function(ev){ev.stopPropagation();showPopup(e,ev);});
    svg.appendChild(hit);
  });
}

function showPopup(e,ev){
  var popup=document.getElementById('edgePopup');
  popup.style.display='block';
  var rect=document.getElementById('mapWrap').getBoundingClientRect();
  var px=ev.clientX-rect.left+12,py=ev.clientY-rect.top+12;
  var mw=document.getElementById('mapWrap').clientWidth;
  var mh=document.getElementById('mapWrap').clientHeight;
  if(px+220>mw)px=px-240;if(py+200>mh)py=py-220;
  popup.style.left=Math.max(0,px)+'px';popup.style.top=Math.max(0,py)+'px';
  document.getElementById('popupTitle').textContent='Cells '+e.i+' \u2194 '+e.j+' ('+e.px+'px contact)';
  var col=CMAP[e.morph]||'#888';
  document.getElementById('popupMorph').innerHTML='PiMorph: <b style="color:'+col+'">'+e.morph+'</b>';
  var patchUrl='/demo_asset?run='+run+'&img='+imgName+'&file=../patches/patches/'+imgName+'__i_'+e.i+'__j_'+e.j+'.png';
  document.getElementById('popupPatch').innerHTML='<img src="'+patchUrl+'" style="width:96px;height:96px;image-rendering:pixelated;border:1px solid var(--border)" onerror="this.style.display=\'none\'">';
  var btns='';
  CLASSES.forEach(function(c){
    var col=CMAP[c]||'#888';
    var sel=c===e.morph?'border-color:'+col+';color:'+col:'';
    btns+='<button class="tbtn" style="font-size:9px;padding:2px 6px;'+sel+'" onclick="relabelEdge('+e.i+','+e.j+',\''+c+'\')">'+c+'</button>';
  });
  document.getElementById('popupBtns').innerHTML=btns;
}
window.relabelEdge=function(i,j,cls){
  fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cls:cls,map_i:i,map_j:j,map_run:run,map_img:imgName})})
  .then(function(){location.reload();});
};
document.addEventListener('click',function(ev){
  var p=document.getElementById('edgePopup');
  if(p.style.display==='block'&&!p.contains(ev.target)&&ev.target.tagName!=='circle')
    p.style.display='none';
});
window.downloadView=function(){
var fmt=prompt('Format: png or tiff','png');
if(!fmt)return;
fmt=fmt.toLowerCase().trim();
var svgEl=document.getElementById('mapSvg');
var vb=svgEl.viewBox.baseVal;
var w=vb.width||cropW,h=vb.height||cropH;
var clone=svgEl.cloneNode(true);
clone.setAttribute('width',w);clone.setAttribute('height',h);
clone.setAttribute('xmlns','http://www.w3.org/2000/svg');
clone.setAttribute('xmlns:xlink','http://www.w3.org/1999/xlink');
var imgs=clone.querySelectorAll('image');
var toLoad=imgs.length;
if(toLoad===0){renderAndSave(clone,w,h,fmt);return;}
imgs.forEach(function(img){
var href=img.getAttribute('href')||img.getAttributeNS('http://www.w3.org/1999/xlink','href');
if(!href){toLoad--;if(toLoad===0)renderAndSave(clone,w,h,fmt);return;}
var xhr=new XMLHttpRequest();
xhr.open('GET',href,true);xhr.responseType='blob';
xhr.onload=function(){
var reader=new FileReader();
reader.onloadend=function(){
img.removeAttributeNS('http://www.w3.org/1999/xlink','href');
img.setAttribute('href',reader.result);
toLoad--;if(toLoad===0)renderAndSave(clone,w,h,fmt);
};reader.readAsDataURL(xhr.response);
};xhr.send();
});
};
function renderAndSave(svgNode,w,h,fmt){
var xml=new XMLSerializer().serializeToString(svgNode);
var blob=new Blob([xml],{type:'image/svg+xml;charset=utf-8'});
var url=URL.createObjectURL(blob);
var img=new Image();
img.onload=function(){
var c=document.createElement('canvas');c.width=w;c.height=h;
var ctx=c.getContext('2d');ctx.drawImage(img,0,0,w,h);
URL.revokeObjectURL(url);
if(fmt==='tiff'){
c.toBlob(function(b){
var a=document.createElement('a');a.href=URL.createObjectURL(b);
a.download=imgName+'_annotated.png';a.click();
alert('TIFF export requires server-side conversion. Saved as PNG at full resolution.');
},'image/png',1.0);
}else{
c.toBlob(function(b){
var a=document.createElement('a');a.href=URL.createObjectURL(b);
a.download=imgName+'_annotated.png';a.click();
},'image/png',1.0);
}
};img.src=url;
}
init();
window.addEventListener('resize',function(){if(edges.length>0)updateLayers();});
})();
</script>
{% endif %}
</div></div></div>
{% elif page=='label' %}
<div class="toolbar"><a href="/" class="tbtn">Home</a><div class="sep-v"></div>
<button class="tbtn" onclick="nav(-1)">&#9664; Prev</button><button class="tbtn" onclick="nav(1)">Next &#9654;</button><div class="sep-v"></div>
<button class="tbtn" onclick="skip_labeled(-1)">&#9664;&#9664;</button><button class="tbtn" onclick="skip_labeled(1)">&#9654;&#9654;</button><div class="sep-v"></div>
<button class="tbtn" onclick="gotoIdx()">Go #</button><div class="info">{{ patch.idx+1 }} / {{ total }}</div></div>
<div class="main">
<div class="sidebar"><div class="stitle"><span>Patches</span><span style="font-size:9px;color:var(--blue)">{{ total }}</span></div>
<div class="filter-bar"><button class="fbtn {% if not filter_cls %}active{% endif %}" onclick="window.location='/label_view?filt='">All</button>
{% for c in classes %}<button class="fbtn {% if filter_cls==c %}active{% endif %}" style="--cc:{{cmap[c]}}" onclick="window.location='/label_view?filt={{c}}'">{{ABBR.get(c,c[:3])}}</button>{% endfor %}
<button class="fbtn {% if filter_cls=='unlabeled' %}active{% endif %}" onclick="window.location='/label_view?filt=unlabeled'">???</button></div>
<div class="slist">{% for p in vis %}<div class="sitem {% if p.idx==patch.idx %}current{% endif %}" onclick="window.location='/label_view?goto={{p.idx}}'">
<span style="color:{% if p.labeled %}var(--green){% else %}var(--gutter){% endif %};font-size:10px">{{ p.short }}</span>
{% if p.label %}<span class="slbl" style="background:{{ cmap.get(p.label,'#555') }};color:#fff;font-weight:bold">{{ ABBR.get(p.label,p.label[:3]) }}</span>{% endif %}
</div>{% endfor %}</div></div>
<div class="center"><div class="tabs"><div class="tab active">{{ patch.image_id }}__i{{ patch.cell_i }}__j{{ patch.cell_j }}.png</div></div>
<div class="editor"><div class="patch-view">
<img src="/patch_img?idx={{ patch.idx }}" alt="patch">
<div class="patch-info"><span style="color:var(--accent2)">{{ patch.image_id }}</span> &nbsp; cells <span style="color:var(--blue)">{{ patch.cell_i }} &#8596; {{ patch.cell_j }}</span>
{% if patch.auto %}<br>PiMorph: <span style="color:{{ cmap.get(patch.auto,'#999') }}">{{ patch.auto }}</span>{% endif %}
{% if patch.label %}<br>Your label: <b style="color:{{ cmap.get(patch.label,'#999') }}">{{ patch.label }}</b>{% endif %}
</div>
<div class="class-bar">{% for i,c in enumerate(classes) %}<button class="cbtn {% if patch.label==c %}selected{% endif %}" style="--cc:{{cmap[c]}};border-left-color:{{cmap[c]}}" onclick="assign('{{c}}')" title="Key: {{ '0' if i==9 else i+1 }}"><span class="key">{{ '0' if i==9 else i+1 }}</span>{{c}}</button>{% endfor %}</div>
</div><div class="nav-hint">&#8592; &#8594; nav &nbsp; 1-9,0 label &nbsp; Shift+arrow skip &nbsp; ? help</div></div></div>
<div class="right-panel"><div class="stitle">Statistics</div><div class="stats">
{% for c in classes %}<div class="row"><span><span class="dot" style="background:{{ cmap[c] }}"></span>{{ c }}</span><span class="cnt">{{ stats.get(c,0) }}</span></div>{% endfor %}
</div><div class="stitle">Progress</div><div class="progress-bar"><div class="fill" style="width:{{ pct }}%"></div></div><div class="ptext">{{ n_labeled }} / {{ total }} ({{ pct }}%)</div>
<div class="stitle">Session</div><div style="padding:8px 12px;font-size:10px;color:var(--gutter)"><div>ID: {{ sid[:8] }}</div><div style="margin-top:4px">Dataset: {{ csv_name }}</div></div></div>
</div>
{% endif %}
<div class="statusbar">
<div style="display:flex;gap:16px;align-items:center">
{% if page=='label' %}<span class="mode">LABEL</span><span>{{ patch.idx+1 }}/{{ total }}</span><span>{{ n_labeled }} labeled</span>
{% elif page=='demo' %}<span class="mode">DEMO</span><span>{{ sel_img|default('S-BIAD1540') }}</span>
{% else %}<span class="mode">HOME</span>{% endif %}
</div>
<span>PiMorph v2.0</span>
</div>
<div id="uploadModal" class="modal-bg" style="display:none" onclick="if(event.target===this)this.style.display='none'">
<div class="modal">
<h3>Upload Microscopy Image</h3>
<form id="uploadForm" action="/upload" method="POST" enctype="multipart/form-data">
<p style="font-size:11px;color:var(--gutter);margin-bottom:8px">Upload TIFF/PNG images. PiMorph will segment cells, extract interfaces, and classify AJ morphology.</p>
<div class="upload-zone" id="dropZone">
<div class="drag-halo"></div>
<div class="drop-icon">&#128464;</div>
<div class="drop-text">Drag & drop images here</div>
<div class="drop-or">or</div>
<button type="button" class="tbtn" onclick="document.getElementById('fileInput').click()">Browse Files</button>
<input type="file" id="fileInput" name="image" accept=".tif,.tiff,.png" multiple style="display:none">
<div class="file-info" id="fileInfo"></div>
</div>
<div class="actions">
<button type="button" class="tbtn" onclick="this.closest('.modal-bg').style.display='none'">Cancel</button>
<button type="submit" class="tbtn" id="uploadBtn" style="background:var(--sel);opacity:0.4;pointer-events:none">Upload & Process</button>
</div>
</form>
</div>
</div>
<div id="helpModal" class="modal-bg" style="display:none" onclick="if(event.target===this)this.style.display='none'">
<div class="modal">
<h3>Keyboard Shortcuts</h3>
<div style="font-size:11px;line-height:2;color:var(--text)">
<b style="color:var(--accent)">1-9</b> / <b style="color:var(--accent)">0</b> &mdash; assign class (straight..other)<br>
<b style="color:var(--accent)">&larr; &rarr;</b> &mdash; previous / next patch<br>
<b style="color:var(--accent)">Shift+&larr; Shift+&rarr;</b> &mdash; skip to unlabeled<br>
<b style="color:var(--accent)">Del</b> &mdash; clear current label<br>
<b style="color:var(--accent)">Ctrl+Z</b> &mdash; undo last label<br>
<b style="color:var(--accent)">+/-</b> &mdash; zoom in/out<br>
<b style="color:var(--accent)">?</b> &mdash; this help<br>
</div>
<div class="actions"><button class="tbtn" onclick="this.closest('.modal-bg').style.display='none'">Close</button></div>
</div>
</div>
<script>
var zoomLevel=1;
function nav(d){window.location='/label_view?d='+d}
function skip_labeled(d){window.location='/label_view?skip='+d}
function gotoIdx(){var n=prompt("Go to patch #:");if(n)window.location='/label_view?goto='+(parseInt(n)-1)}
var ABBR={"straight":"STR","reticular":"RET","fingers":"FIN","thick":"THK","thick_to_reticular":"T2R","other":"OTH"};
var CUR_LABEL='{{patch.label if patch is defined and patch.label else ""}}';
function assign(cls){
var c=(cls===CUR_LABEL)?'__clear__':cls;
fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cls:c}),credentials:'same-origin'})
.then(r=>r.json()).then(d=>{if(d.ok)window.location='/label_view?d=1';else alert('Label failed: '+(d.error||'unknown'))})
.catch(e=>{console.error('label error',e);alert('Label error: '+e)})
}
function clearLabel(){
fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cls:'__clear__'}),credentials:'same-origin'})
.then(r=>r.json()).then(d=>{window.location='/label_view'})
}
function undo(){fetch('/api/undo',{method:'POST',credentials:'same-origin'}).then(()=>window.location='/label_view')}
function togglePanel(cls){var el=document.querySelector('.'+cls);if(el)el.style.display=el.style.display==='none'?'':'none'}
function zoomIn(){zoomLevel=Math.min(4,zoomLevel*1.25);applyZoom()}
function zoomOut(){zoomLevel=Math.max(0.25,zoomLevel/1.25);applyZoom()}
function applyZoom(){var img=document.querySelector('.patch-view img');if(img)img.style.transform='scale('+zoomLevel+')'}
function showHelp(){document.getElementById('helpModal').style.display='flex'}
document.addEventListener('keydown',function(e){
if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
if(e.key==='?'){showHelp();e.preventDefault();return}
if(e.ctrlKey&&e.key==='z'){undo();e.preventDefault();return}
if(e.ctrlKey&&e.key==='u'){document.getElementById('uploadModal').style.display='flex';e.preventDefault();return}
if(e.ctrlKey&&e.key==='g'){gotoIdx();e.preventDefault();return}
if(e.key==='ArrowLeft'&&e.shiftKey){skip_labeled(-1);e.preventDefault()}
else if(e.key==='ArrowRight'&&e.shiftKey){skip_labeled(1);e.preventDefault()}
else if(e.key==='ArrowLeft'){nav(-1);e.preventDefault()}
else if(e.key==='ArrowRight'){nav(1);e.preventDefault()}
else if(e.key>='1'&&e.key<='9'){var cls=CLASSES_JS[parseInt(e.key)-1];assign(cls);e.preventDefault()}
else if(e.key==='0'){assign('other');e.preventDefault()}
else if(e.key==='Delete'){clearLabel();e.preventDefault()}
else if(e.key==='+'||e.key==='='){zoomIn();e.preventDefault()}
else if(e.key==='-'){zoomOut();e.preventDefault()}
});
var CLASSES_JS={{ classes|tojson }};
var cur=document.querySelector('.sitem.current');
if(cur)cur.scrollIntoView({block:'center',behavior:'auto'});
(function(){
var sb=document.querySelector('.sidebar');
var rp=document.querySelector('.right-panel');
function mkHandle(el,side){
var h=document.createElement('div');h.className='drag-handle';
h.style[side==='right'?'right':'left']='0';
el.appendChild(h);
var dragging=false,startX,startW;
h.addEventListener('mousedown',function(e){
dragging=true;startX=e.clientX;startW=el.offsetWidth;
h.classList.add('active');
document.body.style.cursor='col-resize';
document.body.style.userSelect='none';
e.preventDefault();
});
document.addEventListener('mousemove',function(e){
if(!dragging)return;
var dx=e.clientX-startX;
var nw=side==='right'?startW+dx:startW-dx;
el.style.width=Math.max(120,Math.min(500,nw))+'px';
});
document.addEventListener('mouseup',function(){
if(!dragging)return;
dragging=false;h.classList.remove('active');
document.body.style.cursor='';document.body.style.userSelect='';
});
}
if(sb)mkHandle(sb,'right');
if(rp)mkHandle(rp,'left');
})();
(function(){
var dz=document.getElementById('dropZone');
var fi=document.getElementById('fileInput');
var info=document.getElementById('fileInfo');
var btn=document.getElementById('uploadBtn');
var halo=dz?dz.querySelector('.drag-halo'):null;
if(!dz)return;
var dc=0;
function setFiles(fl){
if(!fl||fl.length===0)return;
var dt=new DataTransfer();
for(var i=0;i<fl.length;i++)dt.items.add(fl[i]);
fi.files=dt.files;
if(fl.length===1){
info.textContent=fl[0].name+' ('+Math.round(fl[0].size/1024)+' KB)';
}else{
var total=0;for(var i=0;i<fl.length;i++)total+=fl[i].size;
info.textContent=fl.length+' files ('+Math.round(total/1024)+' KB total)';
}
info.classList.add('show');
btn.style.opacity='1';btn.style.pointerEvents='auto';
dc=0;dz.classList.remove('drag','drag-over');
}
fi.addEventListener('change',function(){setFiles(fi.files);});
dz.addEventListener('dragenter',function(e){
e.preventDefault();e.stopPropagation();
dc++;dz.classList.add('drag','drag-over');
});
dz.addEventListener('dragover',function(e){
e.preventDefault();e.stopPropagation();
if(halo){
var r=dz.getBoundingClientRect();
halo.style.left=(e.clientX-r.left)+'px';
halo.style.top=(e.clientY-r.top)+'px';
}
});
dz.addEventListener('dragleave',function(e){
e.preventDefault();e.stopPropagation();
dc--;if(dc<=0){dc=0;dz.classList.remove('drag','drag-over');}
});
dz.addEventListener('drop',function(e){
e.preventDefault();e.stopPropagation();
dc=0;dz.classList.remove('drag','drag-over');
setFiles(e.dataTransfer.files);
});
dz.addEventListener('click',function(e){
if(e.target===dz||e.target.classList.contains('drop-text')||e.target.classList.contains('drop-icon')||e.target.classList.contains('drop-or'))
fi.click();
});
})();
</script>
</body></html>'''







_label_states={}
def _get_label_state(sid):
    if sid not in _label_states:
        _label_states[sid]={"patches":[],"idx":0,"csv":"","filter":None,"filter_img":None}
    return _label_states[sid]

@app.route("/api/debug_session")
def debug_session():
    sid=session.get("sid","none")
    db=get_db()
    n_labels=db.execute("SELECT COUNT(*) as c FROM labels WHERE session_id=?",(sid,)).fetchone()["c"]
    n_all=db.execute("SELECT COUNT(*) as c FROM labels").fetchone()["c"]
    db.close()
    ls=_label_states.get(sid,{})
    return jsonify(sid=sid,n_labels=n_labels,n_all_labels=n_all,
                   patches_loaded=len(ls.get("patches",[])),
                   idx=ls.get("idx",0),db_path=str(DB_PATH),
                   db_exists=DB_PATH.exists(),data_dir=str(DATA_DIR))

@app.route("/")
def index():
    sid=get_or_create_session()
    ds=discover_datasets()
    db=get_db()
    active=db.execute("SELECT id,filename,status,created FROM uploads WHERE status='processing' ORDER BY created DESC").fetchall()
    recent=db.execute("SELECT id,filename,status,created FROM uploads WHERE status='done' ORDER BY created DESC LIMIT 10").fetchall()
    db.close()
    return render_page("home",datasets=ds,active_uploads=[dict(r) for r in active],recent_uploads=[dict(r) for r in recent])

@app.route("/demo")
def demo():
    get_or_create_session()
    all_ds=load_demo_datasets()
    run=request.args.get("run","")
    sel=request.args.get("img","")
    view=request.args.get("view","info")
    cur_ds=next((ds for ds in all_ds if ds["name"]==run),None) if run else None
    images=cur_ds["images"] if cur_ds else []
    sel_data=next((im for im in images if im["id"]==sel),None) if sel else None
    return render_page("demo",all_ds=all_ds,cur_ds=cur_ds or {},images=images,
                       sel_img=sel,sel_data=sel_data or {},view=view,run=run)

@app.route("/demo_asset")
def demo_asset():
    run=request.args.get("run","")
    img=request.args.get("img","")
    fname=request.args.get("file","")
    if ".." in run or ".." in img or ".." in fname: abort(400)
    p=_resolve_run(run,img,fname)
    if p.exists(): return send_file(str(p))
    abort(404)

@app.route("/api/cropped_bg")
def api_cropped_bg():
    import numpy as np
    from PIL import Image as PILImage
    run=request.args.get("run","")
    img=request.args.get("img","")
    mode=request.args.get("mode","seg")
    if ".." in run or ".." in img: abort(400)
    raw=_resolve_run(run,img,"raw_display.png")
    seg=_resolve_run(run,img,"seg_overlay.png")
    qc=_resolve_run(run,img,"qc_cells.png")
    if mode=="seg":
        if seg.exists(): return send_file(str(seg),mimetype="image/png")
    else:
        if raw.exists(): return send_file(str(raw),mimetype="image/png")
    if qc.exists():
        im=PILImage.open(str(qc))
        arr=np.array(im.convert("L")).astype(float)/255
        rd=np.where(np.mean(arr,axis=1)<0.5)[0]
        cd=np.where(np.mean(arr,axis=0)<0.5)[0]
        if len(rd)>0 and len(cd)>0:
            t,b=int(rd[0]),int(rd[-1])+1
            l,r=int(cd[0]),int(cd[-1])+1
            im=im.crop((l,t,r,b))
        if mode=="plain":
            from PIL import Image as PILImg2
            gray=im.convert("L")
            buf=io.BytesIO()
            gray.save(buf,format="PNG")
            buf.seek(0)
            return send_file(buf,mimetype="image/png")
        buf=io.BytesIO()
        im.save(buf,format="PNG")
        buf.seek(0)
        return send_file(buf,mimetype="image/png")
    abort(404)

@app.route("/api/map_data")
def api_map_data():
    import pandas as pd
    run=request.args.get("run","")
    img=request.args.get("img","")
    if ".." in run or ".." in img: abort(400)
    base=_resolve_run(run,img)
    edges_f=base/"edges.csv"
    cells_f=base/"cells.csv"
    if not edges_f.exists(): return jsonify(edges=[],cells=[])
    edf=pd.read_csv(edges_f)
    cells=[]
    cmap_pos={}
    if cells_f.exists():
        cdf=pd.read_csv(cells_f)
        y_col=next((c for c in ("centroid_y","cy") if c in cdf.columns),"cy")
        x_col=next((c for c in ("centroid_x","cx") if c in cdf.columns),"cx")
        for _,r in cdf.iterrows():
            cid=int(r["cell_id"])
            cy,cx=float(r[y_col]),float(r[x_col])
            cells.append({"id":cid,"y":cy,"x":cx})
            cmap_pos[cid]=(cy,cx)
    edges=[]
    has_iface="iface_cy" in edf.columns and "iface_cx" in edf.columns
    morph_col=next((c for c in ("aj_morph","AJ_morph_label") if c in edf.columns),None)
    for _,r in edf.iterrows():
        ci,cj=int(r["cell_i"]),int(r["cell_j"])
        if has_iface:
            ey,ex=float(r["iface_cy"]),float(r["iface_cx"])
        else:
            p1=cmap_pos.get(ci,(0,0));p2=cmap_pos.get(cj,(0,0))
            ey=(p1[0]+p2[0])/2;ex=(p1[1]+p2[1])/2
        m=str(r[morph_col]) if morph_col else "unknown"
        edges.append({"i":ci,"j":cj,"cy":ey,"cx":ex,"px":int(r["contact_px"]),"morph":m})
    patch_dir=_resolve_run(run,"patches","patches")
    has_patches=patch_dir.exists()
    import numpy as np
    from PIL import Image as PILImage
    raw_f=_resolve_run(run,img,"raw_display.png")
    seg_f=_resolve_run(run,img,"seg_overlay.png")
    qc_f=_resolve_run(run,img,"qc_cells.png")
    if raw_f.exists():
        w,h=PILImage.open(str(raw_f)).size
    elif seg_f.exists():
        w,h=PILImage.open(str(seg_f)).size
    elif qc_f.exists():
        qc=PILImage.open(str(qc_f)).convert("L")
        arr=np.array(qc).astype(float)/255
        rd=np.where(np.mean(arr,axis=1)<0.5)[0]
        cd=np.where(np.mean(arr,axis=0)<0.5)[0]
        if len(rd)>0 and len(cd)>0:
            w=int(cd[-1]-cd[0]+1);h=int(rd[-1]-rd[0]+1)
        else:
            w,h=qc.size
    else:
        w,h=1024,1024
    return jsonify(edges=edges,cells=cells,has_patches=has_patches,
                   cmap=CMAP,classes=CLASSES,img_w=w,img_h=h,
                   crop_w=w,crop_h=h)

@app.route("/label_view")
def label_view():
    sid=get_or_create_session()
    label_state=_get_label_state(sid)
    csv_path=request.args.get("load","")
    if csv_path and csv_path!=label_state["csv"]:
        label_state["patches"]=load_patches(csv_path)
        label_state["csv"]=csv_path
        label_state["idx"]=0
        label_state["filter"]=None
        label_state["filter_img"]=None
        db=get_db()
        db.execute("UPDATE sessions SET dataset=? WHERE id=?",(csv_path,sid))
        db.commit()
        db.close()
    fi=request.args.get("filter_img")
    if fi: label_state["filter_img"]=fi
    filt=request.args.get("filt")
    if filt is not None: label_state["filter"]=filt if filt else None
    d=request.args.get("d")
    if d: label_state["idx"]=max(0,min(label_state["idx"]+int(d),len(label_state["patches"])-1))
    goto=request.args.get("goto")
    if goto is not None: label_state["idx"]=max(0,min(int(goto),len(label_state["patches"])-1))
    skip=request.args.get("skip")
    if skip:
        labels=get_session_labels(sid)
        d2=int(skip)
        i=label_state["idx"]+d2
        while 0<=i<len(label_state["patches"]):
            p=label_state["patches"][i]
            k=f"{p['image_id']}__{p['cell_i']}__{p['cell_j']}"
            if k not in labels:
                label_state["idx"]=i
                break
            i+=d2
    if not label_state["patches"]:
        return redirect("/")
    labels=get_session_labels(sid)
    idx=label_state["idx"]
    p=label_state["patches"][idx]
    k=f"{p['image_id']}__{p['cell_i']}__{p['cell_j']}"
    patch={"idx":idx,"image_id":p["image_id"],"cell_i":p["cell_i"],"cell_j":p["cell_j"],
           "auto":p.get("aj_morph",""),"label":labels.get(k,""),"key":k}
    vis=[]
    for i,pp in enumerate(label_state["patches"]):
        kk=f"{pp['image_id']}__{pp['cell_i']}__{pp['cell_j']}"
        lbl=labels.get(kk,"")
        labeled=kk in labels
        if label_state["filter_img"] and pp["image_id"]!=label_state["filter_img"]: continue
        if label_state["filter"]:
            if label_state["filter"]=="unlabeled" and labeled: continue
            elif label_state["filter"]!="unlabeled" and lbl!=label_state["filter"]: continue
        short=f"{pp['image_id'][:14]}..i{pp['cell_i']}_j{pp['cell_j']}"
        vis.append({"idx":i,"short":short,"label":lbl,"labeled":labeled})
    stats={c:0 for c in CLASSES}
    for v in labels.values():
        if v in stats: stats[v]+=1
    n_labeled=sum(stats.values())
    total=len(label_state["patches"])
    pct=round(n_labeled/total*100,1) if total>0 else 0
    csv_name=Path(label_state["csv"]).parent.parent.name if label_state["csv"] else ""
    return render_page("label",patch=patch,vis=vis,stats=stats,n_labeled=n_labeled,total=total,pct=pct,
                       csv_name=csv_name,filter_cls=label_state["filter"])

@app.route("/patch_img")
def patch_img():
    sid=get_or_create_session()
    label_state=_get_label_state(sid)
    idx=int(request.args.get("idx",0))
    if 0<=idx<len(label_state["patches"]):
        p=label_state["patches"][idx]
        return send_file(p["_abs"],mimetype="image/png")
    abort(404)

@app.route("/api/label",methods=["POST"])
def api_label():
    sid=get_or_create_session()
    label_state=_get_label_state(sid)
    data=request.get_json()
    cls=data.get("cls","")
    if data.get("map_i") is not None:
        mi=int(data["map_i"]);mj=int(data["map_j"])
        mrun=data.get("map_run","");mimg=data.get("map_img","")
        if cls in CLASSES:
            save_label(sid,mimg,mi,mj,cls,"map")
        return jsonify(ok=True)
    if not label_state["patches"]: return jsonify(ok=False,error="no_patches",sid=sid,n_states=len(_label_states))
    p=label_state["patches"][label_state["idx"]]
    if cls=="__clear__":
        db=get_db()
        db.execute("DELETE FROM labels WHERE session_id=? AND image_id=? AND cell_i=? AND cell_j=?",
                   (sid,p["image_id"],int(p["cell_i"]),int(p["cell_j"])))
        db.commit()
        db.close()
        return jsonify(ok=True)
    if cls not in CLASSES: return jsonify(ok=False)
    save_label(sid,p["image_id"],p["cell_i"],p["cell_j"],cls,p.get("aj_morph",""))
    return jsonify(ok=True)

@app.route("/api/undo",methods=["POST"])
def api_undo():
    sid=get_or_create_session()
    label_state=_get_label_state(sid)
    db=get_db()
    last=db.execute("SELECT id,image_id,cell_i,cell_j FROM labels WHERE session_id=? ORDER BY id DESC LIMIT 1",(sid,)).fetchone()
    if last:
        db.execute("DELETE FROM labels WHERE id=?",(last["id"],))
        db.commit()
    db.close()
    if label_state["idx"]>0: label_state["idx"]-=1
    return jsonify(ok=True)

@app.route("/export_labels")
def export_labels():
    sid=get_or_create_session()
    db=get_db()
    rows=db.execute("SELECT image_id,cell_i,cell_j,label,auto_label,timestamp FROM labels WHERE session_id=? ORDER BY id",(sid,)).fetchall()
    db.close()
    si=io.StringIO()
    w=csv.writer(si)
    w.writerow(["image_id","cell_i","cell_j","aj_label","auto_label","timestamp"])
    for r in rows: w.writerow([r["image_id"],r["cell_i"],r["cell_j"],r["label"],r["auto_label"],r["timestamp"]])
    mem=io.BytesIO(si.getvalue().encode())
    return send_file(mem,mimetype="text/csv",as_attachment=True,download_name=f"pimorph_labels_{sid[:8]}.csv")

@app.route("/export_all")
def export_all():
    sid=get_or_create_session()
    db=get_db()
    rows=db.execute("SELECT image_id,cell_i,cell_j,label,auto_label,timestamp FROM labels WHERE session_id=?",(sid,)).fetchall()
    db.close()
    data={"session":sid,"exported":datetime.utcnow().isoformat(),"labels":[dict(r) for r in rows]}
    mem=io.BytesIO(json.dumps(data,indent=2).encode())
    return send_file(mem,mimetype="application/json",as_attachment=True,download_name=f"pimorph_export_{sid[:8]}.json")

@app.route("/download_run/<run_name>")
def download_run(run_name):
    import zipfile
    if ".." in run_name: abort(400)
    rd=_resolve_run(run_name)
    if not rd.exists(): abort(404)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for f in rd.rglob("*"):
            if f.is_file() and f.stat().st_size<50*1024*1024:
                zf.write(f,f"{run_name}/{f.relative_to(rd)}")
    buf.seek(0)
    return send_file(buf,mimetype="application/zip",as_attachment=True,download_name=f"{run_name}.zip")

@app.route("/download_image/<run_name>/<img_name>/<fname>")
def download_image(run_name,img_name,fname):
    if ".." in run_name or ".." in img_name or ".." in fname: abort(400)
    p=_resolve_run(run_name,img_name,fname)
    if not p.exists(): abort(404)
    return send_file(str(p),as_attachment=True)

import threading
MAX_DIM=1024

_upload_progress={}
PIPELINE_STEPS=["Reading image","Resizing","Processing (segment + graph + classify)","Done"]
def _set_progress(uid,step_idx,detail=""):
    _upload_progress[uid]={"step":step_idx,"total":len(PIPELINE_STEPS),"label":PIPELINE_STEPS[min(step_idx,len(PIPELINE_STEPS)-1)],"detail":detail,"pct":int(100*step_idx/len(PIPELINE_STEPS))}
def _check_cancel(uid):
    if uid in _cancel_set:
        _cancel_set.discard(uid)
        raise RuntimeError("Cancelled by user")
def _run_pipeline_bg(uid,run_name,fpath,run_dir,filename):
    try:
        import numpy as np
        from endopigraph.io import read_image
        from endopigraph.pipeline import process_one_image
        from skimage.transform import resize
        import tifffile
        _set_progress(uid,0,"Loading file...")
        _check_cancel(uid)
        arr,ch_names=read_image(fpath)
        h,w=arr.shape[1],arr.shape[2]
        _set_progress(uid,1,f"{h}x{w} px, {arr.shape[0]} channels")
        if max(h,w)>MAX_DIM:
            scale=MAX_DIM/max(h,w)
            new_h,new_w=int(h*scale),int(w*scale)
            _set_progress(uid,1,f"Resizing {h}x{w} -> {new_h}x{new_w}")
            resized=np.zeros((arr.shape[0],new_h,new_w),dtype=arr.dtype)
            for c in range(arr.shape[0]):
                resized[c]=resize(arr[c],(new_h,new_w),preserve_range=True,anti_aliasing=True).astype(arr.dtype)
            arr=resized
            resized_path=run_dir/(Path(filename).stem+"_resized.tif")
            tifffile.imwrite(str(resized_path),arr)
            fpath=resized_path
        else:
            _set_progress(uid,1,"No resize needed")
        n_ch=arr.shape[0]
        try:
            from cellpose.models import CellposeModel
            use_cp=True
        except ImportError:
            use_cp=False
        if use_cp:
            if n_ch>=2:
                seg_cfg={"method":"cellpose","cellpose":{"model_type":"cyto2","diameter":30,"channels":{"cyto":{"channel_index":n_ch-1},"nuclei":{"channel_index":0}}}}
            else:
                seg_cfg={"method":"cellpose","cellpose":{"model_type":"cyto2","diameter":30,"channels":{"cyto":{"channel_index":0}}}}
        else:
            seg_cfg={"method":"watershed","watershed":{"nuclei":{"channel_index":0},"membrane":{"channel_index":0}}}
        if n_ch>=2:
            jm_cfg={"AJ":{"channel_index":n_ch-1,"threshold":"otsu","dilate_px":2,"min_occupancy":0.05}}
        else:
            jm_cfg={"AJ":{"channel_index":0,"threshold":"otsu","dilate_px":2,"min_occupancy":0.05}}
        cfg={"segmentation":seg_cfg,"junction_markers":jm_cfg,"graph":{"min_contact_px":5}}
        img_id=Path(filename).stem
        _check_cancel(uid)
        _set_progress(uid,2,"Running PiMorph pipeline...")
        process_one_image(img_id,fpath,cfg,run_dir)
        import shutil
        img_dir=run_dir/img_id
        img_dir.mkdir(exist_ok=True)
        t=run_dir/"tables"
        g=run_dir/"graphs"
        q=run_dir/"qc"
        for src,dst in [
            (t/f"{img_id}__edges.csv",img_dir/"edges.csv"),
            (t/f"{img_id}__cells.csv",img_dir/"cells.csv"),
            (g/f"{img_id}.json",img_dir/"graph.json"),
            (g/f"{img_id}.graphml",img_dir/"graph.graphml"),
            (q/f"{img_id}__qc_seg.png",img_dir/"qc_cells.png"),
            (q/f"{img_id}__qc_graph.png",img_dir/"qc_graph.png"),
        ]:
            if src.exists(): shutil.copy2(str(src),str(dst))
        _set_progress(uid,len(PIPELINE_STEPS)-1)
        db=get_db()
        db.execute("UPDATE uploads SET status='done' WHERE id=?",(uid,))
        db.commit();db.close()
    except Exception as ex:
        import traceback;traceback.print_exc()
        _upload_progress[uid]={"step":-1,"total":len(PIPELINE_STEPS),"label":"Error","detail":str(ex)[:200],"pct":0}
        db=get_db()
        db.execute("UPDATE uploads SET status=? WHERE id=?",(f"error: {str(ex)[:200]}",uid))
        db.commit();db.close()

_cancel_set=set()

@app.route("/upload",methods=["POST"])
def upload():
    sid=get_or_create_session()
    files=request.files.getlist("image")
    if not files or not files[0].filename: return redirect("/")
    uids=[]
    for f in files:
        uid=str(uuid.uuid4())[:8]
        run_name=f"upload_{uid}"
        run_dir=DATA_DIR/"runs"/run_name
        run_dir.mkdir(parents=True,exist_ok=True)
        fpath=run_dir/f.filename
        f.save(str(fpath))
        db=get_db()
        db.execute("INSERT INTO uploads(id,session_id,filename,created,status) VALUES(?,?,?,?,?)",
                   (uid,sid,f.filename,datetime.utcnow().isoformat(),"processing"))
        db.commit();db.close()
        t=threading.Thread(target=_run_pipeline_bg,args=(uid,run_name,fpath,run_dir,f.filename),daemon=True)
        t.start()
        uids.append({"uid":uid,"run":run_name,"filename":Path(f.filename).stem})
    if len(uids)==1:
        u=uids[0]
        return redirect(f"/upload_status?uid={u['uid']}&run={u['run']}&filename={u['filename']}")
    return redirect(f"/upload_batch?uids={','.join(u['uid'] for u in uids)}")

@app.route("/api/cancel_upload",methods=["POST"])
def api_cancel_upload():
    uid=request.json.get("uid","")
    _cancel_set.add(uid)
    db=get_db()
    db.execute("UPDATE uploads SET status='cancelled' WHERE id=? AND status='processing'",(uid,))
    db.commit();db.close()
    return jsonify(ok=True)

@app.route("/upload_batch")
def upload_batch():
    get_or_create_session()
    uids=request.args.get("uids","").split(",")
    db=get_db()
    items=[]
    for uid in uids:
        row=db.execute("SELECT id,filename,status FROM uploads WHERE id=?",(uid,)).fetchone()
        if row: items.append(dict(row))
    db.close()
    return render_template_string(r'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Batch Processing</title>
<meta http-equiv="refresh" content="5">
<style>*{margin:0;padding:0;box-sizing:border-box}
body{background:#2b2b2b;color:#a9b7c6;font-family:'JetBrains Mono',Consolas,monospace;display:flex;align-items:center;justify-content:center;height:100vh}
.card{background:#3c3d3f;border:1px solid #515151;border-radius:8px;padding:24px 32px;min-width:400px}
h3{color:#ffc66d;margin-bottom:16px;font-weight:normal;font-size:14px}
.item{display:flex;justify-content:space-between;padding:6px 0;font-size:12px;border-bottom:1px solid rgba(255,255,255,0.05)}
.s-done{color:#6a8759}.s-err{color:#bc3f3c}.s-proc{color:#cc7832}
a{color:#6897bb;text-decoration:none;font-size:12px}a:hover{text-decoration:underline}
</style></head><body><div class="card">
<h3>Batch Processing ({{ items|length }} images)</h3>
{% for it in items %}
<div class="item"><span>{{ it.filename }}</span>
<span class="{% if it.status=='done' %}s-done{% elif it.status.startswith('error') %}s-err{% else %}s-proc{% endif %}">
{% if it.status=='done' %}<a href="/demo?run=upload_{{ it.id }}&img={{ it.filename.rsplit('.',1)[0] }}">done &rarr;</a>
{% elif it.status.startswith('error') %}error{% else %}processing...{% endif %}</span></div>
{% endfor %}
<div style="margin-top:16px;text-align:center"><a href="/">Home</a></div>
</div></body></html>''',items=items)

@app.route("/api/upload_progress")
def api_upload_progress():
    uid=request.args.get("uid","")
    db=get_db()
    row=db.execute("SELECT status FROM uploads WHERE id=?",(uid,)).fetchone()
    db.close()
    st=row["status"] if row else "unknown"
    prog=_upload_progress.get(uid,{"step":0,"total":len(PIPELINE_STEPS),"label":"Queued","detail":"","pct":0})
    return jsonify(status=st,**prog)

@app.route("/upload_status")
def upload_status():
    get_or_create_session()
    uid=request.args.get("uid","")
    run_name=request.args.get("run","")
    filename=request.args.get("filename","")
    steps_json=json.dumps(PIPELINE_STEPS)
    return render_template_string(r'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Processing...</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#2b2b2b;--bg2:#3c3d3f;--bg3:#313335;--border:#515151;--text:#a9b7c6;--accent:#cc7832;--accent2:#ffc66d;--blue:#6897bb;--green:#6a8759;--gutter:#606366}
body{background:var(--bg);color:var(--text);font-family:'JetBrains Mono',Consolas,'Courier New',monospace;display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:28px 36px;min-width:420px;max-width:520px}
.title{color:var(--accent2);font-size:15px;margin-bottom:16px;display:flex;align-items:center;gap:10px}
.spinner{width:18px;height:18px;border:2px solid var(--border);border-top:2px solid var(--accent);border-radius:50%;animation:spin 0.8s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
.bar-wrap{height:8px;background:var(--bg);border-radius:4px;overflow:hidden;margin:12px 0 16px}
.bar-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:4px;transition:width 0.4s ease;width:0%}
.steps{list-style:none}
.steps li{padding:5px 0;font-size:11px;color:var(--gutter);display:flex;align-items:center;gap:8px;transition:color 0.3s}
.steps li .dot{width:8px;height:8px;border-radius:50%;border:1.5px solid var(--border);flex-shrink:0;transition:all 0.3s}
.steps li.done{color:var(--green)}
.steps li.done .dot{background:var(--green);border-color:var(--green)}
.steps li.active{color:var(--accent2)}
.steps li.active .dot{background:var(--accent);border-color:var(--accent);box-shadow:0 0 6px rgba(204,120,50,0.5)}
.steps li .detail{color:var(--gutter);font-size:10px;margin-left:auto;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pct{font-size:11px;color:var(--gutter);text-align:right;margin-top:4px}
.err{color:#bc3f3c;font-size:11px;margin-top:12px;padding:8px;background:rgba(188,63,60,0.1);border:1px solid rgba(188,63,60,0.3);border-radius:4px}
.done-msg{text-align:center;margin-top:16px}
.done-msg a{color:var(--blue);font-size:12px}
</style></head><body>
<div class="card">
<div class="title"><div class="spinner" id="spin"></div><span>Processing {{ filename }}</span></div>
<div class="bar-wrap"><div class="bar-fill" id="barFill"></div></div>
<ul class="steps" id="stepList"></ul>
<div class="pct" id="pctText"></div>
<div class="err" id="errBox" style="display:none"></div>
<div style="display:flex;gap:8px;justify-content:center;margin-top:12px"><a href="/" style="background:#3c3d3f;border:1px solid #515151;color:#a9b7c6;padding:4px 14px;font-size:11px;cursor:pointer;font-family:inherit;border-radius:2px;text-decoration:none;transition:all 0.15s" onmouseenter="this.style.background='#214283';this.style.borderColor='#6897bb'" onmouseleave="this.style.background='#3c3d3f';this.style.borderColor='#515151'">Home</a><button id="cancelBtn" onclick="cancelUpload()" style="background:#3c3d3f;border:1px solid #bc3f3c;color:#bc3f3c;padding:4px 14px;font-size:11px;cursor:pointer;font-family:inherit;border-radius:2px;transition:all 0.15s" onmouseenter="this.style.background='#bc3f3c';this.style.color='#fff'" onmouseleave="this.style.background='#3c3d3f';this.style.color='#bc3f3c'">Cancel Upload</button></div>
<div class="done-msg" id="doneBox" style="display:none"><p style="color:var(--green);margin-bottom:8px">&#10003; Processing complete</p><a id="doneLink" href="#">View Results &rarr;</a><br><a id="dlLink" href="#" style="font-size:11px;margin-top:4px;display:inline-block">Download Results (.zip)</a></div>
</div>
<script>
var STEPS={{ steps_json|safe }};
function cancelUpload(){
fetch('/api/cancel_upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({uid:uid})})
.then(function(){window.location='/';});
}
var uid='{{ uid }}',run='{{ run_name }}',fname='{{ filename }}';
var list=document.getElementById('stepList');
STEPS.forEach(function(s){
var li=document.createElement('li');
li.innerHTML='<span class="dot"></span><span>'+s+'</span>';
list.appendChild(li);
});
function poll(){
fetch('/api/upload_progress?uid='+uid).then(function(r){return r.json()}).then(function(d){
document.getElementById('barFill').style.width=d.pct+'%';
document.getElementById('pctText').textContent=d.pct+'%';
var items=list.querySelectorAll('li');
items.forEach(function(li,i){
li.className='';
var old=li.querySelector('.detail');
if(old)old.remove();
if(i<d.step){li.className='done';}
else if(i===d.step){
li.className='active';
if(d.detail){var sp=document.createElement('span');sp.className='detail';sp.textContent=d.detail;li.appendChild(sp);}
}
});
if(d.status==='done'){
document.getElementById('spin').style.display='none';
document.getElementById('barFill').style.width='100%';
document.getElementById('pctText').textContent='100%';
items.forEach(function(li){li.className='done';});
document.getElementById('doneBox').style.display='block';
document.getElementById('doneLink').href='/demo?run='+run+'&img='+fname;
document.getElementById('dlLink').href='/download_run/'+run;
document.getElementById('cancelBtn').style.display='none';
return;
}
if(d.status&&d.status.startsWith('error')){
document.getElementById('spin').style.display='none';
document.getElementById('errBox').style.display='block';
document.getElementById('errBox').textContent=d.detail||d.status;
return;
}
setTimeout(poll,800);
}).catch(function(){setTimeout(poll,2000);});
}
poll();
</script>
</body></html>''',uid=uid,run_name=run_name,filename=filename,steps_json=steps_json)

@app.route("/admin")
def admin():
    key=request.args.get("key","")
    if key!=ADMIN_KEY: abort(403)
    db=get_db()
    sessions=db.execute("SELECT * FROM sessions ORDER BY created DESC").fetchall()
    labels=db.execute("SELECT l.*,s.ip,s.user_agent FROM labels l JOIN sessions s ON l.session_id=s.id ORDER BY l.timestamp DESC LIMIT 500").fetchall()
    total=db.execute("SELECT COUNT(*) as c FROM labels").fetchone()["c"]
    by_class=db.execute("SELECT label,COUNT(*) as c FROM labels GROUP BY label ORDER BY c DESC").fetchall()
    db.close()
    html=f"""<html><head><title>PiMorph Admin</title>
    <style>body{{background:#1e1e1e;color:#a9b7c6;font-family:monospace;padding:20px}}
    table{{border-collapse:collapse;width:100%;margin:12px 0}}
    td,th{{border:1px solid #515151;padding:4px 8px;font-size:11px;text-align:left}}
    th{{background:#313335}}h2{{color:#cc7832;margin-top:20px}}</style></head><body>
    <h1 style="color:#ffc66d">PiMorph Admin Panel</h1>
    <p>{total} total labels across {len(sessions)} sessions</p>
    <h2>Label Distribution</h2><table><tr><th>Class</th><th>Count</th></tr>"""
    for r in by_class: html+=f"<tr><td>{r['label']}</td><td>{r['c']}</td></tr>"
    html+="</table><h2>Sessions</h2><table><tr><th>ID</th><th>Created</th><th>IP</th><th>Dataset</th></tr>"
    for s in sessions: html+=f"<tr><td>{s['id']}</td><td>{s['created']}</td><td>{s['ip']}</td><td>{s['dataset']}</td></tr>"
    html+="</table><h2>Recent Labels (last 500)</h2><table><tr><th>Session</th><th>Image</th><th>Pair</th><th>Label</th><th>Auto</th><th>Time</th></tr>"
    for l in labels: html+=f"<tr><td>{l['session_id'][:8]}</td><td>{l['image_id']}</td><td>{l['cell_i']}-{l['cell_j']}</td><td>{l['label']}</td><td>{l['auto_label']}</td><td>{l['timestamp']}</td></tr>"
    html+="</table><h2>Export All</h2><p><a href='/admin/export?key="+key+"' style='color:#6897bb'>Download all labels as CSV</a></p></body></html>"
    return html

@app.route("/admin/export")
def admin_export():
    key=request.args.get("key","")
    if key!=ADMIN_KEY: abort(403)
    db=get_db()
    rows=db.execute("SELECT l.*,s.ip,s.dataset FROM labels l JOIN sessions s ON l.session_id=s.id ORDER BY l.timestamp").fetchall()
    db.close()
    si=io.StringIO()
    w=csv.writer(si)
    w.writerow(["session_id","image_id","cell_i","cell_j","label","auto_label","timestamp","ip","dataset"])
    for r in rows: w.writerow([r["session_id"],r["image_id"],r["cell_i"],r["cell_j"],r["label"],r["auto_label"],r["timestamp"],r["ip"],r["dataset"]])
    mem=io.BytesIO(si.getvalue().encode())
    return send_file(mem,mimetype="text/csv",as_attachment=True,download_name="pimorph_all_labels.csv")

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--port",type=int,default=5050)
    pa.add_argument("--admin-key",type=str,default=None)
    pa.add_argument("--debug",action="store_true")
    args=pa.parse_args()
    if args.admin_key: ADMIN_KEY=args.admin_key
    print(f"PiMorph Labeler v2: http://localhost:{args.port}")
    print(f"Admin panel: http://localhost:{args.port}/admin?key={ADMIN_KEY}")
    app.run(host="0.0.0.0",port=args.port,debug=args.debug)
