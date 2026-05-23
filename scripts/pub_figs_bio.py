from __future__ import annotations
from pathlib import Path
import json, math, sys, argparse, os
sys.path.insert(0,os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.collections import LineCollection
from matplotlib import gridspec
import tifffile
from cell_positions import all_positions,positions_xenium_breast,positions_mibi,positions_stereoseq

mpl.rcParams.update({
    "figure.dpi":110, "savefig.dpi":200, "savefig.bbox":"tight",
    "font.family":"DejaVu Sans","font.size":8.5,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.titleweight":"normal","axes.titlesize":10.5,"axes.labelsize":9,
    "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.5,
    "axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,
    "pdf.fonttype":42,"ps.fonttype":42,
})

OUT=Path("runs/cross_dataset_v10/publication_figures")
OUT.mkdir(parents=True,exist_ok=True)

PALETTE_LIN={"unassigned":"#dcdcdc","tumor_breast":"#e41a1c","fibroblast":"#ff7f00","tcell":"#4daf4a","myeloid":"#984ea3",
             "epithelial":"#377eb8","endothelial":"#a65628","basal_myoep":"#f781bf","bcell":"#999999","mural_smc":"#fdbf6f",
             "mast":"#ffff33","nk":"#1b9e77","neural":"#386cb0","glia":"#7fc97f","hepatocyte":"#beaed4","proliferating":"#fb8072",
             "tumor_prolif":"#fb8072"}

def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png",bbox_inches="tight",dpi=200)
    plt.close(fig)
    import matplotlib.image as mpimg
    img=mpimg.imread(OUT/f"{name}.png")
    h,w=img.shape[:2]; print(f"  {name}.png: {w}x{h} px",end="")
    if max(w,h)>1800:
        from PIL import Image
        scale=1800/max(w,h); im=Image.open(OUT/f"{name}.png")
        im=im.resize((int(w*scale),int(h*scale)),Image.LANCZOS); im.save(OUT/f"{name}.png")
        img=mpimg.imread(OUT/f"{name}.png"); print(f" -> {img.shape[1]}x{img.shape[0]}",end="")
    print()

def _scatter_cells(ax,df,s=0.4,alpha=0.85,bg="white"):
    ax.set_facecolor(bg)
    counts=df["lineage"].value_counts()
    order=counts.index.tolist()
    if "unassigned" in order:
        order.remove("unassigned"); order=["unassigned"]+order
    for L in order:
        sub=df[df["lineage"]==L]
        ax.scatter(sub["x"],sub["y"],c=PALETTE_LIN.get(L,"#888"),s=s,alpha=alpha if L!="unassigned" else 0.4,
                   linewidth=0,rasterized=True)
    ax.set_aspect("equal","datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

# ============ FIG 6: SPATIAL CELL-TYPE ATLAS ============
def fig6_spatial_atlas():
    print("loading positions...")
    pos=all_positions(verbose=False)
    allen_section="C57BL6J-638850.37"
    if "allen_merfish" in pos:
        d=pos["allen_merfish"]
        sections=d["brain_section_label"].value_counts()
        allen_section=sections.index[0]
        d=d[d["brain_section_label"]==allen_section].copy()
        pos["allen_merfish"]=d
        print(f"  allen subset to section {allen_section}: {len(d):,} cells")

    # subsample large for plotting density
    def _sub(df,n=80_000):
        if len(df)<=n: return df
        return df.sample(n=n,random_state=0)

    order=[("xenium_breast","Xenium · human breast tumor"),
           ("visium_breast","Visium · human breast tumor"),
           ("allen_merfish","Allen MERFISH · mouse brain"),
           ("stereoseq_brain","Stereo-seq · adult mouse brain"),
           ("stereoseq_embryo_e95","Stereo-seq · E9.5 mouse embryo"),
           ("mibi_glioma","MIBI · human glioma (1 FOV)")]
    order=[o for o in order if o[0] in pos]
    n=len(order); cols=3; rows=int(np.ceil(n/cols))
    fig=plt.figure(figsize=(9,5.6))
    gs=gridspec.GridSpec(rows,cols,figure=fig,hspace=0.18,wspace=0.05)
    for i,(key,label) in enumerate(order):
        r,c=divmod(i,cols)
        ax=fig.add_subplot(gs[r,c])
        d=_sub(pos[key])
        s=0.25 if len(d)>50000 else (0.6 if len(d)>5000 else 4.0)
        _scatter_cells(ax,d,s=s,alpha=0.9)
        ax.text(0.5,1.01,label,transform=ax.transAxes,ha="center",va="bottom",fontsize=8.5,fontweight="bold")
        ax.text(0.99,0.01,f"n = {len(pos[key]):,}",transform=ax.transAxes,ha="right",va="bottom",
                fontsize=7,color="#555",bbox=dict(facecolor="white",edgecolor="none",alpha=0.85,pad=1.5))
    # legend
    all_lins=set()
    for k,v in pos.items(): all_lins.update(v["lineage"].unique())
    common=["neural","glia","endothelial","epithelial","fibroblast","myeloid","tcell","bcell","tumor_breast","mural_smc","proliferating","unassigned"]
    common=[L for L in common if L in all_lins]
    leg=[Patch(color=PALETTE_LIN.get(L,"#888"),label=L) for L in common]
    fig.legend(handles=leg,loc="lower center",bbox_to_anchor=(0.5,-0.04),ncol=len(common),
               frameon=False,fontsize=7.5,handlelength=0.9,handleheight=0.7,columnspacing=1.0)
    save(fig,"fig6_spatial_atlas")

# ============ FIG 7: Xenium breast zoom (cells + pi-graph + transcripts) ============
def fig7_xenium_zoom():
    print("loading xenium breast positions + edges...")
    pos=positions_xenium_breast()
    if pos is None: print("  no data"); return
    edges=pd.read_parquet("runs/xenium_full_ec2/edges_typed.parquet",columns=["cell_i","cell_j","edge_type","contact_px"])
    # pick a zoom region with diverse cell types and decent density
    # find a ~600 µm window dense with tumor + fibroblast + immune
    cx,cy=3600,2500
    W=400
    sub=pos[(pos["x"]>cx-W/2)&(pos["x"]<cx+W/2)&(pos["y"]>cy-W/2)&(pos["y"]<cy+W/2)].copy()
    print(f"  zoom region cells: {len(sub)}")
    ids=set(sub["cell_id"].tolist())
    e_sub=edges[edges["cell_i"].isin(ids)&edges["cell_j"].isin(ids)].copy()
    pos_lkup=dict(zip(sub["cell_id"],zip(sub["x"],sub["y"])))

    fig=plt.figure(figsize=(9,5.5))
    gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.06,width_ratios=[1.0,1.0,1.0])

    ax0=fig.add_subplot(gs[0,0])
    _scatter_cells(ax0,pos,s=0.25,alpha=0.85)
    ax0.add_patch(plt.Rectangle((cx-W/2,cy-W/2),W,W,fill=False,edgecolor="red",linewidth=1.4))
    ax0.text(0.5,1.01,"full tissue — 167,780 cells",transform=ax0.transAxes,ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    ax0.text(cx+W/2+50,cy,"zoom ↑",fontsize=7,color="red",va="center")
    ax0.invert_yaxis()

    ax1=fig.add_subplot(gs[0,1])
    _scatter_cells(ax1,sub,s=14,alpha=0.95)
    ax1.set_xlim(cx-W/2,cx+W/2); ax1.set_ylim(cy+W/2,cy-W/2)
    ax1.text(0.5,1.01,f"{int(W)} µm region — {len(sub)} cells",transform=ax1.transAxes,ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    ax1.plot([cx-W/2+20,cx-W/2+20+100],[cy+W/2-25,cy+W/2-25],color="black",lw=1.8)
    ax1.text(cx-W/2+70,cy+W/2-35,"100 µm",ha="center",fontsize=6.5)

    ax2=fig.add_subplot(gs[0,2])
    et_palette={"tumor_breast__tumor_breast":"#e41a1c","fibroblast__fibroblast":"#ff7f00",
                "fibroblast__tumor_breast":"#feb24c","fibroblast__myeloid":"#fb6a4a",
                "myeloid__tcell":"#984ea3","tcell__tcell":"#4daf4a","myeloid__myeloid":"#beaed4",
                "fibroblast__tcell":"#a6d854","epithelial__epithelial":"#377eb8"}
    # plot cells dimmed in background
    ax2.scatter(sub["x"],sub["y"],c=[PALETTE_LIN.get(L,"#888") for L in sub["lineage"]],
                s=12,alpha=0.55,linewidth=0)
    # plot edges
    segs_main=[]; colors_main=[]; segs_other=[]
    for _,r in e_sub.iterrows():
        if int(r["cell_i"]) not in pos_lkup or int(r["cell_j"]) not in pos_lkup: continue
        x1,y1=pos_lkup[int(r["cell_i"])]; x2,y2=pos_lkup[int(r["cell_j"])]
        et=r["edge_type"]
        if et in et_palette:
            segs_main.append([(x1,y1),(x2,y2)]); colors_main.append(et_palette[et])
        else:
            segs_other.append([(x1,y1),(x2,y2)])
    if segs_other:
        ax2.add_collection(LineCollection(segs_other,colors=["#ccc"]*len(segs_other),linewidths=0.4,alpha=0.4,zorder=2))
    if segs_main:
        ax2.add_collection(LineCollection(segs_main,colors=colors_main,linewidths=1.0,alpha=0.85,zorder=3))
    ax2.set_xlim(cx-W/2,cx+W/2); ax2.set_ylim(cy+W/2,cy-W/2)
    ax2.set_aspect("equal","datalim")
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values(): sp.set_visible(False)
    ax2.text(0.5,1.01,f"typed π-graph ({len(e_sub)} edges)",transform=ax2.transAxes,ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    et_leg=[Patch(color=c,label=k.replace("__"," — ")) for k,c in et_palette.items()]
    et_leg.append(Patch(color="#ccc",label="other edge type"))
    ax2.legend(handles=et_leg,loc="lower center",bbox_to_anchor=(0.5,-0.18),ncol=2,
               frameon=False,fontsize=6.5,handlelength=1.2,columnspacing=1.0,labelspacing=0.3)
    save(fig,"fig7_xenium_breast_zoom")

# ============ FIG 8: MIBI multichannel ============
def fig8_mibi():
    print("loading MIBI multi-channel...")
    fov_dir=Path("data/mibi/fov")
    if not fov_dir.exists(): print("  no data"); return
    def _norm(arr,p=99.5):
        v=np.percentile(arr,p)
        return np.clip(arr/max(v,1e-9),0,1)
    chans={}
    for ch in("CD45","CD31","GFAP","Nuclear"):
        p=fov_dir/f"{ch}.tiff"
        if p.exists(): chans[ch]=tifffile.imread(p).astype(np.float32)
    mask=tifffile.imread(fov_dir/"seg_wholecell.tiff").astype(np.int32)
    if mask.ndim==3: mask=mask[0]

    H,W=mask.shape
    # composite: R=CD45, G=GFAP, B=CD31, +DAPI gray bg
    rgb=np.zeros((H,W,3),dtype=np.float32)
    if "CD45" in chans: rgb[...,0]=_norm(chans["CD45"])
    if "GFAP" in chans: rgb[...,1]=_norm(chans["GFAP"])
    if "CD31" in chans: rgb[...,2]=_norm(chans["CD31"])
    nuc=_norm(chans.get("Nuclear",np.zeros_like(mask,dtype=np.float32)))*0.5
    rgb=np.clip(rgb+nuc[...,None]*0.5,0,1)

    pos=positions_mibi()
    fig=plt.figure(figsize=(9,4.4))
    gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.06,top=0.88,bottom=0.18)

    ax0=fig.add_subplot(gs[0,0])
    ax0.imshow(rgb,interpolation="nearest")
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values(): sp.set_visible(False)
    leg_items=[Patch(color="#e41a1c",label="CD45 (immune)"),
               Patch(color="#4daf4a",label="GFAP (astrocyte)"),
               Patch(color="#377eb8",label="CD31 (endothelial)"),
               Patch(color="#999",label="Nuclear (DAPI)")]
    ax0.legend(handles=leg_items,loc="upper center",bbox_to_anchor=(0.5,-0.02),ncol=2,
               frameon=False,fontsize=7,handlelength=0.9,columnspacing=1.0)
    ax0.text(0.5,1.01,"multi-channel composite",
             transform=ax0.transAxes,ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    ax0.plot([60,60+128],[1990,1990],color="white",lw=2.5)
    ax0.text(60+64,1960,"50 µm",ha="center",fontsize=7,color="white")

    ax1=fig.add_subplot(gs[0,1])
    cmap=np.array([PALETTE_LIN.get(l,"#888") for l in pos["lineage"]])
    n_label=int(mask.max())+1
    color_arr=np.zeros((n_label,3),dtype=np.float32)
    color_arr[0]=[0.07,0.07,0.07]
    for _,r in pos.iterrows():
        cid=int(r["cell_id"])
        if cid<n_label:
            h=PALETTE_LIN.get(r["lineage"],"#888").lstrip("#")
            color_arr[cid]=np.array([int(h[i:i+2],16)/255 for i in (0,2,4)])
    lab_rgb=color_arr[mask]
    ax1.imshow(lab_rgb,interpolation="nearest")
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values(): sp.set_visible(False)
    ax1.text(0.5,1.01,f"PiMorph lineage on cell mask ({len(pos)} cells)",
             transform=ax1.transAxes,ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    lin_keep=["myeloid","tcell","bcell","endothelial","neural","glia","fibroblast","tumor_prolif","unassigned"]
    lin_keep=[L for L in lin_keep if L in pos["lineage"].unique()]
    ax1.legend(handles=[Patch(color=PALETTE_LIN.get(L,"#888"),label=L) for L in lin_keep],
               loc="upper center",bbox_to_anchor=(0.5,-0.02),ncol=4,frameon=False,fontsize=6.5,
               handlelength=0.8,columnspacing=0.7)

    ax2=fig.add_subplot(gs[0,2])
    ax2.imshow(rgb*0.55,interpolation="nearest")
    e=pd.read_parquet("runs/mibi_glioma_local/edges_typed.parquet")
    et_palette={"endothelial__endothelial":"#fdae61","endothelial__myeloid":"#f46d43",
                "endothelial__bcell":"#d73027","myeloid__myeloid":"#7a3789",
                "tumor_prolif__tumor_prolif":"#e7298a","bcell__bcell":"#666666",
                "neural__neural":"#386cb0","endothelial__neural":"#377eb8"}
    pos_lkup=dict(zip(pos["cell_id"],zip(pos["x"],pos["y"])))
    segs=[]; cs=[]; segs_o=[]
    for _,r in e.iterrows():
        if int(r["cell_i"]) not in pos_lkup or int(r["cell_j"]) not in pos_lkup: continue
        x1,y1=pos_lkup[int(r["cell_i"])]; x2,y2=pos_lkup[int(r["cell_j"])]
        et=r["edge_type"]
        if et in et_palette: segs.append([(x1,y1),(x2,y2)]); cs.append(et_palette[et])
        else: segs_o.append([(x1,y1),(x2,y2)])
    if segs_o: ax2.add_collection(LineCollection(segs_o,colors=["#888"]*len(segs_o),linewidths=0.35,alpha=0.55))
    if segs: ax2.add_collection(LineCollection(segs,colors=cs,linewidths=1.0,alpha=0.95))
    ax2.set_xlim(0,W); ax2.set_ylim(H,0)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values(): sp.set_visible(False)
    ax2.text(0.5,1.01,f"typed π-graph ({len(e)} edges)",
             transform=ax2.transAxes,ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    et_leg=[Patch(color=c,label=k.replace("__"," — ")) for k,c in et_palette.items()]
    ax2.legend(handles=et_leg,loc="upper center",bbox_to_anchor=(0.5,-0.02),ncol=2,
               frameon=False,fontsize=6,handlelength=0.9,columnspacing=0.7)
    save(fig,"fig8_mibi_multichannel")

# ============ FIG 9: pi-graph network for stereoseq brain region ============
def fig9_pigraph_network():
    print("building stereoseq brain pi-graph network...")
    pos=positions_stereoseq("brain")
    if pos is None: print("  no data"); return
    edges=pd.read_parquet("runs/stereoseq_brain_local/edges_typed.parquet",columns=["cell_i","cell_j","edge_type","dist_um"])
    cx=pos["x"].mean(); cy=pos["y"].mean()
    W=1800
    sub=pos[(pos["x"]>cx-W/2)&(pos["x"]<cx+W/2)&(pos["y"]>cy-W/2)&(pos["y"]<cy+W/2)].copy()
    print(f"  region cells: {len(sub)}")
    ids=set(sub["cell_id"].tolist())
    e_sub=edges[edges["cell_i"].isin(ids)&edges["cell_j"].isin(ids)].copy()
    pos_lkup=dict(zip(sub["cell_id"],zip(sub["x"],sub["y"])))
    print(f"  region edges: {len(e_sub)}")

    fig=plt.figure(figsize=(9,5.0))
    gs=gridspec.GridSpec(1,2,figure=fig,width_ratios=[1.0,1.0],wspace=0.08)

    ax0=fig.add_subplot(gs[0,0])
    _scatter_cells(ax0,sub,s=3.0,alpha=0.9)
    ax0.text(0.5,1.01,f"cells coloured by lineage  ({len(sub):,} cells)",
             transform=ax0.transAxes,ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    lin_keep=["neural","glia","endothelial","myeloid","fibroblast","mural_smc","unassigned"]
    lin_keep=[L for L in lin_keep if L in sub["lineage"].unique()]
    ax0.legend(handles=[Patch(color=PALETTE_LIN.get(L,"#888"),label=L) for L in lin_keep],
               loc="lower center",bbox_to_anchor=(0.5,-0.07),ncol=4,frameon=False,fontsize=7,
               handlelength=0.8,columnspacing=0.7)

    ax1=fig.add_subplot(gs[0,1])
    ax1.scatter(sub["x"],sub["y"],c=[PALETTE_LIN.get(l,"#888") for l in sub["lineage"]],
                s=3.0,alpha=0.7,linewidth=0,rasterized=True)
    et_palette={"neural__neural":"#386cb0","glia__neural":"#7fc97f","glia__glia":"#1b9e77",
                "endothelial__neural":"#a65628","endothelial__glia":"#d95f02","endothelial__endothelial":"#fb6a4a",
                "myeloid__neural":"#984ea3","myeloid__glia":"#beaed4"}
    segs=[]; cs=[]; segs_o=[]
    for _,r in e_sub.iterrows():
        if int(r["cell_i"]) not in pos_lkup or int(r["cell_j"]) not in pos_lkup: continue
        x1,y1=pos_lkup[int(r["cell_i"])]; x2,y2=pos_lkup[int(r["cell_j"])]
        et=r["edge_type"]
        if et in et_palette: segs.append([(x1,y1),(x2,y2)]); cs.append(et_palette[et])
        else: segs_o.append([(x1,y1),(x2,y2)])
    if segs_o: ax1.add_collection(LineCollection(segs_o,colors=["#ddd"]*len(segs_o),linewidths=0.25,alpha=0.5))
    if segs: ax1.add_collection(LineCollection(segs,colors=cs,linewidths=0.6,alpha=0.7))
    ax1.set_xlim(cx-W/2,cx+W/2); ax1.set_ylim(cy+W/2,cy-W/2)
    ax1.set_aspect("equal","datalim")
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values(): sp.set_visible(False)
    ax1.text(0.5,1.01,f"typed π-graph ({len(e_sub):,} edges; selected types coloured)",
             transform=ax1.transAxes,ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    et_leg=[Patch(color=c,label=k.replace("__"," — ")) for k,c in et_palette.items()]
    et_leg.append(Patch(color="#ddd",label="other types"))
    ax1.legend(handles=et_leg,loc="lower center",bbox_to_anchor=(0.5,-0.07),ncol=3,
               frameon=False,fontsize=6.5,handlelength=0.8,columnspacing=0.8)
    save(fig,"fig9_stereoseq_pigraph")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--only",nargs="*",default=None)
    args=ap.parse_args()
    funcs={"fig6":fig6_spatial_atlas,"fig7":fig7_xenium_zoom,"fig8":fig8_mibi,"fig9":fig9_pigraph_network}
    for n,f in funcs.items():
        if args.only and n not in args.only: continue
        f()
