from __future__ import annotations
from pathlib import Path
import json, math, sys, argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib import gridspec
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import mannwhitneyu, rankdata

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

MAX_PIX=1800
def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png",bbox_inches="tight",dpi=200)
    plt.close(fig)
    import matplotlib.image as mpimg
    img=mpimg.imread(OUT/f"{name}.png")
    h,w=img.shape[:2]
    print(f"  {name}.png: {w}x{h} px",end="")
    if max(w,h)>MAX_PIX:
        scale=MAX_PIX/max(w,h)
        from PIL import Image
        im=Image.open(OUT/f"{name}.png")
        im=im.resize((int(w*scale),int(h*scale)),Image.LANCZOS)
        im.save(OUT/f"{name}.png")
        img2=mpimg.imread(OUT/f"{name}.png")
        print(f" -> resized {img2.shape[1]}x{img2.shape[0]} px",end="")
    print()

PALETTE_PLATFORM={"Xenium":"#1b9e77","MERFISH":"#d95f02","Stereo-seq":"#7570b3","CosMx":"#e7298a","Visium":"#66a61e","MIBI":"#e6ab02"}
PALETTE_TISSUE={"breast":"#fb9a99","colon":"#ff7f00","brain":"#a6cee3","brain_tumor":"#1f78b4","pancreas":"#b2df8a","embryo":"#fdbf6f"}
PALETTE_LIN={"unassigned":"#cccccc","tumor_breast":"#e41a1c","fibroblast":"#ff7f00","tcell":"#4daf4a","myeloid":"#984ea3",
             "epithelial":"#377eb8","endothelial":"#a65628","basal_myoep":"#f781bf","bcell":"#999999","mural_smc":"#fdbf6f",
             "mast":"#ffff33","nk":"#1b9e77","neural":"#386cb0","glia":"#7fc97f","hepatocyte":"#beaed4","proliferating":"#fb8072"}

meta=pd.read_csv("runs/cross_dataset_v10/R_analysis/tables/dataset_metadata.csv")
js=pd.read_csv("runs/cross_dataset_v10/R_analysis/tables/js_distance_matrix.csv",index_col=0); js.columns=js.index
lin=pd.read_csv("runs/cross_dataset_v10/R_analysis/tables/lineage_fractions.csv")
pair=pd.read_csv("runs/cross_dataset_v10/R_analysis/tables/pair_classification.csv")
edge_cons=pd.read_csv("runs/cross_dataset_v10/R_analysis/tables/edge_type_conservation.csv")

def clean_bnd(df,n_min=100):
    d=df[df["n_cells_used"]>=n_min].copy()
    if "gene" in d.columns:
        d=d[~d["gene"].astype(str).str.match(r"^(BLANK_|NegControl|UnassignedCodeword|antisense_)",na=False)]
    sort_col="mean_boundary_frac" if "mean_boundary_frac" in d.columns else "median_ratio_bnd_over_body"
    return d.sort_values(sort_col,ascending=False).reset_index(drop=True)

bnd_col=clean_bnd(pd.read_csv("runs/cross_dataset_v10/boundary_vs_body/xenium_colon/boundary_vs_body.csv"))
bnd_col["rank"]=np.arange(1,len(bnd_col)+1)
bnd_br=clean_bnd(pd.read_csv("runs/cross_dataset_v10/boundary_vs_body/xenium_mouse_brain/boundary_vs_body.csv"))
bnd_br["rank"]=np.arange(1,len(bnd_br)+1)
bnd_mibi=clean_bnd(pd.read_csv("runs/cross_dataset_v10/boundary_vs_body/mibi_glioma/boundary_vs_body_mibi.csv"))
bnd_mibi["rank"]=np.arange(1,len(bnd_mibi)+1)

# ---------------- FIG 1: corpus -----------------
def fig1():
    fig=plt.figure(figsize=(9,5.0))
    gs=gridspec.GridSpec(2,1,figure=fig,height_ratios=[2.6,1.0],hspace=0.95)

    ax=fig.add_subplot(gs[0,0])
    ds_order=meta.sort_values(["tissue","species","platform","label"])["label"].tolist()
    ds_order=[d for d in ds_order if d in lin["dataset"].unique()]
    lin_order=lin.groupby("lineage")["frac"].sum().sort_values(ascending=False).index.tolist()
    bottom=np.zeros(len(ds_order))
    for L in lin_order:
        vals=np.array([lin[(lin["dataset"]==d)&(lin["lineage"]==L)]["frac"].sum() for d in ds_order])
        ax.bar(np.arange(len(ds_order)),vals,bottom=bottom,width=0.78,
               color=PALETTE_LIN.get(L,"#999"),edgecolor="white",linewidth=0.4,label=L)
        bottom+=vals
    ax.set_xticks(np.arange(len(ds_order)))
    ax.set_xticklabels(ds_order,rotation=35,ha="right",fontsize=7.5)
    ax.set_ylim(0,1.02); ax.set_ylabel("fraction of cells",fontsize=9)
    for i,d in enumerate(ds_order):
        row=meta[meta["label"]==d].iloc[0]
        ax.text(i,1.04,row["platform"],ha="center",va="bottom",fontsize=6.8,
                color=PALETTE_PLATFORM.get(row["platform"],"k"),fontweight="bold")
        ax.text(i,1.10,f"{row['tissue']}{'·m' if row['species']=='mouse' else ''}",
                ha="center",va="bottom",fontsize=6.3,color="#555")
    ax.set_ylim(0,1.16)
    leg=ax.legend(handles=[Patch(color=PALETTE_LIN.get(L,"#999"),label=L) for L in lin_order if PALETTE_LIN.get(L)],
                  bbox_to_anchor=(1.005,1.0),loc="upper left",ncol=1,frameon=False,fontsize=7,
                  handlelength=0.9,handleheight=0.8,labelspacing=0.25)

    ax2=fig.add_subplot(gs[1,0])
    def _c(d):
        s=summaries.get(d,{})
        for k in("n_cells_used","n_cells","n_cells_total","n_spots"):
            if s.get(k): return int(s[k])
        return 1
    summaries={}
    for d in ds_order:
        p=meta[meta.label==d].path.iloc[0]+"/pilot_summary.json"
        if Path(p).exists(): summaries[d]=json.load(open(p))
    cells=np.array([_c(d) for d in ds_order])
    edges=np.array([max(int(summaries.get(d,{}).get("n_edges",1) or 1),1) for d in ds_order])
    x=np.arange(len(ds_order))
    ax2.bar(x-0.2,cells/1e3,width=0.4,color="#6c8ebf",label="cells",edgecolor="white",linewidth=0.4)
    ax2.bar(x+0.2,edges/1e3,width=0.4,color="#d6a26b",label="edges",edgecolor="white",linewidth=0.4)
    ax2.set_yscale("log")
    ax2.set_ylim(0.5,2000)
    ax2.set_xticks(x); ax2.set_xticklabels(ds_order,rotation=35,ha="right",fontsize=7.5)
    ax2.set_ylabel("count × 1,000\n(log scale)",fontsize=8.5)
    ax2.legend(loc="upper left",frameon=False,fontsize=7.5,ncol=2)
    save(fig,"fig1_corpus")

# ---------------- FIG 2: cross-platform ---------------
def fig2():
    fig=plt.figure(figsize=(9,4.5))
    gs=gridspec.GridSpec(1,2,figure=fig,width_ratios=[1.25,1.0],wspace=0.45)

    ax=fig.add_subplot(gs[0,0])
    M=js.values.astype(float)
    order=leaves_list(linkage(M[np.triu_indices_from(M,k=1)],method="average"))
    M2=M[np.ix_(order,order)]; labels=js.index[order].tolist()
    im=ax.imshow(M2,cmap="RdYlBu",vmin=0,vmax=1,aspect="equal")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v=M2[i,j]
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6.5,
                    color="black" if 0.3<v<0.85 else "white")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels,rotation=45,ha="right",fontsize=7.5)
    ax.set_yticklabels(labels,fontsize=7.5)
    ax.set_xlabel(""); ax.set_ylabel("")
    cbar=fig.colorbar(im,ax=ax,fraction=0.045,pad=0.04,shrink=0.7)
    cbar.set_label("Jensen-Shannon distance",fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax2=fig.add_subplot(gs[0,1])
    pair_p=pair[pair["group"]!="same tissue, same platform"].copy()
    group_order=["same tissue, different platform","different tissue, same platform","different tissue, different platform"]
    palette={"same tissue, different platform":"#1b9e77","different tissue, same platform":"#d95f02","different tissue, different platform":"#bababa"}
    positions=range(len(group_order))
    for i,g in enumerate(group_order):
        sub=pair_p[pair_p["group"]==g]["js"].values
        if len(sub)==0: continue
        bp=ax2.boxplot([sub],positions=[i],widths=0.55,patch_artist=True,showfliers=False,
                       medianprops={"color":"#222","linewidth":1.3})
        for patch in bp["boxes"]: patch.set_facecolor(palette[g]); patch.set_alpha(0.7)
        rng=np.random.default_rng(0)
        jit=rng.uniform(-0.15,0.15,size=len(sub))
        ax2.scatter([i+j for j in jit],sub,color="#222",alpha=0.7,s=18,linewidth=0,zorder=3)
        med=np.median(sub)
        ax2.annotate(f"n={len(sub)}\nmed={med:.2f}",xy=(i+0.36,med),xytext=(i+0.45,med),
                     fontsize=7.5,va="center",ha="left",
                     bbox=dict(facecolor="white",edgecolor="#888",alpha=0.95,pad=2,linewidth=0.5,boxstyle="round,pad=0.25"))
    same_t=pair_p[pair_p["group"]=="same tissue, different platform"]["js"].values
    diff_t=pair_p[pair_p["group"].astype(str).str.startswith("different tissue")]["js"].values
    u,p=mannwhitneyu(same_t,diff_t,alternative="less")
    ax2.plot([0,0,2,2],[1.02,1.05,1.05,1.02],color="#333",lw=0.8,clip_on=False)
    ax2.text(1,1.07,f"Wilcoxon: p = {p:.3f}",ha="center",fontsize=8,color="#333")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(["same tissue\nXplatform","diff tissue\nsame platform","diff tissue\nXplatform"],fontsize=7.5)
    ax2.set_ylabel("JS distance")
    ax2.set_ylim(0.4,1.13)
    ax2.set_xlim(-0.5,2.9)

    fig.text(0.005,0.97,"a",fontsize=14,fontweight="bold")
    fig.text(0.58,0.97,"b",fontsize=14,fontweight="bold")
    save(fig,"fig2_cross_platform")

# ---------------- FIG 3: boundary-vs-body (split into colon + brain/mibi) ---------------
def _ladder(ax,df,xcol,ycol,highlights,base_label,title,xtotal,baseline=None):
    """ladder rank plot with side-annotated key genes."""
    n=len(df)
    ax.scatter(df[xcol],df[ycol],s=7,color="#bdbdbd",alpha=0.55,linewidth=0,zorder=2)
    if baseline is not None:
        ax.axhline(baseline,linestyle="--",color="#777",linewidth=0.7,alpha=0.6)
        ax.text(n*0.99,baseline*0.93,f"panel mean = {baseline:.2f}",ha="right",fontsize=7,color="#666",style="italic")
    for g,(cat,color) in highlights.items():
        row=df[df["gene"]==g] if "gene" in df.columns else df[df["marker"]==g]
        if not len(row): continue
        r=int(row[xcol].iloc[0]); v=float(row[ycol].iloc[0])
        ax.scatter([r],[v],s=70,color=color,edgecolor="white",linewidth=0.9,zorder=4)
    ax.set_xlim(-n*0.04,n*1.04)
    ax.set_title(title,fontsize=10,pad=5)

def fig3a_colon():
    fig=plt.figure(figsize=(9,5.0))
    gs=gridspec.GridSpec(1,2,figure=fig,width_ratios=[1.4,0.85],wspace=0.05)
    ax=fig.add_subplot(gs[0,0])
    CAT={"EPCAM":("AJ/membrane","#e41a1c"),"CDH1":("AJ/membrane","#e41a1c"),"CTNNB1":("AJ/membrane","#e41a1c"),
         "CD24":("GPI/apical","#377eb8"),"PIGR":("transcytosis","#4daf4a"),"MUC12":("surface mucin","#984ea3"),
         "KRT8":("cytokeratin","#ff7f00"),"REG4":("secretory","#a65628"),
         "ACTA2":("cytoplasmic","#999"),"CD3D":("TCR complex","#666"),"COL1A1":("ECM","#fdbf6f")}
    base=bnd_col["mean_boundary_frac"].mean()
    _ladder(ax,bnd_col,"rank","mean_boundary_frac",CAT,"",
            f"Xenium colon — n={len(bnd_col)} expressed non-control genes",len(bnd_col),baseline=base)
    ax.set_xlabel("gene rank by mean boundary fraction"); ax.set_ylabel("mean boundary fraction")

    ax2=fig.add_subplot(gs[0,1])
    ax2.axis("off")
    ax2.text(0.0,0.98,"highlighted genes",fontsize=9.5,fontweight="bold",transform=ax2.transAxes)
    y=0.92
    for g,(cat,color) in CAT.items():
        row=bnd_col[bnd_col["gene"]==g]
        if not len(row): continue
        r=int(row["rank"].iloc[0]); v=float(row["mean_boundary_frac"].iloc[0])
        ax2.add_patch(plt.Circle((0.03,y),0.013,color=color,transform=ax2.transAxes,clip_on=False))
        ax2.text(0.08,y,f"{g}",fontsize=9.5,fontweight="bold",color=color,va="center",transform=ax2.transAxes)
        ax2.text(0.32,y,f"#{r:3d}",fontsize=9,family="monospace",color="#222",va="center",transform=ax2.transAxes)
        ax2.text(0.50,y,f"{v:.2f}",fontsize=9,family="monospace",color="#222",va="center",transform=ax2.transAxes)
        ax2.text(0.66,y,cat,fontsize=8,color="#555",va="center",transform=ax2.transAxes)
        y-=0.065
    ax2.text(0.32,y-0.01,"rank",fontsize=7.5,color="#888",va="center",fontstyle="italic",transform=ax2.transAxes,ha="center")
    ax2.text(0.50,y-0.01,"score",fontsize=7.5,color="#888",va="center",fontstyle="italic",transform=ax2.transAxes,ha="center")
    save(fig,"fig3a_colon_boundary")

def fig3b_brain_mibi():
    fig=plt.figure(figsize=(9,7.0))
    gs=gridspec.GridSpec(2,2,figure=fig,width_ratios=[1.4,0.85],height_ratios=[1.0,1.0],wspace=0.05,hspace=0.6)
    CAT_B={"Slc17a7":("synaptic","#e41a1c"),"Gad1":("synaptic","#e41a1c"),"Gad2":("synaptic","#e41a1c"),
           "Aqp4":("astrocyte endfoot","#ff7f00"),"Gfap":("astrocyte","#ffb280"),"Cldn5":("tight junction","#8856a7"),
           "Pecam1":("endothelial","#377eb8"),"Dcn":("perivascular ECM","#a65628"),"Igf2":("growth factor","#4daf4a"),
           "Acta2":("mural","#fdbf6f"),"Nrn1":("neurite","#fb9a99"),"Nr2f2":("venous EC","#1f78b4"),
           "Calb1":("neuron subtype","#999"),"Pvalb":("neuron subtype","#999"),"Lamp5":("neuron subtype","#999"),
           "Rorb":("neuron subtype","#999"),"Epha4":("axon guidance","#984ea3")}
    ax=fig.add_subplot(gs[0,0])
    base=bnd_br["mean_boundary_frac"].mean()
    _ladder(ax,bnd_br,"rank","mean_boundary_frac",CAT_B,"",
            f"Xenium mouse brain — n={len(bnd_br)} expressed non-control genes",len(bnd_br),baseline=base)
    ax.set_xlabel("gene rank"); ax.set_ylabel("mean boundary fraction")

    ax2=fig.add_subplot(gs[0,1]); ax2.axis("off")
    ax2.text(0.0,0.98,"highlighted genes",fontsize=9.5,fontweight="bold",transform=ax2.transAxes)
    y=0.93
    for g,(cat,color) in CAT_B.items():
        row=bnd_br[bnd_br["gene"]==g]
        if not len(row): continue
        r=int(row["rank"].iloc[0]); v=float(row["mean_boundary_frac"].iloc[0])
        ax2.add_patch(plt.Circle((0.03,y),0.012,color=color,transform=ax2.transAxes,clip_on=False))
        ax2.text(0.08,y,f"{g}",fontsize=9,fontweight="bold",color=color,va="center",transform=ax2.transAxes)
        ax2.text(0.34,y,f"#{r:3d}",fontsize=8.5,family="monospace",color="#222",va="center",transform=ax2.transAxes)
        ax2.text(0.50,y,f"{v:.2f}",fontsize=8.5,family="monospace",color="#222",va="center",transform=ax2.transAxes)
        ax2.text(0.66,y,cat,fontsize=7.5,color="#555",va="center",transform=ax2.transAxes)
        y-=0.052

    ax3=fig.add_subplot(gs[1,0])
    CAT_M={"CD45":("pan-leukocyte","#e41a1c"),"CD40":("co-stim","#377eb8"),"CD86":("co-stim","#377eb8"),
           "CD133":("stem/membrane","#4daf4a"),"CD47":("membrane","#984ea3"),
           "CD31":("endothelial","#ff7f00"),"CD3":("TCR","#a65628"),"CD8":("TCR","#a65628"),
           "CD20":("BCR","#f781bf"),"CD68":("myeloid","#a6cee3"),"CD163":("myeloid","#a6cee3"),
           "GFAP":("astrocyte","#1b9e77"),"FoxP3":("nuclear","#999"),"Ki67":("nuclear","#999"),"Chym_Tryp":("granule","#000")}
    n=len(bnd_mibi)
    ax3.scatter(bnd_mibi["rank"],bnd_mibi["median_ratio_bnd_over_body"],s=10,color="#bdbdbd",alpha=0.6,linewidth=0)
    ax3.axhline(1.0,linestyle="--",color="#666",linewidth=0.7,alpha=0.7)
    ax3.text(n*0.99,1.04,"boundary = body",ha="right",fontsize=7,color="#666",style="italic")
    for m,(cat,color) in CAT_M.items():
        row=bnd_mibi[bnd_mibi["marker"]==m]
        if not len(row): continue
        r=int(row["rank"].iloc[0]); v=float(row["median_ratio_bnd_over_body"].iloc[0])
        ax3.scatter([r],[v],s=70,color=color,edgecolor="white",linewidth=0.9,zorder=4)
    ax3.set_xlim(-n*0.04,n*1.04)
    ax3.set_xlabel("marker rank"); ax3.set_ylabel("median boundary / body intensity")
    ax3.set_title(f"MIBI glioma — n={n} protein markers",fontsize=10,pad=5)

    ax4=fig.add_subplot(gs[1,1]); ax4.axis("off")
    ax4.text(0.0,0.98,"highlighted markers",fontsize=9.5,fontweight="bold",transform=ax4.transAxes)
    y=0.93
    for m,(cat,color) in CAT_M.items():
        row=bnd_mibi[bnd_mibi["marker"]==m]
        if not len(row): continue
        r=int(row["rank"].iloc[0]); v=float(row["median_ratio_bnd_over_body"].iloc[0])
        ax4.add_patch(plt.Circle((0.03,y),0.012,color=color,transform=ax4.transAxes,clip_on=False))
        ax4.text(0.08,y,f"{m}",fontsize=9,fontweight="bold",color=color,va="center",transform=ax4.transAxes)
        ax4.text(0.40,y,f"#{r:2d}",fontsize=8.5,family="monospace",color="#222",va="center",transform=ax4.transAxes)
        ax4.text(0.55,y,f"{v:.2f}",fontsize=8.5,family="monospace",color="#222",va="center",transform=ax4.transAxes)
        ax4.text(0.71,y,cat,fontsize=7.5,color="#555",va="center",transform=ax4.transAxes)
        y-=0.052
    save(fig,"fig3b_brain_mibi_boundary")

# ---------------- FIG 4: null deflation -----------------
def fig4():
    fig=plt.figure(figsize=(9,4.0))
    gs=gridspec.GridSpec(1,2,figure=fig,width_ratios=[1.0,0.8],wspace=0.30)
    ax=fig.add_subplot(gs[0,0])
    panels=[
        ("Brain 3-platform","cell junction",5.3e-14),
        ("Brain Xe↔Allen","cell junction",1.2e-15),
        ("Brain Xe↔Stereo","neuron projection",2.7e-7),
        ("Brain Allen↔Stereo","synaptic signaling",6.4e-8),
        ("Colon Xe↔CosMx","extracellular space",8.2e-15),
        ("Breast Xe↔Visium","extracellular exosome",6.0e-9),
    ]
    y=np.arange(len(panels))[::-1]
    wg=[-math.log10(p[2]) for p in panels]
    ax.barh(y,wg,color="#d6604d",height=0.62,edgecolor="white",linewidth=0.6)
    for i,(name,term,p) in enumerate(panels):
        yi=y[i]; v=-math.log10(p)
        if v>3: ax.text(v*0.5,yi,term,va="center",ha="center",fontsize=7.5,color="white",fontweight="bold")
        else: ax.text(v+0.3,yi,term,va="center",ha="left",fontsize=7.5,color="#333")
        ax.text(v+0.3 if v<=3 else v+0.3,yi,f"  p={p:.0e}",va="center",fontsize=7,color="#555")
    ax.set_yticks(y); ax.set_yticklabels([p[0] for p in panels],fontsize=8)
    ax.set_xlabel("−log10(p), top GO term")
    ax.set_title("Whole-genome background (the wrong null)",fontsize=10,pad=4)
    ax.set_xlim(0,18)

    ax2=fig.add_subplot(gs[0,1]); ax2.axis("off")
    def box(x,y,w,h,t,fc,fs=8.5,fw="normal",ec="#444"):
        ax2.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.03",
                                      facecolor=fc,edgecolor=ec,linewidth=0.8))
        ax2.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight=fw,wrap=True)
    box(0.0,0.82,1.0,0.14,"top-50 conserved genes",fc="#fff0d6",fs=9,fw="bold")
    ax2.annotate("",xy=(0.25,0.66),xytext=(0.35,0.82),arrowprops=dict(arrowstyle="->",color="#666",lw=1.2))
    ax2.annotate("",xy=(0.75,0.66),xytext=(0.65,0.82),arrowprops=dict(arrowstyle="->",color="#666",lw=1.2))
    box(0.0,0.46,0.46,0.22,"whole-genome bg\n(circular — panel was\ndesigned for these genes)\n730 enriched",fc="#fce4d6",fs=7.5)
    box(0.54,0.46,0.46,0.22,"shared-panel bg\n(proper test)\n0 enriched",fc="#e6e6e6",fs=7.5)
    box(0.0,0.10,1.0,0.30,"PiMorph's top-50 are NOT\nstatistically distinct from\nthe rest of the panel under\nthe correct background.",
        fc="#fff4cc",fs=8.5,fw="bold",ec="#a87b00")
    ax2.set_xlim(0,1); ax2.set_ylim(0,1)
    ax2.set_title("How the GO claim deflated",fontsize=10,pad=4)

    fig.text(0.005,0.97,"a",fontsize=14,fontweight="bold")
    fig.text(0.54,0.97,"b",fontsize=14,fontweight="bold")
    save(fig,"fig4_null_deflation")

# ---------------- FIG 5: cross-platform gene conservation ----------------
def _rank_scatter(ax,gt,c1,c2,color,top_n=8,lab_c1=None,lab_c2=None):
    s1=gt[c1].values; s2=gt[c2].values
    r1=rankdata(-s1); r2=rankdata(-s2)
    rho=np.corrcoef(r1,r2)[0,1]
    ax.scatter(r1,r2,s=11,color="#bbb",alpha=0.55,linewidth=0)
    top=gt.sort_values("rank_mean").head(top_n).copy()
    top_idx=top.index.tolist()
    rs1=[int(r1[i]) for i in top_idx]; rs2=[int(r2[i]) for i in top_idx]
    ax.scatter(rs1,rs2,s=55,color=color,edgecolor="white",linewidth=0.9,zorder=4)
    from adjustText import adjust_text
    ts=[ax.text(rs1[k],rs2[k],top.iloc[k]["gene"],fontsize=8.5,fontweight="bold",color=color,zorder=5) for k in range(len(top_idx))]
    adjust_text(ts,ax=ax,arrowprops=dict(arrowstyle="-",color="#888",lw=0.5,alpha=0.5),
                expand=(1.4,1.5),force_text=(0.5,0.6),min_arrow_len=3,max_move=18,
                only_move={"text":"xy","static":"xy","explode":"xy"})
    m=int(max(r1.max(),r2.max()))
    ax.plot([0,m],[0,m],"--",color="#888",linewidth=0.7,alpha=0.6)
    ax.set_xlim(-m*0.03,m+m*0.05); ax.set_ylim(-m*0.03,m+m*0.05)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel(lab_c1 or c1); ax.set_ylabel(lab_c2 or c2)
    return rho,len(gt)

def fig5():
    fig=plt.figure(figsize=(9,4.5))
    gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.55)

    ax=fig.add_subplot(gs[0,0])
    et=edge_cons[~edge_cons["edge_type"].astype(str).str.contains("unassigned")]
    et=et.sort_values(["n_datasets","mean_frac"],ascending=[False,False]).head(12)
    y=np.arange(len(et))[::-1]
    norm=mpl.colors.Normalize(vmin=2,vmax=8); cmap=mpl.cm.YlOrRd
    for i,(_,r) in enumerate(et.iterrows()):
        ax.barh(y[i],r["mean_frac"],color=cmap(norm(r["n_datasets"])),edgecolor="white",linewidth=0.5,height=0.7)
        ax.text(r["mean_frac"]+0.003,y[i],f"{int(r['n_datasets'])}/10",va="center",fontsize=7,color="#222")
    ax.set_yticks(y); ax.set_yticklabels(et["edge_type"].astype(str).tolist(),fontsize=7.5)
    ax.set_xlabel("mean fraction across datasets",fontsize=8.5)
    sm=mpl.cm.ScalarMappable(norm=norm,cmap=cmap); sm.set_array([])
    cb=fig.colorbar(sm,ax=ax,fraction=0.04,pad=0.04,shrink=0.65)
    cb.set_label("# datasets ≥0.5%",fontsize=7.5); cb.ax.tick_params(labelsize=7)
    ax.set_title("Universal contact classes",fontsize=10,pad=4)

    ax2=fig.add_subplot(gs[0,1])
    gt=pd.read_csv("runs/cross_dataset_v10/R_analysis/tables/brain_xenium_stereo_gene_table.csv")
    rho,n=_rank_scatter(ax2,gt,"signal_xenium_mouse_brain","signal_stereoseq_brain","#e41a1c",top_n=12,
                        lab_c1="Xenium MB rank (1=top)",lab_c2="Stereo-seq MB rank (1=top)")
    ax2.set_title(f"Brain  (ρ={rho:.2f}, n={n})",fontsize=10,pad=4)

    ax3=fig.add_subplot(gs[0,2])
    gt=pd.read_csv("runs/cross_dataset_v10/R_analysis/tables/colon_xenium_cosmx_gene_table.csv")
    rho,n=_rank_scatter(ax3,gt,"signal_xenium_colon","signal_cosmx_colon","#3690c0",top_n=12,
                        lab_c1="Xenium colon rank (1=top)",lab_c2="CosMx colon rank (1=top)")
    ax3.set_title(f"Colon  (ρ={rho:.2f}, n={n})",fontsize=10,pad=4)
    save(fig,"fig5_conserved_biology")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--only",nargs="*",default=None)
    args=ap.parse_args()
    funcs={"fig1":fig1,"fig2":fig2,"fig3a":fig3a_colon,"fig3b":fig3b_brain_mibi,"fig4":fig4,"fig5":fig5}
    for n,f in funcs.items():
        if args.only and n not in args.only: continue
        f()
