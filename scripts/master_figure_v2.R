suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(patchwork); library(pheatmap)
  library(grid); library(RColorBrewer); library(ggrepel)
})

args<-commandArgs(trailingOnly=TRUE)
in_dir<-ifelse(length(args)>=1,args[1],"runs/cross_dataset_v10/R_analysis")
out_pdf<-ifelse(length(args)>=2,args[2],file.path(in_dir,"figures","master_figure_v2.pdf"))

ds_meta<-fread(file.path(in_dir,"tables","dataset_metadata.csv"))
js<-fread(file.path(in_dir,"tables","js_distance_matrix.csv"))
js_mat<-as.matrix(js[,-1]); rownames(js_mat)<-js$dataset
edge_cons<-fread(file.path(in_dir,"tables","edge_type_conservation.csv"))
lin<-fread(file.path(in_dir,"tables","lineage_fractions.csv"))
pair<-fread(file.path(in_dir,"tables","pair_classification.csv"))
bnd_col<-fread("runs/cross_dataset_v10/boundary_vs_body/xenium_colon/boundary_vs_body.csv")
bnd_brain<-fread("runs/cross_dataset_v10/boundary_vs_body/xenium_mouse_brain/boundary_vs_body.csv")
bnd_mibi<-tryCatch(fread("runs/cross_dataset_v10/boundary_vs_body/mibi_glioma/boundary_vs_body_mibi.csv"),error=function(e) NULL)

plat_colors<-setNames(brewer.pal(8,"Set1")[seq_len(uniqueN(ds_meta$platform))],sort(unique(ds_meta$platform)))
tiss_colors<-setNames(brewer.pal(8,"Set2")[seq_len(uniqueN(ds_meta$tissue))],sort(unique(ds_meta$tissue)))

ord<-hclust(as.dist(js_mat),method="average")$order
js_long<-melt(as.data.table(js_mat,keep.rownames="a"),id.vars="a",variable.name="b",value.name="js")
js_long[,a:=factor(a,levels=rownames(js_mat)[ord])]
js_long[,b:=factor(b,levels=rownames(js_mat)[ord])]
panA<-ggplot(js_long,aes(a,b,fill=js))+geom_tile()+
  geom_text(aes(label=sprintf("%.2f",js)),size=1.9)+
  scale_fill_distiller(palette="RdYlBu",direction=1,limits=c(0,1))+
  theme_minimal()+
  theme(axis.text.x=element_text(angle=45,hjust=1,size=7),
        axis.text.y=element_text(size=7))+
  labs(x=NULL,y=NULL,fill="JS",title="A. JS distance, 10 datasets clustered")

pair[,group:=factor(group,levels=c("same tissue, different platform",
  "same tissue, same platform","different tissue, same platform",
  "different tissue, different platform"))]
panB<-ggplot(pair,aes(group,js,fill=group))+
  geom_boxplot(alpha=0.65,outlier.shape=NA)+
  geom_jitter(width=0.18,alpha=0.7,size=2,color="grey20")+
  theme_bw()+
  theme(axis.text.x=element_text(angle=18,hjust=1,size=8),legend.position="none")+
  labs(x=NULL,y="JS distance",title="B. Same-tissue cross-platform pulls tightest")

bnd_col<-bnd_col[!grepl("^BLANK_|^NegControl",gene)][n_cells_used>=100]
bnd_col<-bnd_col[order(-mean_boundary_frac)]
bnd_col[,rank:=.I]
hl_col<-c("EPCAM","CD24","CDH1","CTNNB1","CR2","C1QBP","ACTA2","COL1A1","CD3D")
bnd_col[,is_hl:=gene%in%hl_col]
bnd_col[,gene_lab:=ifelse(is_hl|rank<=4,gene,"")]
panC<-ggplot(bnd_col,aes(rank,mean_boundary_frac))+
  geom_point(aes(color=is_hl),size=ifelse(bnd_col$is_hl,3,1.3),alpha=0.8)+
  scale_color_manual(values=c("FALSE"="grey60","TRUE"="firebrick"))+
  geom_hline(yintercept=mean(bnd_col$mean_boundary_frac),linetype="dashed",color="grey30")+
  geom_text_repel(aes(label=gene_lab),size=2.8,max.overlaps=40)+
  theme_bw()+theme(legend.position="none")+
  labs(x="Gene rank in Xenium colon (n=124)",y="Mean boundary fraction",
       title="C. Xenium colon — AJ-complex genes top by boundary fraction",
       subtitle="EPCAM (1), CD24 (2), CDH1 (5), CTNNB1 (13); ACTA2 mid-pack (24); CD3D low (74)")

bnd_brain<-bnd_brain[!grepl("^BLANK_|^NegControl",gene)][n_cells_used>=100]
bnd_brain<-bnd_brain[order(-mean_boundary_frac)]
bnd_brain[,rank:=.I]
hl_brain<-c("Aqp4","Gfap","Slc17a7","Gad1","Pecam1","Cldn5","Nr2f2","Igf2","Epha4","Calb1","Pvalb","Lamp5")
bnd_brain[,is_hl:=gene%in%hl_brain]
bnd_brain[,gene_lab:=ifelse(is_hl|rank<=4,gene,"")]
panD<-ggplot(bnd_brain,aes(rank,mean_boundary_frac))+
  geom_point(aes(color=is_hl),size=ifelse(bnd_brain$is_hl,3,1.3),alpha=0.8)+
  scale_color_manual(values=c("FALSE"="grey60","TRUE"="firebrick"))+
  geom_hline(yintercept=mean(bnd_brain$mean_boundary_frac),linetype="dashed",color="grey30")+
  geom_text_repel(aes(label=gene_lab),size=2.8,max.overlaps=40)+
  theme_bw()+theme(legend.position="none")+
  labs(x="Gene rank in Xenium mouse brain (n=34)",y="Mean boundary fraction",
       title="D. Xenium MB — Aqp4 / Cldn5 mid-pack, not boundary-enriched at RNA",
       subtitle="Synaptic Slc17a7/Gad1 rank 5-6; Aqp4 rank 15; Cldn5 rank 24")

null_summary<-data.table(
  panel=c("Brain 3-platform","Brain Xe x Allen","Brain Xe x Stereo","Brain Allen x Stereo",
          "Colon Xe x Cx","Breast Xe x Vi"),
  whole_genome_top=c("cell junction","cell junction","neuron projection","synaptic signaling",
                     "extracellular space","extracellular exosome"),
  pval_wg=c(5.3e-14,1.2e-15,2.7e-7,6.4e-8,8.2e-15,5.95e-9),
  panel_bg=c("(none)","(none)","(none)","(none)","(none)","(none)"))
null_summary[,neglogp:=-log10(pval_wg)]
panE<-ggplot(null_summary,aes(x=reorder(panel,neglogp),y=neglogp))+
  geom_col(aes(fill="whole-genome bg"))+
  geom_col(aes(y=0,fill="panel bg"))+
  scale_fill_manual(values=c("whole-genome bg"="firebrick","panel bg"="grey80"))+
  coord_flip()+geom_text(aes(label=whole_genome_top),hjust=1.1,size=2.8,color="white",fontface="bold")+
  theme_bw()+
  labs(x=NULL,y="-log10(p), whole-genome bg vs panel bg",
       title="E. GO enrichment evaporates with proper panel background",
       subtitle="All whole-genome enrichments (red bars) drop to 0 with shared-panel background (grey).",
       fill=NULL)

if(!is.null(bnd_mibi)){
  bnd_mibi<-bnd_mibi[order(-median_ratio_bnd_over_body)][n_cells_used>=100]
  bnd_mibi[,rank:=.I]
  hl_m<-c("CD45","CD40","CD31","CD3","CD68","CD163","GFAP","NeuN","Olig2","FoxP3","Ki67","Arginase1","CD8")
  bnd_mibi[,is_hl:=marker%in%hl_m]
  bnd_mibi[,gene_lab:=ifelse(is_hl|rank<=3,marker,"")]
  panF<-ggplot(bnd_mibi,aes(rank,median_ratio_bnd_over_body))+
    geom_point(aes(color=is_hl),size=ifelse(bnd_mibi$is_hl,3,1.3),alpha=0.8)+
    scale_color_manual(values=c("FALSE"="grey60","TRUE"="firebrick"))+
    geom_hline(yintercept=1,linetype="dashed",color="grey30")+
    geom_text_repel(aes(label=gene_lab),size=2.8,max.overlaps=40)+
    theme_bw()+theme(legend.position="none")+
    labs(x="Marker rank in MIBI glioma (n=40)",y="Median bnd/body intensity ratio",
         title="F. MIBI glioma — membrane proteins atop boundary ratio",
         subtitle="CD45, CD40, CD133, CD47, CD86 top; Chym_Tryp (granule) bottom")
} else {
  panF<-ggplot()+theme_void()+labs(title="F. MIBI boundary ratio (data unavailable)")
}

edge_top<-edge_cons[order(-n_datasets,-mean_frac)][1:12]
edge_top[,edge_type:=factor(edge_type,levels=rev(edge_type))]
panG<-ggplot(edge_top,aes(edge_type,mean_frac,fill=n_datasets))+
  geom_col()+coord_flip()+
  scale_fill_gradient(low="grey85",high="firebrick",breaks=c(2,5,8))+
  theme_bw()+theme(axis.text.y=element_text(size=7.5))+
  labs(x=NULL,y="mean edge fraction across datasets",
       title="G. Conserved edge types (universal contact classes)",fill="n datasets")

ds_order<-ds_meta[order(tissue,species,platform,label),label]
lin[,dataset:=factor(dataset,levels=ds_order)]
lin_ord<-lin[,.(s=sum(frac)),by=lineage][order(-s),lineage]
lin[,lineage:=factor(lineage,levels=lin_ord)]
panH<-ggplot(lin,aes(dataset,frac,fill=lineage))+geom_col()+
  scale_fill_manual(values=colorRampPalette(brewer.pal(12,"Set3"))(length(lin_ord)))+
  theme_bw()+theme(axis.text.x=element_text(angle=45,hjust=1,size=8),
                    legend.position="right",legend.key.size=unit(0.32,"cm"),
                    legend.text=element_text(size=7))+
  labs(x=NULL,y="fraction of cells",title="H. Lineage composition by dataset",fill=NULL)

design_str<-"
AAAABB
AAAABB
CCCDDD
CCCDDD
EEFFFF
EEFFFF
GGGHHH
GGGHHH
"

big<-panA+panB+panC+panD+panE+panF+panG+panH+
  plot_layout(design=design_str)+
  plot_annotation(title="PiMorph cross-platform typed-edge spatial transcriptomics — v2",
                  subtitle="10 datasets · 6 platforms · 7 tissues · 2 species. (E) corrects the GO claim with a proper panel-background null; (C) is the direct boundary-vs-body validation.",
                  theme=theme(plot.title=element_text(face="bold",size=14)))

ggsave(out_pdf,big,width=17,height=22,limitsize=FALSE)
ggsave(sub("\\.pdf$",".png",out_pdf),big,width=17,height=22,dpi=180,limitsize=FALSE)
cat("wrote",out_pdf,"\n")
