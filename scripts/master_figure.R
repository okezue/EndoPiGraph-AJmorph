suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(patchwork); library(pheatmap)
  library(grid); library(RColorBrewer); library(ggrepel)
})

args<-commandArgs(trailingOnly=TRUE)
in_dir<-ifelse(length(args)>=1,args[1],"runs/cross_dataset_v10/R_analysis")
out_pdf<-ifelse(length(args)>=2,args[2],file.path(in_dir,"figures","master_figure.pdf"))

ds_meta<-fread(file.path(in_dir,"tables","dataset_metadata.csv"))
js<-fread(file.path(in_dir,"tables","js_distance_matrix.csv"))
js_mat<-as.matrix(js[,-1]); rownames(js_mat)<-js$dataset
edge_cons<-fread(file.path(in_dir,"tables","edge_type_conservation.csv"))
lin<-fread(file.path(in_dir,"tables","lineage_fractions.csv"))
pair<-fread(file.path(in_dir,"tables","pair_classification.csv"))
panel_concord<-fread(file.path(in_dir,"tables","panel_concordance_summary.csv"))
all_enrich<-fread(file.path(in_dir,"tables","enrichment","enrich_all_panels.csv"))

plat_colors<-setNames(brewer.pal(8,"Set1")[seq_len(uniqueN(ds_meta$platform))],sort(unique(ds_meta$platform)))
tiss_colors<-setNames(brewer.pal(8,"Set2")[seq_len(uniqueN(ds_meta$tissue))],sort(unique(ds_meta$tissue)))

panA_dat<-ds_meta[,.(label,tissue,platform,species)][order(tissue,platform)]
panA_dat[,row:=.I]
panA<-ggplot(panA_dat,aes(x=1,y=-row))+
  geom_tile(aes(fill=tissue),width=0.4)+
  geom_text(aes(label=label),x=1.3,hjust=0,size=2.8)+
  geom_text(aes(label=platform,x=2.3),hjust=0,size=2.5,color="grey30")+
  geom_text(aes(label=species,x=2.85),hjust=0,size=2.5,color="grey50")+
  scale_fill_manual(values=tiss_colors)+
  xlim(0.7,3.5)+theme_void()+
  theme(legend.position="bottom")+labs(fill="tissue",title="A. 10-dataset corpus")

ord<-hclust(as.dist(js_mat),method="average")$order
js_long<-melt(as.data.table(js_mat,keep.rownames="a"),id.vars="a",variable.name="b",value.name="js")
js_long[,a:=factor(a,levels=rownames(js_mat)[ord])]
js_long[,b:=factor(b,levels=rownames(js_mat)[ord])]
panB<-ggplot(js_long,aes(a,b,fill=js))+geom_tile()+
  geom_text(aes(label=sprintf("%.2f",js)),size=1.9,color="black")+
  scale_fill_distiller(palette="RdYlBu",direction=1,limits=c(0,1))+
  theme_minimal()+
  theme(axis.text.x=element_text(angle=45,hjust=1,size=7),
        axis.text.y=element_text(size=7),
        legend.position="right")+
  labs(x=NULL,y=NULL,fill="JS",title="B. Edge-type Jensen-Shannon distance (clustered)")

pair[,group:=factor(group,levels=c("same tissue, different platform",
  "same tissue, same platform","different tissue, same platform",
  "different tissue, different platform"))]
panC<-ggplot(pair,aes(group,js,fill=group))+
  geom_boxplot(alpha=0.65,outlier.shape=NA)+
  geom_jitter(width=0.18,alpha=0.7,size=2,color="grey20")+
  theme_bw()+
  theme(axis.text.x=element_text(angle=18,hjust=1,size=8),legend.position="none")+
  labs(x=NULL,y="JS distance",title="C. Cross-dataset distance by comparison class",
       subtitle="Same-tissue cross-platform: median 0.51 vs ~0.84 otherwise")

permanova_lab<-data.table(model=c("Tissue","Platform","Tissue+Platform"),
                           r2=c(0.808,0.637,1.0),pval=c(0.0097,0.097,NA))
panD<-ggplot(permanova_lab,aes(model,r2,fill=pval))+
  geom_col(width=0.6)+geom_text(aes(label=sprintf("R²=%.2f\np=%s",r2,ifelse(is.na(pval),"-",signif(pval,2)))),
                                 vjust=-0.5,size=3)+
  scale_fill_gradient(low="firebrick",high="grey90",na.value="grey80",limits=c(0,0.1))+
  ylim(0,1.15)+theme_bw()+
  labs(x=NULL,y="R² (variance explained)",title="D. PERMANOVA on JS distance",
       fill="p-value")

edge_top<-edge_cons[order(-n_datasets,-mean_frac)][1:15]
edge_top[,edge_type:=factor(edge_type,levels=rev(edge_type))]
panE<-ggplot(edge_top,aes(edge_type,mean_frac,fill=n_datasets))+
  geom_col()+coord_flip()+
  scale_fill_gradient(low="grey85",high="firebrick",breaks=c(2,5,8))+
  theme_bw()+theme(axis.text.y=element_text(size=7.5))+
  labs(x=NULL,y="mean edge fraction across datasets",
       title="E. Conserved edge types (most universal)",fill="datasets ≥0.5%")

ds_order<-ds_meta[order(tissue,species,platform,label),label]
lin[,dataset:=factor(dataset,levels=ds_order)]
lin_ord<-lin[,.(s=sum(frac)),by=lineage][order(-s),lineage]
lin[,lineage:=factor(lineage,levels=lin_ord)]
panF<-ggplot(lin,aes(dataset,frac,fill=lineage))+geom_col()+
  scale_fill_manual(values=colorRampPalette(brewer.pal(12,"Set3"))(length(lin_ord)))+
  theme_bw()+theme(axis.text.x=element_text(angle=45,hjust=1,size=8),
                    legend.position="right",legend.key.size=unit(0.35,"cm"),
                    legend.text=element_text(size=7))+
  labs(x=NULL,y="fraction of cells",title="F. Lineage composition by dataset",fill=NULL)

brain_tab<-fread(file.path(in_dir,"tables","brain_3plat_gene_table.csv"))
brain_top<-brain_tab[order(rank_mean)][1:20]
brain_long<-melt(brain_top,id.vars="gene",measure.vars=patterns("^signal_"),
                 variable.name="dataset",value.name="signal")
brain_long[,dataset:=sub("signal_","",dataset)]
brain_long[,gene:=factor(gene,levels=rev(brain_top$gene))]
panG<-ggplot(brain_long,aes(dataset,gene,fill=log1p(signal)))+geom_tile()+
  scale_fill_distiller(palette="Reds",direction=1)+
  theme_minimal()+theme(axis.text.x=element_text(angle=30,hjust=1,size=8),
                         axis.text.y=element_text(size=7.5))+
  labs(x=NULL,y=NULL,fill="log1p\nsignal",
       title="G. Mouse brain — top 20 junctional genes (3 platforms)")

enr_focus<-all_enrich[panel%in%c("brain_3plat","colon_xenium_cosmx","breast_xenium_visium","dev_vs_adult")]
enr_focus[,neglog_p:=-log10(p_value)]
enr_top<-enr_focus[order(panel,p_value)][,.SD[1:6],by=panel]
enr_top[,label:=paste0(substr(name,1,42),ifelse(nchar(name)>42,"…",""))]
enr_top[,panel:=factor(panel,levels=c("brain_3plat","colon_xenium_cosmx","breast_xenium_visium","dev_vs_adult"),
       labels=c("Mouse brain","Colon (Xe vs Cx)","Breast (Xe vs Vi)","Embryo vs adult"))]
panH<-ggplot(enr_top,aes(neglog_p,reorder(label,neglog_p),color=panel,size=intersection_size))+
  geom_point()+facet_wrap(~panel,scales="free_y",ncol=2)+
  scale_color_brewer(palette="Dark2")+scale_size_continuous(range=c(2,7))+
  theme_bw()+theme(axis.text.y=element_text(size=7),legend.position="none",
                   strip.background=element_rect(fill="grey95"))+
  labs(x="-log10(adj p)",y=NULL,size="intersect",
       title="H. Pathway enrichment (g:Profiler, top-50 conserved genes)")

design_str<-"
AAAABBBBBBBB
AAAABBBBBBBB
AAAABBBBBBBB
CCCCDDDDEEEE
CCCCDDDDEEEE
FFFFFFGGGGGG
FFFFFFGGGGGG
HHHHHHHHHHHH
HHHHHHHHHHHH
"

big<-panA+panB+panC+panD+panE+panF+panG+panH+
  plot_layout(design=design_str)+
  plot_annotation(title="PiMorph cross-platform typed-edge spatial transcriptomics",
                  subtitle="10 datasets · 6 platforms · 7 tissues · 2 species · 1.3M edges · 122 edge types",
                  theme=theme(plot.title=element_text(face="bold",size=14)))

ggsave(out_pdf,big,width=18,height=22,limitsize=FALSE)
cat("wrote",out_pdf,"\n")
ggsave(sub("\\.pdf$",".png",out_pdf),big,width=18,height=22,dpi=180,limitsize=FALSE)
cat("wrote PNG too\n")
