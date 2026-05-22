suppressPackageStartupMessages({
  library(arrow); library(data.table); library(ggplot2); library(pheatmap)
  library(vegan); library(uwot); library(ape); library(RColorBrewer)
  library(ggrepel); library(patchwork)
})

args<-commandArgs(trailingOnly=TRUE)
if(length(args)<2) stop("usage: Rscript deep_cross_analysis.R <out_dir> <label1=path1> [...]")
out_dir<-args[1]; runs<-args[-1]
dir.create(out_dir,showWarnings=FALSE,recursive=TRUE)
dir.create(file.path(out_dir,"figures"),showWarnings=FALSE)
dir.create(file.path(out_dir,"tables"),showWarnings=FALSE)

meta<-data.table(label=sapply(strsplit(runs,"="),`[`,1),path=sapply(strsplit(runs,"="),`[`,2))
meta[,platform:=fcase(
  grepl("^xenium",label),"Xenium",
  grepl("^cosmx",label),"CosMx",
  grepl("^stereoseq",label),"Stereo-seq",
  grepl("^allen",label),"MERFISH",
  grepl("^mibi",label),"MIBI",
  grepl("^visium",label),"Visium",
  default="other")]
meta[,tissue:=fcase(
  grepl("breast",label),"breast",
  grepl("colon",label),"colon",
  grepl("brain",label) & !grepl("embryo",label),"brain",
  grepl("^allen",label),"brain",
  grepl("glioma",label),"brain_tumor",
  grepl("pancreas",label),"pancreas",
  grepl("embryo",label),"embryo",
  default="other")]
meta[,species:=fcase(
  grepl("mouse_brain|allen|stereoseq_brain|stereoseq_embryo",label),"mouse",
  default="human")]
fwrite(meta,file.path(out_dir,"tables","dataset_metadata.csv"))

norm_etype<-function(s){ parts<-strsplit(s,"__")[[1]]; paste(sort(parts),collapse="__") }
vecs<-list()
for(i in seq_len(nrow(meta))){
  p<-file.path(meta$path[i],"edge_type_summary.parquet")
  if(!file.exists(p)) next
  d<-as.data.table(read_parquet(p))
  d[,edge_type:=sapply(edge_type,norm_etype)]
  d<-d[,.(n_edges=sum(as.numeric(n_edges))),by=edge_type]
  d[,frac:=n_edges/sum(n_edges)]
  vecs[[meta$label[i]]]<-d
}
all_types<-unique(unlist(lapply(vecs,function(x) x$edge_type)))
M<-matrix(0,nrow=length(all_types),ncol=length(vecs),dimnames=list(all_types,names(vecs)))
for(nm in names(vecs)){
  d<-vecs[[nm]]
  M[d$edge_type,nm]<-d$frac
}
fwrite(data.table(edge_type=rownames(M),M),file.path(out_dir,"tables","edge_type_fraction_matrix.csv"))

js<-function(p,q){ p<-p/sum(p); q<-q/sum(q); m<-(p+q)/2
  kl<-function(a,b){ ok<-a>0; sum(a[ok]*log2(a[ok]/b[ok])) }
  sqrt(pmax(0,(kl(p,m)+kl(q,m))/2)) }
n<-ncol(M); D<-matrix(0,n,n,dimnames=list(colnames(M),colnames(M)))
for(i in 1:n) for(j in 1:n) D[i,j]<-js(M[,i],M[,j])
fwrite(data.table(dataset=rownames(D),D),file.path(out_dir,"tables","js_distance_matrix.csv"))

plat_colors<-setNames(brewer.pal(7,"Set1")[seq_len(length(unique(meta$platform)))],unique(meta$platform))
tiss_colors<-setNames(brewer.pal(8,"Set2")[seq_len(length(unique(meta$tissue)))],unique(meta$tissue))
ann_col<-data.frame(platform=meta$platform[match(colnames(D),meta$label)],
                    tissue=meta$tissue[match(colnames(D),meta$label)],
                    species=meta$species[match(colnames(D),meta$label)],
                    row.names=colnames(D))
ann_colors<-list(platform=plat_colors,tissue=tiss_colors,
                 species=c(mouse="#1f78b4",human="#33a02c"))
pdf(file.path(out_dir,"figures","js_heatmap.pdf"),width=10,height=9)
pheatmap(as.dist(D),clustering_method="average",
         display_numbers=round(D,2),annotation_col=ann_col,annotation_colors=ann_colors,
         main=sprintf("Jensen-Shannon distance, %d datasets",n))
dev.off()

hc<-hclust(as.dist(D),method="average")
pdf(file.path(out_dir,"figures","dendrogram.pdf"),width=12,height=6)
par(mar=c(8,4,2,2))
plot(as.phylo(hc),type="unrooted",no.margin=FALSE,
     tip.color=plat_colors[ann_col$platform[match(hc$labels,rownames(ann_col))]],
     cex=0.9,lab4ut="axial",label.offset=0.02)
dev.off()

pca<-prcomp(t(M),scale.=FALSE)
varexp<-round(100*pca$sdev^2/sum(pca$sdev^2),1)
pca_df<-data.table(label=colnames(M),PC1=pca$x[,1],PC2=pca$x[,2],PC3=pca$x[,3])
pca_df<-merge(pca_df,meta,by="label")
g_pca<-ggplot(pca_df,aes(PC1,PC2,color=platform,shape=tissue,label=label))+
  geom_point(size=4)+geom_text_repel(size=2.8,box.padding=0.3,max.overlaps=20)+
  scale_color_manual(values=plat_colors)+theme_bw()+
  labs(x=sprintf("PC1 (%.1f%%)",varexp[1]),y=sprintf("PC2 (%.1f%%)",varexp[2]),
       title="Edge-type composition PCA")
ggsave(file.path(out_dir,"figures","pca.pdf"),g_pca,width=9,height=6)

if(ncol(M)>=8){
  set.seed(0)
  u<-umap(t(M),n_neighbors=min(5,ncol(M)-1),metric="cosine")
  ud<-data.table(label=colnames(M),U1=u[,1],U2=u[,2])
  ud<-merge(ud,meta,by="label")
  g_u<-ggplot(ud,aes(U1,U2,color=platform,shape=tissue,label=label))+
    geom_point(size=4)+geom_text_repel(size=2.8,box.padding=0.3,max.overlaps=20)+
    scale_color_manual(values=plat_colors)+theme_bw()+
    labs(title=sprintf("UMAP edge-type composition (n=%d datasets)",ncol(M)))
  ggsave(file.path(out_dir,"figures","umap.pdf"),g_u,width=9,height=6)
}

set.seed(1)
adn_tiss<-adonis2(D~tissue,data=meta[match(colnames(D),meta$label)],permutations=9999)
adn_plat<-adonis2(D~platform,data=meta[match(colnames(D),meta$label)],permutations=9999)
adn_both<-adonis2(D~tissue+platform,data=meta[match(colnames(D),meta$label)],permutations=9999)
adn_inter<-adonis2(D~tissue*platform,data=meta[match(colnames(D),meta$label)],permutations=9999)
sink(file.path(out_dir,"tables","permanova_results.txt"))
cat("=== PERMANOVA on JS distance ===\n\n")
cat("Tissue alone:\n"); print(adn_tiss)
cat("\nPlatform alone:\n"); print(adn_plat)
cat("\nTissue + Platform:\n"); print(adn_both)
cat("\nTissue * Platform:\n"); print(adn_inter)
sink()

n_datasets<-ncol(M); thresh<-0.005
conserv<-rowSums(M>=thresh)
conservation<-data.table(edge_type=rownames(M),
  n_datasets=conserv,
  mean_frac=rowMeans(M),
  max_frac=apply(M,1,max),
  cv_frac=apply(M,1,function(x){ if(mean(x)==0) return(NA); sd(x)/mean(x) }))
setorder(conservation,-n_datasets,-mean_frac)
fwrite(conservation,file.path(out_dir,"tables","edge_type_conservation.csv"))

cons_top<-conservation[n_datasets>=ceiling(0.5*n_datasets)][1:25]
g_cons<-ggplot(cons_top,aes(x=reorder(edge_type,mean_frac),y=mean_frac))+
  geom_col(aes(fill=n_datasets))+coord_flip()+
  scale_fill_gradient(low="grey80",high="firebrick")+theme_bw()+
  labs(x=NULL,y="mean edge fraction across datasets",
       title=sprintf("Top conserved edge types (present in >=%d of %d datasets at >=%.1f%%)",
                     ceiling(0.5*n_datasets),n_datasets,100*thresh))
ggsave(file.path(out_dir,"figures","conserved_edge_types.pdf"),g_cons,width=10,height=7)

pairs_dt<-data.table()
labs<-colnames(D)
for(i in 1:(n-1)) for(j in (i+1):n){
  pairs_dt<-rbind(pairs_dt,data.table(
    a=labs[i],b=labs[j],js=D[i,j],
    a_tissue=meta$tissue[match(labs[i],meta$label)],
    b_tissue=meta$tissue[match(labs[j],meta$label)],
    a_platform=meta$platform[match(labs[i],meta$label)],
    b_platform=meta$platform[match(labs[j],meta$label)],
    a_species=meta$species[match(labs[i],meta$label)],
    b_species=meta$species[match(labs[j],meta$label)]))
}
pairs_dt[,same_tissue:=a_tissue==b_tissue]
pairs_dt[,same_platform:=a_platform==b_platform]
pairs_dt[,same_species:=a_species==b_species]
pairs_dt[,group:=fcase(
  same_tissue & !same_platform,"same tissue, different platform",
  same_tissue & same_platform,"same tissue, same platform",
  !same_tissue & same_platform,"different tissue, same platform",
  !same_tissue & !same_platform,"different tissue, different platform")]
fwrite(pairs_dt,file.path(out_dir,"tables","pair_classification.csv"))

g_box<-ggplot(pairs_dt,aes(x=group,y=js,fill=group))+
  geom_boxplot(alpha=0.7,outlier.shape=NA)+geom_jitter(width=0.18,alpha=0.6,size=1.8)+
  theme_bw()+theme(axis.text.x=element_text(angle=25,hjust=1))+
  labs(x=NULL,y="JS distance",title="JS distance by comparison class")+
  guides(fill="none")
ggsave(file.path(out_dir,"figures","pair_class_boxplot.pdf"),g_box,width=8,height=6)

wt_tiss<-wilcox.test(pairs_dt[same_tissue==TRUE,js],pairs_dt[same_tissue==FALSE,js])
wt_plat<-wilcox.test(pairs_dt[same_platform==TRUE,js],pairs_dt[same_platform==FALSE,js])

lin<-list()
for(i in seq_len(nrow(meta))){
  p<-file.path(meta$path[i],"cell_types.parquet")
  if(!file.exists(p)) p<-file.path(meta$path[i],"spot_types.parquet")
  if(!file.exists(p)) next
  d<-as.data.table(read_parquet(p))
  if(!"lineage"%in%names(d)) next
  tab<-d[,.N,by=lineage]; tab[,frac:=N/sum(N)]
  tab[,dataset:=meta$label[i]]
  lin[[meta$label[i]]]<-tab
}
lin_dt<-rbindlist(lin,fill=TRUE)
fwrite(lin_dt,file.path(out_dir,"tables","lineage_fractions.csv"))

ord_lin<-lin_dt[,.(s=sum(frac)),by=lineage][order(-s),lineage]
lin_dt[,lineage:=factor(lineage,levels=ord_lin)]
ds_order<-meta[order(tissue,species,platform,label),label]
lin_dt[,dataset:=factor(dataset,levels=ds_order)]
g_lin<-ggplot(lin_dt,aes(dataset,frac,fill=lineage))+geom_col()+
  scale_fill_manual(values=colorRampPalette(brewer.pal(12,"Set3"))(length(ord_lin)))+
  theme_bw()+theme(axis.text.x=element_text(angle=45,hjust=1))+
  labs(x=NULL,y="fraction of cells",title="Cell-type composition across datasets")
ggsave(file.path(out_dir,"figures","lineage_composition.pdf"),g_lin,width=11,height=6)

sink(file.path(out_dir,"tables","summary_stats.txt"))
cat("=== Summary statistics ===\n\n")
cat(sprintf("Datasets: %d\n",ncol(M)))
cat(sprintf("Edge types observed: %d\n",nrow(M)))
cat(sprintf("Platforms: %s\n",paste(sort(unique(meta$platform)),collapse=", ")))
cat(sprintf("Tissues: %s\n",paste(sort(unique(meta$tissue)),collapse=", ")))
cat(sprintf("Species: %s\n",paste(sort(unique(meta$species)),collapse=", ")))
cat("\nJS distance summary by class:\n")
print(pairs_dt[,.(n=.N,median=median(js),mean=mean(js),min=min(js),max=max(js)),by=group])
cat("\nWilcoxon: same vs different tissue: p=",signif(wt_tiss$p.value,3),"\n")
cat("Wilcoxon: same vs different platform: p=",signif(wt_plat$p.value,3),"\n")
cat("\nMost conserved edge types (top 10):\n")
print(conservation[1:10])
sink()

cat("=== analysis complete ===\n")
cat("outputs in",out_dir,"\n")
cat("  figures: js_heatmap.pdf dendrogram.pdf pca.pdf umap.pdf pair_class_boxplot.pdf conserved_edge_types.pdf lineage_composition.pdf\n")
cat("  tables: js_distance_matrix.csv edge_type_conservation.csv pair_classification.csv lineage_fractions.csv permanova_results.txt summary_stats.txt\n")
