suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(ggrepel)
  library(pheatmap); library(RColorBrewer); library(patchwork)
})

args<-commandArgs(trailingOnly=TRUE)
if(length(args)<2) stop("usage: Rscript deep_gene_conservation_v2.R <out_dir> <signal_dir>")
out_dir<-args[1]; sig_dir<-args[2]
dir.create(file.path(out_dir,"figures"),showWarnings=FALSE,recursive=TRUE)
dir.create(file.path(out_dir,"tables"),showWarnings=FALSE,recursive=TRUE)

files<-list.files(sig_dir,pattern="_gene_signal\\.csv$",full.names=TRUE)
sigs<-list()
for(f in files){
  lab<-sub("_gene_signal\\.csv$","",basename(f))
  d<-fread(f)
  sigs[[lab]]<-d
  cat(lab,": n_genes=",nrow(d),"  n_edges=",d$n_edges[1],"\n")
}

cross_panel<-function(labels,title_tag,out_prefix){
  if(!all(labels%in%names(sigs))){
    cat(title_tag,"missing some labels:",setdiff(labels,names(sigs)),"\n"); return(NULL)
  }
  inter<-Reduce(intersect,lapply(labels,function(l) sigs[[l]]$gene))
  cat(title_tag,"intersecting genes:",length(inter),"\n")
  if(length(inter)<5) return(NULL)
  M<-sapply(labels,function(l){
    d<-sigs[[l]]; setkey(d,gene); d[inter,mean]
  })
  rownames(M)<-inter
  M_log<-log1p(M)
  rk<-apply(M,2,function(x) rank(-x)/length(x))
  mean_rank<-rowMeans(rk)
  cv<-apply(M,1,function(x){ m<-mean(x); if(m==0) return(NA); sd(x)/m })
  gene_tab<-data.table(gene=inter,mean_signal=rowMeans(M),rank_mean=mean_rank,
                       cv=cv,max_signal=apply(M,1,max),min_signal=apply(M,1,min))
  for(l in labels) gene_tab[[paste0("signal_",l)]]<-as.numeric(M[,l])
  setorder(gene_tab,rank_mean)
  fwrite(gene_tab,file.path(out_dir,"tables",paste0(out_prefix,"_gene_table.csv")))

  if(ncol(M)==2){
    rho<-cor(M_log[,1],M_log[,2])
    sr<-cor(M[,1],M[,2],method="spearman")
    pdf<-as.data.frame(M_log); pdf$gene<-rownames(pdf)
    topg<-gene_tab[1:30,gene]
    pdf$label_g<-ifelse(pdf$gene%in%topg,pdf$gene,"")
    g<-ggplot(pdf,aes(x=.data[[labels[1]]],y=.data[[labels[2]]],label=label_g))+
      geom_point(alpha=0.35,size=1)+geom_abline(slope=1,intercept=0,color="grey50",linetype="dashed")+
      geom_smooth(method="lm",se=FALSE,color="firebrick")+
      geom_text_repel(size=2.6,max.overlaps=80,segment.alpha=0.4)+
      theme_bw()+labs(x=paste("log1p mean junctional",labels[1]),
                       y=paste("log1p mean junctional",labels[2]),
                       title=sprintf("%s — gene-level junctional concordance",title_tag),
                       subtitle=sprintf("Pearson(log1p)=%.3f, Spearman=%.3f, n=%d shared genes",rho,sr,length(inter)))
    ggsave(file.path(out_dir,"figures",paste0(out_prefix,"_gene_scatter.pdf")),g,width=8.5,height=8.5)
    return(list(gene_tab=gene_tab,pearson=rho,spearman=sr,n=length(inter)))
  } else {
    cor_mat<-cor(M_log,use="pairwise.complete.obs")
    sp_mat<-cor(M,use="pairwise.complete.obs",method="spearman")
    fwrite(data.table(dataset=rownames(cor_mat),cor_mat),
           file.path(out_dir,"tables",paste0(out_prefix,"_pearson_log.csv")))
    fwrite(data.table(dataset=rownames(sp_mat),sp_mat),
           file.path(out_dir,"tables",paste0(out_prefix,"_spearman.csv")))
    pdf(file.path(out_dir,"figures",paste0(out_prefix,"_correlation_heatmap.pdf")),width=8,height=7)
    pheatmap(cor_mat,display_numbers=round(cor_mat,3),
             main=sprintf("%s gene-level Pearson(log1p) (n=%d shared genes)",title_tag,length(inter)))
    dev.off()
    topg<-gene_tab[1:30,gene]
    M_top<-M_log[topg,]
    pdf(file.path(out_dir,"figures",paste0(out_prefix,"_top_gene_heatmap.pdf")),width=8,height=9)
    pheatmap(M_top,scale="row",cluster_cols=TRUE,
             main=sprintf("%s — top 30 conserved junctional genes",title_tag))
    dev.off()
    return(list(gene_tab=gene_tab,pearson_mat=cor_mat,spearman_mat=sp_mat,n=length(inter)))
  }
}

panels<-list(
  brain_3=list(labels=c("xenium_mouse_brain","allen_merfish","stereoseq_brain"),
               tag="Mouse brain (3 platforms)",prefix="brain_3plat"),
  brain_xenium_allen=list(labels=c("xenium_mouse_brain","allen_merfish"),
               tag="Xenium MB vs Allen MERFISH MB",prefix="brain_xenium_allen"),
  brain_xenium_stereo=list(labels=c("xenium_mouse_brain","stereoseq_brain"),
               tag="Xenium MB vs Stereo-seq MB",prefix="brain_xenium_stereo"),
  brain_allen_stereo=list(labels=c("allen_merfish","stereoseq_brain"),
               tag="Allen MERFISH MB vs Stereo-seq MB",prefix="brain_allen_stereo"),
  colon=list(labels=c("xenium_colon","cosmx_colon"),
               tag="Human colon (Xenium vs CosMx)",prefix="colon_xenium_cosmx"),
  breast=list(labels=c("xenium_breast","visium_breast"),
               tag="Human breast (Xenium vs Visium)",prefix="breast_xenium_visium"),
  dev_vs_adult=list(labels=c("stereoseq_embryo_e95","stereoseq_brain"),
               tag="Embryo E9.5 vs adult brain (Stereo-seq, devel axis)",prefix="dev_vs_adult"),
  cosmx_tissues=list(labels=c("cosmx_colon","cosmx_pancreas"),
               tag="CosMx colon vs pancreas (within-platform tissue)",prefix="cosmx_colon_pancreas")
)

results<-list()
for(nm in names(panels)){
  cat("\n--- ",nm," ---\n",sep="")
  r<-cross_panel(panels[[nm]]$labels,panels[[nm]]$tag,panels[[nm]]$prefix)
  if(!is.null(r)) results[[nm]]<-r
}

summary_dt<-data.table()
for(nm in names(results)){
  r<-results[[nm]]
  if(!is.null(r$pearson)){
    summary_dt<-rbind(summary_dt,data.table(panel=nm,
      tag=panels[[nm]]$tag,n_shared=r$n,pearson=r$pearson,spearman=r$spearman),fill=TRUE)
  } else if(!is.null(r$pearson_mat)){
    pm<-r$pearson_mat; sm<-r$spearman_mat
    mean_p<-mean(pm[upper.tri(pm)]); mean_s<-mean(sm[upper.tri(sm)])
    summary_dt<-rbind(summary_dt,data.table(panel=nm,
      tag=panels[[nm]]$tag,n_shared=r$n,pearson=mean_p,spearman=mean_s),fill=TRUE)
  }
}
fwrite(summary_dt,file.path(out_dir,"tables","panel_concordance_summary.csv"))

sink(file.path(out_dir,"tables","gene_conservation_summary.txt"))
cat("=== Gene-level junctional concordance across panels ===\n\n")
print(summary_dt[,.(panel,n_shared,pearson=round(pearson,3),spearman=round(spearman,3))])
cat("\n--- top conserved junctional genes per panel ---\n")
for(nm in names(results)){
  r<-results[[nm]]
  cat(sprintf("\n=== %s (n=%d shared) ===\n",panels[[nm]]$tag,r$n))
  print(r$gene_tab[1:min(15,nrow(r$gene_tab)),
                   .(gene,rank_mean=round(rank_mean,3),mean=round(mean_signal,3),cv=round(cv,3))])
}
sink()

cat("\n=== analysis complete ===\n")
