options(bitmapType='cairo')
.libPaths("/mnt/aiongpfs/users/sjanmahanthi/Rlibs")

library(Seurat)
library(ggplot2)
library(dplyr)
library(tidyr)
library(pheatmap)

# ── Paths ─────────────────────────────────────────────────────────────────────
DAT_DIR     <- "/home/users/sjanmahanthi/Data_Test/Final_Dataset"
RESULTS_DIR <- "/home/users/sjanmahanthi/final_final_final_sctranslator/Temporal_scTranslator/new_results/film_pre_combined_loss_deg1000_leakfix"
OUT_DIR     <- "/home/users/sjanmahanthi/Data_Test/Results"
dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

# ── Colors ────────────────────────────────────────────────────────────────────
TIME_COLORS <- c("0"="lightgrey", "2"="lightblue", "4"="skyblue",
                 "6"="steelblue", "60"="navy", "180"="black")

ORANGE_BLUE_DIVERGING <- c("#B94701","#E67E22","#F6C28B","#FFFFFF",
                            "#BFDDF2","#4C97C9","#08519C")
ORANGE_BLUE_SEQUENTIAL <- colorRampPalette(c(
    "#D95F02","#F4A261","#F6E8D5","#F7F7F7","#DCEAF7","#6BAED6","#2171B5"
))(100)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Protein Expression Heatmap by Timepoint
# ══════════════════════════════════════════════════════════════════════════════
cat("Loading data for expression heatmap...\n")

rna_df  <- read.csv(file.path(DAT_DIR, "rna_clean.csv"),  row.names=1, check.names=FALSE)
prot_df <- read.csv(file.path(DAT_DIR, "prot_clean.csv"), row.names=1, check.names=FALSE)
meta_df <- read.csv(file.path(DAT_DIR, "meta_clean.csv"), row.names=1)

rna_mat  <- t(as.matrix(rna_df))
prot_mat <- t(as.matrix(prot_df))

seurat_obj <- CreateSeuratObject(counts=rna_mat, meta.data=meta_df)
seurat_obj[["Protein"]] <- CreateAssayObject(counts=prot_mat)
seurat_obj$time_min <- factor(seurat_obj$time_min, levels=c(0,2,4,6,60,180))

DefaultAssay(seurat_obj) <- "Protein"
seurat_obj <- NormalizeData(seurat_obj, assay="Protein",
                             normalization.method="CLR", margin=2)
seurat_obj <- ScaleData(seurat_obj, assay="Protein")

cat("Building expression heatmap...\n")

prot_scaled <- GetAssayData(seurat_obj, assay="Protein", layer="scale.data")
prot_scaled[prot_scaled > 2]  <- 2
prot_scaled[prot_scaled < -2] <- -2

cell_order  <- order(seurat_obj$time_min)
prot_scaled <- prot_scaled[, cell_order]

annotation_col <- data.frame(Timepoint = seurat_obj$time_min[cell_order])
rownames(annotation_col) <- colnames(prot_scaled)
annotation_colors <- list(Timepoint = TIME_COLORS)

grDevices::png(file.path(OUT_DIR, "expression_heatmap.png"),
    width=3600, height=3000, res=300, type="cairo-png")
pheatmap(prot_scaled,
    color             = ORANGE_BLUE_SEQUENTIAL,
    cluster_cols      = FALSE,
    cluster_rows      = TRUE,
    annotation_col    = annotation_col,
    annotation_colors = annotation_colors,
    show_colnames     = FALSE,
    fontsize_row      = 6,
    border_color      = NA,
    main              = "Protein Expression Heatmap by Timepoint")
dev.off()

cat("Saved expression_heatmap.png\n")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Per-protein Pearson R Heatmap
# ══════════════════════════════════════════════════════════════════════════════
cat("Computing per-protein Pearson R from prediction/truth files...\n")

# All 10 timepoint pairs used in the thesis
pair_list <- list(
    c(0, 2), c(0, 4), c(0, 6), c(0, 60), c(0, 180),
    c(2, 6), c(4, 60), c(6, 60), c(6, 180), c(60, 180)
)

per_protein_r_list <- list()

for (pair in pair_list) {
    t_src <- pair[1]
    t_tgt <- pair[2]
    pair_tag <- paste0("t", t_src, "to", t_tgt)

    pred_path  <- file.path(RESULTS_DIR, paste0("joint_y_pred_", pair_tag, ".csv"))
    truth_path <- file.path(RESULTS_DIR, paste0("joint_y_truth_", pair_tag, ".csv"))

    if (!file.exists(pred_path) || !file.exists(truth_path)) {
        cat(paste0("  Skipping ", pair_tag, " (files not found)\n"))
        next
    }

    pred_df  <- read.csv(pred_path)
    truth_df <- read.csv(truth_path)

    # Protein columns = everything except t_src, t_tgt
    protein_cols <- setdiff(colnames(pred_df), c("t_src", "t_tgt"))

    r_values <- sapply(protein_cols, function(protein) {
        pred_vals  <- pred_df[[protein]]
        truth_vals <- truth_df[[protein]]
        if (sd(pred_vals) == 0 || sd(truth_vals) == 0) {
            return(NA)
        }
        cor(pred_vals, truth_vals, method = "pearson")
    })

    per_protein_r_list[[pair_tag]] <- data.frame(
        timepoint_pair  = paste0(t_src, "->", t_tgt),
        protein         = protein_cols,
        mean_pearson_r  = as.numeric(r_values)
    )

    cat(paste0("  Done: ", pair_tag, " (", length(protein_cols), " proteins)\n"))
}

results_df <- bind_rows(per_protein_r_list)

# Save the computed per-protein R table for reuse (avoids recomputing later)
write.csv(results_df, file.path(OUT_DIR, "per_protein_pearson_r_all_pairs.csv"),
          row.names = FALSE)
cat("Saved per_protein_pearson_r_all_pairs.csv\n")

# ── PTM vs Total-protein grouping ─────────────────────────────────────────────
ptm_channels <- c("p-p65","p-JNK","p-p38","p-PLC-y2","p-SHP-2","p-SHP-1","p-S6",
    "p-TOR","p-BLNK","p-PKC-b1","p-IRAK4","p-CD79a","p-CDK1","p-CDK4",
    "p-AMPK-a1/2","p-AMPK-b1","p-JAK1","p-CDK6","p-STAT1","p-STAT3","p-STAT5","p-STAT6")

results_df <- results_df %>%
    mutate(group = ifelse(protein %in% ptm_channels, "PTM", "Total protein"))

# Order timepoint pairs to match thesis convention
pair_order <- sapply(pair_list, function(p) paste0(p[1], "->", p[2]))
results_df$timepoint_pair <- factor(results_df$timepoint_pair, levels = pair_order)

vmax <- max(abs(results_df$mean_pearson_r), na.rm = TRUE)

cat("Building Pearson R heatmap...\n")

p <- ggplot(results_df, aes(x = timepoint_pair, y = protein, fill = mean_pearson_r)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = sprintf("%.2f", mean_pearson_r)), size = 2.0, color = "black") +
    scale_fill_gradientn(colors = ORANGE_BLUE_DIVERGING, limits = c(-vmax, vmax),
                          name = "Pearson R", na.value = "grey90") +
    facet_grid(rows = vars(group), scales = "free_y", space = "free_y", switch = "y") +
    labs(title = "Per-protein Pearson R across timepoint pairs",
         x = "Timepoint pair", y = NULL) +
    theme_minimal(base_size = 9) +
    theme(
        plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 7),
        panel.grid = element_blank(),
        strip.placement = "outside",
        strip.text.y.left = element_text(face = "bold", angle = 90, size = 10),
        strip.background = element_blank(),
        panel.spacing = unit(0.15, "lines")
    )

ggsave(file.path(OUT_DIR, "pearson_r_heatmap.png"), p,
       width = 9, height = 12, dpi = 300, bg = "white")

cat("Saved pearson_r_heatmap.png\n")

cat("\n=== DONE ===\n")
cat(paste("Results saved to:", OUT_DIR, "\n"))