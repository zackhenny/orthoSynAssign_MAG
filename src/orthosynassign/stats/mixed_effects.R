#!/usr/bin/env Rscript
# mixed_effects.R
#
# Mixed-effects logistic regression to account for genome-level clustering.
#
# Model:
#   P(is_split) ~ flank_score + flank_completeness + (1 | genome)
#
# The genome random intercept captures genome-wide fragmentation level,
# completeness, and contamination, giving unbiased fixed-effect estimates.
#
# Usage:
#   Rscript mixed_effects.R \
#       --table sog_gene_edge_long.csv \
#       --output_dir results/
#
# Outputs:
#   mixed_effects_results.tsv   — fixed-effect coefficients + 95 % CI
#   genome_random_effects.tsv   — per-genome random intercepts
#   mixed_effects_icc.txt       — intra-class correlation (ICC) value

suppressPackageStartupMessages({
    library(lme4)
    library(methods)
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
    result <- list(
        table      = "sog_gene_edge_long.csv",
        output_dir = ".",
        verbose    = FALSE
    )
    i <- 1
    while (i <= length(args)) {
        switch(args[i],
            "--table"      = { result$table      <- args[i + 1]; i <- i + 2 },
            "--output_dir" = { result$output_dir <- args[i + 1]; i <- i + 2 },
            "--verbose"    = { result$verbose <- TRUE; i <- i + 1 },
            { warning(paste("Unknown argument:", args[i])); i <- i + 1 }
        )
    }
    result
}

opts <- parse_args(args)
if (opts$verbose) {
    message("Table:      ", opts$table)
    message("Output dir: ", opts$output_dir)
}

dir.create(opts$output_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if (!file.exists(opts$table)) {
    stop(paste("Training table not found:", opts$table))
}

data <- read.csv(opts$table, stringsAsFactors = FALSE)
data_internal <- subset(data, edge_type == "internal")

if (nrow(data_internal) == 0) {
    stop("No internal-edge genes found in the training table.")
}
message(sprintf("Fitting model on %d internal genes across %d genomes",
                nrow(data_internal), length(unique(data_internal$genome))))

# Ensure response is numeric 0/1
data_internal$is_split <- as.integer(data_internal$is_split)

# ---------------------------------------------------------------------------
# Fit model
# ---------------------------------------------------------------------------
model <- tryCatch(
    glmer(
        is_split ~ flank_score + flank_completeness + (1 | genome),
        data   = data_internal,
        family = binomial,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    ),
    error = function(e) {
        stop(paste("glmer failed:", conditionMessage(e)))
    }
)

message("Model converged successfully.")

# ---------------------------------------------------------------------------
# Extract results
# ---------------------------------------------------------------------------

# Fixed effects with Wald 95% CI
fe <- as.data.frame(summary(model)$coefficients)
fe$term <- rownames(fe)
colnames(fe) <- c("Estimate", "Std_Error", "z_value", "Pr_z", "term")
fe$CI_lower <- fe$Estimate - 1.96 * fe$Std_Error
fe$CI_upper <- fe$Estimate + 1.96 * fe$Std_Error
fe <- fe[, c("term", "Estimate", "Std_Error", "z_value", "Pr_z", "CI_lower", "CI_upper")]

fixed_path <- file.path(opts$output_dir, "mixed_effects_results.tsv")
write.table(fe, fixed_path, sep = "\t", row.names = FALSE, quote = FALSE)
message("Wrote fixed-effect results to ", fixed_path)

# Random effects (genome intercepts)
re <- ranef(model)$genome
re$genome <- rownames(re)
colnames(re)[1] <- "random_intercept"
re <- re[, c("genome", "random_intercept")]
re <- re[order(re$random_intercept), ]

random_path <- file.path(opts$output_dir, "genome_random_effects.tsv")
write.table(re, random_path, sep = "\t", row.names = FALSE, quote = FALSE)
message("Wrote genome random effects to ", random_path)

# ICC: tau0^2 / (tau0^2 + pi^2/3)
vc <- as.data.frame(VarCorr(model))
tau0_sq <- vc$vcov[vc$grp == "genome"]
icc <- tau0_sq / (tau0_sq + (pi^2 / 3))
message(sprintf("ICC (genome random effect): %.4f", icc))
message(sprintf("  Interpretation: %.1f%% of residual variance explained by genome identity", icc * 100))

icc_path <- file.path(opts$output_dir, "mixed_effects_icc.txt")
cat(sprintf("ICC\t%.6f\ntau0_sq\t%.6f\n", icc, tau0_sq), file = icc_path)
message("Wrote ICC to ", icc_path)
