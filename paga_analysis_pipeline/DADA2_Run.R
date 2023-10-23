# rm(list=ls())
# R on server = /usr/local/lib/R-4.1.2/bin/R
.libPaths(c('/specific/elhanan/PROJECTS/BV_MANIFOLD_MT/R_Packages/', .libPaths()))

library(dada2)
library(dplyr)
library(stringr)
library(ShortRead)
library(ggplot2)
library(reshape2)

############
#### Variables ####
fastq_path = '/specific/elhanan/PROJECTS/BV_MANIFOLD_MT/Ravel2013/fastq'
taxonomy_path = '/specific/elhanan/PROJECTS/BV_MANIFOLD_MT/Taxonomy/Training_OLD/silva_nr99_v138.1_train_set.fa.gz'
species_path = '/specific/elhanan/PROJECTS/BV_MANIFOLD_MT/Taxonomy/Training_OLD/silva_species_assignment_v138.1.fa.gz'
qualityFunction_path = '/specific/elhanan/PROJECTS/BV_MANIFOLD_MT/Scripts/R/plotQualityProfile_MT_sumOnly.R'
save_path = '/specific/elhanan/PROJECTS/BV_MANIFOLD_MT/Ravel2013/DADA2_results/'

## Samples path
samples_path <- sort(list.files(fastq_path, pattern = ".fastq", full.names = TRUE))
samples_names <- str_replace(basename(samples_path), '.fastq', '')

## Quality plot
source(qualityFunction_path)
sum_quality_lst = plotQualityProfile(samples_path, aggregate = TRUE)

## Save
write.csv(sum_quality_lst, paste0(save_path, 'quality_stats_sum.csv'))

## Filter and trim
filt_path <- file.path(fastq_path, "filtered", paste0(samples_names, "_filt.fastq.gz"))
names(filt_path) <- samples_names

out_table <- filterAndTrim(samples_path, filt_path, truncLen = 597,
                           maxN = 0, maxEE = 2, truncQ = 2, rm.phix = TRUE,
                           compress = TRUE, multithread = FALSE)

filt_path <- filt_path[file.exists(filt_path)]

## Error rates
err_rate <- learnErrors(filt_path, multithread = TRUE)

## DADA create ASV for each sample
dada_asv <- dada(filt_path, err = err_rate, multithread = TRUE, HOMOPOLYMER_GAP_PENALTY = -1, BAND_SIZE = 32)
seq_asv = makeSequenceTable(dada_asv)

## Remove chimeras
seq_nochim <- removeBimeraDenovo(seq_asv, method = "consensus", multithread = TRUE, verbose = TRUE)

write.csv(seq_nochim, paste0(save_path, 'seq_nochim.csv'))

## Track changes in the process
getN <- function(x) sum(getUniques(x))
track_table <- cbind(out_table, sapply(dada_asv, getN), rowSums(seq_nochim))
colnames(track_table) <- c("input", "filtered", "denoised", "nochim")
rownames(track_table) <- samples_names

write.csv(track_table, paste0(save_path, 'track_process.csv'))

## Assign taxonomy
taxa_assigned <- assignTaxonomy(seq_nochim, taxonomy_path, multithread = TRUE)
taxa_assigned2 <- addSpecies(taxa_assigned, species_path)

write.csv(taxa_assigned, paste0(save_path, 'taxa_assigned.csv'))
write.csv(taxa_assigned2, paste0(save_path, 'taxa_assigned2.csv'))


## Create abundance table
taxa_ASV = taxa_assigned2
abundance_ASV = seq_nochim

names_ASV <- paste0("ASV", seq(nrow(taxa_assigned2))) 
seqs <- rownames(taxa_assigned2)
seq_df = data.frame(sequences = seqs, ASV = names_ASV)

row.names(taxa_ASV) <- names_ASV
colnames(abundance_ASV) <- names_ASV

write.csv(seq_df, paste0(save_path, 'seq_ASV_df.csv'))
write.csv(taxa_ASV, paste0(save_path, 'taxa_ASV2.csv'))
write.csv(abundance_ASV, paste0(save_path, 'abundance_ASV.csv'))


