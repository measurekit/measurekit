# Identifiability of the DINA. Reference: CDM's din_identifiability, on 200 seeded random Q-matrices of 2 to 5 attributes
# and up to 12 items, every attribute in an item. Each leaves at least two items once a single-attribute item of each
# attribute is set aside, where din_identifiability's comparison of columns is defined.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(CDM))

set.seed(20261105)
identifiability <- lapply(seq_len(200), function(index) {
  repeat {
    K <- sample(2:5, 1)
    J <- sample((K + 2):12, 1)
    Q <- t(sapply(seq_len(J), function(j) {
      row <- integer(K)
      row[sample.int(K, sample(c(1, 1, 1, 2, 3), 1) |> min(K))] <- 1L
      row
    }))
    if (all(colSums(Q) > 0)) break
  }
  colnames(Q) <- paste0("a", seq_len(K))
  result <- din_identifiability(Q)
  stopifnot(!is.na(result$submat_distinct))
  rows <- lapply(seq_len(J), function(j) I(colnames(Q)[Q[j, ] == 1]))
  list(
    qMatrix = setNames(rows, paste0("item", seq_len(J))),
    singleItems = as.list(result$is_single),
    threeItems = as.list(result$is_three_items),
    distinctColumns = result$submat_distinct,
    dinaIdentified = as.logical(result$dina_identified)
  )
})
write_fixture("cdm-identifiability", list(source = "CDM::din_identifiability on seeded random Q-matrices.", cases = identifiability))
