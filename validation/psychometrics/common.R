# Helpers for the fixture scripts. Run the scripts from the repository root,
# e.g. `Rscript validation/psychometrics/ctt.R`.

suppressPackageStartupMessages(library(jsonlite))

# One BLAS thread, as validation/README.md runs every script: several sum in another order, which moves the last digits
# of some fixtures, such as the exploratory cases of multidimensional.R. R cannot set it once it has started.
for (variable in c("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")) {
  if (Sys.getenv(variable) != "1") stop("Set ", variable, "=1: the references run with one thread.")
}

# Writes `x` as packages/psychometrics/test/fixtures/<name>.json, with 17 significant digits and NA as null.
write_fixture <- function(name, x) {
  path <- file.path("packages/psychometrics/test/fixtures", paste0(name, ".json"))
  write_json(x, path, digits = I(17), auto_unbox = TRUE, na = "null", null = "null")
  message("Wrote ", path)
}

# The key of mirt::SAT12, whose responses are option numbers, 8 for an omission.
SAT12_KEY <- c(1, 4, 5, 2, 3, 1, 2, 1, 3, 1, 2, 4, 2, 1, 5, 3, 4, 4, 1, 4, 3, 3, 4, 1, 3, 5, 1, 3, 1, 5, 4, 5)

# Returns mirt's model of `data` with the items set to slope-intercept coefficients, one data frame column per name,
# NA where an item lacks the name. The items are set, not estimated: mirt evaluates its starting values when TOL = NaN.
set_items <- function(data, itemtype, coefficients) {
  values <- mirt::mirt(data, 1, itemtype = itemtype, pars = "values")
  for (i in seq_len(nrow(coefficients))) {
    for (name in names(coefficients)) {
      value <- coefficients[[name]][i]
      if (!is.na(value)) values$value[values$item == colnames(data)[i] & values$name == name] <- value
    }
  }
  mirt::mirt(data, 1, itemtype = itemtype, pars = values, TOL = NaN, verbose = FALSE)
}

# measurekit's parameters of an item from its IRT coefficients; for the rsm, whose thresholds are b_j - c / a1, the
# location and the steps around it.
psychometrics_parameters <- function(model, irt) {
  numbered <- function(prefix) unname(irt[grep(paste0("^", prefix, "[0-9]+$"), names(irt))])
  switch(model,
    "2pl" = , "3pl" = , "4pl" = list(a = irt[["a"]], b = irt[["b"]], c = irt[["g"]], d = irt[["u"]]),
    grm = , gpcm = , pcm = list(a = irt[["a"]], b = I(numbered("b"))),
    nrm = list(a = I(numbered("a")), c = I(numbered("c"))),
    rsm = {
      thresholds <- numbered("b") - irt[["c"]] / irt[["a1"]]
      list(a = irt[["a1"]], b = mean(thresholds), steps = I(thresholds - mean(thresholds)))
    }
  )
}

# Weights of persons in the group of the `fraction` * n highest totals. Persons tied at the boundary share the
# remaining places equally. Pass -total for the lower group.
group_weights <- function(total, fraction) {
  remaining <- fraction * length(total)
  weights <- numeric(length(total))
  for (score in sort(unique(total), decreasing = TRUE)) {
    tied <- which(total == score)
    weights[tied] <- min(max(remaining, 0), length(tied)) / length(tied)
    remaining <- remaining - length(tied)
  }
  weights
}

weighted_share <- function(weights, x) sum(weights * x) / sum(weights)

# Pearson correlation, NA when either variable is constant.
correlation <- function(x, y) suppressWarnings(cor(x, y))

# The probability of each total score at each node, by score then node, by the recursion of Lord and Wingersky (1984),
# from mirt's probtrace.
score_likelihoods <- function(fit, nodes) {
  likelihoods <- matrix(1, 1, length(nodes))
  for (i in seq_len(extract.mirt(fit, "nitems"))) {
    trace <- probtrace(extract.item(fit, i), nodes)
    scores <- nrow(likelihoods)
    updated <- matrix(0, scores + ncol(trace) - 1, length(nodes))
    for (k in seq_len(ncol(trace))) {
      rows <- k:(k + scores - 1)
      updated[rows, ] <- updated[rows, ] + likelihoods * matrix(trace[, k], scores, length(nodes), byrow = TRUE)
    }
    likelihoods <- updated
  }
  likelihoods
}

# The Q-matrix's columns in measurekit's order: attributes by first appearance over the items, so that both libraries
# order the profiles and each item's latent groups alike.
psychometrics_order <- function(Q) {
  first <- unique(unlist(lapply(seq_len(nrow(Q)), function(j) which(Q[j, ] == 1))))
  Q[, first, drop = FALSE]
}

# The Q-matrix as measurekit takes it: the attributes of each item, by item id.
qmatrix <- function(Q, items) {
  setNames(lapply(seq_len(nrow(Q)), function(j) I(colnames(Q)[Q[j, ] == 1])), items)
}
