# Fit of cognitive diagnosis models. References: GDINA's modelfit (M2, RMSEA2, SRMSR) and itemfit, whose expected values
# are a simulation of N.resampling persons, CDM's modelfit.cor.din, whose expected correlations are exact, and base R,
# which writes out GDINA's item statistics with the exact expected values. Data: GDINA's sim10GDINA (1000 persons, 10
# items, 3 attributes) under each model, and seeded sets of a true G-DINA model for its rejections.
#
# Where GDINA's Δ has a rank below its columns, as for the G-DINA and the A-CDM here, modelfit takes nrow - ncol columns
# of qr.Q's complement, which then depend on the order of Δ's columns: its M2 is not the statistic. The reference is then
# the statistic on the whole complement, from GDINA's own Δ, Ξ and e, with nrow - rank degrees of freedom; where Δ has
# full rank the same code reproduces GDINA's M2.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(GDINA)
  library(CDM)
})

CONTROL <- list(conv.crit = 1e-9, maxitr = 20000)
ADDITIVE_CONTROL <- list(conv.crit = 1e-8, maxitr = 5000) # as cdm.R: the additive M-steps stop near 1e-8
RESAMPLING <- 1e6
ZERO_CELL <- 0.5 # what GDINA's fitstats counts an empty observed cell as

data_matrix <- as.matrix(sim10GDINA$simdat)
q_matrix <- sim10GDINA$simQ
items <- paste0("item", seq_len(ncol(data_matrix)))
colnames(data_matrix) <- items
attributes <- paste0("a", seq_len(ncol(q_matrix)))
q_list <- setNames(lapply(seq_len(nrow(q_matrix)), function(j) I(attributes[q_matrix[j, ] == 1])), items)

fit_gdina <- function(data, model) {
  additive <- any(model %in% c("ACDM", "LLM", "RRUM"))
  fit <- GDINA(data, q_matrix, model = model, verbose = 0, control = if (additive) ADDITIVE_CONTROL else CONTROL)
  stopifnot(extract(fit, "convergence"))
  fit
}

# The statistics of `itemfit`, with the expected values of the fit written out as exact sums over the profiles.
item_statistics <- function(fit, data) {
  probabilities <- t(extract(fit, "LCprob.parm"))
  weights <- extract(fit, "posterior.prob")
  weights <- drop(weights)
  N <- nrow(data)
  J <- ncol(data)
  expected <- drop(crossprod(weights, probabilities))
  z_proportion <- abs(colMeans(data) - expected) / sqrt(expected * (1 - expected) / N)
  z_correlation <- z_odds <- p_correlation <- p_odds <- matrix(NA_real_, J, J)
  for (i in seq_len(J - 1)) {
    for (j in (i + 1):J) {
      both <- sum(weights * probabilities[, i] * probabilities[, j])
      cells <- c(both, expected[i] - both, expected[j] - both, 1 - expected[i] - expected[j] + both)
      counts <- c(sum(data[, i] * data[, j]), sum(data[, i] * (1 - data[, j])), sum((1 - data[, i]) * data[, j]),
                  sum((1 - data[, i]) * (1 - data[, j])))
      counts[counts == 0] <- ZERO_CELL
      implied <- (both - expected[i] * expected[j]) / sqrt(expected[i] * (1 - expected[i]) * expected[j] * (1 - expected[j]))
      z <- abs(atanh(cor(data[, i], data[, j])) - atanh(implied)) * sqrt(N - 3)
      odds <- abs(log(counts[1] * counts[4] / (counts[2] * counts[3])) - log(cells[1] * cells[4] / (cells[2] * cells[3])))
      w <- odds / sqrt(sum(1 / (N * cells)))
      z_correlation[i, j] <- z_correlation[j, i] <- z
      z_odds[i, j] <- z_odds[j, i] <- w
    }
  }
  tail <- function(z) 2 * pnorm(z, lower.tail = FALSE)
  largest <- function(table) apply(table, 1, max, na.rm = TRUE)
  adjusted <- function(table) sapply(seq_len(J), function(i) {
    p <- tail(table[i, -i])
    p.adjust(p, "holm")[which.max(table[i, -i])]
  })
  list(observed = colMeans(data), expected = expected, proportionZ = z_proportion, correlationZ = largest(z_correlation),
       logOddsZ = largest(z_odds), significance = pmin(adjusted(z_correlation), adjusted(z_odds)))
}

# modelfit up to its statistic: GDINA's Δ, Ξ, e and N.
modelfit_parts <- local({
  source <- deparse(GDINA::modelfit)
  line <- grep("tmp <- qr.Q\\(qr\\(delt\\)", source)
  stopifnot(length(line) == 1)
  source[line] <- "return(list(delt = delt, Xi2 = Xi2, p = p, e = e, N = N))"
  parts <- eval(parse(text = source))
  environment(parts) <- asNamespace("GDINA")
  parts
})

RANK_TOLERANCE <- 1e-9

# M2 on the whole complement of Δ, with its degrees of freedom, and the same statistic with Δ's columns reversed as
# GDINA computes it, which differs when Δ's rank is below its columns.
whole_complement <- function(fit) {
  parts <- modelfit_parts(fit)
  delta <- parts$delt
  decomposition <- qr(delta, tol = RANK_TOLERANCE)
  complement <- qr.Q(decomposition, complete = TRUE)[, (decomposition$rank + 1):nrow(delta), drop = FALSE]
  form <- function(basis) {
    residuals <- parts$p - parts$e
    drop(parts$N * t(residuals) %*% basis %*% solve(t(basis) %*% parts$Xi2 %*% basis) %*% t(basis) %*% residuals)
  }
  reversed <- qr.Q(qr(delta[, ncol(delta):1]), complete = TRUE)[, (ncol(delta) + 1):nrow(delta), drop = FALSE]
  list(statistic = form(complement), degreesOfFreedom = nrow(delta) - decomposition$rank,
       full = decomposition$rank == ncol(delta), reversed = form(reversed), persons = parts$N)
}

case_of <- function(name, models) {
  fit <- fit_gdina(data_matrix, models)
  fit_statistics <- modelfit(fit)
  wider <- modelfit(fit, CI = 0.95)
  whole <- whole_complement(fit)
  if (whole$full) {
    stopifnot(abs(whole$statistic - fit_statistics$M2) < 1e-6, whole$degreesOfFreedom == fit_statistics$M2.df)
  } else {
    stopifnot(abs(whole$reversed - fit_statistics$M2) > 0.1)
    m2 <- whole$statistic
    df <- whole$degreesOfFreedom
    fit_statistics$M2 <- m2
    fit_statistics$M2.df <- df
    fit_statistics$M2.pvalue <- pchisq(m2, df, lower.tail = FALSE)
    fit_statistics$RMSEA2 <- sqrt(max((m2 - df) / (whole$persons * df), 0))
    fit_statistics$RMSEA2.CI <- GDINA:::RMSEAfun(m2, df, whole$persons, 0.9)
    wider$RMSEA2.CI <- GDINA:::RMSEAfun(m2, df, whole$persons, 0.95)
  }
  resampled <- itemfit(fit, N.resampling = RESAMPLING, digits = 12, randomseed = 20261104)
  table <- resampled$max.itemlevel.fit
  list(
    name = name,
    model = I(tolower(models)),
    gdina = if (!whole$full) list(m2 = modelfit(fit)$M2, degreesOfFreedom = modelfit(fit)$M2.df, reversed = whole$reversed),
    m2 = fit_statistics$M2, degreesOfFreedom = fit_statistics$M2.df, significance = fit_statistics$M2.pvalue,
    rmsea = fit_statistics$RMSEA2, rmseaInterval = unname(fit_statistics$RMSEA2.CI),
    wideInterval = unname(wider$RMSEA2.CI), srmsr = fit_statistics$SRMSR,
    exact = item_statistics(fit, data_matrix),
    resampled = list(proportionZ = unname(table[, "z.prop"]), correlationZ = unname(table[, "max[z.r]"]),
                     logOddsZ = unname(table[, "max[z.logOR]"]))
  )
}

MODELS <- c("GDINA", "DINA", "DINO", "ACDM", "LLM", "RRUM")
mixed <- rep(c("GDINA", "DINA", "DINO", "ACDM", "LLM"), 2)
cases <- c(lapply(MODELS, function(model) case_of(model, rep(model, ncol(data_matrix)))), list(case_of("one model per item", mixed)))

# CDM's exact residual correlations of the DINA, on its own estimates.
din_fit <- din(data_matrix, q.matrix = q_matrix, rule = "DINA", progress = FALSE, conv.crit = 1e-9, dev.crit = 1e-9, maxit = 20000)
pairs <- modelfit.cor.din(din_fit)
cdm_pairs <- pairs$itempairs
cdm_largest <- sapply(seq_len(ncol(data_matrix)), function(j) {
  rows <- cdm_pairs$item1 == items[j] | cdm_pairs$item2 == items[j]
  max(abs(cdm_pairs$fcor_z[rows]))
})

# M2 of a true model on sets of its own data.
SETS <- 60
PERSONS <- 400
set.seed(20261104)
true_items <- lapply(seq_len(nrow(q_matrix)), function(j) {
  mastered <- rowSums(attributepattern(sum(q_matrix[j, ])))
  0.15 + 0.7 * mastered / max(mastered)
})
rejections <- lapply(seq_len(SETS), function(set) {
  simulated <- simGDINA(PERSONS, q_matrix, catprob.parm = true_items)
  data <- extract(simulated, "dat")
  fit <- GDINA(data, q_matrix, model = "GDINA", verbose = 0, control = list(conv.crit = 1e-6, maxitr = 5000))
  whole <- whole_complement(fit)
  list(responses = apply(data, 1, paste, collapse = ""), statistic = whole$statistic, degreesOfFreedom = whole$degreesOfFreedom)
})

write_fixture("cdm-fit", list(
  source = "GDINA's modelfit and itemfit (N.resampling = 1e6), CDM's modelfit.cor.din, base R's exact sums.",
  responses = unname(lapply(seq_len(nrow(data_matrix)), function(i) as.numeric(data_matrix[i, ]))),
  items = items,
  qMatrix = q_list,
  cases = cases,
  din = list(srmsr = pairs$modelfit.stat["SRMSR", "est"], correlationZ = cdm_largest),
  rejections = list(sets = rejections, alpha = 0.05)
))
