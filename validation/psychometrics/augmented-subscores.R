# Augmented subscore fixtures. Reference: the subscore package's subscore.Wainer (Wainer et al., 2001, in Thissen and
# Wainer), checked against the same computation written out in base R: with x the section sums of the persons who
# answered every item, S their covariance matrix and S_true that matrix with each variance replaced by alpha times it,
# the augmented subscores are mean(x) + B (x - mean(x)) with B = S_true S^-1, and the reliability of subscore j is
# [B B S_true]_jj / [B S_true]_jj. A seeded simulation with known true subscores shows what augmenting buys.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(subscore))

# Cronbach's alpha of the columns of `x`.
alpha_of <- function(x) {
  k <- ncol(x)
  k / (k - 1) * (1 - sum(apply(x, 2, var)) / var(rowSums(x)))
}

# The augmented subscores of the persons without missing responses to the items of `sections`, in base R.
wainer_base <- function(x, sections) {
  x <- na.omit(x[, unlist(sections), drop = FALSE])
  sums <- sapply(sections, function(columns) rowSums(x[, match(columns, colnames(x)), drop = FALSE]))
  rho <- sapply(sections, function(columns) alpha_of(x[, match(columns, colnames(x)), drop = FALSE]))
  observed <- cov(sums)
  true <- observed
  diag(true) <- rho * diag(observed)
  weights <- true %*% solve(observed)
  centered <- sweep(sums, 2, colMeans(sums))
  scores <- sweep(centered %*% t(weights), 2, colMeans(sums), "+")
  explained <- weights %*% true
  list(scores = scores, weights = weights, reliability = unname(rho),
       augmentedReliability = diag(weights %*% explained) / diag(explained))
}

# subscore.Wainer on the same persons, items side by side as data.prep takes them.
wainer_package <- function(x, sections) {
  x <- as.data.frame(na.omit(x[, unlist(sections), drop = FALSE]))
  test_data <- data.prep(x, c(length(sections), lengths(sections)), c(names(sections), "total"))
  subscore.Wainer(test_data)
}

wainer_case <- function(name, x, sections, truth = NULL) {
  base <- wainer_base(x, sections)
  reference <- wainer_package(x, sections)
  stopifnot(max(abs(base$scores - reference$subscore.augmented)) < 1e-9,
            max(abs(base$augmentedReliability - reference$summary[, "Augmented.reliability"])) < 1e-12,
            max(abs(colMeans(base$scores) - reference$summary[, "mean"])) < 1e-9,
            max(abs(apply(base$scores, 2, sd) - reference$summary[, "SD"])) < 1e-9)
  persons <- which(complete.cases(x[, unlist(sections)]))
  message(name, ": ", length(persons), " complete cases; reliability ", paste(round(base$reliability, 3), collapse = ", "),
          " augmented to ", paste(round(base$augmentedReliability, 3), collapse = ", "))
  result <- list(name = name, itemIds = colnames(x),
                 responses = unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))),
                 sections = sections, sectionNames = names(sections), persons = persons - 1L,
                 scores = unname(lapply(seq_len(nrow(base$scores)), function(i) base$scores[i, ])),
                 weights = unname(lapply(seq_len(nrow(base$weights)), function(i) base$weights[i, ])),
                 reliability = base$reliability, augmentedReliability = unname(base$augmentedReliability),
                 mean = unname(colMeans(base$scores)), standardDeviation = unname(apply(base$scores, 2, sd)))
  if (!is.null(truth)) {
    truth <- truth[persons, , drop = FALSE]
    observed <- sapply(sections, function(columns) rowSums(x[persons, columns, drop = FALSE]))
    result$truth <- unname(lapply(seq_len(nrow(truth)), function(i) truth[i, ]))
    result$observedCorrelation <- sapply(seq_len(ncol(truth)), function(j) cor(observed[, j], truth[, j]))
    result$augmentedCorrelation <- sapply(seq_len(ncol(truth)), function(j) cor(base$scores[, j], truth[, j]))
    message("  correlation with the true subscores: observed ", round(mean(result$observedCorrelation), 3),
            ", augmented ", round(mean(result$augmentedCorrelation), 3))
  }
  result
}

# subscore's scored.data: 150 persons, 50 items in sections of 15, 15 and 20.
data(scored.data, package = "subscore")
scored <- as.matrix(scored.data)
colnames(scored) <- sprintf("v%02d", 1:50)
scored_sections <- list(Algebra = colnames(scored)[1:15], Geometry = colnames(scored)[16:30],
                        Measurement = colnames(scored)[31:50])

# subscore's TIMSS 2011 grade 8 mathematics: 765 persons; items 1 to 30 in four content domains, items 31 and 32 in none.
data(TIMSS11G8M.data, package = "subscore")
timss <- as.matrix(TIMSS11G8M.data)
colnames(timss) <- sprintf("q%02d", 1:32)
timss_sections <- list(Number = colnames(timss)[1:9], Algebra = colnames(timss)[10:18],
                       Geometry = colnames(timss)[19:24], Data = colnames(timss)[25:30])

# A simulation: 2,000 persons answer 6 items in each of three sections whose abilities correlate 0.8, 2PL with slope
# 1.1; 1% of the responses are missing. The true subscore is the expected score on the section's items.
set.seed(20261029)
sizes <- c(6, 6, 6)
abilities <- matrix(rnorm(2000 * 3), 2000) %*% chol(matrix(0.8, 3, 3) + diag(0.2, 3))
dimension <- rep(1:3, sizes)
difficulties <- rnorm(sum(sizes))
probability <- plogis(1.1 * (abilities[, dimension] - rep(difficulties, each = 2000)))
simulated <- matrix(rbinom(length(probability), 1, probability), 2000)
true_subscores <- sapply(1:3, function(j) rowSums(probability[, dimension == j]))
simulated[matrix(runif(length(probability)) < 0.01, 2000)] <- NA
colnames(simulated) <- sprintf("s%02d", seq_len(sum(sizes)))
simulated_sections <- list(Reading = colnames(simulated)[1:6], Listening = colnames(simulated)[7:12],
                           Writing = colnames(simulated)[13:18])

write_fixture("ctt-augmented-subscores", list(
  source = "subscore's scored.data and TIMSS11G8M.data; a simulation of three abilities correlated 0.8 with known true subscores, set.seed(20261029).",
  cases = list(
    wainer_case("scored.data", scored, scored_sections),
    wainer_case("TIMSS11G8M.data", timss, timss_sections),
    wainer_case("simulated", simulated, simulated_sections, true_subscores)
  )
))
