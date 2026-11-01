# Dimensionality of psych's bfi, 25 items of 6 categories with missing responses: polychoric correlations by psych's
# polychoric, parallel analysis of components and factors by psych's fa.parallel, with 20 random data sets, five factors
# by psych's fa and the minimum average partial test by psych's VSS, from Pearson correlations.

suppressPackageStartupMessages(library(psych))

data(bfi, package = "psych")
items <- bfi[, 1:25] - 1L
ITERATIONS <- 20

bench_fixture("psychometrics", "dimensionality", list(
  source = "psych's bfi, the 25 items scored from 0, with missing responses as null.",
  itemIds = colnames(items),
  responses = unname(as.matrix(items)),
  replications = ITERATIONS
))

size <- sprintf("%s persons, %d items of 6 categories", count(nrow(items)), ncol(items))
bench_task("dimensionality.polychoricCorrelations", "Polychoric correlations of every pair of items", size,
           reference("psych", "polychoric"),
           function() suppressWarnings(suppressMessages(polychoric(items, smooth = FALSE, progress = FALSE))))
bench_task("dimensionality.parallelAnalysis", "Parallel analysis of components and factors from polychoric correlations, 20 data sets",
           size, reference("psych", "fa.parallel(cor = \"poly\")"),
           function() invisible(capture.output(suppressWarnings(suppressMessages(
             fa.parallel(items, n.iter = ITERATIONS, fa = "both", cor = "poly", plot = FALSE))))))
bench_task("dimensionality.exploratoryFactorAnalysis", "Five factors by minimum residuals with oblimin rotation, from Pearson correlations",
           size, reference("psych", "fa(nfactors = 5, fm = \"minres\", rotate = \"oblimin\")"),
           function() suppressWarnings(suppressMessages(fa(items, nfactors = 5, fm = "minres", rotate = "oblimin"))),
           note = "fa also computes fit statistics, and this task leaves them out")
bench_task("dimensionality.minimumAveragePartial", "Velicer's minimum average partial test of 8 components, from Pearson correlations",
           size, reference("psych", "VSS(n = 8)"),
           function() invisible(capture.output(suppressWarnings(suppressMessages(
             VSS(items, n = 8, n.obs = nrow(items), plot = FALSE))))),
           note = "VSS also fits 8 factor solutions with their very simple structure and fit statistics, which this task leaves out")
