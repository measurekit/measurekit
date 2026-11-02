# Factor solutions of psych's bfi, the data of dimensionality.R, 25 items of 6 categories with missing responses, from
# Pearson correlations: five factors by maximum likelihood with oblimin rotation and their fit, against psych's
# fa(fm = "ml"), and McDonald's omega hierarchical of five group factors, against psych's omega.

suppressPackageStartupMessages(library(psych))

data(bfi, package = "psych")
items <- bfi[, 1:25] - 1L

size <- sprintf("%s persons, %d items of 6 categories", count(nrow(items)), ncol(items))
bench_task("dimensionality.maximumLikelihoodFactors", "Five factors by maximum likelihood with oblimin rotation, and their fit",
           size, reference("psych", "fa(nfactors = 5, fm = \"ml\", rotate = \"oblimin\")"),
           function() suppressWarnings(suppressMessages(fa(items, nfactors = 5, fm = "ml", rotate = "oblimin"))),
           note = "fa also computes factor scores' statistics and other fit indices, which measurekit leaves out")
bench_task("dimensionality.omegaHierarchical", "Omega hierarchical and total from five group factors by minimum residuals",
           size, reference("psych", "omega(nfactors = 5, fm = \"minres\")"),
           function() suppressWarnings(suppressMessages(omega(items, nfactors = 5, fm = "minres", flip = FALSE, plot = FALSE))),
           note = "omega also computes the fit of the general factor and of the group factors, which measurekit leaves out")
