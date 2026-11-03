# Item and person fit: infit and outfit of items and persons, the summed-score statistic S-X2, Yen's Q3, Chen and
# Thissen's chi-square and M2* for a calibrated two-parameter model, against mirt's itemfit, personfit, residuals and M2;
# lz* at the posterior modes against PerFit's lzstar, the RMSD of the items against TAM's IRT.RMSD, and kernel-smoothed
# item curves against KernSmoothIRT's ksIRT. The data are those of the irt script, with mirt's estimates of the items and
# the posterior means of the abilities. PerFit, KernSmoothIRT and TAM are not attached, so that the scripts after this one
# keep the functions they attach.

suppressPackageStartupMessages(library(mirt))

source("validation/benchmarks/simulation.R")

set.seed(20261029)
persons <- 2000
item_count <- 40

data <- simulate_2pl(persons, item_count)$responses
fit <- mirt(data, 1, itemtype = "2PL", TOL = 1e-4, quadpts = 61, verbose = FALSE)
thetas <- fscores(fit, method = "EAP", quadpts = 61, verbose = FALSE)
size <- sprintf("%s persons, %d items", count(persons), item_count)

bench_task("fit.itemFit", "Infit and outfit of the items", size, reference("mirt", "itemfit(fit_stats = \"infit\")"),
           function() itemfit(fit, fit_stats = "infit", Theta = thetas, method = "EAP"))
bench_task("fit.personFit", "Infit, outfit and the standardized log-likelihood of the persons", size,
           reference("mirt", "personfit"), function() personfit(fit, Theta = thetas))
bench_task("fit.summedScoreFit", "The summed-score statistic S-X2 of the items", size,
           reference("mirt", "itemfit(fit_stats = \"S_X2\")"), function() itemfit(fit, fit_stats = "S_X2"))
bench_task("fit.yenQ3", "Yen's Q3 of every pair of items", size, reference("mirt", "residuals(type = \"Q3\")"),
           function() residuals(fit, type = "Q3", verbose = FALSE))

modes <- fscores(fit, method = "MAP", quadpts = 61, verbose = FALSE)[, "F1"]
coefficients <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
nodes <- seq(-6, 6, length.out = 61)
tam_fit <- TAM::tam.mml.2pl(data, irtmodel = "2PL", control = list(nodes = nodes, conv = 1e-4, convD = 1e-4),
                            verbose = FALSE)

bench_task("fit.correctedPersonFit", "Snijders' lz* of the persons at their posterior modes", size,
           reference("PerFit", "lzstar(Ability.PModel = \"BM\")"),
           function() PerFit::lzstar(data, IP = cbind(coefficients[, "a"], coefficients[, "b"], 0), Ability = modes,
                                     Ability.PModel = "BM"))
bench_task("fit.kernelItemCurves", "Kernel-smoothed curves of the categories of the items at 51 points", size,
           reference("KernSmoothIRT", "ksIRT(format = 2)"),
           function() KernSmoothIRT::ksIRT(data, key = rep(1, item_count), format = 2, nevalpoints = 51),
           note = "ksIRT also computes the expected scores, the correlation of each item with the total and the maximum likelihood theta of each person on the curves")
bench_task("fit.limitedInformationFit", "M2* with its RMSEA, SRMSR, TLI and CFI", size, reference("mirt", "M2"),
           function() M2(fit))
# IRT.RMSD, of CDM, which TAM imports, reads the expected counts that TAM's calibration stored. measurekit takes items,
# so it computes each person's posterior at the 61 nodes of N(0, 1) from the responses. This does the same in R with
# matrix products.
weights <- dnorm(nodes) / sum(dnorm(nodes))
rmsd_from_responses <- time_calls(function() {
  correct <- plogis(sweep(outer(nodes, coefficients[, "b"], "-"), 2, coefficients[, "a"], "*"))
  log_posterior <- data %*% t(log(correct)) + (1 - data) %*% t(log(1 - correct)) + rep(log(weights), each = nrow(data))
  posterior <- exp(log_posterior - apply(log_posterior, 1, max))
  posterior <- posterior / rowSums(posterior)
  observed <- t(data) %*% posterior / matrix(colSums(posterior), item_count, length(nodes), byrow = TRUE)
  deviation <- t(observed) - correct
  list(rmsd = sqrt(colSums(deviation^2 * colMeans(posterior))), mean = colSums(deviation * colMeans(posterior)))
})
bench_task("fit.rootMeanSquareDeviation", "The RMSD and mean deviation of the items, at 61 nodes", size,
           reference("TAM", "IRT.RMSD"), function() CDM::IRT.RMSD(tam_fit),
           note = sprintf(paste("IRT.RMSD reads the expected counts that the calibration stored, which is work of the",
                                "calibration; measurekit takes items, so it computes each person's posterior from the",
                                "responses. The same computation in R, with matrix products, takes %.1f ms"),
                          rmsd_from_responses$median))
bench_task("fit.localDependenceChiSquare", "Chen and Thissen's chi-square of every pair of items", size,
           reference("mirt", "residuals(type = \"LD\")"), function() residuals(fit, type = "LD", verbose = FALSE))
