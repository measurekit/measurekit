# Item and person fit: infit and outfit of items and persons, the summed-score statistic S-X2 and Yen's Q3 for a calibrated
# two-parameter model, against mirt's itemfit, personfit and residuals. The data are those of the irt script, with mirt's
# estimates of the items and the posterior means of the abilities.

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
