# The fit of a G-DINA calibration of the data of cdm.R, 1,000 persons, 20 items and 4 attributes: M2 with its RMSEA and
# SRMSR against GDINA's modelfit, and each item's statistics against GDINA's itemfit. Each side times the fit of its own
# calibration, which is not timed.

suppressPackageStartupMessages({
  library(GDINA)
  library(jsonlite)
})

fixture <- fromJSON("packages/psychometrics/test/fixtures/bench-cdm.json", simplifyVector = TRUE)
responses <- fixture$responses
attributes <- sort(unique(unlist(fixture$qMatrix)))
Q <- t(sapply(fixture$itemIds, function(id) as.integer(attributes %in% fixture$qMatrix[[id]])))
fit <- GDINA(responses, Q, model = "GDINA", control = list(conv.crit = 1e-4, maxitr = 2000), verbose = 0)

size <- sprintf("%s persons, %d items, %d attributes", count(nrow(responses)), ncol(responses), length(attributes))
bench_task("cdm.modelFit", "M2 with its RMSEA and SRMSR of a G-DINA calibration", size, reference("GDINA", "modelfit"),
           function() modelfit(fit))
bench_task("cdm.itemFit", "The fit statistics of each item of a G-DINA calibration", size, reference("GDINA", "itemfit"),
           function() itemfit(fit),
           note = "itemfit's expected values come from a simulation of persons, and measurekit's are exact sums over the profiles")
