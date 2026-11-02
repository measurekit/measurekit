# Steps of a Mokken scale analysis of the data of mokken.R, mokken's acl, 20 items of 5 categories: the Guttman errors of
# each person against mokken's check.errors, the conditional association of every pair of items against
# check.ca(Windex = TRUE), and two-way imputation of the same data with one response in ten missing against twoway.

suppressPackageStartupMessages(library(mokken))

data(acl, package = "mokken")
items <- as.matrix(acl[, 1:20])
items <- items[complete.cases(items), ]
items <- sweep(items, 2, apply(items, 2, min))
colnames(items) <- ids(ncol(items))

set.seed(20261102)
missing <- items
missing[sample(length(missing), round(length(missing) / 10))] <- NA

bench_fixture("psychometrics", "mokken-steps", list(
  source = "mokken.R's acl data, with one response in ten missing at random, set.seed(20261102).",
  itemIds = colnames(missing),
  responses = unname(missing)
))

size <- sprintf("%s persons, %d items of 5 categories", count(nrow(items)), ncol(items))
bench_task("mokken.guttmanErrors", "The Guttman errors of every person", size, reference("mokken", "check.errors"),
           function() check.errors(items))
bench_task("mokken.conditionalAssociation", "The W1, W2 and W3 indices of conditional association", size,
           reference("mokken", "check.ca(Windex = TRUE)"), function() suppressWarnings(check.ca(items, Windex = TRUE)),
           note = "check.ca removes the item with the most flags and runs again until none is flagged, and measurekit runs its first pass")
bench_task("mokken.twoWayImputation", "Two-way imputation of the missing responses", sprintf("%s, one in ten missing", size),
           reference("mokken", "twoway"), function() twoway(missing))
