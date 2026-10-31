# Mokken scale analysis of mokken's acl, the first 20 items of 5 categories answered by the persons who answered all of them:
# scalability coefficients with standard errors, automated item selection, and the checks of monotonicity and of
# invariant item ordering, against mokken's coefH, aisp, check.monotonicity and check.iio.

suppressPackageStartupMessages(library(mokken))

data(acl, package = "mokken")
items <- as.matrix(acl[, 1:20])
items <- items[complete.cases(items), ]
items <- sweep(items, 2, apply(items, 2, min))
colnames(items) <- ids(ncol(items))

bench_fixture("psychometrics", "mokken", list(
  source = "mokken's acl, the first 20 items of the complete cases, scored from 0.",
  itemIds = colnames(items),
  responses = unname(items)
))

size <- sprintf("%s persons, %d items of 5 categories", count(nrow(items)), ncol(items))
bench_task("mokken.scalability", "Scalability coefficients with standard errors", size,
           reference("mokken", "coefH(se = TRUE)"), function() coefH(items, se = TRUE))
bench_task("mokken.automatedItemSelection", "Automated item selection into Mokken scales", size,
           reference("mokken", "aisp"), function() suppressWarnings(aisp(items, lowerbound = 0.3)))
bench_task("mokken.monotonicity", "The check of monotonicity of every item", size,
           reference("mokken", "check.monotonicity"), function() suppressWarnings(check.monotonicity(items)))
bench_task("mokken.invariantItemOrdering", "The check of invariant item ordering, with selection of items", size,
           reference("mokken", "check.iio(method = \"MIIO\")"),
           function() suppressWarnings(check.iio(items, method = "MIIO", item.selection = TRUE)))
