# The search for the decay of a proportion of successes and the exponent of a recency by maximum likelihood, on the log of
# the validation script, against LKT, which searches the rates by L-BFGS-B over fits of LiblineaR (seedpars). LKT's fits
# are penalized by LiblineaR's cost of 512 and stop at epsilon 1e-4, so its rates differ from the maximum in the second
# decimal; measurekit's maximize the log-likelihood.

suppressPackageStartupMessages({
  library(LKT)
  library(jsonlite)
})

validation <- fromJSON("packages/knowledge-tracing/test/fixtures/logistic-rates.json", simplifyVector = FALSE)
lkt_log <- data.frame(
  Anon.Student.Id = sapply(validation$events, `[[`, "learner"),
  KC..Default. = sapply(validation$events, function(e) e$skills[[1]]),
  CF..ansbin. = sapply(validation$events, `[[`, "outcome"),
  CF..Time. = sapply(validation$events, `[[`, "time"),
  stringsAsFactors = FALSE
)
lkt_log$CF..reltime. <- lkt_log$CF..Time.
lkt_log$Outcome <- ifelse(lkt_log$CF..ansbin. == 1, "CORRECT", "INCORRECT")
lkt_log$Duration..sec. <- 10
lkt_log <- lkt_log[order(lkt_log$Anon.Student.Id, lkt_log$CF..Time.), ]
lkt_log <- data.table::setDT(computeSpacingPredictors(lkt_log, "KC..Default."))

bench_task("logistic.searchRates", "Search for the decay of a proportion of successes and the exponent of a recency",
           sprintf("%s events of %d learners at 4 skills, 2 rates", count(nrow(lkt_log)), length(unique(lkt_log$Anon.Student.Id))),
           reference("LKT", "LKT(seedpars = c(0.5, 0.5))"),
           function() LKT(data = lkt_log, interc = FALSE, components = rep("KC..Default.", 3),
                          features = c("intercept", "propdec$", "recency$"), fixedpars = c(NA, NA),
                          seedpars = c(0.5, 0.5), verbose = FALSE),
           note = "each fit of the search is one of LiblineaR in LKT and one of Newton's method in measurekit")
