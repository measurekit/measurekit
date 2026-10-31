# Classical test theory on simulated 2PL responses: item statistics and Cronbach's alpha by psych's alpha, McDonald's omega
# by psych's omega, and the value of subscores by subscore's CTTsub. Items are scored 0 or 1.

suppressPackageStartupMessages({
  library(psych)
  library(subscore)
})

source("validation/benchmarks/simulation.R")

set.seed(20261029)
persons <- 3000
item_count <- 50
omega_items <- 20
section_size <- 10

data <- simulate_2pl(persons, item_count)$responses
sections <- split(seq_len(item_count), rep(seq_len(item_count / section_size), each = section_size))

bench_fixture("psychometrics", "ctt", list(
  source = "A 2PL simulation, set.seed(20261029).",
  itemIds = ids(item_count),
  responses = unname(data),
  sectionSize = section_size,
  omegaItems = omega_items
))

size <- sprintf("%s persons, %d items", count(persons), item_count)
bench_task("ctt.itemStatistics", "Item difficulty, discrimination and the statistics of the items", size,
           reference("psych", "alpha"), function() alpha(data, check.keys = FALSE, warnings = FALSE))
bench_task("ctt.cronbachAlpha", "Cronbach's alpha", size, reference("psych", "alpha"),
           function() alpha(data, check.keys = FALSE, warnings = FALSE))
bench_task("ctt.mcdonaldOmega", "McDonald's omega total of one factor", sprintf("%s persons, %d items", count(persons), omega_items),
           reference("psych", "omega(nfactors = 1)"),
           function() invisible(capture.output(suppressMessages(suppressWarnings(
             omega(data[, seq_len(omega_items)], nfactors = 1, plot = FALSE))))))
bench_task("ctt.subscoreValue", "The value of five subscores of ten items over the total score", size,
           reference("subscore", "CTTsub"), function() {
             x <- as.data.frame(data[, unlist(sections)])
             names <- paste0("section", seq_along(sections))
             CTTsub(data.prep(x, c(length(sections), lengths(sections)), c(names, "total")))
           })
