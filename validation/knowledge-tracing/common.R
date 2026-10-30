# Helpers for the fixture scripts. Run the scripts from the repository root, e.g. `Rscript validation/knowledge-tracing/elo.R`.

suppressPackageStartupMessages(library(jsonlite))

# Writes `x` as packages/knowledge-tracing/test/fixtures/<name>.json, with 17 significant digits and NA as null.
write_fixture <- function(name, x) {
  path <- file.path("packages/knowledge-tracing/test/fixtures", paste0(name, ".json"))
  write_json(x, path, digits = I(17), auto_unbox = TRUE, na = "null", null = "null")
  message("Wrote ", path)
}

# The logistic function written as the package computes it, so that both round alike.
logistic <- function(x) 1 / (1 + exp(-x))

# A log as a list of events, the form the package takes: learner, item, skills (always an array), time, outcome.
events_of <- function(learner, item, skills, time, outcome) {
  lapply(seq_along(learner), function(i) {
    list(learner = learner[i], item = item[i], skills = I(skills[[i]]), time = time[i], outcome = outcome[i])
  })
}
