# Helpers for the fixture scripts. Run the scripts from the repository root,
# e.g. `Rscript validation/knowledge-tracing/elo.R`.

suppressPackageStartupMessages(library(jsonlite))

# One BLAS thread, as validation/README.md runs every script: several sum in another order, which moves the last digits
# of some fixtures, such as the standard errors of logistic-terms.R. R cannot set it once it has started.
for (variable in c("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")) {
  if (Sys.getenv(variable) != "1") stop("Set ", variable, "=1: the references run with one thread.")
}

# Writes `x` as packages/knowledge-tracing/test/fixtures/<name>.json, with 17 significant digits and NA as null.
write_fixture <- function(name, x) {
  path <- file.path("packages/knowledge-tracing/test/fixtures", paste0(name, ".json"))
  write_json(x, path, digits = I(17), auto_unbox = TRUE, na = "null", null = "null")
  message("Wrote ", path)
}

# The logistic function written as measurekit computes it, so that both round alike.
logistic <- function(x) 1 / (1 + exp(-x))

# A log as a list of events, the form measurekit takes: learner, item, skills (always an array), time, outcome.
events_of <- function(learner, item, skills, time, outcome) {
  lapply(seq_along(learner), function(i) {
    list(learner = learner[i], item = item[i], skills = I(skills[[i]]), time = time[i], outcome = outcome[i])
  })
}
