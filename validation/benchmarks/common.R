# Helpers of the benchmark scripts: the fixtures with the inputs of the tasks, the timing of the references, and the
# record of each task that docs/performance.md is made from. Run the scripts from the repository root, with
# `Rscript validation/benchmarks.R`.

suppressPackageStartupMessages(library(jsonlite))

RESULTS_DIR <- "validation/benchmarks/results"
# One task of the benchmarks per key, by package, in the order that the scripts run.
RECORDS <- list()
CURRENT_PACKAGE <- NULL

# Writes `x` as packages/<package>/test/fixtures/bench-<name>.json, with 17 significant digits and NA as null.
bench_fixture <- function(package, name, x) {
  path <- file.path("packages", package, "test/fixtures", paste0("bench-", name, ".json"))
  write_json(x, path, digits = I(17), auto_unbox = TRUE, na = "null", null = "null")
  message("Wrote ", path, " (", round(file.size(path) / 1024), " KB)")
}

# Calls of `fn` are timed one after another, on a quiet machine. The first call warms up and sizes the rest: a call under
# 50 ms is repeated, and a sample is the time of one call. The samples are 15 for a call under a second, 7 under 15
# seconds, 3 under a minute and 2 for slower ones. Returns the median and the fastest sample in milliseconds.
time_calls <- function(fn) {
  start <- proc.time()[["elapsed"]]
  fn()
  first <- proc.time()[["elapsed"]] - start
  inner <- max(1, ceiling(0.05 / max(first, 1e-4)))
  runs <- if (first < 1) 15 else if (first < 15) 7 else if (first < 60) 3 else 2
  samples <- vapply(seq_len(runs), function(i) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    for (j in seq_len(inner)) fn()
    (proc.time()[["elapsed"]] - start) * 1000 / inner
  }, 0)
  list(median = median(samples), min = min(samples), samples = runs)
}

# The name, version and function of a reference, as the table of docs/performance.md shows it.
reference <- function(package, call) {
  sprintf("%s %s `%s`", package, as.character(packageVersion(package)), call)
}

# Records a task of the benchmarks of the current package. `fn` runs the reference on the data of the task and is timed
# here; leave it out where the reference is timed elsewhere (a Python reference) or where there is none (`reference` is
# `NULL`). `key` is the name of the benchmark of measurekit, `module.task`.
bench_task <- function(key, task, size, reference = NULL, fn = NULL, note = NULL) {
  entry <- list(task = task, size = size, reference = reference, note = note)
  if (!is.null(fn)) {
    entry <- c(entry, time_calls(fn))
    message(sprintf("%-32s median %10.3f ms   fastest %10.3f ms   %s", key, entry$median, entry$min, size))
  }
  RECORDS[[CURRENT_PACKAGE]][[key]] <<- entry
}

# Merges the records of this run into the results file, which keeps the tasks of the modules that did not run.
save_records <- function(file = file.path(RESULTS_DIR, "r.json")) {
  dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
  results <- if (file.exists(file)) fromJSON(file, simplifyVector = FALSE) else list()
  for (package in names(RECORDS)) {
    for (key in names(RECORDS[[package]])) results[[package]][[key]] <- RECORDS[[package]][[key]]
  }
  write_json(results, file, auto_unbox = TRUE, null = "null", pretty = TRUE, digits = NA)
  message("Wrote ", file)
}

# A count with thousands separators, as the sizes of the table show it.
count <- function(x) format(x, big.mark = ",", scientific = FALSE, trim = TRUE)

# The ids of `n` columns: item1, item2, ...
ids <- function(n, prefix = "item") paste0(prefix, seq_len(n))
