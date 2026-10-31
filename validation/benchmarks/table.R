# The tables of docs/performance.md: for each task, the median of measurekit from the JSON reports of Vitest, the median of
# its reference from r.json and python.json, and their ratio. The rows where measurekit is more than 1.5 times slower are
# in bold. The tables replace what lies between the markers of the document.
#
#   pnpm --filter @measurekit/core bench --reporter json --outputFile validation/benchmarks/results/measurekit-core-1.json
#   pnpm --filter @measurekit/psychometrics bench --reporter json --outputFile validation/benchmarks/results/measurekit-psychometrics-1.json
#   pnpm --filter @measurekit/knowledge-tracing bench --reporter json --outputFile validation/benchmarks/results/measurekit-knowledge-tracing-1.json

SLOW <- 1.5
DOCUMENT <- "docs/performance.md"
START <- "<!-- benchmarks -->"
END <- "<!-- /benchmarks -->"
PACKAGES <- c(core = "Core", psychometrics = "Psychometrics", `knowledge-tracing` = "Knowledge tracing")

# The median of each benchmark over the reports of Vitest of a package, by name: the median of the medians of its runs.
vitest_medians <- function(package) {
  files <- list.files(RESULTS_DIR, pattern = paste0("^measurekit-", package, "-[0-9]+\\.json$"), full.names = TRUE)
  if (length(files) == 0) stop("No report of Vitest for ", package, " in ", RESULTS_DIR, ".")
  runs <- list()
  for (file in files) {
    for (suite in fromJSON(file, simplifyVector = FALSE)$testResults) {
      for (test in suite$assertionResults) for (benchmark in test$benchmarks) {
        for (task in benchmark$tasks) runs[[task$name]] <- c(runs[[task$name]], task$latency$p50)
      }
    }
  }
  lapply(runs, median)
}

milliseconds <- function(x) {
  digits <- if (x >= 100) 0 else if (x >= 10) 1 else if (x >= 1) 2 else 3
  formatC(x, format = "f", digits = digits, big.mark = ",")
}

# The medians of the references in r.json and python.json: python.json holds those that R does not time.
references <- fromJSON(file.path(RESULTS_DIR, "r.json"), simplifyVector = FALSE)
python_file <- file.path(RESULTS_DIR, "python.json")
if (file.exists(python_file)) {
  for (package in names(fromJSON(python_file, simplifyVector = FALSE))) {
    timed <- fromJSON(python_file, simplifyVector = FALSE)[[package]]
    for (key in names(timed)) references[[package]][[key]]$median <- timed[[key]]$median
  }
}

section <- function(package) {
  ours <- vitest_medians(package)
  rows <- character()
  notes <- character()
  for (key in names(references[[package]])) {
    entry <- references[[package]][[key]]
    own <- ours[[key]]
    if (is.null(own)) stop("No benchmark of measurekit has the name ", key, ".")
    theirs <- entry$median
    ratio <- if (is.null(theirs)) NA else own / theirs
    slow <- !is.na(ratio) && ratio > SLOW
    cell <- function(text) if (slow) paste0("**", text, "**") else text
    rows <- c(rows, sprintf("| %s | %s | %s | %s | %s | %s |", cell(entry$task), entry$size, milliseconds(own),
                            if (is.null(entry$reference)) "none" else entry$reference,
                            if (is.null(theirs)) "" else milliseconds(theirs),
                            if (is.na(ratio)) "" else cell(sprintf("%s×", trimws(formatC(ratio, format = "g", digits = 2))))))
    if (!is.null(entry$note)) notes <- c(notes, sprintf("- %s: %s.", entry$task, entry$note))
  }
  c(sprintf("### %s", PACKAGES[[package]]), "",
    "| Task | Size | measurekit (ms) | Reference | Reference (ms) | Ratio |",
    "| ---- | ---- | --------------: | --------- | -------------: | ----: |", rows, "",
    if (length(notes) > 0) c(notes, ""))
}

command <- function(...) tryCatch(trimws(system2(..., stdout = TRUE, stderr = FALSE)[1]), error = function(e) "unknown")
environment <- c(
  sprintf("Measured on %s with %s cores, %s GB of memory, Node.js %s and %s.",
          command("sysctl", c("-n", "machdep.cpu.brand_string")), command("sysctl", c("-n", "hw.ncpu")),
          round(as.numeric(command("sysctl", c("-n", "hw.memsize"))) / 2^30), sub("^v", "", command("node", "--version")),
          R.version.string))

tables <- c(environment, "", unlist(lapply(names(PACKAGES), function(package) {
  if (is.null(references[[package]])) NULL else section(package)
})))

lines <- readLines(DOCUMENT)
first <- which(lines == START)
last <- which(lines == END)
stopifnot(length(first) == 1, length(last) == 1, first < last)
writeLines(c(lines[seq_len(first)], "", tables, lines[last:length(lines)]), DOCUMENT)
message("Wrote the tables of ", DOCUMENT)
