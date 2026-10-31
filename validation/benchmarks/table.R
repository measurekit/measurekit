# The charts and tables of docs/performance.md: for each task, the median of measurekit from the JSON reports of Vitest, the
# median of its reference from r.json and python.json, and their ratio. The charts and the tables replace what lies between
# their markers in the document. The rows where measurekit is more than 1.5 times slower are in bold.
#
#   pnpm --filter @measurekit/core bench --reporter json --outputFile validation/benchmarks/results/measurekit-core-1.json
#   pnpm --filter @measurekit/psychometrics bench --reporter json --outputFile validation/benchmarks/results/measurekit-psychometrics-1.json
#   pnpm --filter @measurekit/knowledge-tracing bench --reporter json --outputFile validation/benchmarks/results/measurekit-knowledge-tracing-1.json
#
# Run each three times, with -1, -2 and -3 in the name of the file: the median of the runs is the time of measurekit.

SLOW <- 1.5
DOCUMENT <- "docs/performance.md"
PACKAGES <- c(core = "Core", psychometrics = "Psychometrics", `knowledge-tracing` = "Knowledge tracing")
# The most tasks that a chart shows, so that the names of the tasks stay readable.
CHART_TASKS <- 18
# The most characters of the modules in the title of a chart, so that the title is not clipped.
TITLE_CHARACTERS <- 36

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
  timed <- fromJSON(python_file, simplifyVector = FALSE)
  for (package in names(timed)) {
    for (key in names(timed[[package]])) references[[package]][[key]]$median <- timed[[package]][[key]]$median
  }
}

# One entry for each task of a package, in the order that the scripts run: its labels, both medians and their ratio.
tasks <- function(package) {
  ours <- vitest_medians(package)
  lapply(names(references[[package]]), function(key) {
    entry <- references[[package]][[key]]
    own <- ours[[key]]
    if (is.null(own)) stop("No benchmark of measurekit has the name ", key, ".")
    theirs <- entry$median
    c(entry[c("task", "size", "reference", "note")], list(key = key, own = own, theirs = theirs,
                                                           ratio = if (is.null(theirs)) NA else own / theirs))
  })
}
results <- lapply(setNames(names(PACKAGES), names(PACKAGES)), function(package) {
  if (is.null(references[[package]])) list() else tasks(package)
})

# --- Charts ---------------------------------------------------------------------------------------------------------------

# A horizontal bar chart of ratios, with a line at 1, where the two take the same time. The height grows with the tasks. A
# chart names each task by its key without the module when its title names the modules, and by its whole key otherwise.
chart <- function(title, entries, by_module = TRUE) {
  ratios <- vapply(entries, function(entry) signif(entry$ratio, 2), 0)
  top <- max(2, ceiling(max(ratios)))
  plain <- vapply(ratios, format, "", scientific = FALSE, trim = TRUE)
  labels <- vapply(entries, function(entry) if (by_module) sub("^[^.]*\\.", "", entry$key) else entry$key, "")
  labels <- sub("^multidimensional\\.", "multi.", labels)
    c("```mermaid",
    sprintf('%%%%{init: {"xyChart": {"height": %d, "xAxis": {"labelFontSize": 11}}}}%%%%', 150 + 34 * length(entries)),
    "xychart-beta horizontal",
    sprintf('  title "%s"', title),
    sprintf("  x-axis [%s]", paste0('"', labels, '"', collapse = ", ")),
    sprintf('  y-axis "Time of measurekit over the time of the reference" 0 --> %d', top),
    sprintf("  bar [%s]", paste(plain, collapse = ", ")),
    sprintf("  line [%s]", paste(rep(1, length(entries)), collapse = ", ")),
    "```", "")
}

# The tasks of a package that have a reference, in groups of modules of at most CHART_TASKS tasks.
chart_groups <- function(entries) {
  entries <- Filter(function(entry) !is.na(entry$ratio), entries)
  modules <- vapply(entries, function(entry) sub("\\..*$", "", entry$key), "")
  groups <- list()
  current <- list()
  for (module in unique(modules)) {
    members <- entries[modules == module]
    names_so_far <- unique(vapply(current, function(entry) sub("\\..*$", "", entry$key), ""))
    title_length <- nchar(paste(c(names_so_far, module), collapse = ", "))
    if (length(current) > 0 && (length(current) + length(members) > CHART_TASKS || title_length > TITLE_CHARACTERS)) {
      groups[[length(groups) + 1]] <- current
      current <- list()
    }
    current <- c(current, members)
  }
  c(groups, if (length(current) > 0) list(current))
}

all_entries <- unlist(results, recursive = FALSE)
slow_entries <- Filter(function(entry) !is.na(entry$ratio) && entry$ratio > SLOW, unname(all_entries))
slow_entries <- slow_entries[order(-vapply(slow_entries, function(entry) entry$ratio, 0))]

command <- function(...) tryCatch(trimws(system2(..., stdout = TRUE, stderr = FALSE)[1]), error = function(e) "unknown")
environment <- sprintf("Measured on %s with %s cores, %s GB of memory, Node.js %s and %s.",
                       command("sysctl", c("-n", "machdep.cpu.brand_string")), command("sysctl", c("-n", "hw.ncpu")),
                       round(as.numeric(command("sysctl", c("-n", "hw.memsize"))) / 2^30),
                       sub("^v", "", command("node", "--version")), R.version.string)

charts <- c("### Where measurekit is slower", "",
            chart(sprintf("Tasks that take more than %s times as long as the reference", SLOW), slow_entries, by_module = FALSE),
            "### Every task", "",
            "A bar above the line takes longer than the reference.", "",
            unlist(lapply(names(PACKAGES), function(package) {
              unlist(lapply(chart_groups(results[[package]]), function(group) {
                modules <- unique(vapply(group, function(entry) sub("\\..*$", "", entry$key), ""))
                chart(sprintf("%s: %s", PACKAGES[[package]], paste(modules, collapse = ", ")), group)
              }))
            })))

# --- Tables ---------------------------------------------------------------------------------------------------------------

section <- function(package) {
  rows <- character()
  notes <- character()
  for (entry in results[[package]]) {
    slow <- !is.na(entry$ratio) && entry$ratio > SLOW
    cell <- function(text) if (slow) paste0("**", text, "**") else text
    rows <- c(rows, sprintf("| %s | %s | %s | %s | %s | %s |", cell(entry$task), entry$size, milliseconds(entry$own),
                            if (is.null(entry$reference)) "none" else entry$reference,
                            if (is.null(entry$theirs)) "" else milliseconds(entry$theirs),
                            if (is.na(entry$ratio)) "" else cell(sprintf("%s×", format(signif(entry$ratio, 2), scientific = FALSE, trim = TRUE)))))
    if (!is.null(entry$note)) notes <- c(notes, sprintf("- %s: %s.", entry$task, entry$note))
  }
  c(sprintf("### %s", PACKAGES[[package]]), "",
    "| Task | Size | measurekit (ms) | Reference | Reference (ms) | Ratio |",
    "| ---- | ---- | --------------: | --------- | -------------: | ----: |", rows, "",
    if (length(notes) > 0) c(notes, ""))
}

tables <- c(environment, "", unlist(lapply(names(PACKAGES), function(package) if (length(results[[package]]) == 0) NULL else section(package))))

# --- The document ---------------------------------------------------------------------------------------------------------

# Replaces what lies between the markers of a document with `text`.
replace_between <- function(lines, name, text) {
  first <- which(lines == sprintf("<!-- %s -->", name))
  last <- which(lines == sprintf("<!-- /%s -->", name))
  stopifnot(length(first) == 1, length(last) == 1, first < last)
  c(lines[seq_len(first)], "", text, lines[last:length(lines)])
}

lines <- readLines(DOCUMENT)
lines <- replace_between(lines, "charts", charts)
lines <- replace_between(lines, "tables", tables)
writeLines(lines, DOCUMENT)
message("Wrote the charts and tables of ", DOCUMENT)
