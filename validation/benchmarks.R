# The references of the benchmarks of measurekit: the same tasks on the same data, timed with one thread. Each script of
# validation/benchmarks/<package>/ writes the inputs of its tasks to packages/<package>/test/fixtures/bench-<module>.json,
# which the benchmarks of measurekit read, and times the references on them. The results go to
# validation/benchmarks/results/r.json, and `table` turns them, with the results of Vitest and of benchmarks.py, into
# the table of docs/performance.md.
#
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript validation/benchmarks.R            # every module
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript validation/benchmarks.R irt mokken # the modules named, or package/module
#   Rscript validation/benchmarks.R table                                               # docs/performance.md
#
# The environment variables keep R's BLAS to one thread, which R cannot do once it has started.

source("validation/benchmarks/common.R")

arguments <- commandArgs(trailingOnly = TRUE)

if (identical(arguments, "table")) {
  source("validation/benchmarks/table.R")
  quit(save = "no")
}

for (variable in c("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")) {
  if (Sys.getenv(variable) != "1") stop("Set ", variable, "=1: the references run with one thread.")
}
options(mc.cores = 1)

for (package in c("core", "psychometrics", "knowledge-tracing")) {
  scripts <- sort(list.files(file.path("validation/benchmarks", package), pattern = "\\.R$", full.names = TRUE))
  modules <- sub("\\.R$", "", basename(scripts))
  chosen <- if (length(arguments) == 0) scripts else scripts[modules %in% arguments | paste0(package, "/", modules) %in% arguments]
  CURRENT_PACKAGE <- package
  for (script in chosen) {
    message("--- ", script)
    source(script)
    save_records()
  }
}
