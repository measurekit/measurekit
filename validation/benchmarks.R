# The references of the benchmarks of measurekit: the same tasks on the same data, timed with one thread. Each script of
# validation/benchmarks/<package>/ writes the inputs of its tasks to packages/<package>/test/fixtures/bench-<module>.json,
# which the benchmarks of measurekit read, and times the references on them. The times are printed, and saved to
# validation/benchmarks/results/r.json, which git ignores; the docs publish no measured time.
#
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript validation/benchmarks.R            # every module
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript validation/benchmarks.R irt mokken # the modules named, or package/module
#
# The environment variables keep R's BLAS to one thread, which R cannot do once it has started.

source("validation/benchmarks/common.R")

arguments <- commandArgs(trailingOnly = TRUE)

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
