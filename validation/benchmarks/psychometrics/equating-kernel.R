# Kernel equating of the random groups of equate's ACTmath, the data of equating.R, two forms of 40 items, on log-linear
# fits of degree 3 by glm, with the bandwidths that minimize the penalty, the standard errors of equating and the
# percent relative error, against kequate's kequate("EG"). Each side's time includes its log-linear fits.

suppressPackageStartupMessages({
  library(equate)
  library(kequate)
})

data(ACTmath, package = "equate")
scores <- ACTmath[, 1]
DEGREE <- 3

# The log-linear fit of a form's counts on powers of the scores scaled to [-1, 1].
fit_counts <- function(counts) {
  z <- (scores - 20) / 20
  glm(counts ~ z + I(z^2) + I(z^3), family = "poisson", x = TRUE)
}

bench_task("equating.kernel", "Kernel equating with its standard errors, from log-linear fits of degree 3",
           sprintf("%d scores, two forms", length(scores)), reference("kequate", "kequate(\"EG\")"),
           function() kequate("EG", scores, scores, fit_counts(ACTmath[, 2]), fit_counts(ACTmath[, 3])))
