# Chained links and grade-to-grade growth. References: equateIRT's chainec (links of forms 1 to 2 and 2 to 3 of est2pl
# and est3pl, composed), mirt's multipleGroup (the means and standard deviations of three grades on one scale), and
# base R written out below for growth.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(equateIRT)
  library(mirt)
})

SEED <- 20261029
METHODS <- c(meanMean = "mean-mean", meanSigma = "mean-sigma", haebara = "Haebara", stockingLord = "Stocking-Lord")
POINTS <- 30 # equateIRT's quadrature: 30 Gauss-Hermite points for N(0, 1)
data(est2pl, est3pl, package = "equateIRT")

# --- Chained links: equateIRT ---------------------------------------------------------------------------------------
# Forms 1, 2 and 3 of est2pl and est3pl: forms 1 and 2 share items I1 to I10, forms 2 and 3 share I11 to I20. Each link
# moves the earlier form to the scale of the later one, and chainec composes them as A = A1 A2, B = A2 B1 + B2.

form_items <- function(mods, form, items, model) {
  coef <- mods[[form]]$coef
  lapply(items, function(i) {
    get <- function(name, default) if (paste0(name, ".", i) %in% names(coef)) coef[[paste0(name, ".", i)]] else default
    list(model = model, parameters = list(a = get("Dscrmn", 1), b = get("Dffclt", NA), c = get("Gussng", 0), d = 1))
  })
}

chain_case <- function(name, estimates, model) {
  mods <- modIRT(coef = estimates$coef, var = estimates$var, display = FALSE)
  first <- paste0("I", 1:10)
  second <- paste0("I", 11:20)
  coefficients <- function(link) list(slope = link$A, intercept = link$B)
  by_method <- lapply(METHODS, function(method) {
    direct <- alldirec(mods = mods, method = method, nq = POINTS)
    chain <- chainec(r = 3, direclist = direct)[["T1.T2.T3"]]
    list(links = list(coefficients(direct[["T1.T2"]]), coefficients(direct[["T2.T3"]])), chain = coefficients(chain))
  })
  list(
    name = name,
    links = list(
      list(from = form_items(mods, 1, first, model), to = form_items(mods, 2, first, model)),
      list(from = form_items(mods, 2, second, model), to = form_items(mods, 3, second, model))
    ),
    methods = by_method
  )
}

write_fixture("equating-chain", list(
  source = paste("equateIRT's est2pl and est3pl: form 1 to form 2 over items I1 to I10 and form 2 to form 3 over I11 to",
                 "I20, by alldirec with 30 Gauss-Hermite points; the chain of T1.T2.T3 by chainec(r = 3)."),
  cases = list(chain_case("2PL", est2pl, "2pl"), chain_case("3PL", est3pl, "3pl"))
))

# --- Growth: mirt and base R ------------------------------------------------------------------------------------------
# Three grades on one scale by a concurrent calibration: 20 2PL items per grade, 6 shared with the next grade, 800
# persons per grade, theta ~ N(0, 1), N(0.8, 1.2^2) and N(1.3, 1.1^2). Grade 3 is the reference group; the means and
# variances of the others are free.

set.seed(SEED)
grades <- c("grade 3", "grade 4", "grade 5")
persons <- 800
unique_items <- 14
shared_items <- 6
pool_size <- 3 * unique_items + 2 * shared_items
true_a <- round(runif(pool_size, 0.8, 1.8), 2)
true_b <- round(rnorm(pool_size, 0, 1), 2)
pool_ids <- sprintf("q%02d", seq_len(pool_size))
# Grade 3 takes the first 14 unique items and the first 6 shared ones; grade 4 the shared 6, its own 14 and the next 6;
# grade 5 the second 6 shared and its own 14.
columns <- list(
  c(1:unique_items, 3 * unique_items + seq_len(shared_items)),
  c(3 * unique_items + seq_len(shared_items), unique_items + seq_len(unique_items),
    3 * unique_items + shared_items + seq_len(shared_items)),
  c(3 * unique_items + shared_items + seq_len(shared_items), 2 * unique_items + seq_len(unique_items))
)
population <- list(c(0, 1), c(0.8, 1.2), c(1.3, 1.1))
vertical <- matrix(NA_integer_, 3 * persons, pool_size, dimnames = list(NULL, pool_ids))
for (g in 1:3) {
  rows <- (g - 1) * persons + seq_len(persons)
  theta <- rnorm(persons, population[[g]][1], population[[g]][2])
  for (j in columns[[g]]) vertical[rows, j] <- rbinom(persons, 1, plogis(true_a[j] * (theta - true_b[j])))
}
group <- rep(grades, each = persons)
fit <- multipleGroup(vertical, 1, group = group, itemtype = "2PL",
                     invariance = c(pool_ids, "free_means", "free_var"), optimizer = "NR", TOL = 1e-10,
                     quadpts = 61, verbose = FALSE, technical = list(NCYCLES = 1e5))
stopifnot(extract.mirt(fit, "converged"))
moments <- lapply(grades, function(g) {
  parameters <- coef(extract.group(fit, g), simplify = TRUE)
  list(group = g, mean = unname(parameters$means[1]), sd = sqrt(unname(parameters$cov[1])))
})

# Base R: the growth between consecutive grades, with the effect size of Yen (1986) as Briggs and Betebenner give it,
# the mean change in units of the pooled standard deviation of the two grades.
growth <- lapply(1:2, function(k) {
  from <- moments[[k]]
  to <- moments[[k + 1]]
  list(from = from$group, to = to$group, meanChange = to$mean - from$mean, sdRatio = to$sd / from$sd,
       effectSize = (to$mean - from$mean) / sqrt((from$sd^2 + to$sd^2) / 2))
})

write_fixture("scaling-growth", list(
  source = paste("A concurrent mirt multipleGroup calibration (Newton-Raphson, TOL 1e-10, 61 nodes) of three grades of",
                 "20 2PL items, 6 shared with the next grade, 800 persons each, set.seed(20261029); the growth between",
                 "consecutive grades in base R."),
  groups = moments,
  growth = growth
))
