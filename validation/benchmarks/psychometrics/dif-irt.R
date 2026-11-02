# Differential item functioning by IRT and by SIBTEST on 2PL responses of 2,000 persons in two groups to 15 items, 2 of
# them 0.6 harder for the focal group: the likelihood-ratio test of every item against mirt's multipleGroup and
# DIF(scheme = "drop"), Raju's areas against difR's difRaju, and SIBTEST against difR's difSIBTEST. The calibrations run
# at measurekit's quadrature and tolerance, 61 points and 1e-4.

suppressPackageStartupMessages({
  library(difR)
  library(mirt)
})

set.seed(20261102)
persons <- 2000
item_count <- 15

# --- Data ----------------------------------------------------------------------------------------------------------

group <- rep(c(0, 1), each = persons / 2)
a <- round(rlnorm(item_count, 0, 0.3), 3)
b <- round(rnorm(item_count), 3)
shift <- ifelse(seq_len(item_count) <= 2, 0.6, 0)
theta <- rnorm(persons) - 0.2 * group
difficulties <- outer(group, shift) + matrix(b, persons, item_count, byrow = TRUE)
responses <- matrix(rbinom(persons * item_count, 1, plogis(matrix(a, persons, item_count, byrow = TRUE) *
                                                              (theta - difficulties))),
                    persons, item_count, dimnames = list(NULL, ids(item_count)))

bench_fixture("psychometrics", "dif-irt", list(
  source = "A simulation, set.seed(20261102): 2PL responses of two groups, with 2 items 0.6 harder for the focal group.",
  itemIds = ids(item_count),
  responses = unname(responses),
  focal = I(as.integer(group))
))

# --- References -----------------------------------------------------------------------------------------------------

# The constrained model, every item equal across groups and the focal group's mean and variance free, then each item
# freed in turn.
likelihood_ratio <- function() {
  constrained <- multipleGroup(responses, 1, group = factor(group), itemtype = "2PL", quadpts = 61, TOL = 1e-4,
                               invariance = c(colnames(responses), "free_means", "free_var"), verbose = FALSE)
  DIF(constrained, which.par = c("a1", "d"), scheme = "drop", verbose = FALSE)
}

size <- sprintf("%s persons in two groups, %d items", count(persons), item_count)
bench_task("dif.irtLikelihoodRatio", "The likelihood-ratio test of every item, against the model with all items equal",
           size, reference("mirt", "multipleGroup, DIF(scheme = \"drop\")"), likelihood_ratio)
bench_task("dif.rajuArea", "Raju's signed and unsigned areas of every item, with both groups calibrated by the 2PL", size,
           reference("difR", "difRaju"), function() difRaju(responses, group = group, focal.name = 1, model = "2PL"),
           note = "difRaju calibrates by ltm's marginal maximum likelihood, and measurekit by EM")
bench_task("dif.sibtest", "SIBTEST of every item, matched on the other items", size,
           reference("difR", "difSIBTEST"), function() difSIBTEST(responses, group = group, focal.name = 1, type = "udif"))
