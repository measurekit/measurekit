# Differential item functioning: the Mantel-Haenszel test and logistic regression of difR for dichotomous items, Lord's
# chi-square of difR's difLord, and the ordinal logistic regression of three nested models fitted with rms's lrm. The data
# have 3 of 30 items with a difficulty 0.6 higher for the focal group; the polytomous ones, 10 items of 4 categories.

suppressPackageStartupMessages({
  library(difR)
  library(rms)
})

set.seed(20261029)
persons <- 2000
item_count <- 30

# --- Data ----------------------------------------------------------------------------------------------------------

group <- rep(c(0, 1), each = persons / 2)
a <- round(rlnorm(item_count, 0, 0.3), 3)
b <- round(rnorm(item_count), 3)
shift <- ifelse(seq_len(item_count) <= 3, 0.6, 0)
theta <- rnorm(persons) - 0.2 * group
difficulties <- b[col(matrix(0, persons, item_count))] + shift[col(matrix(0, persons, item_count))] * group
dichotomous <- matrix(rbinom(persons * item_count, 1, plogis(rep(a, each = persons) * (theta - difficulties))),
                      persons, item_count, dimnames = list(NULL, ids(item_count)))

ordinal_persons <- 1000
ordinal_items <- 10
ordinal_group <- rep(c(0, 1), each = ordinal_persons / 2)
ordinal_theta <- rnorm(ordinal_persons) - 0.2 * ordinal_group
cuts <- t(sapply(round(rnorm(ordinal_items), 3), function(m) m + c(-1, 0, 1)))
cuts[1:2, ] <- cuts[1:2, ] + 0.6 * 0
uniform <- matrix(runif(ordinal_persons * ordinal_items), ordinal_persons, ordinal_items)
polytomous <- matrix(0L, ordinal_persons, ordinal_items, dimnames = list(NULL, ids(ordinal_items)))
for (k in 1:3) {
  shifted <- cuts[, k][col(uniform)] + ifelse(col(uniform) <= 2, 0.6, 0) * ordinal_group
  polytomous <- polytomous + (uniform < plogis(ordinal_theta - shifted))
}
polytomous <- matrix(as.integer(polytomous), ordinal_persons, ordinal_items, dimnames = list(NULL, ids(ordinal_items)))

bench_fixture("psychometrics", "dif", list(
  source = "Simulations, set.seed(20261029): 2PL responses of two groups, and graded responses of two groups, with 3 and 2 items with DIF.",
  dichotomous = list(itemIds = ids(item_count), responses = unname(dichotomous), focal = I(as.integer(group))),
  polytomous = list(itemIds = ids(ordinal_items), responses = unname(polytomous), focal = I(as.integer(ordinal_group)))
))

# --- References -----------------------------------------------------------------------------------------------------

ordinal_dif <- function() {
  total <- rowSums(polytomous)
  for (item in seq_len(ordinal_items)) {
    y <- ordered(polytomous[, item])
    lrm(y ~ total)
    lrm(y ~ total + ordinal_group)
    lrm(y ~ total * ordinal_group)
  }
}

size <- sprintf("%s persons in two groups, %d items", count(persons), item_count)
bench_task("dif.mantelHaenszel", "The Mantel-Haenszel test of every item", size, reference("difR", "difMH"),
           function() difMH(dichotomous, group = group, focal.name = 1))
bench_task("dif.logisticRegression", "Logistic regression for uniform and nonuniform DIF of every item", size,
           reference("difR", "difLogistic"), function() difLogistic(dichotomous, group = group, focal.name = 1, type = "both"))
bench_task("dif.lordChiSquare", "Lord's chi-square of every item, with both groups calibrated by the 2PL", size,
           reference("difR", "difLord"), function() difLord(dichotomous, group = group, focal.name = 1, model = "2PL"))
bench_task("dif.ordinalLogistic", "Ordinal logistic regression of every item",
           sprintf("%s persons in two groups, %d items of 4 categories", count(ordinal_persons), ordinal_items),
           reference("rms", "lrm, three models for an item"), ordinal_dif)
