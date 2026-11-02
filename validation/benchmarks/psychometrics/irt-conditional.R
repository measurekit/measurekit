# Conditional maximum likelihood of the Rasch family, with the standard errors of the Hessian, against eRm's RM, PCM and
# RSM, whose sum0 sets the scale as measurekit's does: Rasch responses of 2,000 persons to 15 items, and partial credit
# responses of 1,500 persons to 8 items of 4 categories, which the rating scale model also fits.

suppressPackageStartupMessages(library(eRm))

set.seed(20261102)
rasch_persons <- 2000
rasch_items <- 15
credit_persons <- 1500
credit_items <- 8

# --- Data ----------------------------------------------------------------------------------------------------------

b <- round(rnorm(rasch_items), 3)
rasch <- matrix(rbinom(rasch_persons * rasch_items, 1, plogis(outer(rnorm(rasch_persons), b, "-"))), rasch_persons,
                rasch_items, dimnames = list(NULL, ids(rasch_items)))

# Partial credit responses: category k of an item has the weight exp(k θ - the sum of its first k thresholds).
thresholds <- t(sapply(round(rnorm(credit_items, 0, 0.7), 3), function(m) m + c(-1, 0, 1)))
theta <- rnorm(credit_persons)
credit <- sapply(seq_len(credit_items), function(i) {
  weights <- exp(outer(theta, 0:3) - matrix(cumsum(c(0, thresholds[i, ])), credit_persons, 4, byrow = TRUE))
  rowSums(runif(credit_persons) > t(apply(weights / rowSums(weights), 1, cumsum))[, 1:3])
})
dimnames(credit) <- list(NULL, ids(credit_items))

bench_fixture("psychometrics", "irt-conditional", list(
  source = "Simulations, set.seed(20261102): Rasch responses, and partial credit responses of 4 categories.",
  rasch = list(itemIds = ids(rasch_items), responses = unname(rasch)),
  partialCredit = list(itemIds = ids(credit_items), responses = unname(credit))
))

# --- References -----------------------------------------------------------------------------------------------------

note <- "eRm maximizes with nlm, which stops on its own criteria, and measurekit by Newton's method to a change of 1e-6"
bench_task("irt.conditionalRasch", "The Rasch model by conditional maximum likelihood",
           sprintf("%s persons, %d items", count(rasch_persons), rasch_items), reference("eRm", "RM"),
           function() RM(rasch, sum0 = TRUE), note = note)
credit_size <- sprintf("%s persons, %d items of 4 categories", count(credit_persons), credit_items)
bench_task("irt.conditionalPartialCredit", "The partial credit model by conditional maximum likelihood", credit_size,
           reference("eRm", "PCM"), function() PCM(credit, sum0 = TRUE), note = note)
bench_task("irt.conditionalRatingScale", "The rating scale model by conditional maximum likelihood", credit_size,
           reference("eRm", "RSM"), function() RSM(credit, sum0 = TRUE), note = note)
