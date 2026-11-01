# Rater bias fixtures. References: the marginal likelihood of the many-facet model with the interaction of items and
# raters written out in base R, maximized by BFGS and polished by Newton's method on numDeriv's Hessian, whose inverse
# gives the standard errors of the biases by the delta method; and TAM's tam.mml.mfr with
# formulaA = ~ item + rater + item:rater + step (constraint = "items") on the same scores, as a check of the biases.
# The data: a seeded simulation of 150 persons, 8 items and 3 raters, each person scored by every rater on every item,
# on the rating scale model with thresholds -1, 0 and 1. Rater 1 is 0.8 logits more severe than the main effects predict
# on item 1 and rater 2 is 0.8 less. Both references constrain the interaction to sum to 0 over the items and over the
# raters, as the package does; the planted biases, centered the same way, are the truth that the estimates recover.

source("validation/psychometrics/common.R")
source("validation/psychometrics/many-facet-likelihood.R")
suppressPackageStartupMessages(library(TAM))

SEED <- 20261029
PERSONS <- 150
ITEMS <- 8
RATERS <- 3
THRESHOLDS <- c(-1, 0, 1)

set.seed(SEED)
ability <- rnorm(PERSONS)
difficulty <- round(rnorm(ITEMS, 0, 0.7), 3)
difficulty <- difficulty - mean(difficulty)
severity <- c(-0.3, 0.1, 0.2)
planted <- matrix(0, ITEMS, RATERS)
planted[1, 1] <- 0.8
planted[1, 2] <- -0.8

# The truth: the planted biases centered over the items and the raters, as the interaction is constrained.
center <- function(x) x - rowMeans(x)[row(x)] - colMeans(x)[col(x)] + mean(x)
truth <- center(planted)

d <- expand.grid(person = seq_len(PERSONS), item = seq_len(ITEMS), rater = seq_len(RATERS))
eta <- ability[d$person] - difficulty[d$item] - severity[d$rater] - planted[cbind(d$item, d$rater)]
# A rating scale category k has the probability proportional to exp(sum over j <= k of (eta - threshold j)).
cumulative <- t(sapply(eta, function(e) cumsum(c(0, e - THRESHOLDS))))
probabilities <- exp(cumulative - apply(cumulative, 1, max))
probabilities <- probabilities / rowSums(probabilities)
d$score <- apply(probabilities, 1, function(p) sample(0:3, 1, prob = p))

wide <- reshape(d, idvar = c("person", "rater"), timevar = "item", direction = "wide")
responses <- wide[, grep("^score\\.", names(wide))]
colnames(responses) <- paste0("i", seq_len(ITEMS))
facets <- data.frame(rater = paste0("r", wide$rater))

# TAM's Newton steps for the interaction parameters oscillate by about 1e-4 once the deviance has stopped changing, so a
# criterion below that is never met; the estimates are stable to that size.
model <- tam.mml.mfr(resp = responses, facets = facets, pid = wide$person,
                     formulaA = ~ item + rater + item:rater + step, constraint = "items",
                     control = list(nodes = Z, conv = 1e-4, convD = 1e-4, maxiter = 500), verbose = FALSE)
stopifnot(model$iter < 500)

table <- model$xsi.facets[model$xsi.facets$facet == "item:rater", ]
item_ids <- paste0("i", seq_len(ITEMS))
rater_ids <- paste0("r", seq_len(RATERS))
tam <- do.call(rbind, lapply(seq_len(nrow(table)), function(i) {
  parts <- strsplit(table$parameter[i], ":rater")[[1]]
  data.frame(item = parts[1], rater = parts[2], bias = table$xsi[i], standardError = table$se.xsi[i])
}))
# The interaction sums to 0 over the items and over the raters.
stopifnot(max(abs(tapply(tam$bias, tam$item, sum))) < 1e-6, max(abs(tapply(tam$bias, tam$rater, sum))) < 1e-6)

# --- Base R -------------------------------------------------------------------------------------------------------------
# The shared likelihood with the interaction as more columns of the design: a cell's columns are the product of its
# item's and its rater's contrast rows, so that the free biases are the (items - 1) (raters - 1) of effect coding.
long <- data.frame(person = d$person, item = item_ids[d$item], rater = rater_ids[d$rater], score = d$score)
design <- facet_design(long, c("item", "rater"))
product <- function(row) as.vector(outer(design$contrasts$item[design$combinations[row, 1], ],
                                         design$contrasts$rater[design$combinations[row, 2], ]))
cell_columns <- t(sapply(seq_len(nrow(design$combinations)), product))
main <- ncol(design$X)
design$X <- cbind(design$X, cell_columns)
fit <- maximize(design, agreement = 1e-4)
# The Jacobian of the exact gradient is the more accurate information matrix: numDeriv's Hessian of the likelihood
# differs from it by 3e-5 relative at this size, which moves a standard error by 1e-5 relative.
information <- jacobian(function(phi) -marginal_gradient(design, phi), fit$phi)
covariance <- solve((information + t(information)) / 2)
free <- main + seq_len(ncol(cell_columns))
biases <- drop(cell_columns %*% fit$phi[free])
bias_errors <- sqrt(diag(cell_columns %*% covariance[free, free] %*% t(cell_columns)))
parameters <- unpack(design, fit$phi)

cells <- data.frame(item = design$elements$item[design$combinations[, 1]],
                    rater = design$elements$rater[design$combinations[, 2]], bias = biases, standardError = bias_errors)
cells <- cells[order(match(cells$item, item_ids), match(cells$rater, rater_ids)), ]
# The truth, and the same cells in TAM's table.
cells$truth <- truth[cbind(match(cells$item, item_ids), match(cells$rater, rater_ids))]
at <- match(paste(cells$item, cells$rater), paste(tam$item, tam$rater))
cells$tamBias <- tam$bias[at]
cells$tamStandardError <- tam$standardError[at]

message("Base R: log-likelihood ", round(fit$logLikelihood, 4), "; TAM: ", model$iter, " iterations, log-likelihood ",
        round(-model$deviance / 2, 4), "; largest distance of TAM's bias from base R's ",
        signif(max(abs(cells$bias - cells$tamBias)), 3), ", of base R's from the truth ", round(max(abs(cells$bias - cells$truth)), 4))

write_fixture("rater-bias", list(
  source = "Base R's marginal likelihood and TAM's tam.mml.mfr with formulaA = ~ item + rater + item:rater + step on a simulation, set.seed(20261029): 150 persons, 8 items, 3 raters, rating scale model with thresholds -1, 0, 1; rater 1 is 0.8 logits more severe on item 1 and rater 2 0.8 less.",
  scores = lapply(seq_len(nrow(d)), function(i) {
    list(levels = list(person = paste0("p", d$person[i]), item = item_ids[d$item[i]], rater = rater_ids[d$rater[i]]),
         score = d$score[i])
  }),
  cells = lapply(seq_len(nrow(cells)), function(i) {
    list(levels = c(cells$item[i], cells$rater[i]), bias = cells$bias[i], standardError = cells$standardError[i],
         truth = cells$truth[i], tamBias = cells$tamBias[i], tamStandardError = cells$tamStandardError[i])
  }),
  logLikelihood = fit$logLikelihood,
  tamLogLikelihood = -model$deviance / 2,
  mean = parameters$mu,
  sd = parameters$sigma
))
