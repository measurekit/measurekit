# Calibration with single parameters held fixed (irt-fixed-parameters.json). Reference: mirt with pars = "values" and
# est = FALSE on the held parameters: the lower asymptote g of 3PL items held at 1/4 and 1/5, and the slope a1 of a 2PL
# item held at 1, the others estimated. The script writes the estimates with their Hessian (Richardson) and
# cross-product standard errors, the log-likelihood, AIC and BIC, which count the estimated parameters only, and S-X²,
# whose degrees of freedom count them. fit-global.R draws the same responses for M2*, which it writes out in base R.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(mirt))

POINTS <- 61 # mirt's default quadrature: 61 equally spaced nodes on [-6, 6]
TOL <- 1e-10 # mirt's EM tolerance, tight so that its estimates are the optimum

set.seed(20261104)
persons <- 2000
a <- round(rlnorm(12, 0.1, 0.25), 2)
b <- round(rnorm(12, 0, 0.8), 2)
guessing <- c(rep(0.25, 6), rep(0.2, 2), rep(0, 4))
a[12] <- 1
theta <- rnorm(persons)
probability <- sapply(1:12, function(j) guessing[j] + (1 - guessing[j]) * plogis(a[j] * (theta - b[j])))
data <- (matrix(runif(persons * 12), persons) < probability) * 1
colnames(data) <- paste0("item", 1:12)
models <- c(rep("3pl", 8), rep("2pl", 4))
held <- c(setNames(lapply(guessing[1:8], function(g) list(c = g)), paste0("item", 1:8)), list(item12 = list(a = 1)))

values <- mirt(data, 1, itemtype = c(rep("3PL", 8), rep("2PL", 4)), pars = "values")
for (j in 1:8) {
  rows <- values$item == colnames(data)[j] & values$name == "g"
  values$value[rows] <- guessing[j]
  values$est[rows] <- FALSE
}
slope <- values$item == "item12" & values$name == "a1"
values$value[slope] <- 1
values$est[slope] <- FALSE

fit_with <- function(se_type) {
  fit <- mirt(data, 1, pars = values, SE = TRUE, SE.type = se_type, TOL = TOL, quadpts = POINTS, verbose = FALSE,
              technical = list(NCYCLES = 1e5))
  stopifnot(extract.mirt(fit, "converged"))
  fit
}

# measurekit's items of a fit: the IRT parameters, and the standard errors of the estimated ones, which mirt's delta
# method gives as NA for a held parameter.
items_of <- function(fit) {
  coefficients <- coef(fit, IRTpars = TRUE, printSE = TRUE)
  lapply(colnames(data), function(id) {
    x <- coefficients[[id]]
    se <- setNames(x["SE", c("a", "b", "g", "u")], c("a", "b", "c", "d"))
    list(parameters = list(a = x["par", "a"], b = x["par", "b"], c = x["par", "g"], d = x["par", "u"]),
         standardErrors = as.list(se[!is.na(se)]))
  })
}

calibration_of <- function(fit, method) {
  list(standardErrors = method, items = items_of(fit), logLikelihood = logLik(fit),
       aic = extract.mirt(fit, "AIC"), bic = extract.mirt(fit, "BIC"),
       estimated = sum(mod2values(fit)$est))
}

hessian <- fit_with("Richardson")
cross_product <- fit_with("crossprod")
summed <- itemfit(hessian, fit_stats = "S_X2")

write_fixture("irt-fixed-parameters", list(
  source = paste("mirt 1.48 with est = FALSE on g of items 1 to 8 and a1 of item 12, set.seed(20261104): 2000",
                 "persons, 12 items; SE.type Richardson and crossprod, TOL = 1e-10, 61 points; itemfit(S_X2) of the",
                 "Richardson fit."),
  itemIds = colnames(data),
  responses = unname(lapply(seq_len(nrow(data)), function(i) unname(data[i, ]))),
  models = I(models),
  fixedParameters = held,
  calibrations = list(calibration_of(hessian, "hessian"), calibration_of(cross_product, "cross-product")),
  summedScore = list(chiSquare = summed$S_X2, degreesOfFreedom = summed$df.S_X2, significance = summed$p.S_X2)
))
