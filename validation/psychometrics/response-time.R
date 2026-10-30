# Response time fixtures. References: the lognormal model (van der Linden, 2006) as the linear mixed model
# nlme::lme(log(t) ~ 0 + item, random = ~ 1 | person, weights = varIdent(form = ~ 1 | item), method = "ML"), its
# estimates taken to the maximum of the written-out likelihood by Newton steps with numDeriv's derivatives; LNIRT's
# LNRT by MCMC as a second check; LNIRT's personfitLN for the person statistic of van der Linden and Guo (2008); mirt's
# EAP for effort-moderated scoring; and base R for the residuals, the thresholds and response time effort.
# Data: a seeded simulation, and the first 10 items of LNIRT's CredentialForm1, times of 0 as missing.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(nlme)
  library(LNIRT)
  library(mirt)
})

SEED <- 20261029

# The marginal log-likelihood of the log times written out: each person's y ~ N(β, D + σ² 11ᵀ), D = diag(1 / α²), by
# the matrix determinant lemma and Sherman and Morrison. Parameters: β, ln(1 / α) and ln σ.
lognormal_log_likelihood <- function(par, log_times) {
  k <- ncol(log_times)
  beta <- par[1:k]
  precision <- exp(-2 * par[k + 1:k])
  variance <- exp(2 * par[2 * k + 1])
  residual <- sweep(log_times, 2, beta)
  answered <- !is.na(residual)
  residual[!answered] <- 0
  weights <- drop(answered %*% precision)
  shrink <- 1 + variance * weights
  form <- drop(residual^2 %*% precision) - variance * drop(residual %*% precision)^2 / shrink
  log_determinant <- drop(answered %*% log(1 / precision)) + log(shrink)
  keep <- rowSums(answered) > 0
  -0.5 * sum((rowSums(answered) * log(2 * pi) + log_determinant + form)[keep])
}

# Each person's posterior mean of τ and its precision, and the standard errors of the intensities.
posterior_speeds <- function(log_times, intensity, discrimination, speed_sd) {
  precision <- 1 / speed_sd^2 + drop((!is.na(log_times)) %*% discrimination^2)
  residual <- sweep(log_times, 2, intensity)
  residual[is.na(residual)] <- 0
  information <- Reduce(`+`, lapply(seq_len(nrow(log_times)), function(j) {
    weights <- ifelse(!is.na(log_times[j, ]), discrimination^2, 0)
    diag(weights) - speed_sd^2 * outer(weights, weights) / (1 + speed_sd^2 * sum(weights))
  }))
  list(speed = -drop(residual %*% discrimination^2) / precision, precision = precision,
       intensityErrors = sqrt(diag(solve(information))))
}

# The lognormal model by lme: β_i the item's fixed effect, 1 / α_i its residual SD, the speed τ = -b of the random
# intercept b ~ N(0, σ²). lme's optimizer stops near 1e-7 from the maximum in the log-likelihood; Newton steps on the
# written-out log-likelihood, with numDeriv's Richardson derivatives, take its estimates to the maximum. The
# log-likelihood is that of the log times.
lognormal <- function(log_times) {
  long <- data.frame(
    logt = as.vector(log_times),
    item = factor(rep(colnames(log_times), each = nrow(log_times)), levels = colnames(log_times)),
    person = factor(rep(seq_len(nrow(log_times)), times = ncol(log_times)))
  )
  long <- long[!is.na(long$logt), ]
  fit <- lme(logt ~ 0 + item, random = ~ 1 | person, weights = varIdent(form = ~ 1 | item), data = long, method = "ML",
             control = lmeControl(msMaxIter = 1000, tolerance = 1e-12, msTol = 1e-14, niterEM = 0))
  ratios <- coef(fit$modelStruct$varStruct, unconstrained = FALSE, allCoef = TRUE)
  sds <- fit$sigma * ratios[colnames(log_times)]
  lme_speed_sd <- as.numeric(nlme::VarCorr(fit)["(Intercept)", "StdDev"])
  k <- ncol(log_times)
  par <- c(unname(nlme::fixef(fit)), log(unname(sds)), log(lme_speed_sd))
  stopifnot(abs(lognormal_log_likelihood(par, log_times) - as.numeric(logLik(fit))) < 1e-9)
  objective <- function(p) lognormal_log_likelihood(p, log_times)
  for (step in 1:5) par <- par - solve(numDeriv::hessian(objective, par), numDeriv::grad(objective, par))
  stopifnot(max(abs(numDeriv::grad(objective, par))) < 1e-5)
  message("lme's log-likelihood is below the maximum by ", objective(par) - as.numeric(logLik(fit)))
  intensity <- par[1:k]
  discrimination <- exp(-par[k + 1:k])
  speed_sd <- exp(par[2 * k + 1])
  # The posterior speeds and the intensities' covariance (Σ X' V⁻¹ X)⁻¹, written out; at lme's own estimates they
  # equal its BLUPs -b, within the rounding of its variance parameters, and its vcov.
  at_lme <- posterior_speeds(log_times, unname(nlme::fixef(fit)), unname(1 / sds), lme_speed_sd)
  blup <- nlme::ranef(fit)[as.character(seq_len(nrow(log_times))), "(Intercept)"]
  stopifnot(max(abs(at_lme$speed + blup)) < 1e-7, max(abs(at_lme$intensityErrors - sqrt(diag(vcov(fit))))) < 1e-9)
  posterior <- posterior_speeds(log_times, intensity, discrimination, speed_sd)
  speed <- posterior$speed
  errors <- posterior$intensityErrors
  precision <- posterior$precision
  list(intensity = intensity, discrimination = discrimination, intensityErrors = errors, speedSd = speed_sd,
       logLikelihood = lognormal_log_likelihood(par, log_times), speed = speed, speedErrors = 1 / sqrt(precision))
}

# LNIRT's MCMC estimates of the same model, for a check within its Monte Carlo error.
mcmc_check <- function(log_times, reference) {
  set.seed(SEED)
  invisible(capture.output(mcmc <- LNRT(RT = log_times, XG = 4000, burnin = 25)))
  posterior <- mcmc$Post.Means
  difference <- max(abs(posterior$Time.Intensity - reference$intensity),
                    abs(1 / sqrt(posterior$Sigma2) - reference$discrimination) / reference$discrimination)
  message("LNIRT's posterior means: largest difference ", difference)
  stopifnot(difference < 0.1)
}

response_time_case <- function(name, log_times) {
  reference <- lognormal(log_times)
  mcmc_check(log_times, reference)
  # The residuals and the person statistic at the posterior speeds, as LNIRT's personfitLN computes them.
  residuals <- sweep(log_times, 2, reference$intensity) + reference$speed
  standardized <- sweep(residuals, 2, reference$discrimination, `*`)
  statistic <- rowSums(standardized^2, na.rm = TRUE)
  answered <- rowSums(!is.na(log_times))
  complete <- answered == ncol(log_times)
  fit <- LNIRT:::personfitLN(RT = log_times[complete, ], theta = reference$speed[complete], phi = rep(1, ncol(log_times)),
                             lambda = reference$intensity, sigma2 = 1 / reference$discrimination^2)
  stopifnot(max(abs(fit$lZPT - statistic[complete])) < 1e-10, max(abs(fit$lZP - pchisq(statistic[complete],
            ncol(log_times), lower.tail = FALSE))) < 1e-10)
  c(list(name = name, itemIds = colnames(log_times), times = unname(exp(log_times)),
         residuals = unname(standardized), statistic = statistic,
         significance = pchisq(statistic, answered, lower.tail = FALSE)), reference)
}

# A simulation: 400 persons, 12 items, β from 3 to 4.5, α from 1.2 to 2.5, τ ~ N(0, 0.3²), 5% of the times missing.
set.seed(SEED)
persons <- 400
items <- 12
beta <- round(runif(items, 3, 4.5), 2)
alpha <- round(runif(items, 1.2, 2.5), 2)
tau <- rnorm(persons, 0, 0.3)
simulated <- sapply(seq_len(items), function(i) rnorm(persons, beta[i] - tau, 1 / alpha[i]))
simulated[matrix(runif(length(simulated)) < 0.05, persons)] <- NA
colnames(simulated) <- sprintf("q%02d", seq_len(items))

# CredentialForm1: the first 10 scored items and their times in seconds.
data(CredentialForm1, package = "LNIRT")
durations <- as.matrix(CredentialForm1[, paste0("idur.", 1:10)])
durations[durations <= 0] <- NA
scored <- as.matrix(CredentialForm1[, paste0("iraw.", 1:10)])
colnames(durations) <- colnames(scored) <- sprintf("i%02d", 1:10)

# Rapid guessing on CredentialForm1: normative thresholds (Wise and Ma, 2012), response time effort (Wise and Kong,
# 2005) and effort-moderated EAP estimates (Wise and DeMars, 2006) under mirt's 2PL, rapid guesses as missing.
thresholds <- function(times, percentage, cap) pmin(cap, percentage / 100 * colMeans(times, na.rm = TRUE))
nt10 <- thresholds(durations, 10, 10)
rapid <- sweep(durations, 2, nt10, `<`)
effort <- 1 - rowSums(rapid, na.rm = TRUE) / rowSums(!is.na(durations))
two_pl <- mirt(as.data.frame(scored), 1, itemtype = "2PL", verbose = FALSE)
moderated <- scored
moderated[!is.na(rapid) & rapid] <- NA
eap <- fscores(two_pl, method = "EAP", response.pattern = moderated, quadpts = 61)
coefficients <- coef(two_pl, IRTpars = TRUE, simplify = TRUE)$items

write_fixture("response-time", list(
  source = "A lognormal simulation, set.seed(20261029): 400 persons, 12 items; LNIRT's CredentialForm1, items 1 to 10.",
  cases = list(
    response_time_case("simulated", log(exp(simulated))),
    response_time_case("CredentialForm1", log(durations))
  ),
  rapidGuessing = list(
    itemIds = colnames(durations),
    times = unname(durations),
    responses = unname(scored),
    thresholds = list(
      list(percentage = 10, cap = 10, values = nt10),
      list(percentage = 15, cap = 30, values = thresholds(durations, 15, 30))
    ),
    effort = effort,
    items = lapply(seq_len(nrow(coefficients)), function(i) list(a = coefficients[i, "a"], b = coefficients[i, "b"],
                                                                  c = 0, d = 1)),
    moderated = lapply(seq_len(nrow(eap)), function(r) list(theta = eap[r, 1], standardError = eap[r, 2]))
  )
))
