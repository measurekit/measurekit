# Response time fixtures. References: the lognormal model (van der Linden, 2006) as the linear mixed model
# nlme::lme(log(t) ~ 0 + item, random = ~ 1 | person, weights = varIdent(form = ~ 1 | item), method = "ML"), its
# estimates taken to the maximum of the written-out likelihood by Newton steps with numDeriv's derivatives; LNIRT's
# LNRT by MCMC as a second check; LNIRT's personfitLN for the person statistic of van der Linden and Guo (2008); mirt's
# EAP for effort-moderated scoring; and base R for the residuals, the thresholds and response time effort. The
# hierarchical model of speed and accuracy (response-time-speed-accuracy.json) has its marginal likelihood written out
# in base R, with theta on 61 rectangular points and the speed integrated out analytically, maximized with an exact
# gradient and polished by Newton's method; LNIRT's Gibbs sampler is a second check on the correlation and the time
# parameters. Data: a seeded simulation, and the first 10 items of LNIRT's CredentialForm1, times of 0 as missing; for
# the joint model also LNIRT's simLNIRT, with some responses and times missing.

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

# --- Hierarchical model of speed and accuracy (van der Linden, 2007) ---------------------------------------------
# Responses: the 2PL, P(x = 1 | theta) = plogis(a (theta - b)); times: ln t_ij ~ N(beta_i - tau_j, 1 / alpha_i^2); and
# theta ~ N(0, 1), tau | theta ~ N(rho sigma theta, sigma^2 (1 - rho^2)). Given theta a person's log times are normal
# with mean beta - a theta and covariance diag(1 / alpha^2) + v 11', a = rho sigma, v = sigma^2 (1 - rho^2); the marginal
# likelihood integrates theta over 61 rectangular points. Parameters, in this order: a, b, beta, alpha, sigma, rho.

SA_NODES <- seq(-6, 6, length.out = 61)
SA_LOG_WEIGHTS <- log(dnorm(SA_NODES) / sum(dnorm(SA_NODES)))

sa_unpack <- function(par, k) {
  list(a = par[1:k], b = par[k + 1:k], beta = par[2 * k + 1:k], alpha = par[3 * k + 1:k], sigma = par[4 * k + 1],
       rho = par[4 * k + 2])
}

# The sums of each person's timed responses that the log density of the times is made of.
sa_time_sums <- function(log_times, beta, alpha) {
  observed <- !is.na(log_times)
  centered <- sweep(log_times, 2, beta)
  centered[!observed] <- 0
  weights <- alpha^2
  list(observed = observed, centered = centered, n = rowSums(observed), S = drop(observed %*% weights),
       A0 = drop(centered %*% weights), B0 = drop(centered^2 %*% weights), Lw = drop(observed %*% log(weights)))
}

# ln f(y | theta) of each person at each node, by the matrix determinant lemma and Sherman and Morrison.
sa_time_loglik <- function(sums, a, v) {
  shift <- a * SA_NODES
  shrink <- 1 + v * sums$S
  total <- outer(sums$S, shift) + sums$A0
  form <- sums$B0 + 2 * outer(sums$A0, shift) + outer(sums$S, shift^2) - v * total^2 / shrink
  -0.5 * (sums$n * log(2 * pi) - sums$Lw + log(shrink)) - 0.5 * form
}

# The same density straight from the multivariate normal, for one person and node.
sa_time_loglik_direct <- function(log_times, beta, alpha, a, v, theta) {
  seen <- !is.na(log_times)
  if (!any(seen)) return(0)
  covariance <- diag(1 / alpha[seen]^2, sum(seen)) + v
  error <- log_times[seen] - (beta[seen] - a * theta)
  -0.5 * (sum(seen) * log(2 * pi) + as.numeric(determinant(covariance)$modulus) + drop(t(error) %*% solve(covariance, error)))
}

# The response model: the logistic 2PL, or the normal ogive that LNIRT's responses follow, P(x = 1) = pnorm(eta).
sa_link <- function(link) if (link == "logit") plogis else pnorm

# Each person's log joint density at each node: the responses and the times, without the weights.
sa_person_node_loglik <- function(par, responses, log_times, link = "logit") {
  p <- sa_unpack(par, ncol(responses))
  cdf <- sa_link(link)
  joint <- matrix(0, nrow(responses), length(SA_NODES))
  for (i in seq_len(ncol(responses))) {
    observed <- !is.na(responses[, i])
    x <- ifelse(observed, responses[, i], 0)
    eta <- p$a[i] * (SA_NODES - p$b[i])
    joint <- joint + outer(x, cdf(eta, log.p = TRUE)) + outer(observed - x, cdf(eta, lower.tail = FALSE, log.p = TRUE))
  }
  joint + sa_time_loglik(sa_time_sums(log_times, p$beta, p$alpha), p$rho * p$sigma, p$sigma^2 * (1 - p$rho^2))
}

sa_loglik <- function(par, responses, log_times, link = "logit") {
  joint <- sweep(sa_person_node_loglik(par, responses, log_times, link), 2, SA_LOG_WEIGHTS, "+")
  top <- apply(joint, 1, max)
  sum(top + log(rowSums(exp(joint - top))))
}

# The posterior of theta over the nodes, and its first two moments.
sa_posterior <- function(par, responses, log_times, link = "logit") {
  joint <- sweep(sa_person_node_loglik(par, responses, log_times, link), 2, SA_LOG_WEIGHTS, "+")
  posterior <- exp(joint - apply(joint, 1, max))
  posterior / rowSums(posterior)
}

# The gradient of the marginal log-likelihood by Fisher's identity: the posterior mean of the score of each person's
# responses and times given theta. The score of the times is quadratic in theta, so it needs the posterior mean and
# second moment of theta alone.
sa_gradient <- function(par, responses, log_times, link = "logit") {
  k <- ncol(responses)
  p <- sa_unpack(par, k)
  cdf <- sa_link(link)
  density <- if (link == "logit") dlogis else dnorm
  posterior <- sa_posterior(par, responses, log_times, link)
  theta_mean <- drop(posterior %*% SA_NODES)
  theta_square <- drop(posterior %*% SA_NODES^2)
  grad_a <- grad_b <- numeric(k)
  for (i in seq_len(k)) {
    observed <- !is.na(responses[, i])
    x <- ifelse(observed, responses[, i], 0)
    eta <- p$a[i] * (SA_NODES - p$b[i])
    # The score of one response in eta: x f / F - (1 - x) f / (1 - F), which is x - F for the logistic.
    score <- outer(x, density(eta) / cdf(eta)) - outer(1 - x, density(eta) / cdf(eta, lower.tail = FALSE))
    weighted <- posterior * score * observed
    grad_a[i] <- sum(sweep(weighted, 2, SA_NODES - p$b[i], `*`))
    grad_b[i] <- -p$a[i] * sum(weighted)
  }
  a <- p$rho * p$sigma
  v <- p$sigma^2 * (1 - p$rho^2)
  s <- sa_time_sums(log_times, p$beta, p$alpha)
  shrink <- 1 + v * s$S
  g <- (s$centered - v * s$A0 / shrink) * s$observed
  h <- a / shrink
  weights <- p$alpha^2
  first <- sweep(g + h * theta_mean, 2, weights, `*`) * s$observed
  second <- sweep(g^2 + 2 * g * h * theta_mean + h^2 * theta_square, 2, weights^2, `*`) * s$observed
  inverse <- sweep(1 - v * sweep(matrix(1, nrow(responses), k), 2, weights, `*`) / shrink, 2, weights, `*`) * s$observed
  grad_beta <- colSums(first)
  grad_weight <- colSums(inverse - second) / (2 * weights^2)
  grad_a_time <- -sum((s$A0 * theta_mean + a * s$S * theta_square) / shrink)
  grad_v <- -0.5 * sum(s$S / shrink - (s$A0^2 + 2 * s$A0 * a * s$S * theta_mean + a^2 * s$S^2 * theta_square) / shrink^2)
  c(grad_a, grad_b, grad_beta, 2 * p$alpha * grad_weight,
    p$rho * grad_a_time + 2 * p$sigma * (1 - p$rho^2) * grad_v, p$sigma * grad_a_time - 2 * p$sigma^2 * p$rho * grad_v)
}

# The posterior means and SDs of theta and of the speed tau, tau | theta being normal with precision 1 / v + S and mean
# (a theta / v - A0) / precision, and, as a check, the same by numerical integration over theta and tau.
sa_estimates <- function(par, responses, log_times) {
  p <- sa_unpack(par, ncol(responses))
  a <- p$rho * p$sigma
  v <- p$sigma^2 * (1 - p$rho^2)
  s <- sa_time_sums(log_times, p$beta, p$alpha)
  posterior <- sa_posterior(par, responses, log_times)
  theta_mean <- drop(posterior %*% SA_NODES)
  theta_square <- drop(posterior %*% SA_NODES^2)
  precision <- 1 / v + s$S
  ratio <- a / v
  speed_mean <- (ratio * theta_mean - s$A0) / precision
  speed_square <- (ratio^2 * theta_square - 2 * ratio * s$A0 * theta_mean + s$A0^2) / precision^2 + 1 / precision
  has_times <- s$n > 0
  list(theta = theta_mean, thetaSd = sqrt(theta_square - theta_mean^2),
       speed = ifelse(has_times, speed_mean, NA), speedSd = ifelse(has_times, sqrt(speed_square - speed_mean^2), NA))
}

# Checks the closed forms on a few persons: the density of the times against the multivariate normal, and the posterior
# of theta and tau against numerical integration over theta and a fine grid of tau.
sa_check_closed_forms <- function(par, responses, log_times, estimates) {
  p <- sa_unpack(par, ncol(responses))
  a <- p$rho * p$sigma
  v <- p$sigma^2 * (1 - p$rho^2)
  sums <- sa_time_sums(log_times, p$beta, p$alpha)
  closed <- sa_time_loglik(sums, a, v)
  people <- which(sums$n >= 2)[c(1, 2, 3, 4, 5, 6)]
  for (person in people) {
    for (q in c(5, 30, 55)) {
      stopifnot(abs(closed[person, q] - sa_time_loglik_direct(log_times[person, ], p$beta, p$alpha, a, v, SA_NODES[q])) < 1e-9)
    }
    taus <- seq(-8, 8, length.out = 4001) * p$sigma
    seen <- !is.na(log_times[person, ])
    joint <- sapply(seq_along(SA_NODES), function(q) {
      response_log <- sum(sapply(seq_len(ncol(responses)), function(i) {
        if (is.na(responses[person, i])) return(0)
        plogis(p$a[i] * (SA_NODES[q] - p$b[i]), lower.tail = responses[person, i] == 1, log.p = TRUE)
      }))
      time_log <- sapply(taus, function(tau) sum(dnorm(log_times[person, seen], p$beta[seen] - tau, 1 / p$alpha[seen], log = TRUE)))
      exp(SA_LOG_WEIGHTS[q] + response_log + dnorm(taus, a * SA_NODES[q], sqrt(v), log = TRUE) + time_log - max(time_log))
    })
    joint <- joint / sum(joint)
    theta_mean <- sum(joint %*% SA_NODES)
    speed_mean <- sum(taus * rowSums(joint))
    stopifnot(abs(theta_mean - estimates$theta[person]) < 1e-6, abs(speed_mean - estimates$speed[person]) < 1e-4,
              abs(sqrt(sum(taus^2 * rowSums(joint)) - speed_mean^2) - estimates$speedSd[person]) < 1e-4)
  }
}

sa_fit <- function(responses, log_times, link = "logit") {
  k <- ncol(responses)
  negative <- function(par) -sa_loglik(par, responses, log_times, link)
  negative_gradient <- function(par) -sa_gradient(par, responses, log_times, link)
  probe <- c(rep(1.1, k), seq(-0.4, 0.4, length.out = k), colMeans(log_times, na.rm = TRUE) + 0.1,
             1 / apply(log_times, 2, sd, na.rm = TRUE) * 0.9, 0.35, 0.2)
  stopifnot(max(abs(negative_gradient(probe) - numDeriv::grad(negative, probe))) < 1e-6 * max(abs(negative_gradient(probe))))
  start <- c(rep(1, k), rep(0, k), colMeans(log_times, na.rm = TRUE), 1 / apply(log_times, 2, sd, na.rm = TRUE), 0.3, 0)
  lower <- c(rep(0.05, k), rep(-Inf, k), rep(-Inf, k), rep(0.05, k), 0.01, -0.99)
  upper <- c(rep(Inf, k), rep(Inf, k), rep(Inf, k), rep(Inf, k), Inf, 0.99)
  par <- optim(start, negative, negative_gradient, method = "L-BFGS-B", lower = lower, upper = upper,
               control = list(maxit = 2000, factr = 1, pgtol = 0))$par
  for (iteration in 1:50) {
    step <- solve(numDeriv::jacobian(negative_gradient, par), negative_gradient(par))
    par <- par - step
    if (max(abs(step)) < 1e-11) break
  }
  stopifnot(max(abs(negative_gradient(par))) < 1e-6)
  # The observed information: numDeriv's Richardson Jacobian of the exact gradient, symmetrized. optimHess with its
  # default steps of 1e-3 is off by up to 2e-4 in the standard errors; with steps of 1e-5 it agrees to 2e-8.
  information <- numDeriv::jacobian(negative_gradient, par)
  information <- (information + t(information)) / 2
  list(par = par, logLikelihood = -negative(par), standardErrors = sqrt(diag(solve(information))))
}

# LNIRT's Gibbs sampler, which needs every time, on the persons who have them all. Its time parameters and the speed
# SD do not involve the response model, and are compared with the maximum likelihood estimates within the sampler's
# Monte Carlo error; its correlation is compared where the items are well determined. LNIRT estimates the items with a
# population distribution, which the package leaves out, and where most persons answer correctly the correlation
# differs: not by the link, which the normal ogive maximum likelihood estimate shows, but by the sampler's model.
# `rho_agrees` says whether to require the correlations to agree.
sa_lnirt_check <- function(responses, log_times, rho_agrees) {
  complete <- rowSums(is.na(log_times)) == 0 & rowSums(is.na(responses)) == 0
  kept_responses <- responses[complete, ]
  kept_times <- log_times[complete, ]
  k <- ncol(responses)
  logistic <- sa_fit(kept_responses, kept_times)
  ogive <- sa_fit(kept_responses, kept_times, "probit")
  set.seed(SEED)
  invisible(capture.output(mcmc <- LNIRT(RT = kept_times, Y = kept_responses, XG = 4000, burnin = 25, td = FALSE,
                                          guess = FALSE, WL = FALSE)))
  means <- mcmc$Post.Means
  correlation <- means$Cov.Person.Ability.Speed / sqrt(means$Var.Person.Ability * means$Var.Person.Speed)
  differences <- function(fit) {
    p <- sa_unpack(fit$par, k)
    c(rho = abs(correlation - p$rho), intensity = max(abs(means$Time.Intensity - p$beta)),
      discrimination = max(abs(1 / sqrt(means$Sigma2) - p$alpha) / p$alpha), speedSd = abs(sqrt(means$Var.Person.Speed) - p$sigma))
  }
  from_ogive <- differences(ogive)
  from_logistic <- differences(logistic)
  message("LNIRT against the normal ogive: ", paste(names(from_ogive), signif(from_ogive, 3), collapse = ", "))
  message("LNIRT against the logistic: ", paste(names(from_logistic), signif(from_logistic, 3), collapse = ", "))
  stopifnot(all(from_ogive[-1] < 0.1), all(from_logistic[-1] < 0.1), !rho_agrees || from_ogive[["rho"]] < 0.05)
  list(persons = sum(complete), correlation = correlation, speedSd = sqrt(means$Var.Person.Speed),
       intensity = means$Time.Intensity, discrimination = 1 / sqrt(means$Sigma2),
       ogive = list(correlation = sa_unpack(ogive$par, k)$rho, logLikelihood = ogive$logLikelihood,
                    standardError = sa_unpack(ogive$standardErrors, k)$rho),
       differencesFromOgive = as.list(from_ogive), differencesFromLogistic = as.list(from_logistic),
       bootstrap = if (rho_agrees) NULL else sa_lnirt_bootstrap(ogive, kept_responses, kept_times))
}

# Where the correlations differ: data simulated from the normal ogive estimates, for which the correlation is known,
# fitted by maximum likelihood and by LNIRT.
sa_lnirt_bootstrap <- function(ogive, responses, log_times) {
  k <- ncol(responses)
  n <- nrow(responses)
  p <- sa_unpack(ogive$par, k)
  set.seed(SEED)
  theta <- rnorm(n)
  tau <- p$sigma * (p$rho * theta + sqrt(1 - p$rho^2) * rnorm(n))
  simulated_responses <- sapply(seq_len(k), function(i) as.numeric(runif(n) < pnorm(p$a[i] * (theta - p$b[i]))))
  simulated_times <- sapply(seq_len(k), function(i) p$beta[i] - tau + rnorm(n) / p$alpha[i])
  colnames(simulated_responses) <- colnames(simulated_times) <- colnames(responses)
  recovered <- sa_unpack(sa_fit(simulated_responses, simulated_times, "probit")$par, k)$rho
  invisible(capture.output(mcmc <- LNIRT(RT = simulated_times, Y = simulated_responses, XG = 4000, burnin = 25, td = FALSE,
                                          guess = FALSE, WL = FALSE)))
  means <- mcmc$Post.Means
  shrunk <- means$Cov.Person.Ability.Speed / sqrt(means$Var.Person.Ability * means$Var.Person.Speed)
  message("Simulated from rho = ", signif(p$rho, 4), ": maximum likelihood ", signif(recovered, 4), ", LNIRT ", signif(shrunk, 4))
  stopifnot(abs(recovered - p$rho) < 0.05)
  list(correlation = p$rho, maximumLikelihood = recovered, lnirt = shrunk)
}

speed_accuracy_case <- function(name, responses, log_times, rho_agrees) {
  fit <- sa_fit(responses, log_times)
  estimates <- sa_estimates(fit$par, responses, log_times)
  sa_check_closed_forms(fit$par, responses, log_times, estimates)
  k <- ncol(responses)
  p <- sa_unpack(fit$par, k)
  errors <- sa_unpack(fit$standardErrors, k)
  message(name, ": ", nrow(responses), " persons, ", k, " items; log-likelihood ", round(fit$logLikelihood, 6), ", rho ",
          round(p$rho, 4), ", sigma ", round(p$sigma, 4))
  list(
    name = name, itemIds = colnames(responses), responses = unname(lapply(seq_len(nrow(responses)), function(i) as.numeric(responses[i, ]))),
    times = unname(lapply(seq_len(nrow(responses)), function(i) as.numeric(exp(log_times[i, ])))),
    a = p$a, b = p$b, intensity = p$beta, discrimination = p$alpha, speedSd = p$sigma, correlation = p$rho,
    logLikelihood = fit$logLikelihood,
    standardErrors = list(a = errors$a, b = errors$b, intensity = errors$beta, discrimination = errors$alpha,
                          speedSd = errors$sigma, correlation = errors$rho),
    estimates = estimates, lnirt = sa_lnirt_check(responses, log_times, rho_agrees)
  )
}

# CredentialForm1: the first 10 items, with the persons' times of 0 as missing (the response stays).
# The joint model's maximum, to the tolerance of 1e-11 that the comparison takes, runs for as many iterations as there are
# persons' worth of information to settle, and takes seconds for each hundred persons under coverage: the case keeps the first
# 500 persons. On 300 the base R fit of its bootstrap fails.
credential_responses <- scored[1:500, ]
credential_times <- log(durations)[1:500, ]

# simLNIRT: 400 persons, 12 items, rho = 0.5; 2% of the responses missing, with their times, and 3% of the other times.
set.seed(SEED)
joint_simulation <- simLNIRT(N = 400, K = 12, rho = 0.5, WL = FALSE)
simulated_responses <- joint_simulation$Y
simulated_times <- joint_simulation$RT
colnames(simulated_responses) <- colnames(simulated_times) <- sprintf("s%02d", 1:12)
missing_response <- matrix(runif(length(simulated_responses)) < 0.02, nrow(simulated_responses))
missing_time <- matrix(runif(length(simulated_times)) < 0.03, nrow(simulated_times))
simulated_responses[missing_response] <- NA
simulated_times[missing_response | missing_time] <- NA

write_fixture("response-time-speed-accuracy", list(
  source = "LNIRT's CredentialForm1, its first 500 persons and items 1 to 10, times of 0 as missing; LNIRT's simLNIRT, set.seed(20261029): 400 persons, 12 items, rho = 0.5, with some responses and times missing.",
  cases = list(
    speed_accuracy_case("CredentialForm1", credential_responses, credential_times, rho_agrees = FALSE),
    speed_accuracy_case("simLNIRT", simulated_responses, simulated_times, rho_agrees = TRUE)
  )
))
