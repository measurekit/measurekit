# Cognitive diagnosis fixtures. References: GDINA's GDINA (G-DINA, DINA, DINO, A-CDM, LLM, R-RUM), personparm, CA by MAP
# (by EAP, CA's computation in base R) and Qval(method = "PVAF"); NPCD's AlphaNP with the Hamming distance; base R's
# constrOptim for one M-step of an additive model. Data: Tatsuoka's fraction subtraction data as GDINA ships it (536
# persons, 20 items, 8 attributes), a seeded G-DINA simulation, GDINA's realdata_ECPE (2922 persons, 28 items, 3
# attributes) and sim30GDINA (1000 persons, 30 items, 5 attributes).

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(GDINA)
  library(NPCD)
  library(alabama)
})

SEED <- 20261029
CONTROL <- list(conv.crit = 1e-9, maxitr = 20000)

# The profiles whose ideal responses under the DINA agree: their proportions are not identified apart, only summed.
dina_classes <- function(Q) {
  ideal <- apply(attributepattern(ncol(Q)), 1, function(a) paste(apply(Q, 1, function(q) all(a[q == 1] == 1)) * 1, collapse = ""))
  match(ideal, unique(ideal))
}

# GDINA's CA by EAP: GDINA 2.13.2 matches the EAP profiles to the profiles by whichrow_AinB, which returns 1 for every
# row, so CA(what = "EAP") classifies every person in the first profile. This is CA's computation, from CM, with the
# profiles matched by their patterns.
eap_accuracy <- function(fit) {
  post <- exp(indlogPost(fit))
  pattern <- extract(fit, "attributepattern")
  pp <- personparm(fit, what = "EAP")
  mp <- personparm(fit, what = "mp", digits = 15)
  gr <- match(apply(pp, 1, paste, collapse = ""), apply(pattern, 1, paste, collapse = ""))
  ccm <- crossprod(post, diag(nrow(pattern))[gr, , drop = FALSE])
  ccm <- ccm / pmax(rowSums(ccm), .Machine$double.eps)
  tau_l <- diag(ccm)
  list(tau = sum(tau_l * c(extract(fit, "posterior.prob"))), tau_l = tau_l,
       tau_k = colMeans(pp * mp + (1 - pp) * (1 - mp)), gamma = sum(post * post) / nrow(post),
       gamma_k = colMeans(mp * mp + (1 - mp) * (1 - mp)))
}

# `model` is one model or one per item. With `deltas`, the fit also holds GDINA's coefficients on the link scale.
cdm_fit <- function(data, Q, model, persons = TRUE, deltas = FALSE, control = CONTROL) {
  fit <- GDINA(data, Q, model = model, control = control, verbose = 0)
  stopifnot(extract(fit, "convergence"))
  posterior <- exp(extract(fit, "logposterior.i"))
  map <- personparm(fit, what = "MAP")
  # GDINA breaks ties of the MAP at random: tied persons have no reference MAP, and CA is compared by EAP only. Its
  # additive M-steps are solved to about 1e-6, so there a MAP between posteriors within 1e-4 is a tie as well.
  tied <- map$multimodes
  if (deltas) tied <- tied | apply(posterior, 1, function(x) diff(sort(x, decreasing = TRUE)[2:1]) < 1e-4)
  classifications <- if (any(tied)) "EAP" else c("MAP", "EAP")
  result <- list(
    model = tolower(model),
    items = unname(lapply(coef(fit, digits = 15), function(p) unname(p))),
    guessing = unname(coef(fit, what = "gs", digits = 15)[, 1]),
    slip = unname(coef(fit, what = "gs", digits = 15)[, 2]),
    proportions = unname(c(extract(fit, "posterior.prob"))),
    logLikelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    bic = BIC(fit),
    classes = if (identical(model, "DINA")) dina_classes(Q),
    persons = if (persons) lapply(1:20, function(i) list(
      mastery = unname(personparm(fit, what = "mp", digits = 15)[i, ]),
      map = if (tied[i]) NULL else unname(as.numeric(map[i, seq_len(ncol(Q))])),
      eap = unname(as.numeric(personparm(fit, what = "EAP")[i, ])),
      posterior = unname(posterior[i, ])
    )),
    accuracy = if (persons) lapply(classifications, function(classification) {
      accuracy <- if (classification == "MAP") CA(fit, what = "MAP") else eap_accuracy(fit)
      list(classification = tolower(classification), accuracy = accuracy$tau, profileAccuracy = unname(accuracy$tau_l),
           attributeAccuracy = unname(accuracy$tau_k), consistency = accuracy$gamma,
           attributeConsistency = unname(accuracy$gamma_k))
    }),
    validation = if (identical(model, "GDINA") && persons) {
      q <- Qval(fit, method = "PVAF", eps = 0.95, digits = 15)
      lapply(seq_len(nrow(Q)), function(j) list(
        suggested = I(colnames(Q)[q$sug.Q[j, ] == 1]),
        pvaf = q$PVAF[apply(q$sug.Q[j, , drop = FALSE], 1, paste, collapse = ""), j]
      ))
    }
  )
  if (deltas) result$coefficients <- unname(lapply(coef(fit, what = "delta", digits = 15), function(d) unname(d)))
  result
}

nonparametric <- function(data, Q, gate) {
  result <- AlphaNP(data, Q, gate = gate, method = "Hamming")
  # NPCD's profiles in measurekit's order: GDINA's attributepattern.
  order <- match(apply(attributepattern(ncol(Q)), 1, paste, collapse = ""), apply(result$pattern, 1, paste, collapse = ""))
  distance <- apply(result$loss.matrix, 2, min)
  list(gate = gate, distance = distance, ties = unname(apply(result$loss.matrix, 2, function(x) sum(x == min(x)))),
       profile = lapply(seq_len(nrow(data)), function(i) {
         if (result$n.tie[i] > 0) return(NULL)
         unname(as.numeric(result$alpha.est[i, ]))
       }),
       lossMatrix = unname(t(result$loss.matrix[order, ])))
}

cdm_case <- function(name, data, Q, models, persons) {
  Q <- psychometrics_order(Q)
  colnames(data) <- paste0("item", seq_len(ncol(data)))
  list(
    name = name,
    itemIds = colnames(data),
    responses = unname(as.matrix(data)),
    qmatrix = qmatrix(Q, colnames(data)),
    attributes = colnames(Q),
    fits = lapply(models, function(model) cdm_fit(data, Q, model, persons)),
    nonparametric = lapply(c("AND", "OR"), function(gate) nonparametric(data[1:30, ], Q, gate))
  )
}

data(realdata_Tatsuoka1990, package = "GDINA")
fraction <- realdata_Tatsuoka1990$dat
fraction_q <- as.matrix(realdata_Tatsuoka1990$Q)
colnames(fraction_q) <- paste0("A", seq_len(ncol(fraction_q)))

# A G-DINA simulation: 4 attributes, 15 items, 1500 persons from a saturated distribution over the 16 profiles.
set.seed(SEED)
K <- 4
sim_q <- rbind(diag(K), diag(K), matrix(c(1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
                                          1, 1, 1, 0), ncol = K, byrow = TRUE))
colnames(sim_q) <- c("add", "borrow", "convert", "simplify")
patterns <- attributepattern(K)
weights <- runif(nrow(patterns), 0.2, 1)
alpha <- patterns[sample.int(nrow(patterns), 1500, replace = TRUE, prob = weights / sum(weights)), ]
success <- function(q, a) {
  required <- which(q == 1)
  mastered <- sum(a[required])
  0.15 + 0.7 * mastered / length(required) + if (mastered == length(required)) 0.05 else 0
}
simulated <- sapply(seq_len(nrow(sim_q)), function(j) {
  rbinom(nrow(alpha), 1, apply(alpha, 1, function(a) success(sim_q[j, ], a)))
})

write_fixture("cdm", list(
  source = "GDINA's realdata_Tatsuoka1990 (536 persons, 20 items, 8 attributes); a G-DINA simulation, set.seed(20261029).",
  cases = list(
    # With 8 attributes the DINA's profiles fall in classes the data cannot tell apart, so the persons' posteriors and
    # the profile proportions within a class are not identified: the fixture compares the classes' proportions. The
    # G-DINA with 8 attributes on 536 persons has several maxima, and GDINA's is not the largest.
    cdm_case("fraction subtraction", fraction, fraction_q, "DINA", FALSE),
    cdm_case("simulated", simulated, sim_q, c("GDINA", "DINO"), TRUE)
  )
))

# ---- Additive models ----------------------------------------------------------------------------------------------

# GDINA's additive M-steps are solved to about 1e-6, so its EM does not reach a change below about 5e-9: the criterion is
# the tightest it meets.
ADDITIVE_CONTROL <- list(conv.crit = 1e-8, maxitr = 5000)

# One model per item, cycled: every model GDINA fits with the identity, logit and log links, and the three that
# are not additive, so that the coefficients of every model are compared.
mixed_models <- function(items) rep(c("ACDM", "LLM", "RRUM", "GDINA", "DINA", "DINO"), length.out = items)

additive_case <- function(name, data, Q) {
  Q <- psychometrics_order(Q)
  colnames(data) <- paste0("item", seq_len(ncol(data)))
  models <- list("ACDM", "LLM", "RRUM", mixed_models(ncol(data)))
  list(
    name = name,
    itemIds = colnames(data),
    responses = unname(as.matrix(data)),
    qmatrix = qmatrix(Q, colnames(data)),
    attributes = colnames(Q),
    fits = lapply(models, function(model) cdm_fit(data, Q, model, persons = TRUE, deltas = TRUE, control = ADDITIVE_CONTROL))
  )
}

# The M-step of an additive item on fixed expected counts, written out: the maximum of
# sum_l r_l ln P_l + (n_l - r_l) ln(1 - P_l) with P = g^-1(M delta) and 1e-4 <= P <= 1 - 1e-4. The objective is convex in
# delta, so a point that meets the KKT conditions is the maximum. A candidate from alabama's augmented Lagrangian is
# polished by Newton's method on the face of the constraints it meets, and kept only when it passes the conditions:
# feasible, and a nonnegative multiplier for every bound it meets.
LINKS <- list(
  identity = list(g = function(p) p, inverse = function(e) e, slope = function(r, n, p) (n - r) / (1 - p) - r / p,
                  curvature = function(r, n, p) r / p^2 + (n - r) / (1 - p)^2),
  logit = list(g = qlogis, inverse = plogis, slope = function(r, n, p) n * p - r,
               curvature = function(r, n, p) n * p * (1 - p)),
  log = list(g = log, inverse = exp, slope = function(r, n, p) (n - r) * p / (1 - p) - r,
             curvature = function(r, n, p) (n - r) * p / (1 - p)^2)
)

additive_maximum <- function(link, K, r, n, lower = 1e-4, upper = 1 - 1e-4) {
  L <- LINKS[[link]]
  M <- cbind(1, attributepattern(K))
  probability <- function(d) pmin(pmax(L$inverse(drop(M %*% d)), lower), upper)
  objective <- function(d) { p <- probability(d); -sum(r * log(p) + (n - r) * log1p(-p)) }
  gradient <- function(d) drop(crossprod(M, L$slope(r, n, probability(d))))
  hessian <- function(d) crossprod(M, M * L$curvature(r, n, probability(d)))
  A <- rbind(M, -M)
  b <- c(rep(L$g(lower), nrow(M)), rep(-L$g(upper), nrow(M)))
  start <- c(L$g(0.5), rep(0, ncol(M) - 1))
  candidate <- auglag(start, fn = objective, gr = gradient, hin = function(d) drop(A %*% d - b), hin.jac = function(d) A,
                      control.outer = list(trace = FALSE, kkt2.check = FALSE, eps = 1e-12), control.optim = list(maxit = 5000))
  d <- candidate$par
  active <- which(drop(A %*% d - b) < 1e-5)
  working <- integer(0)
  for (i in active) if (qr(A[c(working, i), , drop = FALSE])$rank == length(working) + 1) working <- c(working, i)
  Aw <- A[working, , drop = FALSE]
  for (iteration in 1:50) {
    k <- length(working)
    kkt <- rbind(cbind(hessian(d), -t(Aw)), cbind(Aw, matrix(0, k, k)))
    step <- solve(kkt, c(-gradient(d), b[working] - drop(Aw %*% d)))[seq_len(ncol(M))]
    d <- d + step
    if (max(abs(step)) < 1e-15) break
  }
  g <- gradient(d)
  lambda <- if (length(working)) qr.solve(t(Aw), g) else numeric(0)
  scale <- 1 + max(abs(g))
  residual <- max(abs(g - if (length(working)) drop(crossprod(Aw, lambda)) else 0))
  certified <- residual <= 1e-9 * scale && all(lambda >= -1e-9 * scale) && min(drop(A %*% d - b)) >= -1e-12
  list(certified = certified, active = length(working), delta = d, probabilities = probability(d), value = objective(d))
}

# Cases are expected counts that make a boundary solution likely: success probabilities of a DINA or DINO item, with and
# without slips and guesses, and mixes of extreme ones, which no additive item fits.
maximum_pattern <- function(kind, K) {
  mastered <- rowSums(attributepattern(K))
  switch(kind,
    dina = ifelse(mastered == K, 0.97, 0.03), dino = ifelse(mastered == 0, 0.03, 0.97),
    dina_exact = ifelse(mastered == K, 1, 0), dino_exact = ifelse(mastered == 0, 0, 1),
    non_monotone = runif(2^K, 0, 1), additive = 0.1 + 0.8 * mastered / K,
    mixed = sample(c(0, 1, 0.02, 0.98, 0.5), 2^K, TRUE))
}

maximum_cases <- function() {
  set.seed(SEED)
  cases <- list()
  for (link in names(LINKS)) for (K in 2:4) for (kind in c("dina", "dino", "dina_exact", "dino_exact", "non_monotone", "additive", "mixed")) {
    persons <- runif(2^K, 50, 500)
    correct <- persons * maximum_pattern(kind, K)
    result <- additive_maximum(link, K, correct, persons)
    if (!result$certified) next
    cases[[length(cases) + 1]] <- list(link = link, attributeCount = K, kind = kind, correct = correct, persons = persons,
                                      activeBounds = result$active, delta = result$delta,
                                      probabilities = result$probabilities, value = result$value)
  }
  # Cases whose reference does not pass the conditions are left out; the tests check the conditions for every case.
  stopifnot(length(cases) >= 40, sum(sapply(cases, function(case) case$activeBounds > 0)) >= 15)
  cases
}

data(realdata_ECPE, package = "GDINA")
data(sim30GDINA, package = "GDINA")
write_fixture("cdm-additive", list(
  source = "GDINA's realdata_ECPE (2922 persons, 28 items, 3 attributes) and sim30GDINA (1000 persons, 30 items, 5 attributes), under the A-CDM, LLM, R-RUM and a mix of all six models per item.",
  cases = list(
    additive_case("ECPE", realdata_ECPE$dat, as.matrix(realdata_ECPE$Q)),
    additive_case("sim30", sim30GDINA$simdat, as.matrix(sim30GDINA$simQ))
  )
))
write_fixture("cdm-maximum", list(
  source = "Expected counts of one item and the maximum of its additive M-step, found by Newton's method on the face of the bounds it meets and certified by the KKT conditions, base R.",
  cases = maximum_cases()
))
