# Cognitive diagnosis fixtures. References: GDINA's GDINA (G-DINA, DINA, DINO), personparm, CA by MAP (by EAP, CA's
# computation in base R) and Qval(method = "PVAF"); NPCD's AlphaNP with the Hamming distance. Data: Tatsuoka's fraction
# subtraction data as GDINA ships it (536 persons, 20 items, 8 attributes) and a seeded G-DINA simulation.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(GDINA)
  library(NPCD)
})

SEED <- 20261029
CONTROL <- list(conv.crit = 1e-9, maxitr = 20000)

# The Q-matrix's columns in the package's order: attributes by first appearance over the items, so that both libraries order
# the profiles and each item's latent groups alike.
psychometrics_order <- function(Q) {
  first <- unique(unlist(lapply(seq_len(nrow(Q)), function(j) which(Q[j, ] == 1))))
  Q[, first, drop = FALSE]
}

qmatrix <- function(Q, items) {
  setNames(lapply(seq_len(nrow(Q)), function(j) I(colnames(Q)[Q[j, ] == 1])), items)
}

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

cdm_fit <- function(data, Q, model, persons = TRUE) {
  fit <- GDINA(data, Q, model = model, control = CONTROL, verbose = 0)
  stopifnot(extract(fit, "convergence"))
  posterior <- exp(extract(fit, "logposterior.i"))
  map <- personparm(fit, what = "MAP")
  # GDINA breaks ties of the MAP at random: tied persons have no reference MAP, and CA is compared by EAP only.
  classifications <- if (any(map$multimodes)) "EAP" else c("MAP", "EAP")
  list(
    model = tolower(model),
    items = unname(lapply(coef(fit, digits = 15), function(p) unname(p))),
    guessing = unname(coef(fit, what = "gs", digits = 15)[, 1]),
    slip = unname(coef(fit, what = "gs", digits = 15)[, 2]),
    proportions = unname(c(extract(fit, "posterior.prob"))),
    logLikelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    bic = BIC(fit),
    classes = if (model == "DINA") dina_classes(Q),
    persons = if (persons) lapply(1:20, function(i) list(
      mastery = unname(personparm(fit, what = "mp", digits = 15)[i, ]),
      map = if (map$multimodes[i]) NULL else unname(as.numeric(map[i, seq_len(ncol(Q))])),
      eap = unname(as.numeric(personparm(fit, what = "EAP")[i, ])),
      posterior = unname(posterior[i, ])
    )),
    accuracy = if (persons) lapply(classifications, function(classification) {
      accuracy <- if (classification == "MAP") CA(fit, what = "MAP") else eap_accuracy(fit)
      list(classification = tolower(classification), accuracy = accuracy$tau, profileAccuracy = unname(accuracy$tau_l),
           attributeAccuracy = unname(accuracy$tau_k), consistency = accuracy$gamma,
           attributeConsistency = unname(accuracy$gamma_k))
    }),
    validation = if (model == "GDINA" && persons) {
      q <- Qval(fit, method = "PVAF", eps = 0.95, digits = 15)
      lapply(seq_len(nrow(Q)), function(j) list(
        suggested = I(colnames(Q)[q$sug.Q[j, ] == 1]),
        pvaf = q$PVAF[apply(q$sug.Q[j, , drop = FALSE], 1, paste, collapse = ""), j]
      ))
    }
  )
}

nonparametric <- function(data, Q, gate) {
  result <- AlphaNP(data, Q, gate = gate, method = "Hamming")
  # NPCD's profiles in the package's order: GDINA's attributepattern.
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
