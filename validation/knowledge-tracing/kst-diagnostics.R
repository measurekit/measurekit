# Knowledge space theory, constraints and diagnostics: the basic local independence model with fixed and equal rates
# and by minimum discrepancy maximum likelihood, its fit and comparison, by pks; local identifiability by the rank of
# pks's Jacobian; forward and backward gradedness by pks; the validity of a structure by kst's kvalidate; and the test of
# a quasi-order's fit, whose reference is the delta method written out in base R, as the variance of DAKS's z_test
# differs from it. DAKS's variance with four lines changed equals the delta method (docs/knowledge-tracing/validation.md).

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages({
  library(kst)
  library(pks)
  library(DAKS)
  library(sets)
})

# A structure as measurekit takes it: the domain and each state's elements.
structure_of <- function(K) {
  domain <- colnames(K)
  list(domain = I(domain), states = lapply(seq_len(nrow(K)), function(r) I(domain[K[r, ] == 1])))
}
famset_of <- function(K) as.set(lapply(seq_len(nrow(K)), function(r) as.set(colnames(K)[K[r, ] == 1])))
# The response patterns of a data matrix, as a count by pattern in the order of first appearance.
patterns_of <- function(data) {
  key <- apply(data, 1, paste, collapse = "")
  as.list(table(factor(key, levels = unique(key))))
}
# The by-item vector of a named list of values: NA where an item has none.
by_item <- function(values, items) {
  out <- setNames(rep(NA_real_, length(items)), items)
  if (length(values) > 0) out[names(values)] <- unlist(values)
  out
}
# An empty object, which JSON writes as {} and not as [].
no_rates <- function() setNames(list(), character(0))
# The positions of the items of each group, as pks's betaequal and etaequal take them.
positions_of <- function(groups, items) lapply(groups, function(group) match(group, items))
# Every subset of m elements a, b, c, ...
subsets <- function(m) {
  K <- as.matrix(expand.grid(rep(list(0:1), m)))
  colnames(K) <- letters[seq_len(m)]
  K
}

data(DoignonFalmagne7)
data(chess)
set.seed(20261029)

# BLIMs with fixed rates, equal rates and the three methods, by pks. Each case's rates are written as measurekit takes
# them: fixed rates by item, and groups of items that share one.
blim_case <- function(name, K, N.R, method, fixed = list(), equal = list()) {
  items <- colnames(K)
  fixed_guess <- fixed$guess %||% no_rates()
  fixed_slip <- fixed$slip %||% no_rates()
  equal_guess <- equal$guess %||% list()
  equal_slip <- equal$slip %||% list()
  fit <- blim(K, N.R, method = method, tol = 1e-13, maxiter = 1e6,
              betafix = by_item(fixed_slip, items), etafix = by_item(fixed_guess, items),
              betaequal = if (length(equal_slip) > 0) positions_of(equal_slip, items),
              etaequal = if (length(equal_guess) > 0) positions_of(equal_guess, items))
  gof <- fit$goodness.of.fit
  message(name, " ", method, ": ", fit$iter, " iterations, G2 ", format(gof["G2"]), ", df ", gof["df"])
  list(name = name, method = tolower(method), structure = structure_of(K), patterns = as.list(N.R),
       fixed = list(guess = fixed_guess, slip = fixed_slip),
       equal = list(guess = lapply(equal_guess, I), slip = lapply(equal_slip, I)),
       probabilities = unname(fit$P.K), guess = as.list(fit$eta), slip = as.list(fit$beta),
       logLikelihood = fit$loglik, parameters = fit$npar, g2 = unname(gof["G2"]),
       degreesOfFreedom = unname(gof["df"]), pValue = unname(gof["pval"]), discrepancy = fit$discrepancy, fit = fit)
}
strip <- function(case) {
  case$fit <- NULL
  case
}

df7 <- DoignonFalmagne7
chess_items <- colnames(chess$dst1)
fixed_zero <- list(guess = list(a = 0, b = 0))
equal_groups <- list(guess = list(c("c", "d", "e")), slip = list(c("a", "b"), c("c", "d", "e")))
equal_rates <- list(guess = list(chess_items), slip = list(chess_items))
free_df7 <- blim_case("free", df7$K, df7$N.R, "ML")
free_chess <- blim_case("free", chess$dst1, chess$N.R, "ML")
equal_df7 <- blim_case("equal", df7$K, df7$N.R, "ML", equal = equal_groups)
equal_chess <- blim_case("equal", chess$dst1, chess$N.R, "ML", equal = equal_rates)
cases <- list(
  blim_case("fixed", df7$K, df7$N.R, "ML", fixed = fixed_zero),
  blim_case("fixed", df7$K, df7$N.R, "MD", fixed = fixed_zero),
  blim_case("fixed", df7$K, df7$N.R, "MDML", fixed = fixed_zero),
  blim_case("fixed at values", df7$K, df7$N.R, "ML", fixed = list(guess = list(a = 0), slip = list(e = 0.05))),
  equal_df7,
  blim_case("equal", df7$K, df7$N.R, "MD", equal = equal_groups),
  blim_case("equal", df7$K, df7$N.R, "MDML", equal = equal_groups),
  blim_case("fixed and equal", df7$K, df7$N.R, "ML", fixed = list(slip = list(e = 0.1)),
            equal = list(guess = list(c("c", "d")))),
  blim_case("free", df7$K, df7$N.R, "MDML"),
  blim_case("free", chess$dst1, chess$N.R, "MDML"),
  equal_chess,
  blim_case("equal in groups", chess$dst1, chess$N.R, "MDML",
            equal = list(slip = list(chess_items[1:8], chess_items[9:16])))
)

# Comparing BLIMs: pks's anova on a restricted model and the model it is nested in, fitted to the same data.
compare <- function(name, restricted, general) {
  table <- anova(restricted$fit, general$fit)
  list(name = name, restricted = strip(restricted), general = strip(general),
       g2 = table[2, "Deviance"], degreesOfFreedom = table[2, "Df"], pValue = table[2, "Pr(>Chi)"])
}
comparisons <- list(
  compare("equal rates against free rates", equal_df7, free_df7),
  compare("one slip and one guess against free rates", equal_chess, free_chess)
)

# Recovery: 20 seeded sets from a chain of 6 items, 400 learners, slip 0.1 and guess 0.05, with one slip and one guess
# estimated by pks, as measurekit does with equal rates.
chain <- outer(0:6, 1:6, ">=") * 1L
colnames(chain) <- paste0("q", 1:6)
recovery <- lapply(1:20, function(k) {
  position <- sample(0:6, 400, replace = TRUE)
  known <- outer(position, 1:6, ">=")
  data <- matrix(as.integer(ifelse(known, runif(400 * 6) > 0.1, runif(400 * 6) < 0.05)), 400, 6,
                 dimnames = list(NULL, colnames(chain)))
  N.R <- table(apply(data, 1, paste, collapse = ""))
  N.R <- setNames(as.integer(N.R), names(N.R))
  fit <- blim(chain, N.R, method = "ML", tol = 1e-13, maxiter = 1e6, betaequal = list(1:6), etaequal = list(1:6))
  list(patterns = as.list(N.R), slip = unname(fit$beta[1]), guess = unname(fit$eta[1]),
       probabilities = unname(fit$P.K), logLikelihood = fit$loglik)
})
message("recovery: mean slip ", mean(sapply(recovery, `[[`, "slip")), ", mean guess ",
        mean(sapply(recovery, `[[`, "guess")))
write_fixture("kst-blim-constraints", list(
  cases = lapply(cases, strip), comparisons = comparisons,
  recovery = list(structure = structure_of(chain), slip = 0.1, guess = 0.05, sets = recovery)
))

# Local identifiability: the rank of pks's Jacobian at the parameters, without the columns of fixed rates, at qr's
# tolerance of 1e-7. The parameters are generic draws, so the rank is the structure's, not that of a boundary.
identifiability_case <- function(name, K, fixed = list()) {
  items <- colnames(K)
  cut <- c(0, sort(runif(nrow(K) - 1)), 1)
  probabilities <- diff(cut)
  slip <- setNames(runif(length(items), 0.05, 0.25), items)
  guess <- setNames(runif(length(items), 0.05, 0.25), items)
  betafix <- by_item(fixed$slip %||% list(), items)
  etafix <- by_item(fixed$guess %||% list(), items)
  slip[!is.na(betafix)] <- betafix[!is.na(betafix)]
  guess[!is.na(etafix)] <- etafix[!is.na(etafix)]
  J <- jacobian(list(K = K), P.K = probabilities, beta = slip, eta = guess, betafix = betafix, etafix = etafix)
  free <- c(rep(TRUE, nrow(K) - 1), is.na(betafix), is.na(etafix))
  message("identifiability ", name, ": ", sum(free), " parameters, rank ", qr(J[, free, drop = FALSE])$rank)
  list(name = name, structure = structure_of(K), probabilities = probabilities,
       guess = as.list(guess), slip = as.list(slip),
       fixed = list(guess = as.list(etafix[!is.na(etafix)]), slip = as.list(betafix[!is.na(betafix)])),
       parameters = sum(free), rank = qr(J[, free, drop = FALSE])$rank)
}
two_chains <- local({
  steps <- as.matrix(expand.grid(0:2, 0:2))
  K <- t(apply(steps, 1, function(g) c(1L * (1:2 <= g[1]), 1L * (1:2 <= g[2]))))
  colnames(K) <- c("a", "b", "c", "d")
  K
})
write_fixture("kst-identifiability", list(
  identifiability_case("DoignonFalmagne7", df7$K),
  identifiability_case("DoignonFalmagne7 with two guesses fixed", df7$K, fixed = list(guess = list(a = 0, b = 0))),
  identifiability_case("a chain of 6", chain),
  identifiability_case("a chain of 6 with its end rates fixed", chain,
                       fixed = list(guess = list(q1 = 0.1), slip = list(q6 = 0.1))),
  identifiability_case("two chains of 2", two_chains),
  identifiability_case("all subsets of 3", subsets(3)),
  identifiability_case("chess, the first structure", chess$dst1)
))

# Gradedness: pks's is.forward.graded and is.backward.graded, on the example structures and on random families of
# states that hold the empty state and the whole domain.
random_family <- function(m, size) {
  K <- unique(rbind(matrix(0L, 1, m), matrix(1L, 1, m), matrix(rbinom(size * m, 1, 0.5), size, m)))
  colnames(K) <- letters[seq_len(m)]
  K
}
graded_structures <- c(
  list(doignonFalmagne = df7$K, chessFirst = chess$dst1, chessThird = chess$dst3, chessFourth = chess$dst4),
  lapply(setNames(1:12, paste0("random", 1:12)), function(k) random_family(2 + k %% 5, 3 + k))
)
write_fixture("kst-graded", lapply(names(graded_structures), function(name) {
  K <- graded_structures[[name]]
  list(name = name, structure = structure_of(K), forward = as.list(is.forward.graded(K)),
       backward = as.list(is.backward.graded(K)))
}))

# Validity: kst's kvalidate for gamma, percent, VC and DI, and for DA where the sets package can list the 2^m patterns. The
# distance agreement is also written out in base R, checked against kvalidate where it runs and used where it does not.
distances_to <- function(patterns, K) apply(patterns, 1, function(r) min(rowSums(abs(sweep(K, 2, r)))))
validity_case <- function(name, K, rp) {
  ks <- kstructure(famset_of(K))
  gamma <- kvalidate(ks, rp, "gamma")
  distance <- kvalidate(ks, rp, "DI")
  ratio <- mean(distances_to(rp, K)) / mean(distances_to(subsets(ncol(K)), K))
  if (ncol(K) <= 8) stopifnot(isTRUE(all.equal(ratio, kvalidate(ks, rp, "DA")$DA, tolerance = 1e-12)))
  list(name = name, structure = structure_of(K), patterns = patterns_of(rp),
       gamma = if (is.nan(gamma$gamma)) NA else gamma$gamma, concordant = gamma$nc, discordant = gamma$nd,
       violationalCoefficient = kvalidate(ks, rp, "VC")$vc, discrepancyIndex = distance$di,
       distances = as.list(table(distances_to(rp, K))), distanceAgreement = ratio,
       solutionRates = setNames(as.list(kvalidate(ks, rp, "percent")[colnames(rp), "%"] / 100), colnames(rp)))
}
df7_rp <- as.matrix(as.binmat(df7$N.R))
colnames(df7_rp) <- colnames(df7$K)
simulated <- simu(8, 200, 0.05, 0.05, delta = 0.2)
simulated_data <- as.matrix(simulated$dataset)
colnames(simulated_data) <- paste0("i", 1:8)
simulated_states <- imp2state(simulated$implications, 8)
colnames(simulated_states) <- colnames(simulated_data)
reverse_chain <- outer(0:8, 1:8, ">=") * 1L
colnames(reverse_chain) <- colnames(simulated_data)
subset_data <- simulated_data[, 1:4]
colnames(subset_data) <- letters[1:4]
write_fixture("kst-validity", list(
  validity_case("DoignonFalmagne7", df7$K, df7_rp),
  validity_case("chess, the first structure", chess$dst1, chess$R),
  validity_case("chess, the third structure", chess$dst3, chess$R),
  validity_case("simulated from a quasi order", simulated_states, simulated_data),
  validity_case("a chain that the data do not follow", reverse_chain, simulated_data),
  validity_case("all subsets", subsets(4), subset_data)
))

# The test of a quasi-order's fit. The estimate is diff / n^2, the value DAKS reports. The variance is the delta method for
# the multinomial distribution of the response patterns, by central differences of the estimate over the probabilities of
# the patterns; DAKS's own variance is stored beside it.
estimate_of <- function(p, patterns, relation, version) {
  m <- ncol(patterns)
  rho <- colSums(patterns * p)
  C <- matrix(0, m, m)
  for (i in 1:m) for (j in 1:m) if (i != j) C[i, j] <- sum(p[patterns[, i] == 0 & patterns[, j] == 1])
  pairs <- which(relation & !diag(m), arr.ind = TRUE)
  if (version == 1) {
    x <- rep(0, 4)
    for (i in 1:m) for (j in 1:m) if (i != j) {
      if (relation[i, j]) {
        x[2] <- x[2] - 2 * C[i, j] * rho[j]
        x[4] <- x[4] + 2 * rho[j]^2
      } else if (relation[j, i]) {
        x[1] <- x[1] - 2 * C[i, j] * rho[i] + 2 * rho[i] * rho[j] - 2 * rho[i]^2
        x[3] <- x[3] + 2 * rho[i]^2
      }
    }
    gamma <- -(x[1] + x[2]) / (x[3] + x[4])
  } else {
    gamma <- mean(C[pairs] / rho[pairs[, 2]])
  }
  total <- 0
  for (i in 1:m) for (j in 1:m) if (i != j) {
    expected <- if (relation[i, j]) gamma * rho[j]
                else if (relation[j, i]) rho[j] - rho[i] + gamma * rho[i]
                else (1 - rho[i]) * rho[j]
    total <- total + (C[i, j] - expected)^2
  }
  total / (m * (m - 1))
}
relation_of <- function(imp, m) {
  relation <- diag(m) == 1
  for (pair in lapply(as.list(imp), function(t) unlist(as.list(t)))) relation[pair[1], pair[2]] <- TRUE
  relation
}
delta_variance <- function(data, relations, version) {
  patterns <- as.matrix(expand.grid(rep(list(0:1), ncol(data))))
  key <- apply(patterns, 1, paste, collapse = "")
  p0 <- as.numeric(table(factor(apply(data, 1, paste, collapse = ""), levels = key))) / nrow(data)
  value <- function(p) {
    estimates <- sapply(relations, function(relation) estimate_of(p, patterns, relation, version))
    estimates[1] - (if (length(estimates) > 1) estimates[2] else 0)
  }
  step <- 1e-6
  gradient <- sapply(seq_along(p0), function(t) {
    up <- down <- p0
    up[t] <- up[t] + step
    down[t] <- down[t] - step
    (value(up) - value(down)) / (2 * step)
  })
  list(estimate = value(p0), variance = sum(p0 * gradient^2) - sum(p0 * gradient)^2)
}
# DAKS 2.1.3's variance with the four lines changed that the comparison with the delta method found: tmp1 and tmp2 are set for
# each pattern; the corrected error rate divides by the number of learners; the gradient of a pair related in neither direction
# has the patterns that solve the first item and fail the second; and the derivative of the corrected error rate is subtracted.
corrected_daks_variance <- local({
  text <- paste(deparse(DAKS::variance), collapse = "\n")
  edits <- list(
    c("tmp1 <- 0\n        tmp2 <- 0\n        for \\(i in 2:2\\^items\\) \\{",
      "for (i in 2:2^items) {\n            tmp1 <- 0\n            tmp2 <- 0"),
    c("\\* ncol\\(dataset\\)/sum", "* nrow(dataset)/sum"),
    c("else \\{\n                    if \\(patterns\\[i, h\\] == 1\\) \\{\n                      if \\(patterns\\[i, k\\] == 1\\) \\{",
      paste0("else {\n                    if (patterns[i, k] == 1 && patterns[i, h] == 0) {\n",
             "                      grad[i - 1] <- grad[i - 1] + 2 * (rho_sum_counter[k, h] - (1 - rho_sum[k]) * rho_sum[h]) * rho_sum[h]\n",
             "                    }\n                    if (patterns[i, h] == 1) {\n                      if (patterns[i, k] == 1) {")),
    c("if \\(patterns\\[i, as.integer\\(j\\[2\\]\\)\\] == 1\\) \n                    gamma_deriv\\[i - 1\\] <- gamma_deriv\\[i - 1\\] \\+ \n                      rho_sum_counter",
      "if (patterns[i, as.integer(j[2])] == 1) \n                    gamma_deriv[i - 1] <- gamma_deriv[i - 1] - \n                      rho_sum_counter")
  )
  for (edit in edits) {
    changed <- sub(edit[1], edit[2], text)
    if (identical(changed, text)) stop("DAKS::variance is not the version of DAKS 2.1.3 that the four edits were found in.")
    text <- changed
  }
  fixed <- eval(parse(text = text))
  environment(fixed) <- asNamespace("DAKS")
  fixed
})
# The statistic, the p-value of each alternative and the end of each interval that is not infinite.
tail_values <- function(estimate, variance, n, mu, level) {
  se <- sqrt(variance / n)
  z <- (estimate - mu) / se
  list(z = z,
       pValue = list(`two-sided` = 2 * min(pnorm(z), 1 - pnorm(z)), less = pnorm(z), greater = 1 - pnorm(z)),
       interval = list(`two-sided` = estimate + c(-1, 1) * qnorm(1 - (1 - level) / 2) * se,
                       less = estimate + qnorm(level) * se, greater = estimate - qnorm(level) * se))
}
prerequisites_of <- function(imp, items) {
  pairs <- lapply(as.list(imp), function(t) unlist(as.list(t)))
  setNames(lapply(seq_along(items), function(j) {
    I(items[sort(unlist(lapply(pairs, function(p) if (p[2] == j) p[1])))])
  }), items)
}
data(pisa)
diff_datasets <- list(list(name = "pisa", data = as.matrix(pisa)))
for (k in 1:3) {
  diff_datasets[[k + 1]] <- list(name = paste0("simulated", k), data = as.matrix(simu(7, 300, 0.05, 0.05, delta = 0.2)$dataset))
}
diff_cases <- list()
for (d in diff_datasets) {
  data <- d$data
  items <- paste0("i", seq_len(ncol(data)))
  colnames(data) <- items
  selection <- ind_gen(ob_counter(data))
  for (version in 1:2) {
    chosen <- iita(data, v = version)$implications
    other <- Filter(function(candidate) !set_is_equal(candidate, chosen), selection)[[1]]
    for (alternative in list(NULL, other)) {
      relations <- c(list(relation_of(chosen, ncol(data))),
                     if (!is.null(alternative)) list(relation_of(alternative, ncol(data))))
      delta <- delta_variance(data, relations, version)
      daks <- z_test(data, chosen, imp_alt = alternative, v = version)
      daks_estimate <- if (is.null(alternative)) daks$diff_value else daks$diff_value[1] - daks$diff_value[2]
      if (abs(delta$estimate - daks_estimate) > 1e-12 * abs(delta$estimate)) {
        stop(d$name, " version ", version, ": base R ", delta$estimate, ", DAKS ", daks_estimate)
      }
      daks_variance <- sum(sapply(c(list(chosen), if (!is.null(alternative)) list(alternative)),
                                  function(imp) DAKS::variance(data, imp, version)))
      if (is.null(alternative)) {
        corrected <- corrected_daks_variance(data, chosen, version)
        if (abs(corrected - delta$variance) > 1e-6 * delta$variance) {
          stop(d$name, " version ", version, ": DAKS with four lines changed ", corrected, ", delta method ", delta$variance)
        }
        message("  corrected DAKS variance differs from the delta method by ", format(abs(corrected - delta$variance) / delta$variance))
      }
      message(d$name, " v", version, if (!is.null(alternative)) " two quasi orders", ": delta variance ",
              format(delta$variance), ", DAKS ", format(daks_variance))
      for (mu in c(0, 0.001)) {
        diff_cases[[length(diff_cases) + 1]] <- c(
          list(name = d$name, items = I(items), patterns = patterns_of(data),
               version = c("minimized", "corrected")[version], prerequisites = prerequisites_of(chosen, items),
               alternative = if (!is.null(alternative)) prerequisites_of(alternative, items),
               mean = mu, confidence = 0.9, estimate = delta$estimate, variance = delta$variance,
               daksVariance = daks_variance),
          tail_values(delta$estimate, delta$variance, nrow(data), mu, 0.9))
      }
    }
  }
}
write_fixture("kst-diff-test", diff_cases)
