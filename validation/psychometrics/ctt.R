# Classical test theory fixtures: item statistics, reliability, distractor analysis and the value of subscores.
# References: psych::alpha (alpha, r.drop, raw.r, alpha if dropped), psych::splitHalf (Guttman split-half per split),
# psych::omega (one factor), checked against base R's minres, and base R for everything else; the subscore package's
# CTTsub (Haberman's PRMSEs), checked against base R. measurekit computes score-based statistics on the persons who
# answered every item, so psych runs on na.omit(x).

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(psych)
  library(mirt)
})

GROUP_FRACTION <- 0.27

# Guttman's split-half coefficient of every split by psych::splitHalf, in combn order, when psych enumerates them.
psych_guttman <- function(x, half) {
  k <- ncol(x)
  m <- floor(k / 2)
  if (choose(k, m) > 15000) {
    return(NA)
  }
  raw <- splitHalf(x, raw = TRUE, brute = TRUE, covar = TRUE, check.keys = FALSE)$raw
  combos <- combn(k, m)
  index <- which(apply(combos, 2, function(a) setequal(a, half) || setequal(a, setdiff(seq_len(k), half))))[1]
  raw[index]
}

split_half <- function(x, half) {
  a <- rowSums(x[, half, drop = FALSE])
  b <- rowSums(x[, -half, drop = FALSE])
  r <- cor(a, b)
  guttman <- 4 * cov(a, b) / var(a + b)
  reference <- psych_guttman(x, half)
  if (!is.na(reference)) stopifnot(isTRUE(all.equal(guttman, reference, tolerance = 1e-12)))
  list(correlation = r, spearmanBrown = 2 * r / (1 + r), guttman = if (is.na(reference)) guttman else reference)
}

# McDonald's omega total of one factor, as psych::omega(nfactors = 1): 1 - Σ (1 - h²) / Σ r, with the communalities
# h² of the minres loadings, which minimize the squared off-diagonal residuals of the correlations. Base R
# minimizes them to convergence (BFGS, then Newton); psych's L-BFGS-B stops near the minimum.
omega_total <- function(x) {
  r <- cor(x)
  objective <- function(l) {
    residual <- r - outer(l, l)
    sum(residual[lower.tri(residual)]^2)
  }
  gradient <- function(l) {
    residual <- r - outer(l, l)
    diag(residual) <- 0
    -2 * drop(residual %*% l)
  }
  hessian <- function(l) {
    h <- -2 * r + 4 * outer(l, l)
    diag(h) <- 2 * (sum(l^2) - l^2)
    h
  }
  l <- optim(rowSums(r) / sqrt(sum(r)), objective, gradient, method = "BFGS",
             control = list(reltol = 1e-16, maxit = 1000))$par
  for (step in 1:10) l <- l - solve(hessian(l), gradient(l))
  stopifnot(max(abs(gradient(l))) < 1e-12)
  l <- l * sign(sum(l))
  omega <- 1 - sum(1 - l^2) / sum(r)
  invisible(capture.output(reference <- suppressMessages(suppressWarnings(psych::omega(x, nfactors = 1, plot = FALSE)))))
  stopifnot(abs(omega - reference$omega.tot) < 1e-4)
  list(omega = omega, loadings = l)
}

ctt_fixture <- function(source, x) {
  x <- as.matrix(x)
  storage.mode(x) <- "integer"
  complete <- na.omit(x)
  k <- ncol(x)
  n <- nrow(complete)
  total <- rowSums(complete)
  upper <- group_weights(total, GROUP_FRACTION)
  lower <- group_weights(-total, GROUP_FRACTION)
  psych_alpha <- alpha(complete, warnings = FALSE)
  dichotomous <- all(x %in% c(0L, 1L, NA))
  alpha_value <- psych_alpha$total$raw_alpha
  population_variance <- var(total) * (n - 1) / n

  items <- lapply(seq_len(k), function(i) {
    answered <- x[!is.na(x[, i]), i]
    item <- complete[, i]
    p <- mean(item)
    item_total <- correlation(item, total)
    list(
      id = colnames(x)[i],
      count = length(answered),
      mean = mean(answered),
      sd = sd(answered),
      itemTotal = psych_alpha$item.stats$raw.r[i],
      itemRest = psych_alpha$item.stats$r.drop[i],
      biserial = if (all(answered %in% c(0L, 1L))) item_total * sqrt(p * (1 - p)) / dnorm(qnorm(p)) else NA,
      discriminationIndex = weighted_share(upper, item) - weighted_share(lower, item),
      alphaIfDeleted = psych_alpha$alpha.drop$raw_alpha[i]
    )
  })
  stopifnot(isTRUE(all.equal(sapply(items, `[[`, "itemTotal"), unname(apply(complete, 2, correlation, total)))))

  odd <- seq(1, k, by = 2)
  kr20 <- if (dichotomous) k / (k - 1) * (1 - sum(colMeans(complete) * (1 - colMeans(complete))) / population_variance)
  if (dichotomous) stopifnot(isTRUE(all.equal(kr20, alpha_value)))
  fixture <- list(
    source = source,
    groupFraction = GROUP_FRACTION,
    itemIds = colnames(x),
    responses = unname(x),
    items = items,
    alpha = alpha_value,
    kr20 = if (dichotomous) kr20 else NA,
    kr21 = if (dichotomous) k / (k - 1) * (1 - mean(total) * (k - mean(total)) / (k * population_variance)) else NA,
    splitHalf = list(
      oddEven = split_half(complete, setdiff(seq_len(k), odd)),
      firstSecond = split_half(complete, seq_len(floor(k / 2)))
    ),
    sem = sd(total) * sqrt(1 - alpha_value),
    omega = if (k >= 3) omega_total(complete)
  )
  fixture
}

distractor_fixture <- function(choices, key, labels) {
  scored <- key2binary(choices, key)
  complete_rows <- complete.cases(scored)
  total <- rowSums(scored[complete_rows, ])
  upper <- group_weights(total, GROUP_FRACTION)
  lower <- group_weights(-total, GROUP_FRACTION)
  lapply(seq_len(ncol(choices)), function(i) {
    column <- choices[, i]
    answered <- sum(!is.na(column))
    list(
      id = colnames(choices)[i],
      key = as.character(key[i]),
      options = lapply(labels, function(label) {
        chosen <- as.integer(!is.na(column[complete_rows]) & column[complete_rows] == label)
        count <- sum(column == label, na.rm = TRUE)
        list(
          label = as.character(label),
          count = count,
          proportion = count / answered,
          upper = weighted_share(upper, chosen),
          lower = weighted_share(lower, chosen),
          pointBiserial = correlation(chosen, total)
        )
      })
    )
  })
}

lsat7 <- expand.table(LSAT7)
write_fixture("ctt-lsat7", ctt_fixture(
  "mirt::LSAT7, expanded: 1000 persons, 5 dichotomous items, complete.",
  lsat7
))

bfi <- psych::bfi[, c("N1", "N2", "N3", "N4", "N5")] - 1
write_fixture("ctt-bfi-neuroticism", ctt_fixture(
  "psych::bfi neuroticism items N1-N5, scored 0-5: 2800 persons, 5 polytomous items, with missing responses.",
  bfi
))

set.seed(20261029)
persons <- 300
item_count <- 16
theta <- rnorm(persons)
a <- runif(item_count, 0.5, 2)
b <- rnorm(item_count)
probability <- plogis(sweep(outer(theta, b, "-"), 2, a, "*"))
simulated <- matrix(rbinom(persons * item_count, 1, probability), persons, item_count)
colnames(simulated) <- sprintf("q%02d", seq_len(item_count))
write_fixture("ctt-simulated-2pl", ctt_fixture(
  "Simulated 2PL responses (set.seed(20261029)): 300 persons, 16 dichotomous items, complete.",
  simulated
))

sat12 <- as.matrix(SAT12)
sat12[sat12 == 8] <- NA
sat12_fixture <- ctt_fixture(
  "mirt::SAT12 scored with its key, 8 as missing: 600 persons, 32 items, with omitted responses.",
  key2binary(sat12, SAT12_KEY)
)
choices <- sat12
storage.mode(choices) <- "character"
sat12_fixture$choices <- unname(choices)
sat12_fixture$key <- as.character(SAT12_KEY)
sat12_fixture$optionLabels <- as.character(1:5)
sat12_fixture$options <- distractor_fixture(sat12, SAT12_KEY, 1:5)
write_fixture("ctt-sat12", sat12_fixture)

# --- Subscore value (Haberman, 2008) ---------------------------------------------------------------------------------
# A section's subscore s is the sum of its items and the total x the sum of the items of all sections. With rho the
# alpha of the section's items, PRMSE(s) = rho and PRMSE(x) = [cov(s, x) - (1 - rho) var(s)]^2 / (rho var(s) var(x)):
# the squared correlation of the true subscore and x. Persons who answered every item of the sections are used.

# The PRMSEs of every section in base R; `sections` lists the item columns of each.
haberman_base <- function(x, sections) {
  x <- na.omit(x[, unlist(sections), drop = FALSE])
  total <- rowSums(x)
  t(sapply(sections, function(columns) {
    s <- rowSums(x[, columns, drop = FALSE])
    k <- length(columns)
    alpha <- k / (k - 1) * (1 - sum(apply(x[, columns, drop = FALSE], 2, var)) / var(s))
    true_covariance <- cov(s, total) - (1 - alpha) * var(s)
    c(alpha = alpha, subscore = alpha, total = true_covariance^2 / (alpha * var(s) * var(total)))
  }))
}

# CTTsub on the same persons: the data as a list of the sections' items and, last, the total test; the items of the
# sections are put side by side, as data.prep takes them.
haberman_ctt <- function(x, sections) {
  x <- as.data.frame(na.omit(x[, unlist(sections), drop = FALSE]))
  test_data <- subscore::data.prep(x, c(length(sections), lengths(sections)), c(names(sections), "total"))
  info <- subscore::CTTsub(test_data)$PRMSE
  info[names(sections), c("Alpha", "PRMSE.s", "PRMSE.x", "added.value.s")]
}

subscore_case <- function(name, x, sections) {
  columns <- lapply(sections, function(ids) match(ids, colnames(x)))
  base <- haberman_base(x, columns)
  reference <- haberman_ctt(x, columns)
  stopifnot(max(abs(base[, "alpha"] - reference[, "Alpha"])) < 1e-12,
            max(abs(base[, "subscore"] - reference[, "PRMSE.s"])) < 1e-12,
            max(abs(base[, "total"] - reference[, "PRMSE.x"])) < 1e-12,
            all((base[, "subscore"] > base[, "total"]) == (reference[, "added.value.s"] == 1)))
  message(name, ": ", nrow(na.omit(x[, unlist(columns)])), " complete cases; PRMSE of the subscores ",
          paste(round(reference[, "PRMSE.s"], 3), collapse = ", "), ", of the total ",
          paste(round(reference[, "PRMSE.x"], 3), collapse = ", "))
  list(name = name, itemIds = colnames(x), responses = unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))),
       sections = sections, completeCases = nrow(na.omit(x[, unlist(columns)])),
       sectionNames = names(sections), subscorePrmse = unname(reference[, "PRMSE.s"]),
       totalPrmse = unname(reference[, "PRMSE.x"]), addedValue = unname(reference[, "added.value.s"] == 1))
}

# subscore's scored.data: 150 persons, 50 items in sections of 15, 15 and 20.
data(scored.data, package = "subscore")
scored <- as.matrix(scored.data)
colnames(scored) <- sprintf("v%02d", 1:50)
scored_sections <- list(Algebra = colnames(scored)[1:15], Geometry = colnames(scored)[16:30],
                        Measurement = colnames(scored)[31:50])

# subscore's TIMSS 2011 grade 8 mathematics: 765 persons; items 1 to 30 in four content domains, items 31 and 32 in none.
data(TIMSS11G8M.data, package = "subscore")
timss <- as.matrix(TIMSS11G8M.data)
colnames(timss) <- sprintf("q%02d", 1:32)
timss_sections <- list(Number = colnames(timss)[1:9], Algebra = colnames(timss)[10:18],
                       Geometry = colnames(timss)[19:24], Data = colnames(timss)[25:30])

# psych's bfi: the five domains of five items on a 0 to 5 scale, reverse-keyed items recoded, with missing responses.
bfi_items <- as.matrix(psych::bfi[, 1:25]) - 1
reversed <- c("A1", "C4", "C5", "E1", "E2", "O2", "O5")
bfi_items[, reversed] <- 5 - bfi_items[, reversed]
bfi_sections <- lapply(list(Agreeableness = "A", Conscientiousness = "C", Extraversion = "E", Neuroticism = "N",
                            Openness = "O"), function(prefix) paste0(prefix, 1:5))

# A simulation: 400 persons answer 8, 10 and 12 items of three correlated abilities (correlation 0.45), 2PL; 1% of the
# responses are missing.
set.seed(20261029)
sizes <- c(8, 10, 12)
abilities <- matrix(rnorm(400 * 3), 400) %*% chol(matrix(0.45, 3, 3) + diag(0.55, 3))
slopes <- runif(sum(sizes), 0.6, 1.8)
difficulties <- rnorm(sum(sizes))
dimension <- rep(1:3, sizes)
probability <- plogis(slopes * (abilities[, dimension] - rep(difficulties, each = 400)))
simulated_sections_data <- matrix(rbinom(length(probability), 1, probability), 400)
simulated_sections_data[matrix(runif(length(probability)) < 0.01, 400)] <- NA
colnames(simulated_sections_data) <- sprintf("s%02d", seq_len(sum(sizes)))
simulated_sections <- list(Reading = colnames(simulated_sections_data)[1:8],
                           Listening = colnames(simulated_sections_data)[9:18],
                           Writing = colnames(simulated_sections_data)[19:30])

write_fixture("ctt-subscores", list(
  source = "subscore's scored.data and TIMSS11G8M.data; psych's bfi by domain; a simulation of three correlated abilities, set.seed(20261029).",
  cases = list(
    subscore_case("scored.data", scored, scored_sections),
    subscore_case("TIMSS11G8M.data", timss, timss_sections),
    subscore_case("bfi", bfi_items, bfi_sections),
    subscore_case("simulated", simulated_sections_data, simulated_sections)
  )
))
