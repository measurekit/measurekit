# Cut-score fixtures (standard-setting-cut-scores.json). No package implements the contrasting-groups, Body of Work or
# Beuk methods, so the script writes each out in base R: per boundary between adjacent categories, glm's logistic
# regression of "at or above" on the score and the score where it reaches the probability; the Beuk crossing by uniroot
# on the interpolated failure rates; and the recipes of standard-setting.md: Nedelsky and Ebel.

source("validation/psychometrics/common.R")

GLM_CONTROL <- glm.control(epsilon = 1e-14, maxit = 100)
set.seed(20261102)

# --- Contrasting groups and Body of Work ----------------------------------------------------------------------------

# The score at which a logistic regression of category >= k on the score reaches `probability`, for each boundary k;
# NA where the groups separate, as no maximum likelihood estimate exists there.
boundary_cuts <- function(scores, categories, probability = 0.5) {
  sapply(seq_len(max(categories)), function(k) {
    above <- as.numeric(categories >= k)
    if (max(scores[above == 0]) <= min(scores[above == 1])) return(NA)
    fit <- glm(above ~ scores, family = binomial, control = GLM_CONTROL)
    (qlogis(probability) - coef(fit)[[1]]) / coef(fit)[[2]]
  })
}

contrasting_case <- function(name, scores, categories, probability = NULL) {
  options <- if (is.null(probability)) setNames(list(), character(0)) else list(probability = probability)
  cuts <- boundary_cuts(scores, categories, if (is.null(probability)) 0.5 else probability)
  list(name = name, scores = scores, categories = categories, options = options, cutScores = I(cuts),
       converged = !anyNA(cuts))
}

# Two groups of a 40-item test, the masters' scores higher; four Body of Work categories from a latent cut on noisy
# judgments of 120 collections of work; a boundary whose groups separate.
persons <- 400
scores <- rbinom(persons, 40, plogis(rnorm(persons, 0.2, 1)))
masters <- rbinom(persons, 1, plogis((scores - 23) / 2.5))
work <- round(runif(120, 0, 60), 1)
judged <- findInterval(work + rnorm(120, 0, 4), c(18, 31, 44))
separated_scores <- c(3, 5, 8, 9, 12, 14, 15, 17, 20, 21, 22, 25)
separated <- c(0, 0, 0, 0, 1, 1, 1, 1, 2, 1, 2, 2)

contrasting <- list(
  contrasting_case("two groups", scores, masters),
  contrasting_case("two groups, probability 0.67", scores, masters, 0.67),
  contrasting_case("Body of Work, four categories", work, judged),
  contrasting_case("a separated boundary", separated_scores, separated)
)
names(contrasting) <- sapply(contrasting, `[[`, "name")

# --- Beuk -----------------------------------------------------------------------------------------------------------

# The failure rate F(c), the share below c, interpolated between the scores; and the crossing of the line through the
# panel's mean cut score and failure rate with slope -(passRateSd / cutScoreSd) in failure rate.
failure_rate <- function(distribution) {
  shares <- c(0, cumsum(distribution)) / sum(distribution)
  approxfun(seq_along(distribution) - 1, shares[seq_along(distribution)])
}
beuk <- function(distribution, judgments) {
  failure <- failure_rate(distribution)
  line <- function(c) (1 - judgments$passRate) - judgments$passRateSd / judgments$cutScoreSd * (c - judgments$cutScore)
  cut <- uniroot(function(c) failure(c) - line(c), c(0, length(distribution) - 1), tol = 1e-14)$root
  list(cutScore = cut, failureRate = failure(cut))
}
beuk_distribution <- as.vector(table(factor(scores, levels = 0:40)))
beuk_cases <- lapply(list(
  list(cutScore = 24, passRate = 0.55, cutScoreSd = 3, passRateSd = 0.12),
  list(cutScore = 20.5, passRate = 0.7, cutScoreSd = 4.2, passRateSd = 0.05),
  list(cutScore = 27, passRate = 0.4, cutScoreSd = 2, passRateSd = 0.2)
), function(judgments) list(judgments = judgments, expected = beuk(beuk_distribution, judgments)))

# --- Recipes --------------------------------------------------------------------------------------------------------

# Nedelsky: each of 5 panelists names the options of each of 12 four-option items that a borderline person would rule
# out; the rating is 1 over the options left.
options_left <- matrix(sample(1:4, 5 * 12, replace = TRUE, prob = c(0.2, 0.35, 0.3, 0.15)), nrow = 5)
nedelsky <- rowSums(1 / options_left)
# Ebel: items in cells of relevance by difficulty, each cell with the panel's proportion a borderline person answers.
ebel_items <- c(6, 9, 4, 5, 8, 3, 2, 4, 3)
ebel_proportions <- c(0.9, 0.75, 0.55, 0.8, 0.6, 0.4, 0.6, 0.45, 0.25)

write_fixture("standard-setting-cut-scores", list(
  source = paste("Base R, set.seed(20261102): glm(binomial) per boundary to a deviance change of 1e-14; the Beuk",
                 "crossing by uniroot; the recipes of Nedelsky and Ebel."),
  contrasting = contrasting,
  beuk = list(distribution = beuk_distribution, cases = beuk_cases),
  recipes = list(
    nedelsky = list(optionsLeft = unname(lapply(seq_len(nrow(options_left)), function(i) options_left[i, ])),
                    cutScore = mean(nedelsky)),
    ebel = list(items = ebel_items, proportions = ebel_proportions, cutScore = sum(ebel_items * ebel_proportions))
  )
))
