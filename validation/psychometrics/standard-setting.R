# Standard-setting fixtures. No package among the references implements the modified Angoff, Bookmark or Hofstee
# methods, so the script writes each out in base R: the Bookmark's locations by uniroot on mirt's probtrace, and its
# raw cut scores by mirt's expected.test; the Hofstee crossing by uniroot on the interpolated failure rates.
# Classification accuracy and consistency: cacIRT's class.Rud.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(cacIRT)
})

set.seed(20261029)

# Modified Angoff: 8 panelists rate 30 dichotomous items.
ratings <- matrix(round(runif(8 * 30, 0.3, 0.95), 2), nrow = 8)
panelist_cuts <- rowSums(ratings)
angoff <- list(
  ratings = ratings,
  cutScore = mean(panelist_cuts),
  panelistCutScores = panelist_cuts,
  sd = sd(panelist_cuts),
  standardError = sd(panelist_cuts) / sqrt(length(panelist_cuts))
)

# Bookmark: logistic, graded and gpcm items set in mirt; the data only shape the model.
data <- cbind(expand.table(LSAT7)[seq_len(nrow(Science)), ], Science - 1)
itemtype <- c("2PL", "2PL", "3PL", "4PL", "2PL", "graded", "graded", "gpcm", "gpcm")
models <- c("2pl", "2pl", "3pl", "4pl", "2pl", "grm", "grm", "gpcm", "gpcm")
fit <- set_items(data, itemtype, data.frame(
  a1 = c(1.2, 0.8, 1.7, 2.1, 0.6, 1.1, 0.7, 0.9, 0.8),
  d = c(0.5, -1.1, 2.3, 0, 0.9, NA, NA, NA, NA),
  g = c(NA, NA, 0.2, 0.15, NA, NA, NA, NA, NA),
  u = c(NA, NA, NA, 0.9, NA, NA, NA, NA, NA),
  d1 = c(NA, NA, NA, NA, NA, 4.8, 2.9, 2.8, 1.7),
  d2 = c(NA, NA, NA, NA, NA, 2.6, 0.9, 5.3, 2.6),
  d3 = c(NA, NA, NA, NA, NA, -1.5, -2.3, 4, 0.8)
))
irt <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
ids <- colnames(data)

# The theta where an item's probability of reaching category k or above is rp.
location <- function(i, k, rp) {
  item <- extract.item(fit, i)
  reaching <- function(theta) sum(probtrace(item, matrix(theta))[1, -seq_len(k)]) - rp
  uniroot(reaching, c(-15, 15), tol = 1e-14)$root
}

booklet <- function(rp) {
  pages <- do.call(rbind, lapply(seq_along(ids), function(i) {
    categories <- seq_len(extract.item(fit, i)@ncat - 1)
    data.frame(id = ids[i], category = categories, location = sapply(categories, location, i = i, rp = rp))
  }))
  pages <- pages[order(pages$location), ]
  rownames(pages) <- NULL
  list(
    responseProbability = rp,
    pages = pages,
    cutScores = sapply(pages$location, function(theta) expected.test(fit, matrix(theta)))
  )
}

bookmark <- list(
  items = lapply(seq_along(ids), function(i) {
    list(id = ids[i], model = models[i], parameters = psychometrics_parameters(models[i], irt[i, !is.na(irt[i, ])]))
  }),
  booklets = list(booklet(0.67), booklet(0.5))
)

# Hofstee: the total scores of SAT12, omissions wrong, with the bounds of the panelists' means and with whole scores.
totals <- rowSums(key2binary(SAT12, SAT12_KEY))
distribution <- tabulate(totals + 1, nbins = ncol(SAT12) + 1)
hofstee <- function(bounds) {
  scores <- seq_along(distribution)
  failure <- approxfun(scores - 1, c(0, cumsum(distribution))[scores] / sum(distribution))
  line <- function(c) {
    with(bounds, maximumFailureRate - (maximumFailureRate - minimumFailureRate) *
      (c - minimumCutScore) / (maximumCutScore - minimumCutScore))
  }
  cut <- uniroot(function(c) failure(c) - line(c), c(bounds$minimumCutScore, bounds$maximumCutScore), tol = 1e-14)$root
  list(bounds = bounds, cutScore = cut, failureRate = line(cut))
}
panel <- data.frame(
  minimumCutScore = round(runif(8, 12, 17)),
  maximumCutScore = round(runif(8, 20, 25)),
  minimumFailureRate = round(runif(8, 0.05, 0.2), 2),
  maximumFailureRate = round(runif(8, 0.3, 0.5), 2)
)

# Rudner's classification accuracy and consistency: cacIRT's class.Rud with the EAP estimates and standard errors of
# SAT12's persons under the 2PL; the probabilities of the categories by base R's pnorm.
sat12_scored <- key2binary(SAT12, SAT12_KEY)
sat12_fit <- mirt(sat12_scored, 1, itemtype = "2PL", verbose = FALSE)
sat12_eap <- fscores(sat12_fit, full.scores.SE = TRUE)
classification_case <- function(cuts) {
  reference <- class.Rud(cuts, ability = sat12_eap[, 1], se = sat12_eap[, 2])
  bounds <- c(-Inf, cuts, Inf)
  probabilities <- sapply(seq_len(length(cuts) + 1), function(k) {
    pnorm(bounds[k + 1], sat12_eap[, 1], sat12_eap[, 2]) - pnorm(bounds[k], sat12_eap[, 1], sat12_eap[, 2])
  })
  marginal <- reference$Marginal
  values <- if (length(cuts) == 1) marginal[1, ] else marginal["Simultaneous", ]
  list(cuts = I(cuts), accuracy = unname(values[["Accuracy"]]), consistency = unname(values[["Consistency"]]),
       category = findInterval(sat12_eap[, 1], cuts), probabilities = unname(probabilities))
}

write_fixture("standard-setting", list(
  source = paste(
    "Simulated Angoff ratings (set.seed(20261029)): 8 panelists, 30 items. Bookmark: LSAT7 and Science items set in",
    "mirt. Hofstee: SAT12 total scores, omissions wrong, with simulated panel bounds. Classification: SAT12's EAP",
    "estimates under the 2PL."
  ),
  angoff = angoff,
  bookmark = bookmark,
  classification = list(
    estimates = lapply(seq_len(nrow(sat12_eap)), function(i) list(theta = sat12_eap[i, 1], standardError = sat12_eap[i, 2])),
    cases = list(classification_case(0.5), classification_case(c(-1, 0.3, 1.2)))
  ),
  hofstee = list(
    distribution = distribution,
    cases = list(
      hofstee(as.list(colMeans(panel))),
      hofstee(list(minimumCutScore = 16, maximumCutScore = 24, minimumFailureRate = 0.1, maximumFailureRate = 0.4))
    )
  )
))
