# Interoperability fixtures: mirt's coefficients of the same items in the IRT form, coef(IRTpars = TRUE), and the
# slope-intercept form, coef(IRTpars = FALSE), for every item type measurekit converts, with the category probabilities
# from probtrace. The items are set, not estimated.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages(library(mirt))

THETA <- c(-2, 0, 1.5)

items_of <- function(fit, itemtype, model) {
  irt <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
  slope_intercept <- coef(fit, simplify = TRUE)$items
  lapply(seq_len(nrow(irt)), function(i) {
    row <- function(table) as.list(table[i, !is.na(table[i, ])])
    list(itemtype = itemtype, model = model, irt = row(irt), slopeIntercept = row(slope_intercept),
         parameters = psychometrics_parameters(model, irt[i, !is.na(irt[i, ])]),
         probabilities = unname(probtrace(extract.item(fit, i), THETA)))
  })
}

lsat7 <- expand.table(LSAT7)
science <- mirt::Science - 1
logistic <- set_items(lsat7, c("2PL", "2PL", "3PL", "4PL", "2PL"), data.frame(
  a1 = c(1.2, 0.8, 1.7, 2.1, -0.6),
  d = c(0.5, -1.1, 2.3, 0, 0.9),
  g = c(0, 0, 0.2, 0.15, 0),
  u = c(1, 1, 1, 0.9, 1)
))
graded <- set_items(science, "graded", data.frame(
  a1 = c(1.1, 0.7, 2.2, -0.9),
  d1 = c(4.8, 2.9, 5.2, 3.4),
  d2 = c(2.6, 0.9, 2.2, 1),
  d3 = c(-1.5, -2.3, -2, -1.6)
))
gpcm <- set_items(science, "gpcm", data.frame(
  a1 = c(0.9, 0.8, 2.2, 0.7),
  d1 = c(2.8, 1.7, 4.6, 2.1),
  d2 = c(5.3, 2.6, 6.8, 2.9),
  d3 = c(4, 0.8, 4.9, 1.7)
))
rasch <- set_items(science, "Rasch", data.frame(
  d1 = c(3.1, 1.9, 2.6, 2.4),
  d2 = c(5.7, 2.8, 4.1, 3.3),
  d3 = c(4.3, 0.9, 2.9, 2)
))
rsm <- set_items(science, "rsm", data.frame(
  a1 = c(1.3, 1.3, 1.3, 1.3),
  b1 = c(-3, -3, -3, -3),
  b2 = c(-1.9, -1.9, -1.9, -1.9),
  b3 = c(0.9, 0.9, 0.9, 0.9),
  c = c(0, -1, -0.4, 0.7)
))
nominal <- set_items(science, "nominal", data.frame(
  a1 = c(1, 0.8, 2, 0.8),
  ak1 = c(1.5, 0.7, 0.8, 1),
  ak2 = c(2, 1.5, 1.9, 1.7),
  d1 = c(3.6, 1.5, 3.7, 2.1),
  d2 = c(5.9, 2.3, 5.9, 2.9),
  d3 = c(4.5, 0.3, 3.9, 1.6)
))

write_fixture("interop-mirt", list(
  source = "LSAT7 and Science items set to given slope-intercept coefficients in mirt, one model per item type.",
  theta = THETA,
  items = c(
    items_of(logistic, "2PL", "2pl")[c(1, 2, 5)],
    items_of(logistic, "3PL", "3pl")[3],
    items_of(logistic, "4PL", "4pl")[4],
    items_of(graded, "graded", "grm"),
    items_of(gpcm, "gpcm", "gpcm"),
    items_of(rasch, "Rasch", "pcm"),
    items_of(rsm, "rsm", "rsm"),
    items_of(nominal, "nominal", "nrm")
  )
))
