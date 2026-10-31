# Interoperability fixtures: mirt's coefficients of the same items in the IRT form, coef(IRTpars = TRUE), and the
# slope-intercept form, coef(IRTpars = FALSE), for every item type measurekit converts, with the category probabilities
# from probtrace. The items are set, not estimated.

source("validation/psychometrics/common.R")
source("validation/psychometrics/multidimensional-items.R")
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

# --- Multidimensional items ----------------------------------------------------------------------------------------------
# mirt's slope-intercept coefficients, coef(simplify = TRUE), of items on 2 and 3 dimensions with the category
# probabilities from probtrace at vectors of theta. The IRT form exists for one dimension only.

multidimensional_thetas <- list(
  `2` = rbind(c(-2, 0.5), c(0, 0), c(1.5, -1.2)),
  `3` = rbind(c(-2, 0.5, 1), c(0, 0, 0), c(1.5, -1.2, -0.3))
)

multidimensional_items_of <- function(fit, itemtype, model, dimensions) {
  slope_intercept <- coef(fit, simplify = TRUE)$items
  thetas <- multidimensional_thetas[[as.character(dimensions)]]
  lapply(seq_len(nrow(slope_intercept)), function(i) {
    row <- as.list(slope_intercept[i, !is.na(slope_intercept[i, ])])
    a <- unname(unlist(row[paste0("a", seq_len(dimensions))]))
    parameters <- if (model == "mgrm") {
      list(a = I(a), intercepts = I(unname(unlist(row[grep("^d[0-9]+$", names(row))]))))
    } else {
      list(a = I(a), intercept = row$d, c = row$g, d = row$u)
    }
    list(itemtype = itemtype, model = model, dimensions = dimensions, slopeIntercept = row, parameters = parameters,
         theta = lapply(seq_len(nrow(thetas)), function(j) I(thetas[j, ])),
         probabilities = unname(probtrace(extract.item(fit, i), thetas)))
  })
}

dichotomous_data <- function(count) {
  data <- as.data.frame(sapply(seq_len(count), function(i) rep(0:1, length.out = 16)))
  colnames(data) <- sprintf("q%d", seq_len(count))
  data
}

graded_data <- function(count, categories) {
  data <- as.data.frame(sapply(seq_len(count), function(i) rep(0:(categories - 1), length.out = 24)))
  colnames(data) <- sprintf("q%d", seq_len(count))
  data
}

m2pl_2d <- set_multidimensional_items(dichotomous_data(6), 2, "2PL", data.frame(
  a1 = c(1.2, 0.5, -0.8, 0.9, 0, 1.4), a2 = c(0.7, 0, 1.5, -0.6, 1.1, 0.2),
  d = c(0.3, -0.4, 1, -1.1, 0.6, 0), g = 0, u = 1
))
m3pl_2d <- set_multidimensional_items(dichotomous_data(6), 2, "3PL", data.frame(
  a1 = c(1.2, 0.5, -0.8, 0.9, 0, 1.4), a2 = c(0.7, 0, 1.5, -0.6, 1.1, 0.2),
  d = c(0.3, -0.4, 1, -1.1, 0.6, 0), g = c(0.2, 0.1, 0.25, 0.15, 0.05, 0.3), u = 1
))
graded_2d <- set_multidimensional_items(graded_data(4, 4), 2, "graded", data.frame(
  a1 = c(1.2, 0.5, -0.8, 1), a2 = c(0.7, 1.1, 1.5, 0), d1 = c(2, 1.5, 2.2, 1), d2 = c(0, 0.3, 0.5, -0.4),
  d3 = c(-2, -1.2, -1.9, -1.5)
))
graded_3d <- set_multidimensional_items(graded_data(5, 3), 3, "graded", data.frame(
  a1 = c(1.2, 0.5, 0.9, 0, 1.6), a2 = c(0.7, 1.1, -0.4, 0.8, 0), a3 = c(0.3, 0.8, 1.3, 1.1, 0.4),
  d1 = c(1.5, 1, 0.8, 2, 0.4), d2 = c(-1, -0.5, -1.4, 0.2, -2)
))

write_fixture("interop-multidimensional", list(
  source = "Items on 2 and 3 dimensions set to given slope-intercept coefficients in mirt, one model per item type.",
  items = c(
    multidimensional_items_of(m2pl_2d, "2PL", "m2pl", 2),
    multidimensional_items_of(m3pl_2d, "3PL", "m3pl", 2),
    multidimensional_items_of(graded_2d, "graded", "mgrm", 2),
    multidimensional_items_of(graded_3d, "graded", "mgrm", 3)
  )
))
