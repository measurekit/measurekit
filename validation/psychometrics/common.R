# Helpers for the fixture scripts. Run the scripts from the repository root, e.g. `Rscript validation/psychometrics/ctt.R`.

suppressPackageStartupMessages(library(jsonlite))

# Writes `x` as packages/psychometrics/test/fixtures/<name>.json, with 17 significant digits and NA as null.
write_fixture <- function(name, x) {
  path <- file.path("packages/psychometrics/test/fixtures", paste0(name, ".json"))
  write_json(x, path, digits = I(17), auto_unbox = TRUE, na = "null", null = "null")
  message("Wrote ", path)
}

# The key of mirt::SAT12, whose responses are option numbers, 8 for an omission.
SAT12_KEY <- c(1, 4, 5, 2, 3, 1, 2, 1, 3, 1, 2, 4, 2, 1, 5, 3, 4, 4, 1, 4, 3, 3, 4, 1, 3, 5, 1, 3, 1, 5, 4, 5)

# Returns mirt's model of `data` with the items set to slope-intercept coefficients, one data frame column per name,
# NA where an item lacks the name. The items are set, not estimated: mirt evaluates its starting values when TOL = NaN.
set_items <- function(data, itemtype, coefficients) {
  values <- mirt::mirt(data, 1, itemtype = itemtype, pars = "values")
  for (i in seq_len(nrow(coefficients))) {
    for (name in names(coefficients)) {
      value <- coefficients[[name]][i]
      if (!is.na(value)) values$value[values$item == colnames(data)[i] & values$name == name] <- value
    }
  }
  mirt::mirt(data, 1, itemtype = itemtype, pars = values, TOL = NaN, verbose = FALSE)
}

# the package's parameters of an item from its IRT coefficients; for the rsm, whose thresholds are b_j - c / a1, the
# location and the steps around it.
psychometrics_parameters <- function(model, irt) {
  numbered <- function(prefix) unname(irt[grep(paste0("^", prefix, "[0-9]+$"), names(irt))])
  switch(model,
    "2pl" = , "3pl" = , "4pl" = list(a = irt[["a"]], b = irt[["b"]], c = irt[["g"]], d = irt[["u"]]),
    grm = , gpcm = , pcm = list(a = irt[["a"]], b = I(numbered("b"))),
    nrm = list(a = I(numbered("a")), c = I(numbered("c"))),
    rsm = {
      thresholds <- numbered("b") - irt[["c"]] / irt[["a1"]]
      list(a = irt[["a1"]], b = mean(thresholds), steps = I(thresholds - mean(thresholds)))
    }
  )
}

# Weights of persons in the group of the `fraction` * n highest totals. Persons tied at the boundary share the
# remaining places equally. Pass -total for the lower group.
group_weights <- function(total, fraction) {
  remaining <- fraction * length(total)
  weights <- numeric(length(total))
  for (score in sort(unique(total), decreasing = TRUE)) {
    tied <- which(total == score)
    weights[tied] <- min(max(remaining, 0), length(tied)) / length(tied)
    remaining <- remaining - length(tied)
  }
  weights
}

weighted_share <- function(weights, x) sum(weights * x) / sum(weights)

# Pearson correlation, NA when either variable is constant.
correlation <- function(x, y) suppressWarnings(cor(x, y))
