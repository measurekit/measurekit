# Helper for the fixture scripts of multidimensional items (multidimensional.R and parameterization.R).

# Returns mirt's model of `data` with `dimensions`, a count or a mirt.model, and the items set to slope-intercept
# coefficients, one data frame column per name, NA where an item lacks the name. The items are set, not estimated: mirt
# evaluates its starting values when TOL = NaN.
set_multidimensional_items <- function(data, dimensions, itemtype, coefficients) {
  values <- mirt::mirt(data, dimensions, itemtype = itemtype, pars = "values")
  for (i in seq_len(nrow(coefficients))) {
    for (name in names(coefficients)) {
      value <- coefficients[[name]][i]
      if (!is.na(value)) values$value[values$item == colnames(data)[i] & values$name == name] <- value
    }
  }
  mirt::mirt(data, dimensions, itemtype = itemtype, pars = values, TOL = NaN, verbose = FALSE)
}
