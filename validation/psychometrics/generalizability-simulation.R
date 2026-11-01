# The simulation of scores shared by generalizability.R and generalizability-unbalanced.R.

# One normal draw for each distinct combination of the columns of an effect, added to the scores of its cells: the
# columns of a nested effect include the facets it is nested in, so labels repeat from one element to the next.

simulate_scores <- function(grid, effects, error_sd, mean = 5) {
  score <- rep(mean, nrow(grid))
  for (effect in effects) {
    key <- do.call(paste, c(grid[effect$columns], sep = "|"))
    draws <- setNames(rnorm(length(unique(key)), 0, effect$sd), unique(key))
    score <- score + unname(draws[key])
  }
  score + rnorm(nrow(grid), 0, error_sd)
}

grid_of <- function(...) {
  grid <- expand.grid(..., KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  as.data.frame(lapply(grid, function(column) factor(column, levels = unique(column))))
}
