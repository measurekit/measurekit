# Henderson's Method 1 for the generalizability fixtures and benchmark of unbalanced designs, written out in base R.

# The facets each facet is nested in, directly or through others, by facet.
ancestors_of <- function(facets, nested) {
  ancestors <- setNames(lapply(facets, function(facet) if (is.null(nested[[facet]])) character(0) else nested[[facet]]),
                        facets)
  repeat {
    wider <- lapply(ancestors, function(parents) unique(c(parents, unlist(ancestors[parents], use.names = FALSE))))
    if (identical(lapply(wider, sort), lapply(ancestors, sort))) return(wider)
    ancestors <- wider
  }
}

# The effects of a design: each set of facets that holds every facet its members are nested in, ordered by the number of
# facets and then by their positions in the design, which combn enumerates in turn. An effect is in the facets it is a
# deviation in (own) and those it is nested in.
effects_of <- function(facets, nested) {
  ancestors <- ancestors_of(facets, nested)
  subsets <- unlist(lapply(seq_along(facets), function(k) combn(facets, k, simplify = FALSE)), recursive = FALSE)
  closed <- Filter(function(subset) all(unlist(ancestors[subset], use.names = FALSE) %in% subset), subsets)
  lapply(closed, function(subset) {
    inside <- intersect(subset, unlist(ancestors[subset], use.names = FALSE))
    list(facets = subset, own = setdiff(subset, inside), nested_in = inside)
  })
}

# The components of the data by Method 1. The expectation of T of an effect less that of the grand mean is the sum over
# the effects b of k(effect, b) less the sum of the squared counts of the cells of b over the number of scores, times
# the variance component of b, with k(a, b) the sum over the cells c of a and d of b of n_cd^2 / n_c.
method_one <- function(data, facets, nested) {
  effects <- effects_of(facets, nested)
  size <- length(effects)
  scores <- data$score
  total <- length(scores)
  cells <- lapply(effects, function(effect) interaction(data[effect$facets], drop = TRUE))
  ids <- lapply(cells, as.integer)
  counts <- lapply(cells, function(cell) as.vector(table(cell)))
  t_values <- sapply(cells, function(cell) sum(as.vector(tapply(scores, cell, sum))^2 / as.vector(table(cell)))) -
    sum(scores)^2 / total
  coefficients <- outer(seq_len(size), seq_len(size), Vectorize(function(a, b) {
    # The non-empty cells of the union of a and b, each a cell c of a and d of b with n_cd scores: coded by the numbers
    # of c and d. Their counts are n_cd, and c is recovered from the code.
    code <- as.numeric(ids[[a]]) + (as.numeric(ids[[b]]) - 1) * length(counts[[a]])
    joint <- table(code)
    c_of_joint <- (as.numeric(names(joint)) - 1) %% length(counts[[a]]) + 1
    sum(as.vector(joint)^2 / counts[[a]][c_of_joint]) - sum(counts[[b]]^2) / total
  }))
  variances <- solve(coefficients, t_values)
  sums <- numeric(size)
  degrees <- numeric(size)
  for (a in seq_len(size)) {
    inner <- which(sapply(effects, function(b) {
      !setequal(b$facets, effects[[a]]$facets) && all(b$facets %in% effects[[a]]$facets)
    }))
    sums[a] <- t_values[a] - sum(sums[inner])
    degrees[a] <- nlevels(cells[[a]]) - 1 - sum(degrees[inner])
  }
  stopifnot(all(degrees > 0))
  lapply(seq_len(size), function(a) {
    effect <- effects[[a]]
    list(facets = I(facets[facets %in% effect$own]), nestedIn = I(facets[facets %in% effect$nested_in]),
         degreesOfFreedom = degrees[a], sumOfSquares = sums[a], meanSquare = sums[a] / degrees[a],
         variance = variances[a])
  })
}
