# The marginal likelihood of the many-facet Rasch model written out in base R, shared by the reference scripts of the rater
# module (rater.R and rater-bias.R): theta = mu + sigma z on the 61 rectangular points of the package, the gradient by
# Fisher's identity, and the maximum by BFGS polished by Newton's method on numDeriv's Jacobian of the gradient, whose
# inverse is the covariance of the parameters. A design has one column of `X` for each location parameter, so that an
# interaction is more columns.

suppressPackageStartupMessages(library(numDeriv))

Z <- seq(-6, 6, length.out = 61)
LOG_WEIGHTS <- log(dnorm(Z) / sum(dnorm(Z)))

# The scores in long format: a data frame with a person column, one column per facet and a score column. The design:
# each distinct combination of the elements of the facets other than the person is a virtual item; a facet's measures
# are effect coded, so that its n elements have n - 1 free parameters and the last is minus their sum, and a threshold
# set of K thresholds has K - 1 free ones.
facet_design <- function(long, facets, threshold_facet = NULL) {
  person <- match(long$person, unique(long$person))
  elements <- setNames(lapply(facets, function(f) unique(as.character(long[[f]]))), facets)
  codes <- sapply(facets, function(f) match(as.character(long[[f]]), elements[[f]]))
  codes <- matrix(codes, nrow = nrow(long))
  key <- apply(codes, 1, paste, collapse = " ")
  virtual <- match(key, unique(key))
  combinations <- codes[!duplicated(key), , drop = FALSE]
  contrasts <- lapply(elements, function(ids) {
    if (length(ids) == 1) matrix(0, 1, 0) else rbind(diag(length(ids) - 1), -1)
  })
  X <- do.call(cbind, lapply(seq_along(facets), function(f) contrasts[[f]][combinations[, f], , drop = FALSE]))
  set_of <- if (is.null(threshold_facet)) rep(1, nrow(combinations)) else combinations[, match(threshold_facet, facets)]
  sets <- if (is.null(threshold_facet)) NA_character_ else elements[[threshold_facet]]
  thresholds <- sapply(seq_along(sets), function(s) max(long$score[set_of[virtual] == s]))
  list(long = long, facets = facets, person = person, elements = elements, contrasts = contrasts, virtual = virtual,
       combinations = combinations, X = X, set_of = set_of, sets = sets, thresholds = thresholds,
       threshold_facet = threshold_facet, persons = length(unique(person)))
}

# The parameter vector: location parameters of the facets in order, free thresholds of each set, mu, ln sigma.
unpack <- function(design, phi) {
  p <- ncol(design$X)
  beta <- phi[seq_len(p)]
  at <- p
  thresholds <- vector("list", length(design$thresholds))
  for (s in seq_along(thresholds)) {
    free <- phi[at + seq_len(design$thresholds[s] - 1)]
    at <- at + design$thresholds[s] - 1
    thresholds[[s]] <- c(free, -sum(free))
  }
  list(beta = beta, thresholds = thresholds, mu = phi[at + 1], sigma = exp(phi[at + 2]))
}

# The log probability of every category of every virtual item at every node, theta = mu + sigma z: row (c - 1) * width
# + k + 1 for item c and category k.
node_tables <- function(design, phi) {
  par <- unpack(design, phi)
  theta <- par$mu + par$sigma * Z
  location <- drop(design$X %*% par$beta)
  width <- max(design$thresholds) + 1
  table <- matrix(NA_real_, nrow(design$combinations) * width, length(Z))
  for (c in seq_len(nrow(design$combinations))) {
    steps <- location[c] + par$thresholds[[design$set_of[c]]]
    logits <- rbind(0, outer(seq_along(steps), theta) - cumsum(steps))
    top <- apply(logits, 2, max)
    total <- top + log(colSums(exp(sweep(logits, 2, top))))
    table[(c - 1) * width + seq_len(nrow(logits)), ] <- sweep(logits, 2, total)
  }
  list(par = par, width = width, table = table, rows = (design$virtual - 1) * width + design$long$score + 1)
}

# The log-likelihood of each person's scores at each node.
person_node_loglik <- function(design, phi) {
  tables <- node_tables(design, phi)
  rowsum(tables$table[tables$rows, , drop = FALSE], design$person)
}

marginal_loglik <- function(design, phi) {
  joint <- sweep(person_node_loglik(design, phi), 2, LOG_WEIGHTS, "+")
  top <- apply(joint, 1, max)
  sum(top + log(rowSums(exp(joint - top))))
}

# The gradient of the marginal log-likelihood, by Fisher's identity: the expected complete-data gradient, with the
# posterior of each person over the nodes. For a virtual item with counts n_k of the expected persons in category k at a
# node, N their total, S_j = P(X >= j) and M_j = the counts at or above j, the gradient with respect to step j is
# N S_j - M_j, and with respect to theta sum(k n_k) - N E(X).
marginal_gradient <- function(design, phi) {
  tables <- node_tables(design, phi)
  joint <- sweep(rowsum(tables$table[tables$rows, , drop = FALSE], design$person), 2, LOG_WEIGHTS, "+")
  posterior <- exp(joint - apply(joint, 1, max))
  posterior <- posterior / rowSums(posterior)
  counts <- rowsum(posterior[design$person, , drop = FALSE], tables$rows)
  expected <- matrix(0, nrow(tables$table), length(Z))
  expected[as.integer(rownames(counts)), ] <- counts
  p <- ncol(design$X)
  starts <- p + cumsum(c(0, design$thresholds - 1))
  gradient <- numeric(length(phi))
  for (c in seq_len(nrow(design$combinations))) {
    s <- design$set_of[c]
    K <- design$thresholds[s]
    rows <- (c - 1) * tables$width + seq_len(K + 1)
    n <- expected[rows, , drop = FALSE]
    probabilities <- exp(tables$table[rows, , drop = FALSE])
    persons <- colSums(n)
    tails <- function(x) apply(x, 2, function(column) rev(cumsum(rev(column))))[-1, , drop = FALSE]
    step <- rowSums(sweep(tails(probabilities), 2, persons, "*") - tails(n))
    theta <- colSums((0:K) * n) - persons * colSums((0:K) * probabilities)
    gradient[seq_len(p)] <- gradient[seq_len(p)] + design$X[c, ] * sum(step)
    gradient[starts[s] + seq_len(K - 1)] <- gradient[starts[s] + seq_len(K - 1)] + step[seq_len(K - 1)] - step[K]
    gradient[length(phi) - 1] <- gradient[length(phi) - 1] + sum(theta)
    gradient[length(phi)] <- gradient[length(phi)] + sum(theta * tables$par$sigma * Z)
  }
  gradient
}

# The maximum: BFGS on the exact gradient, then Newton's method on numDeriv's Jacobian of it. The gradient is checked
# against numDeriv's gradient of the likelihood, and the information matrix is numDeriv's Hessian of the likelihood,
# which must agree with the Jacobian of the gradient to `agreement` of its largest entry; numDeriv's Hessian of a model
# with many parameters is accurate to about 1e-5 relative.
maximize <- function(design, agreement = 1e-5) {
  size <- ncol(design$X) + sum(design$thresholds - 1) + 2
  negative <- function(phi) -marginal_loglik(design, phi)
  negative_gradient <- function(phi) -marginal_gradient(design, phi)
  probe <- seq(-0.3, 0.3, length.out = size)
  stopifnot(max(abs(negative_gradient(probe) - grad(negative, probe))) < 1e-7 * max(abs(negative_gradient(probe))))
  phi <- optim(rep(0, size), negative, negative_gradient, method = "BFGS",
               control = list(maxit = 10000, reltol = 1e-15))$par
  for (iteration in 1:100) {
    step <- solve(jacobian(negative_gradient, phi), negative_gradient(phi))
    phi <- phi - step
    if (max(abs(step)) < 1e-11) break
  }
  stopifnot(max(abs(negative_gradient(phi))) < 1e-8)
  information <- hessian(negative, phi)
  stopifnot(max(abs(information - jacobian(negative_gradient, phi))) < agreement * max(abs(information)))
  list(phi = phi, information = information, logLikelihood = -negative(phi), size = size)
}
