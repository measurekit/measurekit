# Factor solutions in base R, which exploratory-factors.R, efa-fit.R and factor-hierarchy.R share: the minimum residual
# and the maximum likelihood solutions of a correlation matrix, each found by L-BFGS-B and polished by Newton steps with a
# Hessian by central differences of the gradient to a projected gradient below 1e-10, and their rotation by GPArotation
# to a gradient of 1e-10, oriented as psych orients a solution.

suppressPackageStartupMessages(library(GPArotation))

# psych's lower bound of a uniqueness, so that a communality is at most 0.995.
UPPER <- 1 - 0.005

# --- The minimum residual solution in base R ---------------------------------------------------------------------------

# Half the squared eigenvalues that the fit of m factors to the matrix with the communalities h on its diagonal leaves
# out, a negative leading eigenvalue counting in full, and its gradient with respect to h, the diagonal of the residuals.
reduced_fit <- function(h, r, m) {
  s <- r
  diag(s) <- h
  e <- eigen(s, symmetric = TRUE)
  excess <- e$values
  excess[seq_len(m)] <- pmin(excess[seq_len(m)], 0)
  list(value = 0.5 * sum(excess^2), gradient = drop((e$vectors^2) %*% excess))
}

# The coordinates of h that the gradient does not push beyond a bound.
free_communalities <- function(h, g) !((h <= 1e-9 & g > 0) | (h >= UPPER - 1e-9 & g < 0))

minres_communalities <- function(r, m) {
  p <- nrow(r)
  gradient <- function(h) reduced_fit(h, r, m)$gradient
  start <- pmin(pmax(1 - 1 / diag(solve(r)), 0), UPPER)
  h <- optim(start, function(h) reduced_fit(h, r, m)$value, gradient, method = "L-BFGS-B", lower = rep(0, p),
             upper = rep(UPPER, p), control = list(factr = 1, pgtol = 0, maxit = 10000))$par
  for (step in 1:20) {
    g <- gradient(h)
    free <- free_communalities(h, g)
    jacobian <- sapply(which(free), function(j) {
      shift <- replace(numeric(p), j, 1e-6)
      (gradient(h + shift)[free] - gradient(h - shift)[free]) / 2e-6
    })
    h[free] <- h[free] - solve(matrix(jacobian, sum(free)), g[free])
    h <- pmin(pmax(h, 0), UPPER)
  }
  g <- gradient(h)
  stopifnot(max(abs(g[free_communalities(h, g)])) < 1e-10)
  h
}

# The loadings on the principal axes of the reduced matrix, each factor's summing to a number that is not negative.
principal_axes <- function(r, m, h) {
  s <- r
  diag(s) <- h
  e <- eigen(s, symmetric = TRUE)
  loadings <- e$vectors[, seq_len(m), drop = FALSE] %*% diag(sqrt(pmax(e$values[seq_len(m)], 0)), m)
  loadings %*% diag(ifelse(colSums(loadings) < 0, -1, 1), m)
}

# psych's orientation of a rotated solution: each factor's loadings sum to a positive number, and the factors are in
# the order of the variance they account for, the diagonal of Phi L'L.
orient <- function(loadings, phi) {
  m <- ncol(loadings)
  signs <- ifelse(colSums(loadings) < 0, -1, 1)
  loadings <- loadings %*% diag(signs, m)
  phi <- diag(signs, m) %*% phi %*% diag(signs, m)
  order <- order(diag(phi %*% crossprod(loadings)), decreasing = TRUE)
  list(loadings = loadings[, order, drop = FALSE], phi = phi[order, order, drop = FALSE])
}

# Rotates unrotated loadings, none, varimax with Kaiser's normalization or oblimin, and orients them as psych does.
rotate_solution <- function(unrotated, rotation) {
  m <- ncol(unrotated)
  rotated <- if (m == 1 || rotation == "none") {
    list(loadings = unrotated, phi = diag(m))
  } else if (rotation == "varimax") {
    v <- Varimax(unrotated, normalize = TRUE, eps = 1e-10, maxit = 100000)
    stopifnot(v$convergence)
    list(loadings = v$loadings, phi = diag(m))
  } else {
    o <- oblimin(unrotated, eps = 1e-10, maxit = 100000)
    stopifnot(o$convergence)
    list(loadings = o$loadings, phi = o$Phi)
  }
  orient(unclass(rotated$loadings), rotated$phi)
}

# --- The maximum likelihood solution in base R --------------------------------------------------------------------------

# Jöreskog's likelihood concentrated on the uniquenesses psi, the sum over the eigenvalues g of the scaled matrix
# psi^-1/2 r psi^-1/2 that m factors leave out of g - log g - 1, and its gradient, as factanal's and psych's FAfn and FAgr.
likelihood_fit <- function(psi, r, m) {
  scaled <- r / sqrt(outer(psi, psi))
  e <- eigen(scaled, symmetric = TRUE)
  left <- e$values[-seq_len(m)]
  loadings <- sqrt(psi) * e$vectors[, seq_len(m), drop = FALSE] %*% diag(sqrt(pmax(e$values[seq_len(m)] - 1, 0)), m)
  list(value = sum(left - log(left) - 1), gradient = (rowSums(loadings^2) + psi - diag(r)) / psi^2,
       loadings = loadings)
}

# The uniquenesses from 0.005 to 1 that maximize the likelihood, from 1 less the squared multiple correlations.
likelihood_uniquenesses <- function(r, m) {
  p <- nrow(r)
  gradient <- function(psi) likelihood_fit(psi, r, m)$gradient
  start <- pmin(pmax(1 / diag(solve(r)), 1 - UPPER), 1)
  psi <- optim(start, function(psi) likelihood_fit(psi, r, m)$value, gradient, method = "L-BFGS-B",
               lower = rep(1 - UPPER, p), upper = rep(1, p), control = list(factr = 1, pgtol = 0, maxit = 10000))$par
  free <- function(psi, g) !((psi <= 1 - UPPER + 1e-9 & g > 0) | (psi >= 1 - 1e-9 & g < 0))
  for (step in 1:20) {
    g <- gradient(psi)
    kept <- free(psi, g)
    if (!any(kept)) break
    jacobian <- sapply(which(kept), function(j) {
      shift <- replace(numeric(p), j, 1e-7)
      (gradient(psi + shift)[kept] - gradient(psi - shift)[kept]) / 2e-7
    })
    psi[kept] <- psi[kept] - solve(matrix(jacobian, sum(kept)), g[kept])
    psi <- pmin(pmax(psi, 1 - UPPER), 1)
  }
  g <- gradient(psi)
  stopifnot(max(abs(g[free(psi, g)])) < 1e-10)
  psi
}

# The loadings of the maximum likelihood solution, each factor's summing to a number that is not negative.
likelihood_axes <- function(r, m, psi) {
  loadings <- likelihood_fit(psi, r, m)$loadings
  loadings %*% diag(ifelse(colSums(loadings) < 0, -1, 1), m)
}
