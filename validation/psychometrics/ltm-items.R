# The helper of dif.R and dif-lord-purification.R that calibrates one group with ltm, in difR's layout of item
# parameters.

suppressPackageStartupMessages(library(ltm))

# difR's itemPar2PL with ltm run to convergence: its default control, 40 EM cycles and then BFGS at optim's default
# reltol, stops up to 1.5e-3 short of the maximum on these data. difLord rescales these estimates and compares them.
item_parameters_2pl <- function(data) {
  J <- ncol(data)
  mod <- ltm(data ~ z1, IRT.param = FALSE, control = list(GHk = 21, iter.em = 2000, iter.qN = 2000))
  beta <- summary(mod)$coefficients[1:J, 1]
  alpha <- summary(mod)$coefficients[(J + 1):(2 * J), 1]
  covariance <- vcov(mod)
  Sig1 <- diag(covariance)[1:J]
  Sig2 <- diag(covariance)[(J + 1):(2 * J)]
  Sig12 <- diag(covariance[1:J, (J + 1):(2 * J)])
  parameters <- cbind(alpha, -beta / alpha, sqrt(Sig2),
                      sqrt(Sig1 / alpha^2 + Sig2 * beta^2 / alpha^4 - 2 * beta * Sig12 / alpha^3),
                      beta * Sig2 / alpha^2 - Sig12 / alpha)
  dimnames(parameters) <- list(colnames(data), c("a", "b", "se(a)", "se(b)", "cov(a,b)"))
  parameters
}
