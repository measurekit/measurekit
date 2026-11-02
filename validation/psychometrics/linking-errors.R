# Standard errors of links between separate calibrations (equating-link-errors.json). References: equateIRT's direc
# for a link of two forms and chainec for a chain of three, whose covariance counts the form between two links once,
# both with 30 Gauss-Hermite points; equateIRT's score(se = TRUE) for the standard errors of equated scores. Data: the
# 2PL calibrations of equateIRT's est2pl, forms T1 to T3, with the covariance of each.
#
# Where equateIRT cannot be the reference, the script writes the delta method out: numDeriv's Jacobian of direc's links,
# or of the true-score equating through them, in each parameter of each form, times the forms' covariances. It
# reproduces equateIRT where equateIRT is right:
# - direc's derivatives of mean-sigma's A divide by the n common items where A's standard deviations divide by n - 1,
#   which shrinks its variances by ((n - 1) / n)², 0.81 for 10 items; mean-mean, Haebara and Stocking-Lord agree;
# - score links the form to equate to onto the scale of the form to equate, where this package links the form to equate
#   onto the other's. The two links are inverse functions for mean-mean, whose equated scores and standard errors then
#   agree; for the other methods the links differ.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(equateIRT)
  library(numDeriv)
})

POINTS <- 30
METHODS <- c(meanMean = "mean-mean", meanSigma = "mean-sigma", haebara = "Haebara", stockingLord = "Stocking-Lord")
data(est2pl, package = "equateIRT")
mods <- modIRT(coef = est2pl$coef, var = est2pl$var, display = FALSE)
FORMS <- c("T1", "T2", "T3")

# A form's items and its covariance in measurekit's terms: b and a of each item, labeled by item and name.
form_parameters <- function(form) mods[[form]]$coef
item_ids <- function(form) sub("Dffclt.", "", grep("^Dffclt", names(form_parameters(form)), value = TRUE))
calibrated_form <- function(form) {
  coef <- form_parameters(form)
  labels <- names(coef)
  list(
    items = lapply(item_ids(form), function(id) {
      list(id = id, model = "2pl", parameters = list(a = coef[[paste0("Dscrmn.", id)]], b = coef[[paste0("Dffclt.", id)]],
                                                     c = 0, d = 1))
    }),
    covariance = list(
      parameters = lapply(labels, function(label) {
        list(item = sub("^[A-Za-z]+\\.", "", label), name = if (startsWith(label, "Dffclt")) "b" else "a")
      }),
      matrix = unname(lapply(seq_along(labels), function(i) unname(mods[[form]]$var[i, ])))
    )
  )
}

link_errors <- function(link) {
  list(slope = link$A, intercept = link$B,
       standardErrors = list(slope = sqrt(link$varAB[1, 1]), intercept = sqrt(link$varAB[2, 2])),
       covariance = as.vector(t(link$varAB)))
}

# --- The delta method written out ----------------------------------------------------------------------------------

# Each form's coefficients after `values` replace them, in the order of the forms, as modIRT objects without the latent
# trait parameterization.
moved_mods <- function(values, forms) {
  sizes <- sapply(forms, function(form) length(form_parameters(form)))
  coef <- lapply(seq_along(forms), function(k) {
    part <- setNames(values[(c(0, cumsum(sizes))[k] + 1):cumsum(sizes)[k]], names(form_parameters(forms[k])))
    ids <- item_ids(forms[k])
    matrix(c(part[paste0("Dffclt.", ids)], part[paste0("Dscrmn.", ids)]), ncol = 2,
           dimnames = list(ids, c("Dffclt", "Dscrmn")))
  })
  modIRT(coef = coef, var = lapply(forms, function(form) mods[[form]]$var), names = forms, ltparam = FALSE,
         display = FALSE)
}

# A and B of the chain from the first form to the last, each link by direc, composed as chainec does.
chain_link <- function(moved, forms, method) {
  Reduce(function(chain, k) {
    link <- direc(mods = moved, which = c(forms[k], forms[k + 1]), method = method, nq = POINTS)
    c(link$A * chain[1], link$A * chain[2] + link$B)
  }, seq_len(length(forms) - 1), c(1, 0))
}

delta_method <- function(f, forms) {
  values <- unlist(lapply(forms, form_parameters), use.names = FALSE)
  jacobian <- numDeriv::jacobian(f, values)
  jacobian %*% as.matrix(Matrix::bdiag(lapply(forms, function(form) mods[[form]]$var))) %*% t(jacobian)
}

written_link <- function(forms, method) {
  link <- chain_link(mods, forms, method)
  covariance <- delta_method(function(v) chain_link(moved_mods(v, forms), forms, method), forms)
  list(A = link[1], B = link[2], varAB = covariance)
}

true_score <- function(a, b, theta) sum(plogis(a * (theta - b)))

# The true-score equivalents on `to` of the scores 1 to n - 1 of `from`, its items moved by the link from `from` to `to`.
linked_true_scores <- function(moved, from, to, method) {
  link <- chain_link(moved, c(from, to), method)
  items <- function(form) {
    coef <- moved[[form]]$coef
    ids <- item_ids(form)
    list(a = coef[paste0("Dscrmn.", ids)], b = coef[paste0("Dffclt.", ids)])
  }
  source <- items(from)
  target <- items(to)
  a <- source$a / link[1]
  b <- link[1] * source$b + link[2]
  sapply(seq_len(length(a) - 1), function(x) {
    theta <- uniroot(function(t) true_score(a, b, t) - x, c(-30, 30), tol = 1e-14)$root
    true_score(target$a, target$b, theta)
  })
}

written_true_score_errors <- function(from, to, method) {
  forms <- c(from, to)
  sqrt(diag(delta_method(function(v) linked_true_scores(moved_mods(v, forms), from, to, method), forms)))
}

score_errors <- function(from, to, method, type) {
  result <- suppressMessages(score(direc(mods = mods, which = c(to, from), method = method, nq = POINTS),
                                   method = type, se = TRUE, nq = POINTS))
  list(scores = result[[from]], standardErrors = result$StdErr)
}

# --- The cases ------------------------------------------------------------------------------------------------------

link_errors <- function(link) {
  list(slope = link$A, intercept = link$B,
       standardErrors = list(slope = sqrt(link$varAB[1, 1]), intercept = sqrt(link$varAB[2, 2])),
       covariance = as.vector(t(link$varAB)))
}

two <- c("T1", "T2")
three <- c("T1", "T2", "T3")
reference_link <- function(method) direc(mods = mods, which = two, method = method, nq = POINTS)
reference_chain <- function(method) {
  chainec(r = 3, direclist = alldirec(mods = mods, method = method, nq = POINTS))[["T1.T2.T3"]]
}

# The write-up reproduces direc and chainec but for mean-sigma, where their variance of A is ((n - 1) / n)² of it.
for (method in METHODS) {
  for (pair in list(list(reference_link(method), written_link(two, method)),
                    list(reference_chain(method), written_link(three, method)))) {
    reference <- pair[[1]]$varAB
    written <- pair[[2]]$varAB
    if (method == "mean-sigma") stopifnot(abs(reference[1, 1] / written[1, 1] - 0.81) < 0.05)
    else stopifnot(max(abs(reference - written) / sqrt(outer(diag(written), diag(written)))) < 0.05)
  }
}

links <- lapply(METHODS, function(method) {
  link_errors(if (method == "mean-sigma") written_link(two, method) else reference_link(method))
})
chains <- lapply(METHODS, function(method) {
  link_errors(if (method == "mean-sigma") written_link(three, method) else reference_chain(method))
})

# score is the reference for mean-mean, and the write-up reproduces it.
mean_mean <- score_errors("T1", "T2", "mean-mean", "TSE")
written_mean_mean <- written_true_score_errors("T1", "T2", "mean-mean")
stopifnot(max(abs(written_mean_mean[mean_mean$scores[mean_mean$scores < 20]] -
                  mean_mean$standardErrors[mean_mean$scores < 20])) < 1e-6)
true_score_cases <- lapply(METHODS, function(method) {
  if (method == "mean-mean") return(mean_mean)
  list(scores = seq_len(length(item_ids("T1")) - 1), standardErrors = written_true_score_errors("T1", "T2", method))
})

quadrature <- statmod::gauss.quad.prob(POINTS, dist = "normal")

write_fixture("equating-link-errors", list(
  source = paste("equateIRT's est2pl, forms T1 to T3 with their covariances: direc and chainec with 30 Gauss-Hermite",
                 "points; score(se = TRUE) for mean-mean; numDeriv over direc's links and base R's true-score",
                 "equating where the script says."),
  forms = setNames(lapply(FORMS, calibrated_form), FORMS),
  quadrature = list(nodes = quadrature$nodes, weights = quadrature$weights),
  links = links,
  chains = chains,
  trueScore = true_score_cases,
  observedScore = list(meanMean = score_errors("T1", "T2", "mean-mean", "OSE"))
))
