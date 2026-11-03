# Standard errors of links between separate calibrations of equateIRT's est2pl, 2PL forms of 20 items with the covariance
# of each calibration, at equateIRT's quadrature of 30 Gauss-Hermite points: the slope and intercept of a chain of three
# forms by Haebara's method with their covariance, against equateIRT's alldirec and chainec, and the standard errors of the
# true-score equating of a form through a mean-mean link, against equateIRT's score(se = TRUE).

suppressPackageStartupMessages(library(equateIRT))

data(est2pl, package = "equateIRT")
FORMS <- c("T1", "T2", "T3")
mods <- modIRT(coef = est2pl$coef[1:3], var = est2pl$var[1:3], names = FORMS, display = FALSE)
POINTS <- 30

# A form's items and the covariance of its parameters, as measurekit's CalibratedForm.
calibrated_form <- function(form) {
  coef <- mods[[form]]$coef
  labels <- names(coef)
  items <- sub("Dffclt.", "", grep("^Dffclt", labels, value = TRUE))
  list(
    items = lapply(items, function(id) list(id = id, a = coef[[paste0("Dscrmn.", id)]], b = coef[[paste0("Dffclt.", id)]])),
    covariance = list(
      parameters = lapply(labels, function(label) {
        list(item = sub("^[A-Za-z]+\\.", "", label), name = if (startsWith(label, "Dffclt")) "b" else "a")
      }),
      matrix = unname(lapply(seq_along(labels), function(i) unname(mods[[form]]$var[i, ])))
    )
  )
}

bench_fixture("psychometrics", "equating-links", list(
  source = "equateIRT's est2pl, forms T1 to T3 with the covariances of their calibrations.",
  forms = setNames(lapply(FORMS, calibrated_form), FORMS),
  points = POINTS
))

size <- sprintf("%d forms of %d items", length(FORMS), length(mods$T1$coef) / 2)
bench_task("equating.linkCovariance", "The link of a chain of three forms by Haebara's method, with its covariance", size,
           reference("equateIRT", "alldirec, chainec"),
           function() chainec(r = 3, direclist = alldirec(mods = mods, method = "Haebara", nq = POINTS)),
           note = paste("alldirec links each pair of forms both ways, twice the links of the chain, as chainec takes no other",
                        "list; equateIRT differentiates the links analytically, and measurekit, for items of any model, by",
                        "central differences of one Newton step from each link's minimum"))
bench_task("equating.linkedErrors", "The standard errors of true-score equating through a mean-mean link",
           sprintf("2 forms of %d items", length(mods$T1$coef) / 2), reference("equateIRT", "score(se = TRUE)"),
           function() invisible(capture.output(score(direc(mods = mods, which = c("T2", "T1"), method = "mean-mean", nq = POINTS),
                                                   method = "TSE", se = TRUE, nq = POINTS))))
