# Item and person fit fixtures. References: eRm (itemfit and personfit: infit and outfit mean squares and their
# standardized values), mirt (itemfit's S-X², personfit's Zh, residuals' Q3). Each fixture holds the responses, the
# items in the package's parameters and the thetas the reference used, so the tests compute the statistics from the same
# inputs.

source("validation/psychometrics/common.R")
suppressPackageStartupMessages({
  library(mirt)
  library(eRm)
  library(difR)
})

TOL <- 1e-10 # mirt's EM tolerance

responses <- function(data) unname(lapply(seq_len(nrow(data)), function(i) as.numeric(data[i, ])))

# the package's parameters of mirt's items: 2PL from coef(IRTpars = TRUE); GPCM and GRM thresholds from the slope-intercept
# values, b_k = -(d_k - d_(k-1)) / a for the GPCM and -d_k / a for the GRM.
mirt_items <- function(fit, itemtype) {
  values <- mod2values(fit)
  lapply(colnames(extract.mirt(fit, "data")), function(item) {
    rows <- values[values$item == item, ]
    v <- setNames(rows$value, rows$name)
    d <- v[grep("^d[0-9]+$", names(v))]
    switch(itemtype,
      "2PL" = list(model = "2pl", parameters = list(a = v[["a1"]], b = -v[["d"]] / v[["a1"]], c = 0, d = 1)),
      gpcm = list(model = "gpcm", parameters = list(a = v[["a1"]], b = I(unname(-diff(c(0, d[-1])) / v[["a1"]])))),
      graded = list(model = "grm", parameters = list(a = v[["a1"]], b = I(unname(-d / v[["a1"]]))))
    )
  })
}

fit_mirt <- function(data, itemtype) {
  fit <- mirt(data, 1, itemtype = itemtype, TOL = TOL, verbose = FALSE, technical = list(NCYCLES = 1e5))
  stopifnot(extract.mirt(fit, "converged"))
  fit
}

na_to_null <- function(x) ifelse(is.finite(x), x, NA)

# --- Rasch infit and outfit: eRm ------------------------------------------------------------------------------------
# eRm estimates the items by conditional maximum likelihood and the persons by maximum likelihood; itemfit and
# personfit leave out persons with extreme scores. coef() gives every person's theta, extrapolated for extreme scores.

data(verbal)
verbal_items <- as.matrix(verbal[, 1:24])
rasch <- RM(verbal_items)
rasch_persons <- person.parameter(rasch)
rasch_items <- eRm::itemfit(rasch_persons)
rasch_person_fit <- eRm::personfit(rasch_persons)
extreme <- rasch_persons$pers.ex
stopifnot(length(extreme) > 0)

science <- as.matrix(Science) - 1
pcm <- PCM(science)
pcm_persons <- person.parameter(pcm)
pcm_items <- eRm::itemfit(pcm_persons)
pcm_person_fit <- eRm::personfit(pcm_persons)
thresholds_pcm <- thresholds(pcm)$threshtable[[1]]

mean_squares <- function(fit, prefix) {
  get <- function(name) unname(fit[[paste0(prefix, name)]])
  list(infitMeanSquare = get("infitMSQ"), outfitMeanSquare = get("outfitMSQ"), infitStandardized = get("infitZ"),
       outfitStandardized = get("outfitZ"))
}

# Persons with extreme scores have no person fit in eRm; their rows are null.
person_mean_squares <- function(fit, persons, count) {
  kept <- setdiff(seq_len(count), persons$pers.ex)
  lapply(mean_squares(fit, "p."), function(values) {
    full <- rep(NA, count)
    full[kept] <- values
    full
  })
}

write_fixture("fit-rasch", list(
  source = "difR's verbal (24 items, 316 persons) under eRm's RM, and mirt's Science recoded 0-3 under eRm's PCM.",
  cases = list(
    list(
      name = "Rasch",
      responses = responses(verbal_items),
      items = lapply(unname(-rasch$betapar), function(b) {
        list(model = "1pl", parameters = list(a = 1, b = b, c = 0, d = 1))
      }),
      thetas = unname(coef(rasch_persons)),
      itemFit = mean_squares(rasch_items, "i."),
      personFit = person_mean_squares(rasch_person_fit, rasch_persons, nrow(verbal_items))
    ),
    list(
      name = "PCM",
      responses = responses(science),
      items = lapply(seq_len(nrow(thresholds_pcm)), function(i) {
        list(model = "pcm", parameters = list(a = 1, b = I(unname(thresholds_pcm[i, -1]))))
      }),
      thetas = unname(coef(pcm_persons)),
      itemFit = mean_squares(pcm_items, "i."),
      personFit = person_mean_squares(pcm_person_fit, pcm_persons, nrow(science))
    )
  )
))

# --- S-X², lz and Q3: mirt ------------------------------------------------------------------------------------------
# S-X² needs complete data. lz and Q3 use mirt's EAP thetas, passed back as Theta so that the statistics depend on the
# same values; missing responses are skipped.

lsat7 <- expand.table(LSAT7)
sat12 <- key2binary(SAT12, SAT12_KEY)
sat12_missing <- sat12
sat12_missing[SAT12 == 8] <- NA
science_items <- as.matrix(Science) - 1

summed_score_case <- function(name, data, itemtype) {
  fit <- fit_mirt(data, itemtype)
  fit_stats <- mirt::itemfit(fit, fit_stats = "S_X2")
  list(
    name = name,
    responses = responses(data),
    items = mirt_items(fit, itemtype),
    chiSquare = na_to_null(fit_stats$S_X2),
    degreesOfFreedom = fit_stats$df.S_X2,
    significance = na_to_null(fit_stats$p.S_X2)
  )
}

residual_case <- function(name, data, itemtype) {
  fit <- fit_mirt(data, itemtype)
  thetas <- fscores(fit, method = "EAP", full.scores = TRUE)
  person <- mirt::personfit(fit, Theta = thetas)
  q3 <- residuals(fit, type = "Q3", Theta = thetas, verbose = FALSE)
  list(
    name = name,
    responses = responses(data),
    items = mirt_items(fit, itemtype),
    thetas = as.numeric(thetas),
    standardizedLogLikelihood = person$Zh,
    q3 = unname(lapply(seq_len(nrow(q3)), function(i) unname(q3[i, ])))
  )
}

write_fixture("fit-summed-score", list(
  source = paste("mirt's LSAT7 and SAT12 (scored, omissions wrong) under the 2PL, and Science recoded 0-3 under the",
                 "GPCM and GRM."),
  cases = list(
    summed_score_case("LSAT7 2PL", lsat7, "2PL"),
    summed_score_case("SAT12 2PL", sat12, "2PL"),
    summed_score_case("Science GPCM", science_items, "gpcm"),
    summed_score_case("Science GRM", science_items, "graded")
  )
))

write_fixture("fit-residuals", list(
  source = "mirt's SAT12 scored with omissions missing under the 2PL, and Science recoded 0-3 under the GPCM.",
  cases = list(
    residual_case("SAT12 2PL", sat12_missing, "2PL"),
    residual_case("Science GPCM", science_items, "gpcm")
  )
))
