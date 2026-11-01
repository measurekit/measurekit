# Cognitive diagnosis: calibration of the G-DINA model and of the models it contains by EM, classification of persons,
# classification accuracy, validation of the Q-matrix and nonparametric classification, against GDINA's GDINA,
# personparm, CA and Qval and NPCD's AlphaNP. GDINA's convergence criterion is the largest change of a success
# probability or a proportion, as measurekit's tolerance, set to measurekit's default of 1e-4.

suppressPackageStartupMessages({
  library(GDINA)
  library(NPCD)
})

source("validation/benchmarks/simulation.R")

set.seed(20261029)
persons <- 1000
item_count <- 20
attribute_count <- 4
CONTROL <- list(conv.crit = 1e-4, maxitr = 2000)

# --- Data ----------------------------------------------------------------------------------------------------------

simulated <- simulate_dina(persons, item_count, attribute_count)
Q <- simulated$Q
attributes <- simulated$attributes
responses <- simulated$responses

# --- References -----------------------------------------------------------------------------------------------------

calibrate <- function(model) GDINA(responses, Q, model = model, control = CONTROL, verbose = 0)

gdina_fit <- calibrate("GDINA")

bench_fixture("psychometrics", "cdm", list(
  source = "A simulation, set.seed(20261029): DINA responses with a guessing of 0.2 and a slip of 0.1 over 4 correlated attributes.",
  itemIds = ids(item_count),
  responses = unname(responses),
  qMatrix = setNames(lapply(seq_len(item_count), function(j) I(attributes[Q[j, ] == 1])), ids(item_count))
))

size <- sprintf("%s persons, %d items, %d attributes", count(persons), item_count, attribute_count)
models <- c(gdina = "GDINA", dina = "DINA", dino = "DINO", additive = "ACDM", logit = "LLM", reduced = "RRUM")
tasks <- c(gdina = "G-DINA", dina = "DINA", dino = "DINO", additive = "A-CDM", logit = "LLM", reduced = "R-RUM")
keys <- c(gdina = "Gdina", dina = "Dina", dino = "Dino", additive = "Acdm", logit = "Llm", reduced = "Rrum")
for (name in names(models)) {
  local({
    model <- models[[name]]
    bench_task(paste0("cdm.calibrate", keys[[name]]), sprintf("Calibration of the %s model by EM", tasks[[name]]), size,
               reference("GDINA", sprintf("GDINA(model = \"%s\")", model)), function() calibrate(model))
  })
}
# personparm reads the log posterior that the fit stored. measurekit's calibration holds none, so attributeProfiles computes
# each person's from the responses. This does the same in R with matrix products, and agrees with the stored posterior.
profile_matrix <- attributepattern(attribute_count)
log_proportions <- log(as.numeric(gdina_fit$struc.parm))
posterior_from_responses <- function() {
  log_posterior <- responses %*% log(gdina_fit$LC.prob) + (1 - responses) %*% log(1 - gdina_fit$LC.prob) +
    rep(log_proportions, each = nrow(responses))
  posterior <- exp(log_posterior - apply(log_posterior, 1, max))
  posterior / rowSums(posterior)
}
stopifnot(max(abs(posterior_from_responses() - exp(gdina_fit$technicals$logposterior.i))) < 1e-12)
from_responses <- time_calls(function() {
  posterior <- posterior_from_responses()
  mastery <- posterior %*% profile_matrix
  list(posterior, mastery, mastery > 0.5, profile_matrix[max.col(posterior, "first"), ])
})
bench_task("cdm.attributeProfiles", "Posterior profiles of the persons of a calibration", size,
           reference("GDINA", "personparm(what = \"EAP\")"), function() personparm(gdina_fit, what = "EAP"),
           note = sprintf(paste("personparm multiplies the log posterior that the fit stored by the profiles, which is work of the",
                                "fit; measurekit's calibration holds no posterior, so it computes each person's from the responses,",
                                "and returns the posterior, the probability of each attribute and the MAP and EAP profiles. The same",
                                "computation in R, with matrix products, takes %.2f ms"), from_responses$median))
bench_task("cdm.accuracy", "Accuracy and consistency of the classification", size,
           reference("GDINA", "CA"), function() CA(gdina_fit))
bench_task("cdm.validateQMatrix", "Validation of the Q-matrix by the proportion of variance accounted for", size,
           reference("GDINA", "Qval(method = \"PVAF\")"), function() Qval(gdina_fit, method = "PVAF", eps = 0.95))
bench_task("cdm.nonparametric", "Nonparametric classification by the Hamming distance", size,
           reference("NPCD", "AlphaNP"), function() AlphaNP(responses, Q, gate = "AND", method = "Hamming"))
bench_task("cdm.compareItemModels", "Wald tests of the DINA, DINO, A-CDM, LLM and R-RUM against the G-DINA, for each item with two or more attributes",
           size, reference("GDINA", "modelcomp(method = \"Wald\")"), function() modelcomp(gdina_fit, method = "Wald"),
           note = "modelcomp computes the covariance of the success probabilities of the fit again for each item; measurekit's comes with its calibration, which cdm.calibrateGdinaStandardErrors times")
bench_task("cdm.calibrateGdinaStandardErrors", "Calibration of the G-DINA model by EM, with the covariance of its success probabilities",
           size, reference("GDINA", "GDINA(model = \"GDINA\"), extract(\"catprob.cov\", SE.type = 2)"),
           function() extract(calibrate("GDINA"), "catprob.cov", SE.type = 2))
