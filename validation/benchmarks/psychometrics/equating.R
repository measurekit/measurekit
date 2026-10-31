# Equating and linking on public data: the random-groups design of equate's ACTmath (two forms of 40 items), the
# nonequivalent groups design of equate's KBneat (36 items, an internal anchor of 12), and equateIRT's est3pl (two forms
# that share 10 items). References: equate's equate, presmoothing and bootstrap, and equateIRT's direc and score.

suppressPackageStartupMessages({
  library(equate)
  library(equateIRT)
})

data(ACTmath, KBneat, package = "equate")
data(est3pl, package = "equateIRT")

REPLICATIONS <- 100
POINTS <- 30 # equateIRT's quadrature: 30 Gauss-Hermite points for N(0, 1)

# --- Data ----------------------------------------------------------------------------------------------------------

act_x <- as.freqtab(ACTmath[, 1:2])
act_y <- as.freqtab(ACTmath[, c(1, 3)])
kb_x <- freqtab(KBneat$x, scales = list(0:36, 0:12))
kb_y <- freqtab(KBneat$y, scales = list(0:36, 0:12))
mods <- modIRT(coef = est3pl$coef, var = est3pl$var, display = FALSE)

form_items <- function(form) {
  coef <- mods[[form]]$coef
  items <- sub("Dffclt.", "", grep("Dffclt", names(coef), value = TRUE))
  list(ids = I(items), a = I(unname(coef[paste0("Dscrmn.", items)])), b = I(unname(coef[paste0("Dffclt.", items)])),
       c = I(unname(coef[paste0("Gussng.", items)])))
}

bench_fixture("psychometrics", "equating", list(
  source = "equate's ACTmath and KBneat, and equateIRT's est3pl: the score distributions, the bivariate distributions and the items.",
  randomGroups = list(from = I(as.numeric(act_x)), to = I(as.numeric(act_y))),
  neat = list(from = matrix(as.numeric(kb_x), nrow = 37), to = matrix(as.numeric(kb_y), nrow = 37)),
  irt = list(forms = list(form_items(1), form_items(2)), common = I(paste0("I", 1:10)), points = POINTS)
))

# --- References -----------------------------------------------------------------------------------------------------

random_groups <- sprintf("%s and %s persons, 40 items", count(sum(act_x)), count(sum(act_y)))
neat <- sprintf("%s and %s persons, 36 items, an anchor of 12", count(sum(kb_x)), count(sum(kb_y)))

bench_task("equating.linear", "Linear equating, random groups", random_groups, reference("equate", "equate(type = \"linear\")"),
           function() equate(act_x, act_y, type = "linear", verbose = FALSE))
bench_task("equating.equipercentile", "Equipercentile equating, random groups", random_groups,
           reference("equate", "equate(type = \"equipercentile\")"),
           function() equate(act_x, act_y, type = "equipercentile", verbose = FALSE))
bench_task("equating.presmoothing", "Log-linear presmoothing of a score distribution, degree 3", "41 scores",
           reference("equate", "presmoothing(smoothmethod = \"loglinear\")"),
           function() presmoothing(act_x, smoothmethod = "loglinear", degrees = list(3)))
bench_task("equating.bootstrapLinear", "Standard errors of the linear equating by 100 bootstrap replications", random_groups,
           reference("equate", "bootstrap"), function() bootstrap(act_x, act_y, reps = REPLICATIONS, args = list(type = "linear")))
bench_task("equating.tucker", "Tucker linear equating, nonequivalent groups", neat,
           reference("equate", "equate(method = \"tucker\")"),
           function() equate(kb_x, kb_y, type = "linear", method = "tucker", verbose = FALSE))
bench_task("equating.frequencyEstimation", "Frequency estimation equipercentile equating", neat,
           reference("equate", "equate(method = \"frequency estimation\")"),
           function() equate(kb_x, kb_y, type = "equipercentile", method = "frequency estimation", verbose = FALSE))
bench_task("equating.chainedEquipercentile", "Chained equipercentile equating", neat,
           reference("equate", "equate(method = \"chained\")"),
           function() equate(kb_x, kb_y, type = "equipercentile", method = "chained", verbose = FALSE))

irt <- "Two forms of 3PL items that share 10"
stocking_lord <- direc(mods = mods, which = c(1, 2), method = "Stocking-Lord", nq = POINTS)
bench_task("equating.stockingLordLinking", "Stocking-Lord linking of the common items", irt,
           reference("equateIRT", "direc(method = \"Stocking-Lord\")"),
           function() direc(mods = mods, which = c(1, 2), method = "Stocking-Lord", nq = POINTS))
bench_task("equating.haebaraLinking", "Haebara linking of the common items", irt,
           reference("equateIRT", "direc(method = \"Haebara\")"),
           function() direc(mods = mods, which = c(1, 2), method = "Haebara", nq = POINTS))
bench_task("equating.trueScore", "IRT true score equating of the linked forms", irt,
           reference("equateIRT", "score(method = \"TSE\")"), function() score(stocking_lord, method = "TSE", se = FALSE))
bench_task("equating.irtObservedScore", "IRT observed score equating of the linked forms", irt,
           reference("equateIRT", "score(method = \"OSE\")"), function() score(stocking_lord, method = "OSE", se = FALSE))
