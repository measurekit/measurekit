# Purified Lord's chi-square fixtures (dif-lord-purification.json). Reference: difR's difLord(purify = TRUE) on ltm's 2PL
# calibrations of each group, run to convergence. The standard errors are ltm's, from the observed information. Data: the
# cases of dif.json, difR's verbal and a 2PL simulation with known DIF, so run dif.R first.

source("validation/psychometrics/common.R")
source("validation/psychometrics/ltm-items.R")
suppressPackageStartupMessages(library(difR))

ALPHA <- 0.05

# difLord's purification on the calibrations of both groups. The anchors of the final statistics are the items the
# round before it left unflagged; with no DIF to begin with, all items.
purified_lord <- function(case) {
  data <- do.call(rbind, lapply(case$responses, unlist))
  focal <- unlist(case$focal)
  colnames(data) <- paste0("item", seq_len(ncol(data)))
  reference <- item_parameters_2pl(data[focal == 0, ])
  focal_parameters <- item_parameters_2pl(data[focal == 1, ])
  result <- difLord(data, group = focal, focal.name = 1, model = "2PL", irtParam = rbind(reference, focal_parameters),
                    same.scale = FALSE, purify = TRUE, alpha = ALPHA)
  anchors <- if (result$nrPur == 0) seq_len(ncol(data)) else which(result$difPur[result$nrPur, ] == 0)
  stopifnot(max(abs(LordChi2(reference, itemRescale(reference, focal_parameters, items = anchors)) - result$LordChi)) < 1e-12)
  slope <- sd(reference[anchors, "b"]) / sd(focal_parameters[anchors, "b"])
  list(
    name = case$name,
    chiSquare = result$LordChi,
    significance = result$p.value,
    iterations = result$nrPur,
    converged = result$convergence,
    anchors = I(anchors),
    transformation = list(slope = slope, intercept = mean(reference[anchors, "b"]) - slope * mean(focal_parameters[anchors, "b"]))
  )
}

cases <- jsonlite::fromJSON("packages/psychometrics/test/fixtures/dif.json", simplifyVector = FALSE)$cases

write_fixture("dif-lord-purification", list(
  source = "difLord(purify = TRUE) on the cases of dif.json: difR's verbal and a 2PL simulation with DIF in items 1 to 3.",
  cases = lapply(cases, purified_lord)
))
