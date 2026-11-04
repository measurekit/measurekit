# Scale scores and standard setting have no reference of comparable computation: mirt reports a scale score for a response
# pattern, not for each summed score. The tasks are measured for measurekit alone, except the joint draws of theta: mvtnorm's
# rmvnorm draws the correlated normal scores, and base R's pnorm and approx map them through the posteriors.

suppressPackageStartupMessages(library(mvtnorm))

set.seed(20261029)
joint_nodes <- seq(-6, 6, length.out = 241)
joint_count <- 6
joint_draws <- 2000
joint_posteriors <- lapply(seq_len(joint_count), function(k) {
  weights <- dnorm(joint_nodes, runif(1, -1, 1), runif(1, 0.3, 0.8))
  list(nodes = I(joint_nodes), weights = I(weights / sum(weights)))
})
joint_correlation <- matrix(0.6, joint_count, joint_count)
diag(joint_correlation) <- 1
# A norm sample of a survey design: 10,000 total scores from 0 to 60, each with a lognormal weight.
norm_persons <- 10000
norm_scores <- rbinom(norm_persons, 60, 0.55)
norm_weights <- round(exp(rnorm(norm_persons, 0, 0.6)), 4)
bench_fixture("psychometrics", "scaling", list(
  posteriors = joint_posteriors, correlation = joint_correlation,
  norms = list(scores = norm_scores, personWeights = norm_weights)
))

# A posterior is a distribution function that is linear within the cell of each node, the cells running between the
# midpoints of neighboring nodes and mirrored at the ends: the draws are its inverse at pnorm of the normal scores. The
# cumulative weights are in order, so approx skips its search for ties.
joint_draw <- function() {
  scores <- rmvnorm(joint_draws, sigma = joint_correlation, method = "eigen")
  sapply(seq_len(joint_count), function(k) {
    nodes <- joint_posteriors[[k]]$nodes
    midpoints <- (nodes[-1] + nodes[-length(nodes)]) / 2
    edges <- c(2 * nodes[1] - midpoints[1], midpoints, 2 * nodes[length(nodes)] - midpoints[length(midpoints)])
    cumulative <- cumsum(joint_posteriors[[k]]$weights)
    approx(c(0, cumulative / cumulative[length(cumulative)]), edges, xout = pnorm(scores[, k]), ties = "ordered")$y
  })
}

bench_task("scaling.drawJointThetas", "Joint draws of theta from separate posteriors by a Gaussian copula",
           sprintf("%s draws, %d posteriors of %d nodes", count(joint_draws), joint_count, length(joint_nodes)),
           reference("mvtnorm", "rmvnorm(method = \"eigen\")"), joint_draw,
           note = "the posteriors' distribution functions are inverted by base R's approx")

bench_task("scaling.normStatisticsWeighted", "Norms of a weighted sample: mean, SD, z and rank of each score, stanines, standard errors",
           sprintf("%s persons, %d distinct scores", count(norm_persons), length(unique(norm_scores))))
bench_task("scaling.conversionTableEapSum", "A conversion table of the posterior mean of theta given each summed score", "40 items")
bench_task("scaling.conversionTableTrueScore", "A conversion table of the theta of each summed true score", "40 items")
bench_task("standardSetting.bookmarkCutScore", "The cut score of a bookmark in an ordered item booklet", "40 items")
bench_task("standardSetting.classificationAccuracy", "Accuracy and consistency of the classification by two cuts", "2,000 persons")
bench_task("parameterization.writeMirtSlopeIntercept", "Conversion of 2PL items to mirt's slope-intercept form", "10,000 items")
bench_task("parameterization.writeMirtIrt", "Conversion of 2PL items to mirt's IRT form", "10,000 items")
