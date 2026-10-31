# The numerics and random numbers of @measurekit/core against base R: qnorm, plogis, chol, runif and rnorm. The inputs
# are formulas, so that measurekit builds the same ones without a fixture.

POINTS <- 1e6
ORDER <- 200
CORRELATION <- 0.9

probabilities <- (seq_len(POINTS) - 0.5) / POINTS
points <- 12 * (seq_len(POINTS) - 1) / POINTS - 6
correlations <- CORRELATION^abs(outer(seq_len(ORDER), seq_len(ORDER), "-"))

bench_task("numerics.normalQuantile", "Normal quantile function", "1,000,000 probabilities",
           reference("stats", "qnorm"), function() qnorm(probabilities))
bench_task("numerics.logistic", "Logistic function", "1,000,000 points", reference("stats", "plogis"),
           function() plogis(points))
bench_task("numerics.cholesky", "Cholesky factorization", "A correlation matrix of order 200",
           reference("base", "chol"), function() chol(correlations),
           note = paste("R's BLAS is OpenBLAS, whose kernels fuse multiplication and addition in vector registers;",
                        "JavaScript has neither, and measurekit's loops are at the speed of scalar code"))
bench_task("random.uniform", "Uniform draws", "1,000,000 draws", reference("stats", "runif"),
           function() runif(POINTS))
bench_task("random.normal", "Normal draws", "1,000,000 draws", reference("stats", "rnorm"), function() rnorm(POINTS))
