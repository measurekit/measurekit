# Knowledge space theory fixtures: quasi-ordinal spaces by DAKS's imp2state and by enumeration in base R; space,
# well-gradedness and fringes by kst; delineation by pks; the basic local independence model fitted by pks's blim; and
# inductive item tree analysis by DAKS's iita and ind_gen. The Bayesian adaptive assessment is written out in base R.

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages({
  library(kst)
  library(pks)
  library(DAKS)
  library(sets)
})

# A structure as the package takes it: the domain and each state's elements.
structure_of <- function(K) {
  domain <- colnames(K)
  list(domain = domain, states = lapply(seq_len(nrow(K)), function(r) I(domain[K[r, ] == 1])))
}
famset_of <- function(K) as.set(lapply(seq_len(nrow(K)), function(r) as.set(colnames(K)[K[r, ] == 1])))
elements_of <- function(s) I(sort(as.character(unlist(as.list(s)))))
# DAKS's implications (i, j), i a prerequisite of j, as each item's prerequisites.
prerequisites_of <- function(imp, items) {
  pairs <- lapply(as.list(imp), function(t) unlist(as.list(t)))
  setNames(lapply(seq_along(items), function(j) {
    I(items[sort(unlist(lapply(pairs, function(p) if (p[2] == j) p[1])))])
  }), items)
}

# Quasi-ordinal spaces: every subset closed under the relation, by enumeration, against imp2state.
closed_subsets <- function(imp, m) {
  pairs <- lapply(as.list(imp), function(t) unlist(as.list(t)))
  subsets <- as.matrix(expand.grid(rep(list(0:1), m)))
  keep <- apply(subsets, 1, function(s) all(vapply(pairs, function(p) s[p[2]] == 0 || s[p[1]] == 1, logical(1))))
  subsets[keep, , drop = FALSE]
}
same_states <- function(a, b) setequal(apply(a, 1, paste, collapse = ""), apply(b, 1, paste, collapse = ""))
set.seed(20261029)
data(pisa)
relations <- list(
  list(name = "pisa", imp = iita(pisa, v = 1)$implications, m = 5),
  list(name = "chain", imp = set(tuple(1L, 2L), tuple(2L, 3L), tuple(1L, 3L)), m = 4),
  list(name = "equivalent", imp = set(tuple(1L, 2L), tuple(2L, 1L), tuple(2L, 3L), tuple(1L, 3L)), m = 4)
)
for (k in 1:3) relations[[length(relations) + 1]] <- list(name = paste0("simulated", k),
                                                          imp = simu(9, 10, 0.05, 0.05, delta = 0.15)$implications, m = 9)
surmise <- lapply(relations, function(r) {
  items <- paste0("i", seq_len(r$m))
  states <- imp2state(r$imp, r$m)
  stopifnot(same_states(states, closed_subsets(r$imp, r$m)))
  colnames(states) <- items
  list(name = r$name, prerequisites = prerequisites_of(r$imp, items), structure = structure_of(states))
})

# Spaces and fringes by kst; learning spaces, the knowledge spaces that are well-graded, equivalently downgradable
# (every non-empty state has an element whose removal leaves a state), by pks's is.downgradable, and by kst's
# well-gradedness test where it runs in reasonable time.
data(DoignonFalmagne7)
data(chess)
three <- matrix(c(0, 0, 0, 1, 1, 0, 1, 1, 1), 3, byrow = TRUE, dimnames = list(NULL, c("a", "b", "c")))
not_space <- DoignonFalmagne7$K[-7, ]
structures <- list(doignonFalmagne = DoignonFalmagne7$K, chess = chess$dst1, gap = three, notSpace = not_space)
spaces <- lapply(names(structures), function(name) {
  K <- structures[[name]]
  ks <- kstructure(famset_of(K))
    space <- kstructure_is_kspace(ks)
  learning <- space && is.downgradable(K)
  if (nrow(K) <= 20) stopifnot(learning == (space && kstructure_is_wellgraded(ks)))
  states <- lapply(seq_len(nrow(K)), function(r) as.set(colnames(K)[K[r, ] == 1]))
  list(name = name, structure = structure_of(K), knowledgeSpace = space,
              learningSpace = learning,
       inner = lapply(states, function(s) elements_of(kfringe_inner(ks, s))),
       outer = lapply(states, function(s) elements_of(kfringe_outer(ks, s))))
})

# Delineation: an item is in a state when all its skills are, from every set of skills (pks's conjunctive model).
skill_map <- data.frame(item = c("p1", "p2", "p3", "p4", "p5"), s1 = c(1, 0, 1, 1, 0), s2 = c(0, 1, 1, 0, 0),
                        s3 = c(0, 0, 0, 1, 1), s4 = c(0, 0, 0, 0, 1))
skills <- c("s1", "s2", "s3", "s4")
power <- as.matrix(expand.grid(rep(list(0:1), 4)))
colnames(power) <- skills
delineated <- delineate(skill_map)$K
write_fixture("kst-structures", list(
  surmise = surmise, spaces = spaces,
  delineation = list(structure = structure_of(power),
                     qMatrix = setNames(lapply(seq_len(nrow(skill_map)), function(r) I(skills[skill_map[r, -1] == 1])),
                                        skill_map$item),
                     items = structure_of(delineated))
))

# BLIM: pks's blim on its data. Each pattern of responses, a string of 0 and 1 over the domain, stands for as many
# learners as its frequency, each with one event per item; the test builds that log.
blim_case <- function(name, K, N.R, method) {
  fit <- blim(K, N.R, method = method, tol = 1e-13, maxiter = 1e6)
    list(name = name, method = tolower(method), structure = structure_of(K), patterns = as.list(N.R),
       probabilities = unname(fit$P.K), guess = as.list(fit$eta), slip = as.list(fit$beta),
       logLikelihood = fit$loglik, iterations = fit$iter)
}
chess_N.R <- chess$N.R
blims <- list(
  blim_case("doignonFalmagne", DoignonFalmagne7$K, DoignonFalmagne7$N.R, "ML"),
  blim_case("doignonFalmagne", DoignonFalmagne7$K, DoignonFalmagne7$N.R, "MD"),
  blim_case("chess", chess$dst1, chess_N.R, "ML")
)
for (b in blims) message(b$name, " ", b$method, ": ", b$iterations, " iterations, log-likelihood ", b$logLikelihood)
write_fixture("kst-blim", blims)

# IITA: DAKS on PISA and on data simulated from quasi orders.
iita_case <- function(name, data) {
  items <- colnames(data) %||% paste0("i", seq_len(ncol(data)))
  colnames(data) <- items
  selection <- ind_gen(ob_counter(data))
  versions <- lapply(c(minimized = 1, corrected = 2, original = 3), function(v) {
    r <- iita(data, v = v)
    list(diff = r$diff, selected = r$selection.set.index, errorRate = r$error.rate,
         prerequisites = prerequisites_of(r$implications, items))
  })
    patterns <- table(apply(data, 1, paste, collapse = ""))
  list(name = name, items = items, patterns = as.list(patterns), selection = lapply(selection, prerequisites_of, items = items),
       versions = versions)
}
iitas <- list(iita_case("pisa", as.matrix(pisa)))
for (k in 1:3) {
  simulated <- simu(7, 300, 0.05, 0.05, delta = 0.2)$dataset
  iitas[[length(iitas) + 1]] <- iita_case(paste0("simulated", k), simulated)
}
write_fixture("kst-iita", iitas)

# Adaptive assessment, written out: the posterior over states, the half-split question (mastery nearest 0.5, the first
# on ties), Bayes' rule on the answer, until the most likely state reaches the threshold.
K <- DoignonFalmagne7$K
ml <- blims[[1]]
guess <- unlist(ml$guess)
slip <- unlist(ml$slip)
assess <- function(truth, threshold) {
    p <- ml$probabilities
  asked <- character(0)
  outcomes <- integer(0)
  while (max(p) < threshold) {
    mastery <- colSums(K * p)
    q <- which.min(abs(mastery - 0.5))
    y <- rbinom(1, 1, if (truth[q] == 1) 1 - slip[q] else guess[q])
        asked <- c(asked, colnames(K)[q])
    outcomes <- c(outcomes, y)
    p <- p * ifelse(K[, q] == 1, if (y == 1) 1 - slip[q] else slip[q], if (y == 1) guess[q] else 1 - guess[q])
        p <- p / sum(p)
  }
  list(questions = I(asked), outcomes = I(outcomes), probabilities = p)
}
assessments <- lapply(seq_len(nrow(K)), function(r) c(list(truth = I(colnames(K)[K[r, ] == 1])), assess(K[r, ], 0.8)))
deterministic <- kassess(kstructure(famset_of(K)), K)
write_fixture("kst-assessment", list(
  threshold = 0.8, assessments = assessments,
  deterministic = list(patterns = lapply(seq_len(nrow(K)), function(r) I(colnames(K)[K[r, ] == 1])),
                       states = unname(lapply(deterministic, elements_of)))
))
