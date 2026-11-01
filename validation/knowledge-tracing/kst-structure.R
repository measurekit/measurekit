# Knowledge structure tools: base, atoms, neighbourhoods, learning paths, well-gradedness, traces and notions by kst; the
# closure under union by kst's kspace; and spaces from clauses by enumeration in base R, whose bases kst's kspace closes
# back to the same space. The element ids are single letters or q01, q02, ...: kst's katoms matches an id as a pattern,
# so an id that another contains would give wrong atoms.

source("validation/knowledge-tracing/common.R")
suppressPackageStartupMessages({
  library(kst)
  library(pks)
  library(DAKS)
  library(sets)
})

# A structure as measurekit takes it: the domain and each state's elements.
structure_of <- function(K) {
  domain <- colnames(K)
  list(domain = I(domain), states = lapply(seq_len(nrow(K)), function(r) I(domain[K[r, ] == 1])))
}
famset_of <- function(K) as.set(lapply(seq_len(nrow(K)), function(r) as.set(colnames(K)[K[r, ] == 1])))
# A set of elements as a sorted vector, and a family of sets as a list of them.
elements_of <- function(s) I(sort(as.character(unlist(as.list(s)))))
family_of <- function(family) lapply(as.list(family), elements_of)
# The 0/1 matrix of a family of sets over a domain.
matrix_of <- function(family, domain) {
  rows <- lapply(as.list(family), function(s) as.integer(domain %in% unlist(as.list(s))))
  K <- matrix(unlist(rows), ncol = length(domain), byrow = TRUE)
  colnames(K) <- domain
  K
}

# Learning paths: the maximal chains of states of kst's lpath, whose steps each add one element, as the order of the
# elements added. A chain with a larger step is not a path in measurekit.
path_orders <- function(ks) {
  chains <- lpath(ks)
  orders <- lapply(chains, function(chain) {
    states <- lapply(as.list(chain), function(s) sort(as.character(unlist(as.list(s)))))
    states <- states[order(lengths(states))]
    added <- lapply(seq_along(states)[-1], function(i) setdiff(states[[i]], states[[i - 1]]))
    if (all(lengths(added) == 1)) unlist(added) else NULL
  })
  unname(Filter(Negate(is.null), orders))
}

# Well-gradedness by its definition: from any state to any other, a path of states that each differ from the one before by
# one element, as many steps as the two states differ by. Each step flips an element in which the state differs from the
# target, so the search tries those.
is_wellgraded <- function(K) {
  keys <- apply(K, 1, paste, collapse = "")
  reaches <- function(state, target) {
    if (all(state == target)) return(TRUE)
    any(vapply(which(state != target), function(e) {
      step <- state
      step[e] <- 1L - step[e]
      paste(step, collapse = "") %in% keys && reaches(step, target)
    }, logical(1)))
  }
  all(vapply(seq_len(nrow(K)), function(i) all(vapply(seq_len(nrow(K))[-i], function(j) reaches(K[i, ], K[j, ]), logical(1))), logical(1)))
}

set.seed(20261029)

# The structures: published ones, quasi-ordinal spaces, and seeded random families that contain the empty state and the
# whole domain, which are mostly not spaces.
data(DoignonFalmagne7)
data(chess)
chess_states <- chess$dst1
colnames(chess_states) <- sprintf("q%02d", seq_len(ncol(chess_states)))
random_family <- function(m) {
  subsets <- as.matrix(expand.grid(rep(list(0:1), m)))
  chosen <- subsets[sample(nrow(subsets), sample(4:min(nrow(subsets), 14), 1)), , drop = FALSE]
  K <- unique(rbind(rep(0L, m), rep(1L, m), chosen))
  colnames(K) <- letters[seq_len(m)]
  K
}
quasi_ordinal <- function(m) {
  K <- imp2state(simu(m, 10, 0.05, 0.05, delta = 0.15)$implications, m)
  colnames(K) <- letters[seq_len(m)]
  K
}
cases <- c(
  list(
    doignonFalmagne = DoignonFalmagne7$K,
    chess = chess_states,
    gap = matrix(c(0, 0, 0, 1, 1, 0, 1, 1, 1), 3, byrow = TRUE, dimnames = list(NULL, c("a", "b", "c"))),
    notSpace = DoignonFalmagne7$K[-7, ],
    quasiOrdinal5 = quasi_ordinal(5),
    quasiOrdinal7 = quasi_ordinal(7)
  ),
  setNames(lapply(c(3, 4, 4, 5, 5, 5, 6, 6), random_family), paste0("random", 1:8))
)

structures <- lapply(names(cases), function(name) {
  K <- cases[[name]]
  domain <- colnames(K)
  ks <- kstructure(famset_of(K))
  space <- kstructure_is_kspace(ks)
  small <- nrow(K) <= 30
  states <- lapply(seq_len(nrow(K)), function(r) as.set(domain[K[r, ] == 1]))
  atoms <- katoms(ks, as.set(domain))
  base <- if (space) {
    base <- kbase(ks)
    stopifnot(identical(as.character(as.set(kspace(base))), as.character(as.set(ks))))
    family_of(base)
  }
  trace_sets <- c(list(domain), list(domain[1]), lapply(1:3, function(i) sort(sample(domain, sample(1:max(1, length(domain) - 1), 1)))))
  list(
    name = name,
    structure = structure_of(K),
    knowledgeSpace = space,
    base = base,
    atoms = setNames(lapply(domain, function(e) family_of(atoms[[e]])), domain),
    wellGraded = if (small) {
      graded <- kstructure_is_wellgraded(ks)
      stopifnot(graded == is_wellgraded(K))
      graded
    },
    paths = if (small) path_orders(ks),
    neighbourhoods = lapply(states, function(s) list(
      state = elements_of(s),
      distance1 = family_of(kneighbourhood(ks, s)),
      distance2 = family_of(knneighbourhood(ks, s, 2)),
      distance3 = family_of(knneighbourhood(ks, s, 3))
    )),
    traces = lapply(trace_sets, function(items) {
      trace <- ktrace(ks, as.set(items))
      list(elements = I(items), structure = structure_of(matrix_of(trace, sort(items))))
    }),
    notions = family_of(knotions(ks))
  )
})

# Closure under union by kst's kspace, from generating families over the union of their elements.
closures <- lapply(1:8, function(i) {
  m <- sample(3:6, 1)
  domain <- letters[seq_len(m)]
  generators <- lapply(seq_len(sample(2:5, 1)), function(g) sort(sample(domain, sample(1:max(1, m - 1), 1))))
  elements <- sort(unique(unlist(generators)))
  space <- kspace(kfamset(as.set(lapply(generators, as.set))))
  list(domain = I(elements), generators = lapply(generators, I),
       space = structure_of(matrix_of(space, elements)))
})

# Spaces from clauses: a subset is a state when its elements can be added one at a time, each with no clauses or with a
# clause inside the elements added before, by enumeration of the subsets. When every element can be learned, kst's kspace
# of the base of that family is the family again.
enumerate_states <- function(clauses, domain) {
  subsets <- as.matrix(expand.grid(rep(list(0:1), length(domain))))
  colnames(subsets) <- domain
  addable <- function(element, built) length(clauses[[element]]) == 0 || any(vapply(clauses[[element]], function(clause) all(clause %in% built), logical(1)))
  keep <- apply(subsets, 1, function(row) {
    target <- domain[row == 1]
    built <- character(0)
    repeat {
      step <- Filter(function(element) addable(element, built), setdiff(target, built))
      if (length(step) == 0) break
      built <- c(built, step[1])
    }
    setequal(built, target)
  })
  subsets[keep, , drop = FALSE]
}
clause_cases <- list(
  list(name = "or", clauses = list(a = list(), b = list(), c = list("a", c("b", "d")), d = list("a"))),
  list(name = "chain", clauses = list(a = list(), b = list("a"), c = list("b"), d = list("c"))),
  list(name = "emptyClause", clauses = list(a = list(character(0)), b = list("a"), c = list(c("a", "b"), "b"))),
  list(name = "two", clauses = list(a = list(), b = list(), c = list(c("a", "b")), d = list("c", "a"), e = list("b", c("c", "d")))),
  list(name = "cycle", clauses = list(a = list(), b = list("c"), c = list("b"), d = list("a"))),
  list(name = "self", clauses = list(a = list("a"), b = list())),
  list(name = "alternativeBreaksCycle", clauses = list(a = list(), b = list("c", "a"), c = list("b"), d = list("b")))
)
for (k in 1:8) {
  m <- sample(4:7, 1)
  domain <- letters[seq_len(m)]
  clause_cases[[length(clause_cases) + 1]] <- list(name = paste0("random", k), clauses = setNames(lapply(seq_len(m), function(j) {
    lapply(seq_len(sample(0:3, 1)), function(c) sort(sample(setdiff(domain, domain[j]), sample(1:3, 1))))
  }), domain))
}
clause_results <- lapply(clause_cases, function(case) {
  domain <- names(case$clauses)
  K <- enumerate_states(case$clauses, domain)
  reached <- domain[colSums(K) > 0]
  unreachable <- setdiff(domain, reached)
  if (length(unreachable) == 0) {
    ks <- kstructure(famset_of(K))
    stopifnot(kstructure_is_kspace(ks), identical(as.character(as.set(kspace(kbase(ks)))), as.character(as.set(ks))))
  }
  list(name = case$name,
       clauses = lapply(case$clauses, function(clauses) lapply(clauses, I)),
       unreachable = I(unreachable),
       structure = if (length(unreachable) == 0) structure_of(K))
})

write_fixture("kst-structure-tools", list(structures = structures, closures = closures))
write_fixture("kst-structure-clauses", clause_results)
