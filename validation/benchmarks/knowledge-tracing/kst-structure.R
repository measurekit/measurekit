# Knowledge structure tools: the closure under union and the base of a space of 125 states, three chains of four of 12
# elements, against kst's kspace and kbase; and the space of 12 elements with clauses, whose subsets base R enumerates.

suppressPackageStartupMessages({
  library(kst)
  library(sets)
})

item_count <- 12
elements <- sprintf("q%02d", seq_len(item_count))

# Three chains of four: the states are the choices of how far to go along each chain.
steps <- as.matrix(expand.grid(rep(list(0:4), 3)))
states <- t(apply(steps, 1, function(g) unlist(lapply(1:3, function(chain) as.integer(seq_len(4) <= g[chain])))))
colnames(states) <- elements
state_sets <- lapply(seq_len(nrow(states)), function(r) as.set(elements[states[r, ] == 1]))
space <- kstructure(as.set(state_sets))
base <- kbase(space)

# Each element after the first three needs the element three before it, or both of the two before it.
clauses <- lapply(seq_len(item_count), function(i) {
  if (i <= 3) list() else list(elements[i - 3], elements[c(i - 1, i - 2)])
})
names(clauses) <- elements

bench_fixture("knowledge-tracing", "kst-structure", list(
  source = "Three chains of four elements; the clauses of 12 elements, each after the third needing the element three before it, or the two before it.",
  domain = I(elements),
  states = lapply(state_sets, function(s) I(sort(as.character(unlist(as.list(s)))))),
  clauses = lapply(clauses, function(c) lapply(c, I))
))

enumerate_states <- function() {
  subsets <- as.matrix(expand.grid(rep(list(0:1), item_count)))
  colnames(subsets) <- elements
  addable <- function(element, built) length(clauses[[element]]) == 0 || any(vapply(clauses[[element]], function(clause) all(clause %in% built), logical(1)))
  keep <- apply(subsets, 1, function(row) {
    target <- elements[row == 1]
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

bench_task("kst.unionClosure", "Closure under union of the 12 atoms of a space",
           sprintf("%d elements, %d states", item_count, nrow(states)), reference("kst", "kspace"),
           function() kspace(kfamset(base)))
bench_task("kst.structureBase", "Base of a space",
           sprintf("%d elements, %d states", item_count, nrow(states)), reference("kst", "kbase"),
           function() kbase(space))
bench_task("kst.clauseStructure", "Space of clauses, by enumeration of the subsets in base R",
           sprintf("%d elements, %d subsets", item_count, 2^item_count), "base R `expand.grid` and `apply`",
           enumerate_states, note = "no package takes clauses")
bench_task("kst.surmiseStructure", "Quasi-ordinal space of a cycle of prerequisites, each element requiring the next",
           "3,000 elements")
