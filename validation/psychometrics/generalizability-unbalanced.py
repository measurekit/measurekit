"""Checks the variance components of generalizability-unbalanced.json against GeneralizIT, whose calculate_anova is
Henderson's Method 1, as the fixture's base R computation is. Writes nothing and fails on a difference above TOLERANCE.

Run from the repository root, after generalizability-unbalanced.R:
validation/psychometrics/.venv/bin/python validation/psychometrics/generalizability-unbalanced.py
"""

import json

import pandas as pd
from generalizit import GeneralizIT

FIXTURE = "packages/psychometrics/test/fixtures/generalizability-unbalanced.json"
# GeneralizIT rounds the T values, the coefficients and the components to 4 decimals, so a component differs from the
# unrounded solution of the fixture by up to 5e-5 and a little more through the system.
TOLERANCE = 1e-4
# GeneralizIT's notation of each design, in the facets' names.
DESIGNS = {
    "p x r": "person x rater",
    "p x r x i": "person x rater x item",
    "r:p": "rater:person",
    "(r:p) x i": "(rater:person) x item",
    "i:p": "item:person",
}

with open(FIXTURE) as file:
    cases = json.load(file)["cases"]

for case in cases:
    scores = pd.DataFrame([{**entry["levels"], "score": entry["score"]} for entry in case["scores"]])
    study = GeneralizIT(scores, DESIGNS[case["name"]], "score")
    study.calculate_anova()
    # The table lists the effects in the order of the fixture, then the grand mean.
    variances = study.design.anova_table["Variance"].to_numpy()[:-1]
    expected = [component["variance"] for component in case["components"]]
    assert len(variances) == len(expected), f"{case['name']}: {len(variances)} components, expected {len(expected)}"
    worst = max(abs(actual - wanted) for actual, wanted in zip(variances, expected))
    print(f"{case['name']}: largest difference {worst:.2e}")
    assert worst < TOLERANCE, f"{case['name']}: GeneralizIT differs from the fixture by {worst}"
