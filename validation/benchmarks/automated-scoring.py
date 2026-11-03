"""Times RSMTool, the reference of the benchmarks of the evaluation of automated scores, on the scores of
rater-automated-scoring.json, which validation/psychometrics/automated-scoring.py writes from PERSUADE 2.0 and ELLIPSE.

    validation/psychometrics/.venv-scoring/bin/python -I validation/benchmarks/automated-scoring.py

Each task starts from the unrounded machine scores and rounds and trims them, as the recipe of rater.md does. On ELLIPSE,
the analyzer's compute_degradation_and_disattenuated_correlations gives the human-system and human-human metrics, the
degradation and the disattenuated correlation; on PERSUADE 2.0, metrics_helper gives the metrics of all essays and of
each group, with the SMD on the population's SDs. The results go to validation/benchmarks/results/python-scoring.json,
which `Rscript validation/benchmarks.R table` reads.
"""

import json
import os
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rsmtool.analyzer import Analyzer  # noqa: E402

FIXTURE = "packages/psychometrics/test/fixtures/rater-automated-scoring.json"
RESULTS = "validation/benchmarks/results/python-scoring.json"


def time_calls(function):
    """The median and the fastest of the calls of `function` in milliseconds, after one call that warms up."""
    start = time.perf_counter()
    function()
    first = time.perf_counter() - start
    inner = max(1, int(0.05 / max(first, 1e-4)) + 1)
    runs = 15 if first < 1 else 7 if first < 15 else 3 if first < 60 else 2
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(inner):
            function()
        samples.append((time.perf_counter() - start) * 1000 / inner)
    return {"median": statistics.median(samples), "min": min(samples), "samples": runs}


with open(FIXTURE) as file:
    cases = json.load(file)["cases"]

ellipse = cases["ELLIPSE"]
ellipse_first = np.array(ellipse["first"], dtype=float)
ellipse_second = np.array(ellipse["second"], dtype=float)
ellipse_machine = np.array(ellipse["machine"])
analyzer = Analyzer()


def evaluate_ellipse():
    low, high = ellipse["scale"]
    frame = pd.DataFrame(
        {"sc1": ellipse_first, "sc2": ellipse_second, "raw_trim_round": np.clip(np.rint(ellipse_machine), low, high)}
    )
    return analyzer.compute_degradation_and_disattenuated_correlations(frame)


persuade = cases["PERSUADE 2.0"]
persuade_human = np.array(persuade["human"], dtype=float)
persuade_machine = np.array(persuade["machine"])
persuade_groups = np.array(persuade["groups"])


def evaluate_persuade():
    low, high = persuade["scale"]
    machine = np.clip(np.rint(persuade_machine), low, high)
    human_sd = np.std(persuade_human, ddof=1)
    machine_sd = np.std(machine, ddof=1)
    overall = Analyzer.metrics_helper(pd.Series(persuade_human), pd.Series(machine), smd_method="pooled")
    groups = [
        Analyzer.metrics_helper(
            pd.Series(persuade_human[rows]), pd.Series(machine[rows]), human_sd, machine_sd, smd_method="williamson"
        )
        for rows in (persuade_groups == group for group in sorted(set(persuade["groups"])))
    ]
    return overall, groups


results = {
    "rater.automatedScoring": time_calls(evaluate_ellipse),
    "rater.automatedScoringGroups": time_calls(evaluate_persuade),
}
for key, result in results.items():
    print(f"{key:32} median {result['median']:10.3f} ms   fastest {result['min']:10.3f} ms")

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
with open(RESULTS, "w") as file:
    json.dump({"psychometrics": results}, file, indent=2)
