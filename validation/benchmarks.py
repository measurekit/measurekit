"""Times pyBKT, the reference of the Bayesian knowledge tracing benchmarks of measurekit, on the log of bench-bkt.json, and
on the same log with a fifth of its attempts as lessons, and with the rates of pairs of items.

    validation/knowledge-tracing/.venv/bin/python validation/benchmarks.py

The results go to validation/benchmarks/results/python.json, which `Rscript validation/benchmarks.R table` reads. pyBKT
fits every skill, five starts each by default, with the processes of its parallel option off, one thread.
"""

import json
import os
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd  # noqa: E402
from pyBKT.models import Model  # noqa: E402

FIXTURE = "packages/knowledge-tracing/test/fixtures/bench-bkt.json"
RESULTS = "validation/benchmarks/results/python.json"


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
    fixture = json.load(file)

data = pd.DataFrame(
    {
        "order_id": range(len(fixture["learner"])),
        "user_id": fixture["learner"],
        "skill_name": fixture["skill"],
        "correct": fixture["outcome"],
        "item": fixture["item"],
    }
)

# The same log with every fifth attempt, from the first, a lesson: correct = -1, pyBKT's no response.
LESSON_EVERY = 5
lessons = data.assign(correct=[-1 if index % LESSON_EVERY == 0 else outcome
                               for index, outcome in enumerate(fixture["outcome"])])

model = Model(seed=1, parallel=False)


def fit():
    Model(seed=1, parallel=False).fit(data=data)


def fit_lessons():
    Model(seed=1, parallel=False).fit(data=lessons)


def fit_pairs():
    Model(seed=1, parallel=False).fit(data=data, multipair=True, defaults={"multipair": "item"})


model.fit(data=data)
results = {
    "bkt.fit": time_calls(fit),
    "bkt.fitLessons": time_calls(fit_lessons),
    "bkt.fitPairs": time_calls(fit_pairs),
    "bkt.replay": time_calls(lambda: model.predict(data=data)),
}
for key, result in results.items():
    print(f"{key:20} median {result['median']:10.3f} ms   fastest {result['min']:10.3f} ms")

os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
with open(RESULTS, "w") as file:
    json.dump({"knowledge-tracing": results}, file, indent=2)
