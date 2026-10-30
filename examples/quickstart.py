"""CE-kNN in a few lines, on synthetic data: fit, read the curves and weights, export the metric, use it elsewhere.

    python examples/quickstart.py
"""

import numpy as np
import pandas as pd

from ceknn import CEKNN, ModelSpace


def synthetic(n: int, seed: int, base: float = -1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(65, 12, n)
    map_mean = rng.normal(80, 10, n)
    surgery = rng.choice(["cardiac", "abdominal", "neuro", "other"], size=n, p=[0.3, 0.3, 0.2, 0.2])
    emergency = (rng.uniform(size=n) < 0.25).astype(float)
    lactate = np.where(rng.uniform(size=n) < 0.3, np.nan, rng.lognormal(0.3, 0.4, n))
    eta = (base + 0.04 * (age - 65) + 0.004 * (map_mean - 80) ** 2 - 0.4 + 0.8 * (surgery == "cardiac")
           + 0.6 * emergency + 0.5 * np.nan_to_num(lactate - 1.4))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    return pd.DataFrame({"patient": np.arange(n).astype(str), "age": age, "map_mean": map_mean, "surgery": surgery,
                         "emergency": emergency, "lactate": lactate, "event": y})


if __name__ == "__main__":
    dev = synthetic(5000, seed=0)

    # 1. choose the inputs, the output and the Rashomon model space
    space = ModelSpace.default(include=["logreg", "spline_logreg", "random_forest"])
    model = CEKNN(inputs={"age": "continuous", "map_mean": "continuous", "surgery": "nominal",
                          "emergency": "binary", "lactate": "continuous"},
                  output="event", group="patient", model_space=space, n_models=12, k=50, n_jobs=2)

    # 2. fit: the metric (curves g_j and weights w_j) is learned, and dev becomes the case base
    model.fit(dev)
    print(model.summary())

    # 3. read the metric
    print(model.shape("map_mean")[["value", "n", "log_odds"]].round(2).to_string(index=False))
    print(model.importance()[["variable", "weight", "shaped", "logodds_range"]].round(3).to_string(index=False))
    print(model.rashomon()[["family", "val_logloss", "accepted", "competitive"]].round(4).head(10))

    # 4. export the metric (never the case base) and use it at another site with its own patients
    model.save("metric.json")
    local = synthetic(3000, seed=1, base=-0.5)              # another site, with a higher event rate
    there = CEKNN.load("metric.json").set_case_base(local)
    new = synthetic(500, seed=2, base=-0.5)
    print("at the other site:", there.evaluate(new))
    print(there.neighbours(new.head(1), k=5)[["rank", "case_index", "distance", "outcome"]])
