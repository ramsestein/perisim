"""The model space of the Rashomon set and the models trained with group-wise feature dropout.

Each model is a surrogate that can predict with any subset of input groups present (Frye et al., 2021): it is
trained on copies of the data with random groups masked. The shape functions are read from the models with every
group present; the masks are what lets SAGE (optional) simulate an absent group without retraining.
- Trees (random forest, LightGBM, XGBoost): an absent group → NaN (all three handle NaN natively).
- Linear and spline-linear models: an absent group → 0 (the mean after standardisation) + a per-group indicator.
- The logistic regression samples its penalty (L2 / L1 / elastic net); the spline logistic regression expands the
  continuous / ordinal columns in a cubic B-spline basis (a penalised GAM: smooth non-linear shapes).

A `ModelSpace` lists the families, how often each is drawn (`weight`) and the range of every hyperparameter:
``{"log": [lo, hi]}`` log-uniform, ``{"uniform": [lo, hi]}``, ``{"int": [lo, hi]}``, ``{"choice": [...]}`` or a
fixed value. A custom family takes a `factory(params, seed, n_jobs)` returning any classifier with `fit` and
`predict_proba`; say with `handles_nan` whether it accepts NaN (then absent groups are NaN, as for the trees).
"""

from __future__ import annotations

import copy
import importlib.util
import time
import warnings

import numpy as np

TREE_FAMILIES = {"random_forest", "lightgbm", "xgboost"}

DEFAULT_FAMILIES: dict[str, dict] = {
    "logreg": {"weight": 1, "params": {"C": {"log": [1e-3, 10.0]}, "penalty": {"choice": ["l2", "l1", "elasticnet"]},
                                       "l1_ratio": {"uniform": [0.1, 0.9]}, "max_iter": 1000}},
    "spline_logreg": {"weight": 1, "params": {"C": {"log": [1e-3, 10.0]}, "n_knots": {"choice": [4, 6, 8]},
                                              "degree": 3, "max_iter": 1000}},
    "random_forest": {"weight": 1, "params": {"n_estimators": {"int": [100, 400]},
                                              "max_depth": {"choice": [6, 10, 16, None]},
                                              "min_samples_leaf": {"int": [5, 50]},
                                              "max_features": {"choice": ["sqrt", 0.3, 0.5]}}},
    "lightgbm": {"weight": 1, "params": {"n_estimators": {"int": [100, 800]}, "learning_rate": {"log": [0.01, 0.2]},
                                         "num_leaves": {"int": [8, 128]}, "min_child_samples": {"int": [10, 200]},
                                         "subsample": {"uniform": [0.5, 1.0]},
                                         "colsample_bytree": {"uniform": [0.5, 1.0]},
                                         "reg_lambda": {"log": [1e-3, 10.0]}}},
    "xgboost": {"weight": 1, "params": {"n_estimators": {"int": [100, 800]}, "learning_rate": {"log": [0.01, 0.2]},
                                        "max_depth": {"int": [2, 8]}, "min_child_weight": {"log": [1.0, 20.0]},
                                        "subsample": {"uniform": [0.5, 1.0]},
                                        "colsample_bytree": {"uniform": [0.5, 1.0]},
                                        "reg_lambda": {"log": [1e-3, 10.0]}}},
}
_NEEDS = {"lightgbm": "lightgbm", "xgboost": "xgboost"}


def installed(family: str) -> bool:
    mod = _NEEDS.get(family)
    return mod is None or importlib.util.find_spec(mod) is not None


class ModelSpace:
    """Families and hyperparameter ranges from which the Rashomon set is drawn."""

    def __init__(self, families: dict[str, dict] | None = None):
        fams = copy.deepcopy(DEFAULT_FAMILIES if families is None else families)
        for name, spec in fams.items():
            if "params" not in spec:
                spec["params"] = {}
            if name not in DEFAULT_FAMILIES and "factory" not in spec:
                raise ValueError(f"family {name!r} is not built in: give it a `factory`")
            if name in DEFAULT_FAMILIES and not installed(name):
                raise ImportError(f"family {name!r} needs the package {_NEEDS[name]!r}")
        if not fams:
            raise ValueError("the model space has no family")
        self.families = fams

    @classmethod
    def default(cls, include: list[str] | None = None, exclude: list[str] | None = None) -> "ModelSpace":
        """The built-in families whose packages are installed (all five with `pip install ceknn[boosting]`)."""
        names = [n for n in DEFAULT_FAMILIES if installed(n)]
        if include is not None:
            unknown = set(include) - set(DEFAULT_FAMILIES)
            if unknown:
                raise ValueError(f"unknown families {sorted(unknown)}: use ModelSpace({{...}}) for custom ones")
            names = [n for n in names if n in include]
            absent = [n for n in include if n not in names]
            if absent:
                raise ImportError(f"families not installed: {absent}")
        names = [n for n in names if not exclude or n not in exclude]
        return cls({n: copy.deepcopy(DEFAULT_FAMILIES[n]) for n in names})

    def add(self, name: str, factory, params: dict | None = None, weight: float = 1.0,
            handles_nan: bool = False) -> "ModelSpace":
        """Adds a custom family; `factory(params, seed, n_jobs)` returns an unfitted classifier."""
        self.families[name] = {"weight": weight, "params": dict(params or {}), "factory": factory,
                               "handles_nan": bool(handles_nan)}
        return self

    def set_weight(self, name: str, weight: float) -> "ModelSpace":
        self.families[name]["weight"] = float(weight)
        return self

    def sample(self, rng: np.random.Generator) -> tuple[str, dict]:
        names = list(self.families)
        wts = np.array([float(self.families[k].get("weight", 1)) for k in names])
        fam = names[rng.choice(len(names), p=wts / wts.sum())]
        return fam, sample_params(self.families[fam]["params"], rng)

    def to_dict(self) -> dict:
        """JSON-safe description (a custom factory is recorded by name only)."""
        out = {}
        for n, s in self.families.items():
            d = {k: v for k, v in s.items() if k != "factory"}
            if "factory" in s:
                d["custom"] = True
            out[n] = copy.deepcopy(d)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpace":
        """Built-in families only; a custom family cannot be rebuilt without its factory and is left out."""
        fams = {n: {k: v for k, v in s.items() if k != "custom"} for n, s in d.items() if not s.get("custom")}
        fams = {n: s for n, s in fams.items() if installed(n)}
        return cls(fams) if fams else cls.default()


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    params = {}
    for k, spec in space.items():
        if isinstance(spec, dict) and "log" in spec:
            lo, hi = spec["log"]
            params[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif isinstance(spec, dict) and "uniform" in spec:
            params[k] = float(rng.uniform(*spec["uniform"]))
        elif isinstance(spec, dict) and "int" in spec:
            lo, hi = spec["int"]
            params[k] = int(rng.integers(lo, hi + 1))
        elif isinstance(spec, dict) and "choice" in spec:
            params[k] = spec["choice"][rng.integers(len(spec["choice"]))]
        else:
            params[k] = spec
    return params


def random_masks(n: int, G: int, rng: np.random.Generator) -> np.ndarray:
    """Mask of present groups: per row p ~ U(0, 1) and each group present with probability 1 − p."""
    p = rng.uniform(0, 1, size=(n, 1))
    return rng.uniform(0, 1, size=(n, G)) >= p


class Surrogate:
    def __init__(self, family: str, params: dict, col_group: np.ndarray, n_groups: int, n_classes: int,
                 seed: int = 0, n_jobs: int = 4, spline_cols: np.ndarray | None = None, space: ModelSpace | None = None):
        self.family = family
        self.params = dict(params)
        self.col_group = np.asarray(col_group)
        self.G = int(n_groups)
        self.C = int(n_classes)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.spline_cols = None if spline_cols is None else np.asarray(spline_cols, dtype=bool)
        spec = (space.families.get(family, {}) if space is not None else {})
        self.factory = spec.get("factory")
        self.nan_absent = family in TREE_FAMILIES or bool(spec.get("handles_nan"))
        self.fit_seconds = float("nan")

    def _apply_mask(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        absent_cols = ~mask[:, self.col_group]
        if self.nan_absent:
            Xm = X.astype(np.float32, copy=True)
            Xm[absent_cols] = np.nan
            return Xm
        Xm = np.where(absent_cols, 0.0, X).astype(np.float32)
        if self.family == "spline_logreg" and getattr(self, "spline", None) is not None and self.spline_cols.any():
            Z = self.spline.transform(Xm[:, self.spline_cols]).astype(np.float32)
            Xm = np.hstack([Z, Xm[:, ~self.spline_cols]])
        return np.hstack([Xm, (~mask).astype(np.float32)])

    def fit(self, X: np.ndarray, y: np.ndarray, augment_copies: int = 2) -> "Surrogate":
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        if self.family == "spline_logreg":
            from sklearn.preprocessing import SplineTransformer

            if self.spline_cols is None:
                self.spline_cols = np.zeros(X.shape[1], dtype=bool)
            self.spline = None
            if self.spline_cols.any():
                self.spline = SplineTransformer(n_knots=int(self.params.get("n_knots", 6)),
                                                degree=int(self.params.get("degree", 3)), knots="quantile",
                                                extrapolation="constant").fit(X[:, self.spline_cols])
        masks = [np.ones((n, self.G), dtype=bool)] + [random_masks(n, self.G, rng) for _ in range(augment_copies)]
        Xa = np.vstack([self._apply_mask(X, m) for m in masks])
        ya = np.tile(np.asarray(y), len(masks))
        self.classes_ = np.unique(ya)
        lut = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([lut[v] for v in ya])
        self.model = self._make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Xa, yi)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def _make(self):
        p = self.params
        if self.factory is not None:
            return self.factory(p, self.seed, self.n_jobs)
        if self.family in ("logreg", "spline_logreg"):
            from sklearn.linear_model import LogisticRegression

            pen = p.get("penalty", "l2")
            if pen == "l2":
                return LogisticRegression(C=p["C"], max_iter=int(p.get("max_iter", 1000)))
            return LogisticRegression(C=p["C"], penalty=pen, solver="saga", max_iter=int(p.get("max_iter", 1000)),
                                      l1_ratio=float(p["l1_ratio"]) if pen == "elasticnet" else None,
                                      random_state=self.seed)
        if self.family == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            mf = p["max_features"]
            return RandomForestClassifier(n_estimators=int(p["n_estimators"]), max_depth=p["max_depth"],
                                          min_samples_leaf=int(p["min_samples_leaf"]),
                                          max_features=mf if isinstance(mf, str) else float(mf),
                                          n_jobs=self.n_jobs, random_state=self.seed)
        if self.family == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(n_estimators=int(p["n_estimators"]), learning_rate=p["learning_rate"],
                                      num_leaves=int(p["num_leaves"]), min_child_samples=int(p["min_child_samples"]),
                                      subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
                                      reg_lambda=p["reg_lambda"], n_jobs=self.n_jobs, random_state=self.seed,
                                      verbose=-1)
        if self.family == "xgboost":
            import xgboost as xgb

            return xgb.XGBClassifier(n_estimators=int(p["n_estimators"]), learning_rate=p["learning_rate"],
                                     max_depth=int(p["max_depth"]), min_child_weight=p["min_child_weight"],
                                     subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                                     reg_lambda=p["reg_lambda"], tree_method="hist", n_jobs=self.n_jobs,
                                     random_state=self.seed, verbosity=0)
        raise ValueError(f"unknown family {self.family!r}")

    def predict_proba(self, X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.ones((X.shape[0], self.G), dtype=bool)
        pr = self.model.predict_proba(self._apply_mask(X, mask))
        out = np.full((X.shape[0], self.C), 1e-6)
        out[:, self.classes_] = pr
        return out / out.sum(1, keepdims=True)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))))
