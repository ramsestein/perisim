"""What a CE-kNN model reads: its inputs (name and type), its output (name and classes) and, optionally, the column
that groups rows of the same patient and the column of the site.

Input types:
- ``continuous``: numeric; missing values are allowed (median imputation + missing indicator);
- ``ordinal``: ordered levels, numeric or given as an ordered list of labels (``levels``);
- ``binary``: 0/1 or boolean;
- ``nominal``: unordered categories (any value, kept as text).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TYPES = ("continuous", "ordinal", "binary", "nominal")


@dataclass
class Variable:
    name: str
    type: str
    levels: list | None = None

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"input {self.name!r}: unknown type {self.type!r} (one of {TYPES})")
        if self.levels is not None and self.type != "ordinal":
            raise ValueError(f"input {self.name!r}: levels are only for ordinal inputs")

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.type}
        if self.levels is not None:
            d["levels"] = list(self.levels)
        return d


def infer_type(s: pd.Series) -> str:
    """Type of a column when none is given: boolean or numeric with two values → binary, other numeric →
    continuous, anything else → nominal. Ordinal is never inferred."""
    if pd.api.types.is_bool_dtype(s):
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.unique(s.dropna())
        return "binary" if len(vals) <= 2 and set(np.asarray(vals, dtype=float)) <= {0.0, 1.0} else "continuous"
    return "nominal"


def _as_variable(name: str, spec) -> Variable:
    if isinstance(spec, Variable):
        return spec
    if isinstance(spec, str):
        return Variable(name, spec)
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
        return Variable(name, spec[0], list(spec[1]))
    if isinstance(spec, dict):
        return Variable(name, spec["type"], spec.get("levels"))
    raise ValueError(f"input {name!r}: cannot read the type {spec!r}")


@dataclass
class Schema:
    inputs: list[Variable]
    output: str
    classes: list | None = None
    group: str | None = None
    site: str | None = None
    pending: list[str] = field(default_factory=list)       # names whose type is inferred at fit

    @classmethod
    def build(cls, inputs, output: str, *, classes=None, group: str | None = None, site: str | None = None,
              types: dict | None = None) -> "Schema":
        """`inputs`: a list of column names (types inferred at fit unless given in `types`), or a mapping
        name → type, (type, levels) or Variable."""
        types = dict(types or {})
        if isinstance(inputs, dict):
            types = {**inputs, **types}
            names = list(inputs)
        else:
            names = list(inputs)
        if output in names:
            raise ValueError(f"the output {output!r} is also listed as an input")
        for c in (group, site):
            if c is not None and c in names:
                raise ValueError(f"{c!r} is the group or site column and cannot be an input")
        vs, pending = [], []
        for n in names:
            if n in types:
                vs.append(_as_variable(n, types[n]))
            else:
                vs.append(Variable(n, "continuous"))      # placeholder until the data are seen
                pending.append(n)
        return cls(vs, output, None if classes is None else list(classes), group, site, pending)

    def resolve(self, df: pd.DataFrame) -> None:
        """Checks the columns and infers the pending types and the classes from the training data."""
        need = [v.name for v in self.inputs] + [self.output] + [c for c in (self.group, self.site) if c]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"columns not in the data: {missing}")
        for i, v in enumerate(self.inputs):
            if v.name in self.pending:
                self.inputs[i] = Variable(v.name, infer_type(df[v.name]))
        self.pending = []
        if self.classes is None:
            vals = pd.unique(df[self.output].dropna())
            try:
                self.classes = sorted(vals.tolist())
            except TypeError:
                self.classes = sorted(vals.tolist(), key=str)
        if len(self.classes) < 2:
            raise ValueError(f"the output {self.output!r} has fewer than two classes")

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def labels(self, df: pd.DataFrame) -> np.ndarray:
        """Class index of each row (an error if a row has no output or a class not declared)."""
        lut = {c: i for i, c in enumerate(self.classes)}
        y = df[self.output].to_numpy()
        out = np.empty(len(y), dtype=np.int64)
        for i, v in enumerate(y):
            if v is None or (isinstance(v, float) and v != v) or v not in lut:
                raise ValueError(f"row {i}: output {v!r} is missing or not one of the classes {self.classes}")
            out[i] = lut[v]
        return out

    def to_dict(self) -> dict:
        return {"inputs": [v.to_dict() for v in self.inputs], "output": self.output,
                "classes": None if self.classes is None else [_jsonable(c) for c in self.classes],
                "group": self.group, "site": self.site, "pending": list(self.pending)}

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls([Variable(v["name"], v["type"], v.get("levels")) for v in d["inputs"]], d["output"],
                   d.get("classes"), d.get("group"), d.get("site"), list(d.get("pending", [])))


def _jsonable(x):
    return x.item() if isinstance(x, np.generic) else x


# ---------------------------------------------------------------------------- the data, as the encoder reads them
class Table:
    """Columns of the inputs as arrays: `num` (float, NaN = missing) for continuous / ordinal / binary inputs and
    `cat` (object) for nominal ones; `groups` holds each row's group (patient) and `sites` its site."""

    def __init__(self, num: dict, cat: dict, variables: list[Variable], groups: np.ndarray, sites: np.ndarray | None):
        self.num, self.cat, self.variables = num, cat, variables
        self.groups, self.sites = groups, sites
        self.n = len(groups)

    def __len__(self) -> int:
        return self.n

    def take(self, idx) -> "Table":
        idx = np.asarray(idx)
        return Table({k: v[idx] for k, v in self.num.items()}, {k: v[idx] for k, v in self.cat.items()},
                     self.variables, self.groups[idx], None if self.sites is None else self.sites[idx])

    @classmethod
    def from_frame(cls, df: pd.DataFrame, schema: Schema) -> "Table":
        num, cat = {}, {}
        for v in schema.inputs:
            if v.name not in df.columns:
                raise KeyError(f"input {v.name!r} is not in the data")
            s = df[v.name]
            if v.type == "nominal":
                cat[v.name] = s.to_numpy(dtype=object)
            elif v.type == "ordinal" and v.levels is not None and not _numeric(v.levels):
                pos = {lv: float(i) for i, lv in enumerate(v.levels)}
                vals = s.to_numpy(dtype=object)
                bad = [x for x in pd.unique(s.dropna()) if x not in pos]
                if bad:
                    raise ValueError(f"input {v.name!r}: values {bad[:5]} are not among its levels")
                num[v.name] = np.array([pos.get(x, np.nan) for x in vals], dtype=float)
            else:
                try:
                    x = pd.to_numeric(s, errors="raise").to_numpy(dtype=float)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"input {v.name!r} ({v.type}) must be numeric") from e
                if v.type == "binary":
                    ok = np.isfinite(x)
                    if not np.isin(x[ok], (0.0, 1.0)).all():
                        raise ValueError(f"input {v.name!r} is binary but has values other than 0 / 1")
                num[v.name] = x
        groups = (df[schema.group].astype(str).to_numpy(dtype=object) if schema.group
                  else np.arange(len(df)).astype(str).astype(object))
        sites = df[schema.site].astype(str).to_numpy(dtype=object) if schema.site else None
        return cls(num, cat, schema.inputs, groups, sites)


def _numeric(levels: list) -> bool:
    try:
        [float(x) for x in levels]
        return not any(isinstance(x, str) for x in levels)
    except (TypeError, ValueError):
        return False
