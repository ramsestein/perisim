"""§6.2b model-transfer validations: patient-level hold-out, pooled case bases, job list."""

import numpy as np

from perisim.features.table import FeatureTable
from perisim.transfer import concat_ft, patient_holdout, transfer_jobs


def test_holdout_is_patient_level_and_has_the_fraction():
    rng = np.random.default_rng(0)
    subjects = np.array([f"MIMIC-{i}" for i in rng.integers(0, 500, 3000)])
    held = patient_holdout(subjects, 0.2, seed=7)
    assert not (set(subjects[held]) & set(subjects[~held]))          # no patient on both sides
    frac = len(set(subjects[held])) / len(set(subjects))
    assert abs(frac - 0.2) < 0.01
    assert (patient_holdout(subjects, 0.2, seed=7) == held).all()    # deterministic: the same split for every source


def test_concat_ft_keeps_features_aligned():
    a = FeatureTable(np.array(["INSPIRE-1", "INSPIRE-2"]), np.array(["INSPIRE-10", "INSPIRE-20"]), [],
                     {"age": np.array([60.0, 70.0])}, {"group": np.array(["cardiac", "neuro"], dtype=object)})
    b = FeatureTable(np.array(["EICU-1"]), np.array(["EICU-9"]), [],
                     {"age": np.array([50.0])}, {"group": np.array(["ortho"], dtype=object)})
    ab = concat_ft(a, b)
    assert len(ab) == 3 and ab.num["age"].tolist() == [60.0, 70.0, 50.0]
    assert ab.cat["group"].tolist() == ["cardiac", "neuro", "ortho"] and ab.subjects[-1] == "EICU-9"


def test_jobs_per_target(cfg):
    every = ["inspire", "mimic", "eicu"]
    assert transfer_jobs(cfg, every, ["t2"]) == [("t2", "mimic", "inspire"), ("t2", "eicu", "inspire"),
                                                 ("t2", "mimic", "multisource")]
    assert transfer_jobs(cfg, every, ["t3"]) == [("t3", "mimic", "inspire")]       # eICU has no full T1
    assert transfer_jobs(cfg, ["inspire", "mimic"], ["t2"]) == [("t2", "mimic", "inspire")]   # no eICU: no multi-source
