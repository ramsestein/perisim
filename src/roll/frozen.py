"""The freeze of experiment `roll` (Phase 3) and its check (Phase 4, at the start and the end of the single run).

The record (<workdir>/experiments/roll/state/freeze.json) holds:
- `roll`'s code hash (src/roll + `hypo`'s code hash, which covers the perisim code it uses) and the hash of
  config/roll.yaml (line endings normalised);
- the freezes it stands on: `hypo`'s (its own check: code, configuration, folds, Phase 2 comparators, final
  artefacts) and T2's (code and configuration hashes), plus the SHA-256 of T2's `final_model.pkl`;
- the digest of every stream of the design (every stream, hospital, N and order): the same patients in the same
  order at the run;
- the git tag and commit, and the SHA-256 of the pre-registration.
The freeze refuses to run when real results already exist, and on a repository with uncommitted changes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time

from .config import REPO, Stop, base_workdir, code_hash, frozen_guard, workdir


def state_dir():
    return workdir() / "state"


def runs_dir():
    return workdir() / "runs"


def sha(path, n: int = 16) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n") if path.suffix in (".md", ".yaml") else
                          path.read_bytes()).hexdigest()[:n]


def streams_digest(cfg: dict, cohorts: dict) -> dict:
    """The digest of every (stream, hospital, N, order) of the design, and one hash over them all."""
    from .run import jobs
    from .streams import build

    out = {}
    for j in jobs(cfg, cohorts):
        for r in range(j["orders"]):
            st = build(j["stream"], j["spec"], cohorts, cfg, r, j["N"], j["hospital"])
            out[f"{j['stream']}|{j['label']}|N{j['N']}|o{r}"] = st.digest() if st is not None else "skipped"
    h = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:16]
    return {"all": h, "n": len(out), "skipped": sum(v == "skipped" for v in out.values())}


def hashes(cfg: dict, cohorts: dict) -> dict:
    g = frozen_guard()
    t2_model = base_workdir() / "experiments" / "t2" / "predictions" / "main" / "external" / "t2" / "final_model.pkl"
    return {"code": code_hash(), "config": sha(REPO / "config" / "roll.yaml"), "hypo": g["hypo"], "t2": g["t2"],
            "t2_final_model": sha(t2_model), "streams": streams_digest(cfg, cohorts)}


def freeze(registration: str | None, cfg: dict, cohorts: dict, *, no_git: bool = False) -> dict:
    if runs_dir().exists() and any(runs_dir().rglob("log.json")):
        raise Stop("real results already exist: freezing again would allow iterating after seeing them")
    stamp = time.strftime("%Y-%m-%d-%H%M%S")
    tag = f"perisim-freeze-roll-{stamp}"
    git = {}
    if not no_git:
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout
        if dirty.strip():
            raise Stop("the repository has uncommitted changes: commit them before freezing\n" + dirty)
        r = subprocess.run(["git", "-C", str(REPO), "-c", "user.name=perisim", "-c", "user.email=perisim@localhost",
                            "tag", "-a", tag, "-m", "PERISIM freeze, experiment roll (docs/prereg_roll.md)"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise Stop(f"could not create the git tag {tag}: {r.stderr.strip()}")
        git = {"tag": tag, "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                                    capture_output=True, text=True).stdout.strip()}
    rec = {"tag": tag, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "registration": registration or "PENDING", "git": git,
           "hashes": hashes(cfg, cohorts), "prereg_sha": sha(REPO / "docs" / "prereg_roll.md")}
    state_dir().mkdir(parents=True, exist_ok=True)
    (state_dir() / "freeze.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def check(cfg: dict, cohorts: dict) -> dict:
    """The freeze record, after checking that nothing it covers has changed."""
    p = state_dir() / "freeze.json"
    if not p.exists():
        raise Stop("no freeze record: the single run happens only on the frozen version (`python -m roll freeze`)")
    rec = json.loads(p.read_text(encoding="utf-8"))
    now = hashes(cfg, cohorts)
    diff = [k for k in rec["hashes"] if now.get(k) != rec["hashes"][k]]
    if diff:
        raise Stop(f"changed since the freeze {rec['tag']}: {', '.join(diff)}. Restore the frozen version "
                   f"(git checkout {rec['tag']}); a change after results exist is a deviation")
    return rec
