"""One experiment (config, seed) produces a self-describing run directory and an MLflow run."""
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import yaml

from . import evaluate as ev
from . import plotting
from .agents import AGENTS
from .agents.common import get_device, seed_everything
from .fingerprint import fingerprint
from .tracking import Tracker


ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    """Output of a git command at the repository root, or None outside a repository."""
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, cwd=ROOT).decode().strip()
    except Exception:
        return None


def _meta(fp, device):
    """Provenance record written next to every run."""
    return dict(**fp, git_sha=_git("rev-parse", "HEAD") or "unknown",
                git_dirty=bool(_git("status", "--porcelain", "--", "src", "configs", "scripts")),
                python=platform.python_version(), host=platform.node(),
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                started=time.strftime("%Y-%m-%dT%H:%M:%S"))


def run(cfg, out_dir, resume=False, force=False):
    """Train one (config, seed), evaluate on val during training and on test at the end.

    A finished run with the same fingerprint is skipped unless `force` is set. With `resume`, training
    continues from the saved checkpoint.
    """
    out = Path(out_dir)
    device = get_device(cfg["device"])
    fp = fingerprint(cfg, device)
    final = out / "final.json"
    if final.exists() and not force and json.loads(final.read_text()).get("fingerprint") == fp["fingerprint"]:
        print(f"[skip] {out} already done with identical fingerprint ({fp['fingerprint']})")
        return json.loads(final.read_text())

    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    seed_everything(seed)
    meta = _meta(fp, device)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    tracker = Tracker(cfg, fp, out, tags={"git_sha": meta["git_sha"], "git_dirty": str(meta["git_dirty"])})

    agent = AGENTS[cfg["algo"]["name"]](cfg, device, seed)
    ckpt, metrics = out / "ckpt.pt", out / "metrics.jsonl"
    if resume and ckpt.exists():
        agent.load_state_dict(torch.load(ckpt, weights_only=False))
        print(f"[resume] from env_steps={agent.env_steps}")
    else:
        metrics.write_text("")

    ecfg, total = cfg["eval"], cfg["train"]["total_steps"]
    val_days = ev.days_for("val", ecfg["val_days"])
    next_eval = (agent.env_steps // ecfg["every_steps"] + 1) * ecfg["every_steps"]
    t0, last_logged = time.time(), [-1]

    def log_eval(stats):
        val = ev.summarise(ev.evaluate(agent.policy(), val_days, cfg["env"]), val_days, cfg["env"])
        row = dict(env_steps=agent.env_steps, wall_s=round(time.time() - t0, 1), **stats,
                   **{f"val_{k}": v for k, v in val.items()})
        with metrics.open("a") as f:
            f.write(json.dumps(row) + "\n")
        tracker.log_metrics(row, row["env_steps"])
        print(f"[{cfg['name']} s{seed}] steps={row['env_steps']:>7} train={row.get('train_return', float('nan')):7.2f} "
              f"val_saving={row['val_saving_per_day']:6.2f} ({row['val_pct_of_oracle_saving']:5.1f}% of oracle) "
              f"{row['wall_s']:.0f}s", flush=True)
        torch.save(agent.state_dict(), ckpt)
        last_logged[0] = agent.env_steps

    # the first evaluation is the untrained "before" policy
    log_eval({})
    stats = {}
    while agent.env_steps < total:
        stats = agent.train_chunk()
        if agent.env_steps >= next_eval:
            log_eval(stats)
            next_eval += ecfg["every_steps"]
    if agent.env_steps != last_logged[0]:
        log_eval(stats)

    test_days = ev.days_for("test", ecfg["test_days"])
    res = ev.summarise(ev.evaluate(agent.policy(), test_days, cfg["env"]), test_days, cfg["env"])
    res.update(**fp, algo=cfg["algo"]["name"], seed=seed, env_steps=agent.env_steps,
               train_wall_s=round(time.time() - t0, 1))
    final.write_text(json.dumps(res, indent=2))
    torch.save(agent.state_dict(), out / "final.pt")

    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    ref = {"rule_based": ev.summarise(ev.evaluate(ev.rule_based, val_days, cfg["env"]), val_days,
                                      cfg["env"])["pct_of_oracle_saving"], "no_control": 0.0}
    tracker.log_figure(plotting.run_curves(rows, ref), "training_curve.png")
    day = test_days[0]
    tracker.log_figure(plotting.dispatch(cfg["env"], {cfg["name"]: agent.policy()}, day), f"dispatch_day{day}.png")
    tracker.log_files([out / f for f in ("config.yaml", "meta.json", "metrics.jsonl", "final.json")])
    tracker.finish(res)
    return res
