"""One experiment (config, seed) produces a self-describing run directory and an MLflow run."""
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import yaml

from . import plotting
from .agents import AGENTS
from .agents.common import get_device, seed_everything
from .fingerprint import fingerprint
from .tasks import make_task
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


def run(cfg, out_dir, resume=False, force=False, stop_at=None):
    """Train one (config, seed), evaluate on val during training and on test at the end.

    A finished run with the same fingerprint is skipped unless `force` is set. With `resume`, training
    continues from the saved checkpoint. `stop_at` interrupts after that many env steps (checkpoint
    written, no final evaluation, returns None) so an interruption can be simulated without changing
    the config.
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
    ckpt, run_id_file = out / "ckpt.pt", out / "mlflow_run_id"
    resuming = resume and ckpt.exists()
    tracker = Tracker(cfg, fp, out, tags={"git_sha": meta["git_sha"], "git_dirty": str(meta["git_dirty"])},
                      run_id=run_id_file.read_text().strip() if resuming and run_id_file.exists() else None)
    if tracker.run_id:
        run_id_file.write_text(tracker.run_id)
    try:
        return _train(cfg, out, device, seed, tracker, fp, resuming, stop_at)
    except BaseException:
        tracker.abort()
        raise


def _train(cfg, out, device, seed, tracker, fp, resuming, stop_at):
    """Training loop body; see `run`."""
    ckpt, metrics, final = out / "ckpt.pt", out / "metrics.jsonl", out / "final.json"
    task = make_task(cfg)
    if cfg["algo"]["name"] not in task.algorithms:
        raise ValueError(f"task {task.name} supports {task.algorithms}, not {cfg['algo']['name']}")
    agent = AGENTS[cfg["algo"]["name"]](cfg, device, seed, task)
    # the policy from the previous evaluation, for tasks that measure improvement against it
    prev = {"weights": None}
    if resuming:
        # the checkpoint holds numpy arrays and RNG states, so it is loaded as a trusted local file
        state = torch.load(ckpt, weights_only=False)
        agent.load_state_dict(state["agent"])
        prev["weights"] = state["prev_weights"]
        print(f"[resume] from env_steps={agent.env_steps}")
    else:
        metrics.write_text("")

    def save_ckpt():
        torch.save({"agent": agent.state_dict(), "prev_weights": prev["weights"]}, ckpt)

    ecfg, total = cfg["eval"], cfg["train"]["total_steps"]
    next_eval = (agent.env_steps // ecfg["every_steps"] + 1) * ecfg["every_steps"]
    t0, last_logged = time.time(), [-1]

    def log_eval(stats):
        val = task.evaluate(agent.policy(), "val", prev_policy=prev["weights"])
        prev["weights"] = agent.policy_state_dict()
        row = dict(env_steps=agent.env_steps, wall_s=round(time.time() - t0, 1), **stats,
                   **{f"val_{k}": v for k, v in val.items()})
        with metrics.open("a") as f:
            f.write(json.dumps(row) + "\n")
        tracker.log_metrics(row, row["env_steps"])
        print(f"[{cfg['name']} s{seed}] steps={row['env_steps']:>7} train={row.get('train_return', float('nan')):7.2f} "
              f"val {task.key_label}={val[task.key_metric]:6.1f} {row['wall_s']:.0f}s", flush=True)
        save_ckpt()
        last_logged[0] = agent.env_steps

    # the first evaluation is the untrained "before" policy
    if not resuming:
        log_eval({})
    stats = {}
    while agent.env_steps < total:
        stats = agent.train_chunk()
        if agent.env_steps >= next_eval:
            log_eval(stats)
            next_eval += ecfg["every_steps"]
        if stop_at is not None and agent.env_steps >= stop_at:
            save_ckpt()
            print(f"[stop] interrupted at env_steps={agent.env_steps}, checkpoint written")
            tracker.finish()
            return None
    if agent.env_steps != last_logged[0]:
        log_eval(stats)

    res = task.evaluate(agent.policy(), "test")
    res.update(**fp, algo=cfg["algo"]["name"], seed=seed, env_steps=agent.env_steps,
               train_wall_s=round(time.time() - t0, 1))
    final.write_text(json.dumps(res, indent=2))
    torch.save(agent.policy_state_dict(), out / "final.pt")

    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    tracker.log_figure(plotting.run_curves(rows, task.reference_lines(), task.key_metric, task.key_label),
                       "training_curve.png")
    task.plots(agent, tracker.log_figure)
    tracker.log_files([out / f for f in ("config.yaml", "meta.json", "metrics.jsonl", "final.json")])
    tracker.finish(res)
    return res
