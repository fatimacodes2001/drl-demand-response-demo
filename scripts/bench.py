"""Measure learner-update throughput on each available device and log the results to MLflow.

    python scripts/bench.py [--profile]

The env and tree search are CPU-bound Python, so a GPU only helps the network updates, and only once nets or
batches are large enough to amortise launch and transfer overhead. This script finds that crossover.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from dr import plotting, tracking
from dr.agents.common import mlp
from dr.env import OBS_DIM, N_ACTIONS

ROOT = Path(__file__).resolve().parents[1]


def devices():
    """Available accelerators plus the CPU."""
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + \
           (["mps"] if torch.backends.mps.is_available() else [])


def bench(dev, hidden, batch, layers, iters):
    """Gradient updates per second for an MLP regression step."""
    net = mlp(OBS_DIM, N_ACTIONS, hidden, layers).to(dev)
    opt = torch.optim.Adam(net.parameters())
    x, y = torch.randn(batch, OBS_DIM, device=dev), torch.randn(batch, N_ACTIONS, device=dev)

    def step():
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    sync = {"cuda": torch.cuda.synchronize, "mps": getattr(getattr(torch, "mps", None), "synchronize", None)}.get(dev) \
        or (lambda: None)
    for _ in range(20):
        step()
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        step()
    sync()
    return iters / (time.perf_counter() - t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    p.add_argument("--no-track", action="store_true", help="do not log to MLflow")
    p.add_argument("--iters", type=int, default=200)
    a = p.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    rows = []
    print(f"{'device':8}{'hidden':>8}{'layers':>8}{'batch':>8}{'updates/s':>12}{'samples/s':>14}")
    for hidden, layers in ((128, 2), (512, 4), (2048, 4)):
        for batch in (128, 1024, 8192):
            for dev in devices():
                ups = bench(dev, hidden, batch, layers, a.iters)
                rows.append(dict(device=dev, hidden=hidden, layers=layers, batch=batch, updates_per_s=ups))
                print(f"{dev:8}{hidden:>8}{layers:>8}{batch:>8}{ups:>12.0f}{ups * batch:>14.0f}")
    (out / "bench.json").write_text(json.dumps(rows, indent=2))
    files = [out / "bench.json"]

    if a.profile:
        dev = devices()[-1]
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
        with profile(activities=acts) as prof:
            bench(dev, 512, 1024, 4, a.iters)
        prof.export_chrome_trace(str(out / "bench_trace.json"))
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
        files.append(out / "bench_trace.json")

    tracking.log_report("bench", "bench", figures={"bench.png": plotting.bench(rows)}, files=files,
                        tracker="none" if a.no_track else "mlflow", out_dir=out)


if __name__ == "__main__":
    main()
