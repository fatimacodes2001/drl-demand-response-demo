"""Aggregate results/<variant>/seed*/ into summary.md and learning_curves.png and log them to MLflow.

Standard deviations are sample standard deviations (ddof=1) across seeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from dr import plotting, tracking
from dr.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "results"))
    out = Path(p.parse_args().out)

    variants = sorted(d for d in out.iterdir() if d.is_dir() and list(d.glob("seed*/final.json")))
    cfgs = {d.name: [yaml.safe_load(f.read_text()) for f in sorted(d.glob("seed*/config.yaml"))] for d in variants}
    cfg0 = cfgs[variants[0].name][0]
    shared = lambda c: (c.get("task", "dr"), c["env"], c["eval"].get("val_days"), c["eval"].get("test_days"),
                        c["train"].get("n_train_days"))
    mismatched = [n for n, cs in cfgs.items() for c in cs if shared(c) != shared(cfg0)]
    if mismatched:
        raise SystemExit(f"variants {sorted(set(mismatched))} differ from {variants[0].name} in task, env or eval "
                         "settings; baselines and tables would not be comparable")
    task = make_task(cfg0)
    base = task.baselines()
    (out / "baselines.json").write_text(json.dumps(base, indent=2))

    cols = list(task.table_metrics)
    rows = [("method", "seeds", *[c[1] for c in cols], "env steps", "train wall (s)")]
    for k in task.baseline_order:
        rows.append((k, "-", *[format(base[k][c[0]], c[2]) for c in cols], "-", "-"))
    curves, metrics = {}, {}
    for d in variants:
        fin = [json.loads(f.read_text()) for f in sorted(d.glob("seed*/final.json"))]
        cells = []
        for key, _, fmt in cols:
            v = np.array([f[key] for f in fin], dtype=float)
            cells.append(f"{v.mean():{fmt}} ± {v.std(ddof=1):{fmt}}" if len(v) > 1 else f"{v.mean():{fmt}}")
        rows.append((d.name, str(len(fin)), *cells, str(fin[0]["env_steps"]),
                     f"{np.mean([f['train_wall_s'] for f in fin]):.0f}"))
        curves[d.name] = [[json.loads(line) for line in f.read_text().splitlines()]
                          for f in sorted(d.glob("seed*/metrics.jsonl"))]
        key = np.array([f[task.key_metric] for f in fin], dtype=float)
        metrics[f"{d.name}_{task.key_metric}_mean"] = float(key.mean())
        if len(key) > 1:
            metrics[f"{d.name}_{task.key_metric}_std"] = float(key.std(ddof=1))

    fig = plotting.compare_curves(curves, task.reference_lines(), task.key_metric, task.key_label)
    fig.savefig(out / "learning_curves.png", dpi=150)

    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    fmt_row = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"
    table = [fmt_row(rows[0]), "|" + "|".join("-" * (x + 2) for x in w) + "|", *map(fmt_row, rows[1:])]
    summary = out / "summary.md"
    summary.write_text(f"Task `{task.name}`, held-out test split, mean ± sample std across seeds.\n\n"
                       + "\n".join(table) + "\n")
    print(summary.read_text())
    tracking.log_report(f"summary:{out.name}", "summary", metrics, {"learning_curves.png": fig},
                        [summary, out / "baselines.json"], tracker=cfg0.get("tracker", "mlflow"),
                        experiment=cfg0.get("experiment", tracking.DEFAULT_EXPERIMENT), out_dir=out)


if __name__ == "__main__":
    main()
