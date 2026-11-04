"""Aggregate results/<variant>/seed*/ into summary.md and learning_curves.png and log them to MLflow."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from dr import evaluate as ev
from dr import plotting, tracking

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "results"))
    out = Path(p.parse_args().out)

    variants = sorted(d for d in out.iterdir() if d.is_dir() and list(d.glob("seed*/final.json")))
    cfg0 = yaml.safe_load((variants[0] / "seed0/config.yaml").read_text())
    days = ev.days_for("test", cfg0["eval"]["test_days"])
    base = ev.baselines(days, cfg0["env"])
    (out / "baselines.json").write_text(json.dumps(base, indent=2))

    rows = [("method", "seeds", "test saving $/day", "% of oracle saving", "env steps", "train wall (s)")]
    for k in ("no_control", "random", "rule_based"):
        b = base[k]
        rows.append((k, "-", f"{b['saving_per_day']:.2f}", f"{b['pct_of_oracle_saving']:.1f}", "-", "-"))
    curves, metrics = {}, {}
    for d in variants:
        fin = [json.loads(f.read_text()) for f in sorted(d.glob("seed*/final.json"))]
        pct = np.array([f["pct_of_oracle_saving"] for f in fin])
        sav = np.array([f["saving_per_day"] for f in fin])
        rows.append((d.name, str(len(fin)), f"{sav.mean():.2f} ± {sav.std():.2f}",
                     f"{pct.mean():.1f} ± {pct.std():.1f}", str(fin[0]["env_steps"]),
                     f"{np.mean([f['train_wall_s'] for f in fin]):.0f}"))
        curves[d.name] = [[json.loads(line) for line in f.read_text().splitlines()]
                          for f in sorted(d.glob("seed*/metrics.jsonl"))]
        metrics[f"{d.name}_pct_of_oracle_mean"], metrics[f"{d.name}_pct_of_oracle_std"] = float(pct.mean()), float(pct.std())

    fig = plotting.compare_curves(curves, {k: base[k]["pct_of_oracle_saving"] for k in ("rule_based", "no_control")})
    fig.savefig(out / "learning_curves.png", dpi=150)

    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    fmt = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"
    table = [fmt(rows[0]), "|" + "|".join("-" * (x + 2) for x in w) + "|", *map(fmt, rows[1:])]
    summary = out / "summary.md"
    summary.write_text(f"Held-out test days ({len(days)}), mean ± std across seeds.\n\n" + "\n".join(table) + "\n")
    print(summary.read_text())
    tracking.log_report("summary", "summary", metrics, {"learning_curves.png": fig},
                        [summary, out / "baselines.json"], out_dir=out)


if __name__ == "__main__":
    main()
