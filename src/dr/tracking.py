"""MLflow experiment tracking behind a small interface that degrades to a no-op."""
import os
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import mlflow
except ImportError:
    mlflow = None

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
DEFAULT_EXPERIMENT = "drl-demand-response"


def init_store(experiment=DEFAULT_EXPERIMENT):
    """Create the tracking database and experiment once, before parallel runs start using it."""
    if mlflow is None:
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
    mlflow.set_experiment(experiment)


def flatten(d, prefix=""):
    """Flatten nested dicts to dotted keys for MLflow params."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


class Tracker:
    """Logs one run's params, tags, metrics, figures and files; every method is a no-op when disabled."""

    def __init__(self, cfg, fp, run_dir, run_name=None, tags=None):
        self.run_dir = Path(run_dir)
        self.active = False
        if cfg.get("tracker", "mlflow") != "mlflow" or mlflow is None:
            return
        try:
            init_store(cfg.get("experiment", DEFAULT_EXPERIMENT))
            mlflow.start_run(run_name=run_name or f"{cfg['name']}-seed{cfg['seed']}-{fp['fingerprint'][:6]}")
            mlflow.log_params(dict(flatten(cfg)))
            mlflow.set_tags({**fp, "algo": cfg["algo"]["name"], "variant": cfg["name"],
                             "seed": cfg["seed"], **(tags or {})})
            self.active = True
        except Exception as e:
            print(f"[WARN] MLflow tracking failed, this run is NOT logged: {e!r}", flush=True)

    def log_metrics(self, row, step):
        """Log the numeric, finite fields of `row` at `step`."""
        if self.active:
            mlflow.log_metrics({k: v for k, v in row.items()
                                if isinstance(v, (int, float)) and v == v and k != "env_steps"}, step=step)

    def log_figure(self, fig, name):
        """Save `fig` under the run directory's plots/ and attach it to the MLflow run."""
        path = self.run_dir / "plots" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        if self.active:
            mlflow.log_artifact(str(path), "plots")
        return path

    def log_files(self, paths, subdir=None):
        """Attach existing files as artifacts."""
        if self.active:
            for p in paths:
                mlflow.log_artifact(str(p), subdir)

    def finish(self, final=None):
        """Log final metrics (prefixed `test_`) and close the MLflow run."""
        if self.active:
            mlflow.log_metrics({f"test_{k}": v for k, v in (final or {}).items() if isinstance(v, (int, float))})
            mlflow.end_run()
            self.active = False


def log_report(name, kind, metrics=None, figures=None, files=(), tracker="mlflow",
               experiment=DEFAULT_EXPERIMENT, out_dir=None):
    """Log a sweep-level or benchmark report as its own MLflow run; returns False if tracking is off."""
    if mlflow is None or tracker != "mlflow":
        return False
    try:
        init_store(experiment)
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"kind": kind, "out_dir": str(out_dir or ROOT / "results")})
            mlflow.log_metrics(metrics or {})
            for fname, fig in (figures or {}).items():
                path = Path(out_dir or ROOT / "results") / fname
                fig.savefig(path, dpi=150)
                mlflow.log_artifact(str(path), "plots")
            for f in files:
                mlflow.log_artifact(str(f))
        return True
    except Exception as e:
        print(f"[WARN] MLflow report not logged: {e!r}", flush=True)
        return False
