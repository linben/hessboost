#!/usr/bin/env python3
"""Compare histogram training with XGBoost on shared synthetic datasets.

Run with uv and a compiled --hessboost example. Both engines time fresh training
matrix construction plus training; file I/O, warmup, and evaluation are excluded.
`--device cuda` trains both on the GPU: XGBoost needs a CUDA build (the PyPI
wheel; the parity requirement's source build is CPU-only), and the example a
`--features cuda` build. The warmup fit absorbs CUDA context creation and
kernel compilation.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

WORKLOADS = {
    "regression": (100_000, 30, "reg:squarederror", 0, "rmse"),
    "wide_regression": (50_000, 128, "reg:squarederror", 0, "rmse"),
    "binary": (100_000, 30, "binary:logistic", 0, "logloss"),
    "multiclass": (50_000, 30, "multi:softprob", 4, "mlogloss"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_dataset(folder, workload, rounds, rows_override):
    rows, cols, objective, classes, metric = WORKLOADS[workload]
    rows = rows_override or rows
    n_test = max(1, rows // 5)
    rng = np.random.default_rng(1234)
    folder.mkdir()
    for suffix, count in [("", rows), ("_test", n_test)]:
        x = rng.random((count, cols), dtype=np.float32)
        if objective == "binary:logistic":
            margin = 4 * (x[:, 0] - 0.5) - 3 * (x[:, 1] - 0.5) + 2 * (x[:, 2] - 0.5)
            y = (rng.random(count) < 1 / (1 + np.exp(-margin))).astype(np.float32)
        elif classes:
            scores = np.column_stack(
                [3 * x[:, i] - x[:, (i + 1) % classes] for i in range(classes)]
            )
            y = np.argmax(scores + 0.1 * rng.standard_normal(scores.shape), axis=1)
        else:
            y = (
                2 * x[:, 0]
                - 3 * x[:, 1] ** 2
                + 0.5 * x[:, 2]
                + x[:, 3] * x[:, 4]
                + 0.05 * rng.standard_normal(count).astype(np.float32)
            )
        np.asarray(x, dtype="<f4").tofile(folder / f"X{suffix}.bin")
        np.asarray(y, dtype="<f4").tofile(folder / f"y{suffix}.bin")
    meta = dict(
        n_rows=rows,
        n_test=n_test,
        n_cols=cols,
        objective=objective,
        num_class=classes,
        metric=metric,
        num_round=rounds,
        max_depth=6,
        eta=0.1,
        **{"lambda": 1.0},
        max_bin=256,
        base_score=0.5,
        seed=1234,
    )
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def test_score(preds, labels, metric):
    preds, labels = preds.astype(np.float64), labels.astype(np.float64)
    if metric == "rmse":
        return float(np.sqrt(np.mean((preds - labels) ** 2)))
    if metric == "mlogloss":
        preds = preds[np.arange(len(labels)), labels.astype(np.int64)]
        return float(np.mean(-np.log(np.clip(preds, 1e-15, 1 - 1e-15))))
    preds = np.clip(preds, 1e-15, 1 - 1e-15)
    return float(np.mean(-(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))))


def xgboost_batch(folder, meta, threads, repeats, device):
    x = np.fromfile(folder / "X.bin", dtype="<f4").reshape(-1, meta["n_cols"])
    y = np.fromfile(folder / "y.bin", dtype="<f4")
    xt = np.fromfile(folder / "X_test.bin", dtype="<f4").reshape(-1, meta["n_cols"])
    yt = np.fromfile(folder / "y_test.bin", dtype="<f4")
    dtest = xgb.DMatrix(xt, label=yt, nthread=threads)
    params = {
        "objective": meta["objective"],
        "tree_method": "hist",
        "device": device,
        "grow_policy": "depthwise",
        "max_depth": meta["max_depth"],
        "eta": meta["eta"],
        "reg_lambda": meta["lambda"],
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "max_bin": meta["max_bin"],
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "base_score": meta["base_score"],
        "nthread": threads,
        "seed": meta["seed"],
    }
    if meta["num_class"]:
        params["num_class"] = meta["num_class"]
    samples, score = [], None
    for run in range(repeats + 1):
        start = time.perf_counter()
        dtrain = xgb.QuantileDMatrix(x, label=y, nthread=threads, max_bin=meta["max_bin"])
        booster = xgb.train(params, dtrain, num_boost_round=meta["num_round"])
        elapsed = time.perf_counter() - start
        if run:
            samples.append(elapsed)
        if run == repeats:
            score = test_score(booster.predict(dtest), yt, meta["metric"])
            config = json.loads(booster.save_config())
        del booster, dtrain
    return {
        "engine": "xgboost",
        "threads": threads,
        "fit_seconds": samples,
        "test_metric": meta["metric"],
        "test_score": score,
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hessboost", type=Path, required=True, help="Compiled bench_compare example"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for data and results"
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument(
        "--repeats", type=int, default=3, help="Measured fits per batch; two batches per engine"
    )
    parser.add_argument("--rows", type=int, help="Override training rows for a smoke check")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Device both engines train on"
    )
    args = parser.parse_args()
    counts = [*args.threads, args.rounds, args.repeats, 1 if args.rows is None else args.rows]
    if min(counts) < 1:
        parser.error("thread, round, repeat, and row counts must be positive")
    executable = args.hessboost.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    library = Path(xgb.core._LIB._name)
    cpu = (
        subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin"
        else platform.processor()
    )
    gpu = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,ecc.mode.current",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if args.device == "cuda"
        else None
    )
    sources = [
        Path(__file__),
        Path("examples/bench_compare.rs"),
        Path("Cargo.toml"),
        Path("Cargo.lock"),
        *sorted(Path("src").rglob("*.rs")),
    ]
    report = {
        "metadata": {
            "started_at": datetime.now(UTC).isoformat(),
            "cpu": cpu,
            "device": args.device,
            "gpu": gpu,
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
            "xgboost_build_info": xgb.build_info(),
            "xgboost_library_sha256": sha256(library),
            "hessboost_executable_sha256": sha256(executable),
            "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
            "source_sha256": {str(p): sha256(p) for p in sources},
            "matrix": "XGBoost QuantileDMatrix; hessboost DMatrix",
            "timing": (
                "Fresh training matrix construction plus train; I/O, evaluation, and "
                "destruction excluded"
            ),
            "order": ["xgboost", "hessboost", "hessboost", "xgboost"],
            "warmup_fits_per_batch": 1,
            "measured_fits_per_batch": args.repeats,
            "threads": args.threads,
        },
        "datasets": {},
        "results": [],
    }
    for workload in args.workloads:
        folder = args.output / workload
        meta = make_dataset(folder, workload, args.rounds, args.rows)
        report["datasets"][workload] = {
            "parameters": meta,
            "sha256": {p.name: sha256(p) for p in sorted(folder.iterdir())},
        }
        for threads in args.threads:
            batches = []
            for engine in report["metadata"]["order"]:
                started = datetime.now(UTC).isoformat()
                if engine == "xgboost":
                    batch = xgboost_batch(folder, meta, threads, args.repeats, args.device)
                else:
                    env = dict(
                        os.environ,
                        BENCH_DIR=str(folder.resolve()),
                        BENCH_REPEATS=str(args.repeats),
                        BENCH_DEVICE=args.device,
                        RAYON_NUM_THREADS=str(threads),
                    )
                    batch = json.loads(subprocess.check_output([str(executable)], env=env))
                    assert batch["threads"] == threads
                batch["started_at"] = started
                batches.append(batch)
                print(
                    f"{workload}, {threads} threads, {engine}: "
                    f"{statistics.median(batch['fit_seconds']):.3f} s",
                    flush=True,
                )
            engines: dict[str, dict[str, Any]] = {}
            for engine in report["metadata"]["order"][:2]:
                selected = [b for b in batches if b["engine"] == engine]
                samples = [t for b in selected for t in b["fit_seconds"]]
                engines[engine] = {
                    "median_fit_seconds": statistics.median(samples),
                    "min_fit_seconds": min(samples),
                    "max_fit_seconds": max(samples),
                    "test_metric": meta["metric"],
                    "test_scores": [b["test_score"] for b in selected],
                }
            report["results"].append(
                {
                    "workload": workload,
                    "threads": threads,
                    "engines": engines,
                    "hessboost_speedup": engines["xgboost"]["median_fit_seconds"]
                    / engines["hessboost"]["median_fit_seconds"],
                    "batches": batches,
                }
            )
            (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    report["metadata"]["finished_at"] = datetime.now(UTC).isoformat()
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
