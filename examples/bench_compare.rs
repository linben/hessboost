//! Fit-time and held-out quality measurements on shared binary datasets.
//!
//! Driven by `scripts/bench_xgb.py`. File I/O and test-data preparation are
//! outside the timer, and each fit constructs a fresh training `DMatrix`.

use hessboost::prelude::*;
use serde_json::json;
use std::path::Path;
use std::time::Instant;

mod common;
use common::{load_meta, read_f32};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("BENCH_DIR")?;
    let dir = Path::new(&dir);
    let meta = load_meta(dir)?;
    let x = read_f32(&dir.join("X.bin"))?;
    let y = read_f32(&dir.join("y.bin"))?;
    let x_test = read_f32(&dir.join("X_test.bin"))?;
    let y_test = read_f32(&dir.join("y_test.bin"))?;
    let dtest = DMatrix::from_dense(&x_test, meta.n_test, meta.n_cols)?.with_labels(&y_test)?;
    // The XGBoost parameter dict `scripts/bench_xgb.py` trains XGBoost with;
    // `BENCH_DEVICE` is its `--device` (`cpu` unless set).
    let device = std::env::var("BENCH_DEVICE").unwrap_or_else(|_| "cpu".to_owned());
    let mut xgboost = vec![
        ("device", json!(device)),
        ("objective", json!(meta.objective)),
        ("tree_method", json!("hist")),
        ("grow_policy", json!("depthwise")),
        ("max_depth", json!(meta.max_depth)),
        ("eta", json!(meta.eta)),
        ("lambda", json!(meta.lambda)),
        ("max_bin", json!(meta.max_bin)),
        ("base_score", json!(meta.base_score)),
        ("seed", json!(meta.seed)),
        ("eval_metric", json!(meta.metric)),
    ];
    if meta.num_class > 0 {
        xgboost.push(("num_class", json!(meta.num_class)));
    }
    let params = TrainingParams::from_xgboost(xgboost)?;
    let metric = params.eval_metric[0].build(meta.num_class.max(1))?;
    let repeats: usize = std::env::var("BENCH_REPEATS")
        .unwrap_or_else(|_| "3".to_owned())
        .parse()?;
    if repeats == 0 {
        return Err("BENCH_REPEATS must be positive".into());
    }
    let mut samples = Vec::with_capacity(repeats);
    let mut score = 0.0;
    // The first complete fit warms the allocator and Rayon pool.
    for run in 0..=repeats {
        let start = Instant::now();
        let dtrain = DMatrix::from_dense(&x, meta.n_rows, meta.n_cols)?.with_labels(&y)?;
        let model = train(&params, &dtrain, meta.num_round)?;
        let elapsed = start.elapsed().as_secs_f64();
        if run > 0 {
            samples.push(elapsed);
        }
        if run == repeats {
            let preds = model.predict(&dtest, Iterations::Best)?;
            score = metric.eval(preds.as_slice(), &y_test, None);
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "engine": "hessboost",
            "threads": rayon::current_num_threads(),
            "fit_seconds": samples,
            "test_metric": meta.metric,
            "test_score": score,
        })
    );
    Ok(())
}
