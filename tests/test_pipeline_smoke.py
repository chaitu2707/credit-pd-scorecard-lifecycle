from pathlib import Path

from src.pipeline import run
from src.synthetic import make_synthetic
from src import config


def test_pipeline_runs_end_to_end(tmp_path: Path):
    df = make_synthetic(n=6000, seed=0)
    result = run(df, out_dir=tmp_path, trials=2)
    tables = tmp_path / "reports" / "tables"
    models = tmp_path / "models"
    for name in ["model_comparison.csv", "cutoff_table.csv", "monitoring_batches.csv",
                 "decile_oot_champion.csv", "sample_summary.csv", "shap_importance_lightgbm.csv"]:
        assert (tables / name).exists(), name
    for name in ["scorecard.json", "champion_challenger.json", "run_summary.json", "calibration.json"]:
        assert (models / name).exists(), name
    comp = result["comparison"]
    assert set(comp["sample"]) == {"train", "validation", "oot"}
    assert comp["gini"].between(-1, 1).all()
    assert result["summary"]["batches_total"] > 0
