"""Tests for utils/doctor.py."""
import numpy as np

from config.settings import CLASS_NAMES, DEFAULT_TRAIN_CONFIG
from models.cnn import DoodleCNN
from training.checkpoint import save_checkpoint
from utils.doctor import FAIL, OK, WARN, Check, main, run_checks


def by_name(checks):
    return {c.name: c for c in checks}


def test_reports_problems_without_raising(tmp_path):
    checks = run_checks(tmp_path / "no.pt", tmp_path / "no_data", tmp_path / "no.json")
    assert all(isinstance(c, Check) and c.status in {OK, WARN, FAIL} for c in checks)
    named = by_name(checks)
    assert named["dataset"].status == FAIL and "download_dataset" in named["dataset"].detail
    assert named["checkpoint"].status == FAIL and "training.train" in named["checkpoint"].detail
    assert named["metrics file"].status == WARN
    assert named["Python"].status == OK


def test_healthy_setup(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in CLASS_NAMES:
        np.save(raw / f"{name}.npy", np.zeros((5, 784), dtype=np.uint8))
    ckpt = tmp_path / "m.pt"
    save_checkpoint(ckpt, DoodleCNN(), CLASS_NAMES, DEFAULT_TRAIN_CONFIG, 3, 0.9)
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}")
    named = by_name(run_checks(ckpt, raw, metrics))
    assert named["dataset"].status == OK and "50" in named["dataset"].detail
    assert named["checkpoint"].status == OK and "epoch 3" in named["checkpoint"].detail
    assert named["metrics file"].status == OK
    assert named["package torch"].status == OK


def test_checkpoint_with_different_classes_warns(tmp_path):
    ckpt = tmp_path / "m.pt"
    save_checkpoint(ckpt, DoodleCNN(num_classes=2), ["a", "b"], DEFAULT_TRAIN_CONFIG, 1, 0.5)
    assert by_name(run_checks(ckpt, tmp_path, tmp_path / "x.json"))["checkpoint"].status == WARN


def test_cli_exit_codes(tmp_path, capsys):
    code = main(["--checkpoint", str(tmp_path / "no.pt"), "--raw-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1 and "[FAIL]" in out and "checks:" in out
