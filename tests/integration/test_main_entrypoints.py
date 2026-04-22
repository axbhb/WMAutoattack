import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "main.py"


def test_main_help():
    result = subprocess.run([sys.executable, str(MAIN), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "attack-search" in result.stdout
    assert "compare-search" in result.stdout


def test_main_train_help():
    result = subprocess.run([sys.executable, str(MAIN), "train", "--help"], capture_output=True, text=True)
    assert result.returncode == 0


def test_main_eval_help():
    result = subprocess.run([sys.executable, str(MAIN), "eval", "--help"], capture_output=True, text=True)
    assert result.returncode == 0


def test_main_attack_search_help():
    result = subprocess.run([sys.executable, str(MAIN), "attack-search", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--checkpoint-path" in result.stdout or "--task-root" in result.stdout


def test_main_compare_search_help():
    result = subprocess.run([sys.executable, str(MAIN), "compare-search", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--baseline-method-name" in result.stdout
    assert "--ours-retrieval-mode" in result.stdout
