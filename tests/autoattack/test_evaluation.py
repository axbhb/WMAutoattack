from autoattack.evaluation import run_baseline, run_eval


def test_evaluation_entrypoints_are_importable():
    assert callable(run_eval)
    assert callable(run_baseline)

