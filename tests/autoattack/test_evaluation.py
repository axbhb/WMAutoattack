from types import SimpleNamespace

from autoattack.evaluation import run_baseline, run_eval
from sheeprl.algos.dreamer_v3.attacks import TwoStageMomentumAttack, build_attack


def test_evaluation_entrypoints_are_importable():
    assert callable(run_eval)
    assert callable(run_baseline)


def test_two_stage_attack_is_buildable():
    cfg = SimpleNamespace(
        attack=SimpleNamespace(
            enabled=True,
            name="two_stage",
            epsilon=8,
            steps=20,
            restarts=1,
            rho=0.75,
            seed=0,
            tau=0.15,
            momentum=0.9,
            beta=0.6,
            stage1_ratio=0.4,
            alpha=0.1,
        ),
        algo=SimpleNamespace(cnn_keys=SimpleNamespace(encoder=("rgb",))),
    )
    attacker = build_attack(cfg)
    assert isinstance(attacker, TwoStageMomentumAttack)
