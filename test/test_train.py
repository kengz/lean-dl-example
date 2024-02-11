from dl import get_cfg
from dl.train import main


def test_train():
    cfg = get_cfg(overrides=['trainer.max_epochs=2'])
    optuna_metric = main(cfg)
    assert optuna_metric > 0
