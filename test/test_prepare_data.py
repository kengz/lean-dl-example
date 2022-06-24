from conftest import get_cfg
from dl.prepare_data import DLDataModule


def test_prepare_data():
    cfg = get_cfg()
    dm = DLDataModule(cfg)
    dm.prepare_data()
    dm.setup('fit')
    assert hasattr(dm, 'train_dataset')
    assert hasattr(dm, 'val_dataset')
