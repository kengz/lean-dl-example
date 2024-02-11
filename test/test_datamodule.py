from dl import cfg
from dl.datamodule import DLDataModule


def test_prepare_data():
    dm = DLDataModule(cfg)
    dm.prepare_data()
    dm.setup('fit')
    assert hasattr(dm, 'train_dataset')
    assert hasattr(dm, 'val_dataset')
