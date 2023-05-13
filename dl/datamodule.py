from dl import DIR
from feature_transform import transform, util
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch


class DLDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.csv = cfg.datamodule.csv  # raw data
        self.data_path = DIR / 'data' / cfg.datamodule.data_filename  # transformed data
        self.train_frac = cfg.datamodule.train_frac

    def prepare_data(self):
        if not self.data_path.exists():
            data_df = pd.read_csv(self.csv)
            mode2data = transform.fit_transform(self.cfg, stage='fit', df=data_df)  # format: {'x': np.ndarray, 'y': np.ndarray}
            util.write(mode2data, self.data_path)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            mode2data = util.read(self.data_path)
            xs, ys = mode2data.values()
            # split with stratify to ensure each split gets the same proportion of target values
            train_xs, val_xs, train_ys, val_ys = train_test_split(xs, ys, train_size=self.train_frac, stratify=ys)
            self.train_dataset = TensorDataset(torch.from_numpy(train_xs), torch.from_numpy(train_ys))
            self.val_dataset = TensorDataset(torch.from_numpy(val_xs), torch.from_numpy(val_ys))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.dataloader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg.dataloader)


@hydra.main(version_base=None, config_path=str(DIR / 'config'), config_name='config')
def main(cfg):
    dm = DLDataModule(cfg)
    dm.prepare_data()


if __name__ == '__main__':
    main()
