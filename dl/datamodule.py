import feature_transform as ft
import hydra
import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from dl import DIR


class DLDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.transform_cfg = cfg.transform
        self.dm_cfg = cfg.datamodule
        self.dataloader_cfg = cfg.dataloader

        self.x_dataset_path = DIR / self.dm_cfg.x.dataset
        self.x_col_tfm_path = DIR / self.dm_cfg.x.col_tfm
        self.y_dataset_path = DIR / self.dm_cfg.y.dataset
        self.y_col_tfm_path = DIR / self.dm_cfg.y.col_tfm

    def prepare_data(self):
        if self.x_dataset_path.exists() and self.y_dataset_path.exists():
            return

        self.x_dataset_path.parent.mkdir(exist_ok=True)
        data_df = pd.read_csv(self.dm_cfg.csv)
        data_x_df, data_y_df = data_df.iloc[:, 0:-1], data_df.iloc[:, -1:]

        x_col_tfm = ft.build(self.transform_cfg.x)
        x_dataset = x_col_tfm.fit_transform(data_x_df)
        np.save(self.x_dataset_path, x_dataset.astype("float32"))
        joblib.dump(x_col_tfm, self.x_col_tfm_path)

        y_col_tfm = ft.build(self.transform_cfg.y)
        y_dataset = y_col_tfm.fit_transform(data_y_df)
        np.save(self.y_dataset_path, y_dataset.astype("float32"))
        joblib.dump(y_col_tfm, self.y_col_tfm_path)

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            xs = np.load(self.x_dataset_path)
            ys = np.load(self.y_dataset_path)
            # split with stratify to ensure each split gets the same proportion of target values
            train_xs, val_xs, train_ys, val_ys = train_test_split(
                xs, ys, train_size=self.dm_cfg.train_frac, stratify=ys
            )
            self.train_dataset = TensorDataset(
                torch.from_numpy(train_xs), torch.from_numpy(train_ys)
            )
            self.val_dataset = TensorDataset(
                torch.from_numpy(val_xs), torch.from_numpy(val_ys)
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_cfg)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_cfg)


@hydra.main(version_base=None, config_path=str(DIR / "config"), config_name="config")
def main(cfg):
    dm = DLDataModule(cfg)
    dm.prepare_data()


if __name__ == "__main__":
    main()
