import hydra
import pydash as ps
import pytorch_lightning as pl
import torch
import torcharc
import torchmetrics

from dl import DIR
from dl.datamodule import DLDataModule


def build_metrics(metric_spec: dict) -> torchmetrics.MetricCollection:
    """Build torchmetrics.MetricCollection from metric spec. Ref: https://torchmetrics.readthedocs.io/en/stable/pages/overview.html?highlight=collection#metriccollection"""
    metrics = torchmetrics.MetricCollection(
        {
            metric_name: getattr(torchmetrics, metric_name)(**(v or {}))
            for metric_name, v in metric_spec.items()
        }
    )
    return metrics


class DLModel(pl.LightningModule):
    """Deep Learning model built from config specifying architecture, loss, and optimizer"""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.spec = hydra.utils.instantiate(cfg, _convert_="all")  # convert to dict
        self.model = torcharc.build(self.spec["model"])
        # for to_onnx to infer input shape
        self.example_input_array = torch.randn(1, 18)

        LossCls = getattr(torch.nn, self.spec["loss"].pop("type"))
        self.criterion = LossCls(**self.spec["loss"])
        self.metrics = build_metrics(self.spec["metric"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.criterion(logit, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.criterion(logit, y)
        pred = logit.sigmoid().round().squeeze()
        target = y.long().squeeze()
        metrics = self.metrics(pred, target)
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)
        self.log(
            "hp_metric", metrics[self.spec["optuna_metric"]]
        )  # tensorboard hparams metric for visualization
        return loss

    def configure_optimizers(self):
        OptimCls = getattr(torch.optim, self.spec["optim"].pop("type"))
        return OptimCls(self.parameters(), **self.spec["optim"])


@hydra.main(version_base=None, config_path=str(DIR / "config"), config_name="config")
def main(cfg):
    dm = DLDataModule(cfg)
    model = DLModel(cfg)

    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, datamodule=dm)
    model.to_onnx(cfg.onnx.path, export_params=True)
    return ps.get(trainer.callback_metrics, cfg.optuna_metric)


if __name__ == "__main__":
    main()
