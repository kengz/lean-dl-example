from contextlib import suppress
from dl import DIR
from dl.prepare_data import DLDataModule
from torch import optim
import hydra
import pydash as ps
import pytorch_lightning as pl
import torch
import torcharc
import torchmetrics


def build_criterion(loss_spec: dict) -> torch.nn.Module:
    '''Build criterion (loss function) from loss spec'''
    criterion_cls = getattr(torch.nn, loss_spec.pop('type'))
    # any numeric arg has to be tensor; scan and try-cast
    for k, v in loss_spec.items():
        with suppress(Exception):
            loss_spec[k] = torch.tensor(v)
    criterion = criterion_cls(**loss_spec)
    return criterion


def build_metrics(metric_spec: dict) -> torchmetrics.MetricCollection:
    '''Build torchmetrics.MetricCollection from metric spec'''
    metrics = torchmetrics.MetricCollection([
        getattr(torchmetrics, metric_name)(**(v or {})) for metric_name, v in metric_spec.items()
    ])
    return metrics


class DLModel(pl.LightningModule):
    '''Deep Learning model built from config specifying architecture, loss, and optimizer'''

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.spec = hydra.utils.instantiate(cfg, _convert_='all')  # convert to dict
        self.model = torcharc.build(self.spec['arc'])
        self.criterion = build_criterion(self.spec['loss'])
        self.metrics = build_metrics(self.spec['metric'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.criterion(logit, y)
        self.log('losses', {'train': loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.criterion(logit, y)
        pred = logit.sigmoid().round().squeeze()
        target = y.long().squeeze()
        metrics = self.metrics(pred, target)
        # log to TensorBoard
        self.log('losses', {'val': loss}, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim_spec = self.spec['optim']
        optim_cls = getattr(optim, optim_spec.pop('type'))
        optimizer = optim_cls(self.parameters(), **optim_spec)
        return optimizer


@hydra.main(version_base=None, config_path=DIR / 'config', config_name='config')
def main(cfg):
    dm = DLDataModule(cfg)
    model = DLModel(cfg)

    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, datamodule=dm)
    return ps.get(trainer.callback_metrics, cfg.optuna_metric)


if __name__ == '__main__':
    main()
