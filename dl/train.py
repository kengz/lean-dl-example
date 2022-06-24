from dl import DIR
from dl.prepare_data import DLDataModule
import hydra
import pydash as ps
import pytorch_lightning as pl
import torcharc
import torchmetrics


def build_metrics(metric_spec: dict) -> torchmetrics.MetricCollection:
    '''Build torchmetrics.MetricCollection from metric spec. Ref: https://torchmetrics.readthedocs.io/en/stable/pages/overview.html?highlight=collection#metriccollection'''
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
        self.criterion = torcharc.build_criterion(self.spec['loss'])
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
        self.log('losses', {'val': loss}, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)
        self.log('hp_metric', metrics[self.spec['optuna_metric']])  # tensorboard hparams metric for visualization
        return loss

    def configure_optimizers(self):
        return torcharc.build_optimizer(self.spec['optim'], self)


@hydra.main(version_base=None, config_path=DIR / 'config', config_name='config')
def main(cfg):
    dm = DLDataModule(cfg)
    model = DLModel(cfg)

    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, datamodule=dm)
    return ps.get(trainer.callback_metrics, cfg.optuna_metric)


if __name__ == '__main__':
    main()
