from dl import DIR
from dl.prepare_data import DLDataModule
from pathlib import Path
from torch import optim
from torch.nn import functional
from torchmetrics import Accuracy
import hydra
import pytorch_lightning as pl
import torcharc


def get_ckpt_path(cfg) -> Path:
    '''Get the ckpt_path for resuming training from the last of lightning_logs/version_x if requested and available'''
    ckpt_path = None  # default
    if not cfg.get('resume'):  # don't resume
        return ckpt_path

    # first, try .pl_auto_save.ckpt from PL_FAULT_TOLERANT_TRAINING
    if (path := DIR / '.pl_auto_save.ckpt').exists():
        ckpt_path = path
    # next, find the latest lightning_logs/version_*/checkpoints/*.ckpt by creation time
    elif (default_dir := DIR / 'lightning_logs').exists():
        ckpts = default_dir.glob('version_*/checkpoints/*.ckpt')
        if latest_ckpt := max(ckpts, key=lambda x: x.stat().st_ctime):
            ckpt_path = latest_ckpt

    if ckpt_path is None:
        raise FileNotFoundError('Trying to resume training from a checkpoint but could not find one')
    else:
        print(f'Resuming training from checkpoint: {ckpt_path}')
        return ckpt_path


class DLModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # convert to dict and build model
        arc = hydra.utils.instantiate(cfg.arc, _convert_='all')
        self.model = torcharc.build(arc)
        self.loss_fn = getattr(functional, cfg.loss)
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.loss_fn(logit, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.loss_fn(logit, y)
        pred = logit.sigmoid().round()
        self.accuracy(pred.squeeze(), y.long().squeeze())

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optim_spec = hydra.utils.instantiate(self.cfg.optim, _convert_='all')
        optim_cls = getattr(optim, optim_spec.pop('type'))
        optimizer = optim_cls(self.parameters(), **optim_spec)
        return optimizer


@hydra.main(version_base=None, config_path=DIR / 'config', config_name='config')
def main(cfg):
    dm = DLDataModule(cfg)
    model = DLModel(cfg)

    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, datamodule=dm, ckpt_path=get_ckpt_path(cfg))


if __name__ == '__main__':
    main()
