# lean-dl-example

Example of a lean deep learning project with a config-driven approach.

## Installation

[Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if you haven't already.

Create a Conda environment and install dependencies. This example uses:

- [Hydra](https://hydra.cc) for composable config
- [feature_transform](https://github.com/kengz/feature_transform) for config-driven feature transformation
- [torcharc](https://github.com/kengz/torcharc) for config-driven model building
- [PyTorch Lightning](https://www.pytorchlightning.ai) for research-focused development
- [Optuna](https://hydra.cc/docs/plugins/optuna_sweeper/) for hyperparameter tuning

```bash
conda create -n dl python=3.10.4 -y
conda activate dl
pip install -r requirements.txt
```

## Usage

Inspect/modify the Hydra config at `config.yaml`. Then run:

```bash
python dl/train.py

# to change configs
python dl/train.py datamodule.batch_size=64 model.hidden_size=128

# to resume (just the the resume config)
python dl/train.py datamodule.batch_size=64 model.hidden_size=128 resume=true
```

This run training with validation at epoch-end, and test when training is done. Metrics will be logged from torchmetrics.
