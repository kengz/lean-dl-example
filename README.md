# lean-dl-example

Example of a lean deep learning project with a config-driven approach.

## Installation

[Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if you haven't already.

Create a Conda environment and install dependencies. This example uses:

- [Hydra](https://hydra.cc) for composable config
- [feature_transform](https://github.com/kengz/feature_transform) for config-driven feature transformation
- [TorchArc](https://github.com/kengz/torcharc) for config-driven model building
- [PyTorch Lightning](https://www.pytorchlightning.ai) for research-focused development
- [PyTorch-TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) for visualization
- [Optuna (with Hydra)](https://hydra.cc/docs/plugins/optuna_sweeper/) for hyperparameter tuning

```bash
conda create -n dl python=3.10.4 -y
conda activate dl
pip install -r requirements.txt
```

## Usage

### Training

Inspect/modify the Hydra config in `config/`. Then run:

```bash
python dl/train.py

# fault tolerant (resumable) training
PL_FAULT_TOLERANT_TRAINING=1 python dl/train.py

# to change configs
python dl/train.py datamodule.batch_size=64 model.hidden_size=128
```

### Monitoring

PyTorch Lightning logs to TensorBoard by default. To launch TensorBoard, run:

```bash
tensorboard --logdir .
```

### Hyperparameter Tuning

By using config-based approach, any variant to the run can be specified as parameter overrides to Hydra configs - hence we can tune hyperparameters without any code changes.

The entrypoint `train.py` returns a float to be used for optimization; the logged metrics in trainer can be accessed via `trainer.callback_metrics`, and the config `cfg.metric` specifies which field.

Hydra has an [Optuna sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/). To run hyperparameter tuning, simply specify the parameter override and search space/details in `config/optuna/`, and run the folllowing:

```bash
# hyperparameter search using Optuna + Hydra. Configure in config/optuna.yaml
# view optuna sweeper config
python dl/train.py hydra/sweeper=optuna +optuna=batch_size -c hydra -p hydra.sweeper
# run optuna sweeper using optuna/batch_size.yaml to search over batch size
python dl/train.py hydra/sweeper=optuna +optuna=batch_size --multirun
```
