# lean-dl-example

Example of a lean deep learning project with a config-driven approach.

## Installation

[Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if you haven't already.

Create a Conda environment and install dependencies. This example uses:

- [Hydra](https://hydra.cc) for composable config
- [feature_transform](https://github.com/kengz/feature_transform) for config-driven feature transformation
- [TorchArc](https://github.com/kengz/torcharc) for config-driven model building
- [PyTorch Lightning](https://www.pytorchlightning.ai) for research-focused development
- [Optuna (with Hydra)](https://hydra.cc/docs/plugins/optuna_sweeper/) for hyperparameter search
- [PyTorch-TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) for visualizing training progress and hyperparameter search

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
python dl/train.py dataloader.batch_size=32 arc.main.layers='[64,64]'
```

### Hyperparameter Search

By using config-based approach, any variant to the run can be specified as parameter overrides to Hydra configs - hence we can tune hyperparameters without any code changes.

The entrypoint `train.py` returns a float to be used for optimization; the logged metrics in trainer can be accessed via `trainer.callback_metrics`, and the config `cfg.metric` specifies which field.

Hydra has an [Optuna sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/). To run hyperparameter tuning, simply specify the parameter override and search space/details in `config/optuna/`, and run the folllowing:

```bash
# hyperparameter search using Optuna + Hydra. Configure in config/optuna.yaml
# view Optuna sweeper config
python dl/train.py hydra/sweeper=optuna +optuna=tune -c hydra -p hydra.sweeper
# run Optuna sweeper using optuna/tune.yaml to search over tune and other hyperparams
python dl/train.py hydra/sweeper=optuna +optuna=tune --multirun
```

Example log from hyperparameter tuning:

```bash
➜ python dl/train.py hydra/sweeper=optuna +optuna=tune --multirun
[I 2022-06-24 19:02:34,839] A new study created in memory with name: tune
[2022-06-24 19:02:34,839][HYDRA] Study name: tune
[2022-06-24 19:02:34,840][HYDRA] Storage: None
[2022-06-24 19:02:34,840][HYDRA] Sampler: TPESampler
[2022-06-24 19:02:34,840][HYDRA] Directions: ['maximize']
[2022-06-24 19:02:34,852][HYDRA] Launching 1 jobs locally
[2022-06-24 19:02:34,852][HYDRA] 	#0 : arc.main.layers=[8,8] arc.main.dropout=0.04679835610086079 loss.pos_weight=1.5227525095137953 optim.type=Adam optim.lr=1.2087541473056957e-05 +optuna=tune
[2022-06-24 19:02:35,083][torch.distributed.nn.jit.instantiator][INFO] - Created a temporary directory at /var/folders/jx/z4vcr3393j537mmdc9jg1gsc0000gn/T/tmpits7qg55
[2022-06-24 19:02:35,084][torch.distributed.nn.jit.instantiator][INFO] - Writing /var/folders/jx/z4vcr3393j537mmdc9jg1gsc0000gn/T/tmpits7qg55/_remote_module_non_sriptable.py
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Missing logger folder: /Users/redrose/lean-dl-example/lightning_logs

  | Name      | Type              | Params
------------------------------------------------
0 | model     | DAGNet            | 265
1 | criterion | BCEWithLogitsLoss | 0
2 | metrics   | MetricCollection  | 0
------------------------------------------------
265       Trainable params
0         Non-trainable params
265       Total params
0.001     Total estimated model params size (MB)
Epoch 99: 100%|█████████████████████████████████████| 15/15 [00:00<00:00, 128.89it/s, loss=0.843, v_num=0, losses={'val': 0.8364414572715759}, Accuracy=0.549, Precision=0.555, Recall=0.909, F1Score=0.688]
[2022-06-24 19:02:49,437][HYDRA] Launching 1 jobs locally
...
[2022-06-24 19:28:52,104][HYDRA] Launching 1 jobs locally
[2022-06-24 19:28:52,104][HYDRA] 	#99 : arc.main.layers=[8,8] arc.main.dropout=0.09820219968782427 loss.pos_weight=2.4295991695810226 optim.type=Adam optim.lr=0.0016705280295178648 +optuna=tune
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs

  | Name      | Type              | Params
------------------------------------------------
0 | model     | DAGNet            | 265
1 | criterion | BCEWithLogitsLoss | 0
2 | metrics   | MetricCollection  | 0
------------------------------------------------
265       Trainable params
0         Non-trainable params
265       Total params
0.001     Total estimated model params size (MB)
Epoch 99: 100%|████████████████████████████████████| 15/15 [00:00<00:00, 79.45it/s, loss=0.377, v_num=99, losses={'val': 0.47382786870002747}, Accuracy=0.848, Precision=0.826, Recall=0.923, F1Score=0.868]
[2022-06-24 19:29:12,666][HYDRA] Best parameters: {'arc.main.layers': '[8]', 'arc.main.dropout': 0.11879921503186516, 'loss.pos_weight': 5.0779681113146555, 'optim.type': 'Adam', 'optim.lr': 0.001365972987748234}
[2022-06-24 19:29:12,667][HYDRA] Best value: 0.912898600101471
```

### Monitoring

PyTorch Lightning logs to TensorBoard by default. To launch TensorBoard, run:

```bash
tensorboard --logdir .
```

![TensorBoard scalar plots](doc/tb_scalars.png)

> TensorBoard plots showing metrics and train/val losses.

![TensorBoard tuning parallel coordinates](doc/tb_tune_parallel_coor.png)

> TensorBoard parallel coordinates plot showing hyperparameter search results.

![TensorBoard tuning scatter](doc/tb_tune_scatter.png)

> TensorBoard scatter plot showing hyperparameter search results.
