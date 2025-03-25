import lightning as pl
from functools import partial
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import torch.distributions as dists

from plmodule import NormalizingFlowModule, VACAModule
from layers import MAFLayer, VACALayer, StackNF, CausalStackNF
from dataset.syn import SynModule
from train.arg_parser import parse_args


def objective(trial, datamodule, epochs = 20):
    dataset = datamodule.dataset
    priors = dists.MultivariateNormal(torch.zeros(dataset.dim, device = "cuda:0"),
                                      torch.eye(dataset.dim, device = "cuda:0"))
    if args.model == "carefl":
        order = dataset.dag.get_topological_order()
        layers = [MAFLayer(dataset.dim,
                           trial.suggest_categorical("hidden_layer",
                                                     [[8, 8], [16, 16],
                                                      [32, 32], [64, 64]]),
                           order = order)
                  for _ in range(args.num_layers)]
        module = NormalizingFlowModule(priors, dataset.dist, StackNF, [layers])
    elif args.model == "causal_nf":
        layers = [MAFLayer(dataset.dim,
                           trial.suggest_categorical("hidden_layer",
                                                     [[8, 8], [16, 16],
                                                      [32, 32], [64, 64]]),
                           adj = dataset.dag.ends_adj)
                  for _ in range(args.num_layers)]
        module = NormalizingFlowModule(priors, dataset.dist, StackNF, [layers])
    elif args.model == "vaca":
        datamodule.setup_normalization()
        model = trial.suggest_categorical("model", ["pna", "gin"])
        num_enc_layers = args.num_layers
        num_dec_layers = 3
        hidden_dim_of_z = 1
        hidden_enc_channels = trial.suggest_categorical("hidden_enc_channels", [16, 32])
        hidden_dec_channels = 16
        dropout = 0.0
        layers = trial.suggest_categorical("pre_post_layers", [0, 1])
        module = VACAModule(model,
                            dataset.dim, dataset.dag.to_coo_format(),
                            num_enc_layers, num_dec_layers,
                            hidden_dim_of_z, hidden_enc_channels, hidden_dec_channels,
                            dropout, layers, layers)
    trainer = pl.Trainer(
        logger = True,
        log_every_n_steps = 1,
        enable_checkpointing = False,
        max_epochs = epochs,
        accelerator = "auto",
        callbacks = PyTorchLightningPruningCallback(trial, monitor = "val_loss"),
    )

    trainer.fit(module, datamodule = datamodule)
    return trainer.callback_metrics["val_loss"]


if __name__ == '__main__':
    args = parse_args()
    torch.set_float32_matmul_precision('high')
    if args.dataset in {"four_node_chain", "nlin_simpson", "network"}:
        datamodule = SynModule(args.dataset)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction = "minimize", pruner = pruner)
    study.optimize(partial(objective,
                           datamodule = datamodule), n_trials = 100, timeout = 600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
