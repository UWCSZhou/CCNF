import numpy as np
import seaborn as sns
import json
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.distributions as dists
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from utils.combined_dists import CombinedDistributions
from dataset.syn import SynModule
from dataset.german.german import GermanModule
from scm.dag import DAG
from layers import MAFLayer, VACALayer, StackNF, CausalStackNF, FlowPlusPlusLayer
from plmodule import NormalizingFlowModule, VACAModule

from utils.metrics import compute_jacobian
from train.arg_parser import parse_args

def create_trainer(args, loc, max_epochs = 1000):
    logger = pl.pytorch.loggers.TensorBoardLogger(loc, name = args.model)
    stop_callback = EarlyStopping(monitor = "val_loss", mode = "min", patience = 50)
    checkpoint_callback = ModelCheckpoint(dirpath = logger.log_dir + "/checkpoints/",
                                          filename = args.model,
                                          enable_version_counter = False)
    trainer = pl.Trainer(log_every_n_steps = 1, max_epochs = max_epochs, \
                         callbacks = [stop_callback, checkpoint_callback],
                         logger = logger)
    return trainer

def record_train_summary(loc, module):
    with open(loc + "/summary", "a") as f:
        record = module.record
        record["version"] = module.logger.version
        record["train_time"] = np.array(record["train_time"]).mean()
        record["valid_time"] = np.array(record["valid_time"]).mean()
        f.write(json.dumps(record))
        f.write("\n")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    args = parse_args()
    loc = "model/{}/".format(args.dataset)
    trainer = create_trainer(args, loc)
    if args.dataset in {"four_node_chain", "nlin_simpson", "nlin_triangle", "m_graph",
                        "network", "backdoor", "eight_node_chain"}:
        if args.model == "vaca":
            batch_size = 1000
        else:
            batch_size = 10000
        datamodule = SynModule(args.dataset, batch_size = batch_size)
        dataset = datamodule.dataset
    elif args.dataset == "german":
        datamodule = GermanModule(noise = True)
        dataset = datamodule
        #print(dataset.mu, dataset.std)
    priors = dists.MultivariateNormal(torch.zeros(dataset.dim, device = "cuda"),
                                      torch.eye(dataset.dim, device = "cuda"))
    if args.model == "my":
        if args.dataset == "german":
            layers = ["gaussian", "flow++", "flow++", "flow++", "flow++", "flow++"]
        elif args.dataset == "nlin_simpson":
            layers = ["gaussian", "maf", "flow++", "maf"]
        elif args.dataset == "nlin_triangle":
            layers = ["gaussian", "flow++", "maf"]
        elif args.dataset == "m_graph":
            layers = ["af", "flow++"]
        elif args.dataset == "network":
            layers = ["af", "flow++", "flow++", "flow++"]
        elif args.dataset == "backdoor":
            layers = ["af", "flow++", "maf", "maf", "flow++"]
        elif args.dataset == "eight_node_chain":
            layers = ["af"] + ["flow++"] * 7
        else:
            layers = None
        module = NormalizingFlowModule(priors, dataset.dist,
                                       CausalStackNF,
                                       [dataset.dag, args.num_layers,
                                        args.hidden_layers, datamodule.limits,
                                        datamodule.mu, datamodule.std, layers])
        #test_module(module)
    elif args.model == "carefl":
        order = dataset.dag.get_topological_order()
        print(order)
        layers = [MAFLayer(dataset.dim, args.hidden_layers, order)
                  for _ in range(args.num_layers[0])]
        module = NormalizingFlowModule(priors, dataset.dist, StackNF,
                                       [datamodule.mu, datamodule.std, layers])
    elif args.model == "causal_nf":
        order = dataset.dag.get_topological_order()
        if args.normalizing_flow == "maf":
            layers = [MAFLayer(dataset.dim, args.hidden_layers,
                               order = order,
                               adj = dataset.dag.ends_adj)
                      for _ in range(args.num_layers[0])]
        elif args.normalizing_flow == "flow++":
            layers = [FlowPlusPlusLayer(dataset.dim, args.hidden_layers,
                                        order = order,
                                        adj = dataset.dag.ends_adj)
                      for _ in range(args.num_layers[0])]
        module = NormalizingFlowModule(priors, dataset.dist, StackNF,
                                       [datamodule.mu, datamodule.std, layers])
    elif args.model == "vaca":
        #datamodule.setup_normalization()
        num_enc_layers = args.num_layers[0]
        num_dec_layers = 4
        hidden_dim_of_z = 4
        hidden_enc_channels = args.hidden_layers[0]
        hidden_dec_channels = args.hidden_layers[0]
        dropout = 0.0
        layers = 1
        model = args.gnn
        module = VACAModule(model, dataset.dim, dataset.dag.to_coo_format(),
                            num_enc_layers, num_dec_layers,
                            hidden_dim_of_z, hidden_enc_channels, hidden_dec_channels,
                            dropout, layers, layers,
                            datamodule.mu, datamodule.std)
    trainer.fit(module, datamodule = datamodule)
    if args.model == "vaca":
        module = VACAModule.load_from_checkpoint(
        "{}{}/version_0/checkpoints/{}.ckpt".
        format(loc, args.model, args.model))
    else:
        module = NormalizingFlowModule.load_from_checkpoint(
        "{}{}/version_0/checkpoints/{}.ckpt".
        format(loc, args.model, args.model))
    trainer.test(module, datamodule = datamodule)
    #record_train_summary(loc + args.model, module)

