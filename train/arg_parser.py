import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = 'args for training model')
    parser.add_argument('--model', '-m', type = str, default = "my")
    parser.add_argument('--dataset', '-d', type = str, default = "four_node_chain")
    parser.add_argument('--num_layers', '-n', nargs = "*", type = int, default = [1])
    parser.add_argument('--hidden_layers', '-hl',  nargs = "*",
                        type = int, default = [64, 64])
    parser.add_argument('--normalizing_flow', '-nf',  type = str, default = "maf")
    parser.add_argument('--gnn', '-gnn',  type = str, default = "pna")
    return parser.parse_args()

