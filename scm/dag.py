import numpy as np
import torch

class CausalLayer():
    def __init__(self):
        self.starts = set() # starts with set() but end with list[]
        self.ends = []
        self.edges = dict() # from end to start {end: [start]}

    def update_edges(self, starts, end):
        self.edges[end] = starts
        self.ends.append(end)
        self.starts.update(starts)

    def sort_starts(self):
        self.starts = sorted(self.starts)


class DAG():
    def __init__(self, adj):
        self.starts_adj = adj # adj is numpy array
        self.ends_adj = np.transpose(self.starts_adj)
        self.dim = len(adj)
        adjv_sum = np.sum(self.ends_adj, axis = 1)
        self.start_nodes = [i for i in range(self.dim) if adjv_sum[i] == 0]
        self.depth, self.diam = self._collect_dag_index()
        self.causal_layers = self._collect_causal_layers()

    def _collect_dag_index(self):
        depth = np.zeros(self.dim, dtype = np.uint32)
        s1, s2, index = set(self.start_nodes), set(), 1
        while s1:
            while s1:
                fro = s1.pop()
                for to, conn in enumerate(self.starts_adj[fro]):
                    if conn != 0:
                        s2.add(to)
            depth[list(s2)], index = index, index + 1
            s1, s2 = s2, s1
            s2.clear()
        self.nodes = [[i for i in range(self.dim) if depth[i] == level] \
                      for level in range(max(depth) + 1)]
        return depth, max(depth)

    def _collect_causal_layers(self):
        causal_layers = [CausalLayer() for _ in range(self.diam)]
        for level in range(1, self.diam + 1):
            for i in range(self.dim):
                if (self.depth[i] == level):
                    starts = [j for j in range(self.dim) if self.ends_adj[i][j] != 0]
                    causal_layers[level - 1].update_edges(starts, i)
            causal_layers[level - 1].sort_starts()
        for layer in causal_layers:
            print(layer.edges, layer.starts, layer.ends)
        print("dag diam: ", self.diam)
        return causal_layers

    def get_topological_order(self):
        return [i for level in self.nodes for i in level]

    def to_coo_format(self):
        tmp = self.starts_adj.copy()
        np.fill_diagonal(tmp, 1)
        return np.vstack(np.nonzero(tmp))
