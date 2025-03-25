import numpy as np
import unittest
import torch

from scm.dag import DAG
from plmodule import VACAModule

class VACATest(unittest.TestCase):
    def setUp(self):
        d = DAG(np.array([[0, 0, 1],
                          [0, 0, 1],
                          [0, 0, 0]]))
        num_enc_layers = 1
        num_dec_layers = 4
        hidden_dim_of_z = 4
        hidden_enc_channels = 16
        hidden_dec_channels = 16
        dropout = 0.0
        layers = 1
        model = "pna"
        self.vaca = VACAModule(model, d.dim, d.to_coo_format(),
                               num_enc_layers, num_dec_layers,
                               hidden_dim_of_z, hidden_enc_channels, hidden_dec_channels,
                               dropout, layers, layers, "cpu")

    def test_forward(self):
        #u = self.trans.prior().sample((1000,))
        u = torch.Tensor([[1, 2, 3],
                          [1, 2, 3]])
        print(self.vaca.model(u))


if __name__ == "__main__":
    unittest.main()
