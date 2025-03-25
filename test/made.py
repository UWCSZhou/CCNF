import numpy as np
import unittest
import torch

from scm.dag import DAG
from layers.made.causal_made import CausalMaskedMLP

class MadeTest(unittest.TestCase):
    def setUp(self):
        d = DAG(np.array([[0, 0, 1, 0, 0, 1],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]))
        self.made = CausalMaskedMLP(d.causal_layers[0], [32, 32])

    def test_forward(self):
        #u = self.trans.prior().sample((1000,))
        u = torch.Tensor([[1, 2, 3],
                          [10, 2, 3],
                          [100, 2, 3],
                          [1, 2, 3],
                          [1, 20, 3],
                          [1, 200, 3],
                          [1, 2, 3],
                          [1, 2, 30],
                          [1, 2, 300]])
        print(self.made(u))


if __name__ == "__main__":
    unittest.main()


'''

a = CausalMaskedMLP(4, d.causal_layers[0])
for p in a.named_parameters():
    print(p)
print(a(torch.Tensor([[1, 2], [3, 4]])))
print(a(torch.Tensor([[10, 2], [12, 4]])))
print(a(torch.Tensor([[1, 10], [3, 12]])))

'''
