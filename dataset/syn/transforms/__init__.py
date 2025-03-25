from dataset.syn.transforms.four_node_chain import FourNodeChainTransform
from dataset.syn.transforms.nlin_simpson import NlinSimpsonTransform
from dataset.syn.transforms.nlin_triangle import NlinTriangleTransform
from dataset.syn.transforms.m_graph import MGraphTransform
from dataset.syn.transforms.network import NetworkTransform
from dataset.syn.transforms.backdoor import BackdoorTransform
from dataset.syn.transforms.eight_node_chain import EightNodeChainTransform

trans_dict = {}
trans_dict["nlin_simpson"] = NlinSimpsonTransform
trans_dict["four_node_chain"] = FourNodeChainTransform
trans_dict["nlin_triangle"] =  NlinTriangleTransform
trans_dict["m_graph"] = MGraphTransform
trans_dict["network"] = NetworkTransform
trans_dict["backdoor"] = BackdoorTransform
trans_dict["eight_node_chain"] = EightNodeChainTransform
