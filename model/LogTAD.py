from . import DomainAdversarial
import torch.nn as nn

class LogTAD(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.emb_dim = options["emb_dim"]
        self.hid_dim = options["hid_dim"]
        self.output_dim = options["output_dim"]
        self.n_layers = options["n_layers"]
        self.dropout = options["dropout"]
        self.bias = options["bias"]
        self.encoder = DomainAdversarial.DA_LSTM(self.emb_dim, self.hid_dim, self.output_dim, self.n_layers, self.dropout, self.bias)