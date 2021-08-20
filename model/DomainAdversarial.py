import torch.nn as nn
import torch
from torch.autograd import Function

class GRL(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DA_LSTM(nn.Module):

    def __init__(self, emb_dim, hid_dim, output_dim, n_layers, dropout, bias):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=False, batch_first=True,
                           bias=bias)

        self.discriminator = nn.Sequential(
            nn.Linear(hid_dim, 64),
            nn.Linear(64, output_dim)
        )

    def forward(self, input, alpha):
        output, (hidden, cell) = self.rnn(input)
        y = GRL.apply(torch.mean(output, dim=1), alpha)
        y = self.discriminator(y)
        return torch.mean(output, dim=1), y