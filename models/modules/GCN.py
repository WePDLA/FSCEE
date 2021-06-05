import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adadelta
from torch.nn import init
import json
import math
import sys
import torchsnooper
# from models.modules.util import BottledOrthogonalLinear, log


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.contiguous().view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0] * size[1]))
        return out.view(-1, size[0], size[1])


class XavierLinear(nn.Module):
    '''
    Simple Linear layer with Xavier init

    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(OrthogonalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.orthogonal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class BottledLinear(Bottle, nn.Linear):
    pass


class BottledXavierLinear(Bottle, XavierLinear):
    pass


class BottledOrthogonalLinear(Bottle, OrthogonalLinear):
    pass


def log(*args, **kwargs):
    print(file=sys.stdout, flush=True, *args, **kwargs)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, edge_types, dropout=0.5, bias=True, use_bn=True,
                 device=torch.device("cuda")):
        """
        Single Layer GraphConvolution

        :param in_features: The number of incoming features
        :param out_features: The number of output features
        :param edge_types: The number of edge types in the whole graph
        :param dropout: Dropout keep rate, if not bigger than 0, 0 or None, default 0.5
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_types = edge_types
        self.dropout = dropout if type(dropout) == float and -1e-7 < dropout < 1 + 1e-7 else None
        # parameters for gates
        self.Gates = nn.ModuleList()
        # parameters for graph convolutions
        self.GraphConv = nn.ModuleList()
        # batch norm
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)

        for _ in range(edge_types):
            self.Gates.append(BottledOrthogonalLinear(in_features=in_features,
                                                      out_features=1,
                                                      bias=bias))
            self.GraphConv.append(BottledOrthogonalLinear(in_features=in_features,
                                                          out_features=out_features,
                                                          bias=bias))
        self.device = device
        self.to(device)
    # @torchsnooper.snoop()
    def forward(self, input, adj):
        """

        :param input: FloatTensor, input feature tensor, (batch_size, seq_len, hidden_size)
        :param adj: FloatTensor (sparse.FloatTensor.to_dense()), adjacent matrix for provided graph of padded sequences, (batch_size, edge_types, seq_len, seq_len)
        :return: output
            - **output**: FloatTensor, output feature tensor with the same size of input, (batch_size, seq_len, hidden_size)
        """

        adj_ = adj.transpose(0, 1)  # (edge_types, batch_size, seq_len, seq_len)
        # print("adj==============",adj_.shape,adj_[0].shape)
        ts = []
        for i in range(self.edge_types):
            gate_status = F.sigmoid(self.Gates[i](input))  # (batch_size, seq_len, 1)
            x = adj_[i] 
            # print(x.shape,gate_status.shape)
            adj_hat_i = torch.mul(x ,gate_status)  # (batch_size, seq_len, seq_len)

            ts.append(torch.bmm(adj_hat_i, self.GraphConv[i](input)))
        ts = torch.stack(ts).sum(dim=0, keepdim=False).to(self.device)
        if self.use_bn:
            ts = ts.transpose(1, 2).contiguous()
            ts = self.bn(ts)
            ts = ts.transpose(1, 2).contiguous()
        ts = F.relu(ts)
        if self.dropout is not None:
            ts = F.dropout(ts, p=self.dropout, training=self.training)
        return ts

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == "__main__":
    device = torch.device("cuda")

    BATCH_SIZE = 1
    SEQ_LEN = 8
    D = 6
    ET = 3
    CLASSN = 2
    adj = torch.sparse.FloatTensor(
        torch.LongTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 1, 2, 3, 4, 5, 6, 7],
                          [1, 3, 0, 2, 1, 5, 0, 4, 3, 7, 2, 6, 5, 7, 4, 6, 0, 1, 2, 3, 4, 5, 6, 7]]),
        torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        torch.Size([BATCH_SIZE, ET, SEQ_LEN, SEQ_LEN])).to_dense().to(device)
    # print(adj.shape)
    input = torch.randn(BATCH_SIZE, SEQ_LEN, D).to(device)
    label = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1]).to(device)

    cc = GraphConvolution(in_features=D, out_features=D, edge_types=ET, device=device, use_bn=True)
    oo = BottledOrthogonalLinear(in_features=D, out_features=CLASSN).to(device)

    optimizer = Adadelta(list(cc.parameters()) + list(oo.parameters()))

    aloss = 1e9
    df = 1e9
    while df > 1e-7:
        output = oo(cc(input, adj)).view(BATCH_SIZE * SEQ_LEN, CLASSN)
        loss = F.cross_entropy(output, label)
        df = abs(aloss - loss.item())
        aloss = loss.item()
        loss.backward()
        optimizer.step()
        log(aloss)

    log(F.softmax(output), dim=2)
