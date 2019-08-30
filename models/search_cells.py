""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i): # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = ops.MixedOp(C, stride)
                self.dag[i].append(op)

        # self.dag是个长度为n_nodes的list，list里每个元素也是个list，self.dag[i]长度为2+i，每个元素为一个MixedOp。
        # self.dag[i] = [
        #       MixedOp
        #         ...   (一共i+2个)
        #       MixedOp
        # ]

    def forward(self, s0, s1, w_dag):
        """
        Args:
        :param s0:
        :param s1:
        :param w_dag: list, len=n_nodes, each element is a weight matrix after softmax.
        :return:
        """
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):   # w_list:[i+2, n_ops]
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)

        s_out = torch.cat(states[2:], dim=1)
        return s_out
