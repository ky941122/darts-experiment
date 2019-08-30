""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
from models.voice_attention import Voice_Attention
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


def sequence_mask(lengths, maxlen):
    batch_size = len(lengths)
    lengths = torch.tensor(lengths).unsqueeze(-1).cuda().float()
    maxlen = torch.arange(maxlen).repeat(batch_size, 1).cuda().float()
    mask = maxlen < lengths
    mask = mask.float()
    return mask


class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self,config,
                 C_in, C, n_classes, n_layers,
                 n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.voice_attention_model = Voice_Attention(config.vocab_size, config)
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.ConstantPad2d((1, 1, (3 - 1) * 1, 0), 0.0),
            nn.Conv2d(C_in, C_cur, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d((200, 1))
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal, weights_reduce,
                voice_embeded, sent_nums, max_sent_nums):

        O, voice_A = self.voice_attention_model(x, voice_embeded, sent_nums, max_sent_nums)
        s0 = s1 = self.stem(O)

        i = 0
        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)  # 前两个cell的输出作为这个cell的输入。
            i += 1

        out = self.gap(s1)
        # out = out.view(out.size(0), -1) # flatten
        out = out.transpose(1, 2).squeeze()
        logits = self.linear(out)
        return logits, voice_A


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, C_in, C, n_classes, n_layers, criterion, config,
                 n_nodes=4, stem_multiplier=3, device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.config = config

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))  # +2是因为有两个input node.
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(config,
                             C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)

    def forward(self, x, voice_embeded, sent_nums, max_sent_nums):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce,
                            voice_embeded, sent_nums, max_sent_nums)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y, sent_nums, voice_embeded):
        logits, voice_A = self.forward(X, voice_embeded, sent_nums, self.config.max_sent_nums)
        logits_for_loss = logits.reshape(-1, self.config.n_classes)
        y_for_loss = y.reshape(-1)
        y_for_loss[y_for_loss==2] = 0

        loss =  self.criterion(logits_for_loss, y_for_loss)  # [batch_size*max_sent_nums]
        loss = loss.reshape(self.config.batch_size, self.config.max_sent_nums)  # [batch_size, max_sent_nums]

        # add a mask based on whether it is 0
        balance_mask_1 = sequence_mask(sent_nums, self.config.max_sent_nums)
        balance_mask_2 = (y == 1).float() * 1
        balance_mask_3 = (y == 0).float() * 2
        temp_mask = (balance_mask_2 > 0).float() + (balance_mask_3 > 0).float()
        balance_mask = balance_mask_1 * (balance_mask_2 + balance_mask_3)
        loss = loss * balance_mask
        loss = torch.sum(loss, dim=1)  # shape: (batch)

        temp_label = y.unsqueeze(-1).float()  # [batch, max_length, 1]
        temp_label_T = temp_label.transpose(1, 2)
        temp_label = torch.matmul(1 - temp_label, temp_label_T) + torch.matmul(temp_label, 1 - temp_label_T)  # [batch, max_len, max_len]

        constraint_mask_1 = (temp_label==1).float()
        constraint_mask_2 = sequence_mask(sent_nums, self.config.max_sent_nums)
        constraint_mask_2 = constraint_mask_2.unsqueeze(-1)  # [batch, max_length, 1]
        constraint_mask_2_T = constraint_mask_2.transpose(1, 2)
        constraint_mask_2 = torch.matmul(constraint_mask_2, constraint_mask_2_T)  # [batch, max_length, max_length]
        constraint_mask = constraint_mask_1 * constraint_mask_2
        # care about the attentions
        attention_loss = voice_A ** 2 * constraint_mask  # [batch, max_length, max_length]
        attention_loss = torch.sum(attention_loss, dim=(1, 2))  # [batch]

        loss = loss + self.config.alpha * attention_loss

        loss = torch.mean(loss / torch.sum(balance_mask, dim=1).float())

        # add the accuracy
        pred_label = torch.argmax(logits, dim=-1).long()

        correct_label = (pred_label == y).float()
        correct_label = correct_label * balance_mask_1 * temp_mask
        accuracy = torch.sum(correct_label).float() / torch.sum(temp_mask * balance_mask_1).float()


        return loss, accuracy


    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return [p for p in self.net.parameters() if p.requires_grad]


    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

