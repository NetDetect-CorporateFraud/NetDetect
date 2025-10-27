import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from attention import AttnSum3d, AttnPooling, MeanPooling
from loss import FocalLoss


class HANLayer(nn.Module):
    """
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    layer_num_heads : number of attention heads
    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features
    """

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout, AttnPool=False):

        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()

        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                in_feats = in_size,
                    out_feats = out_size,
                    num_heads = layer_num_heads,
                    feat_drop = dropout,
                    attn_drop = dropout,
                    activation = F.relu,
                    allow_zero_in_degree = True, # 是否允许有度为0的点
                    bias=False
                )
            )
        self.BN = nn.BatchNorm1d(out_size * layer_num_heads)

        self.AttnPool = AttnPool

        if self.AttnPool:
            self.graph_pooling = AttnSum3d(dim=out_size * layer_num_heads)
        else:
            self.graph_pooling = MeanPooling()
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h, path_mask):
        semantic_embeddings = []

        gat_attns = []

        for i, g in enumerate(gs):
            # gat_attn shape : [N, H, 1]
            g_emb, gat_attn = self.gat_layers[i](g, h, get_attention=True)
            g_emb = self.BN(g_emb.flatten(1) + h)

            # average multiple heads weights
            gat_attn = gat_attn.mean(1).squeeze()
            gat_attn = self.attn_coef_to_mat(g, gat_attn, n_node=h.shape[0])
            gat_attns.append(gat_attn.detach())
            # flatten multiple heads
            semantic_embeddings.append(g_emb)

        # (N, M, D)
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )

        # (N, M, D) -> (N, 1, D) -> (N, D)
        if self.AttnPool:
            sum_embeddings, path_attn = self.graph_pooling(semantic_embeddings, path_mask)
        else:
            sum_embeddings, path_attn = self.graph_pooling(semantic_embeddings, path_mask)

        return sum_embeddings.squeeze(1), path_attn.detach(), gat_attns

    def attn_coef_to_mat(self, g, gat_attn, n_node):
        src, dst = g.edges()

        sparse_attn = torch.sparse_coo_tensor(indices=torch.stack([src, dst]),
                                                values=gat_attn,
                                                size=(n_node, n_node))
        return sparse_attn.coalesce()


class HAN(nn.Module):
    def __init__(self, in_size, hidden_size, num_meta_paths, num_heads, dropout, args, pred_=True):
        super(HAN, self).__init__()

        self.pred_ = pred_

        self.args = args

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(num_meta_paths, hidden_size, int(hidden_size/num_heads[0]), num_heads[0], dropout,
                     AttnPool=args['use_path_pooling'])
        )

        for l in range(1, len(num_heads)): # num_heads = [2, 2]
            self.layers.append(
                HANLayer(num_meta_paths,
                    hidden_size,
                    int(hidden_size / num_heads[l - 1]),
                    num_heads[l],
                    dropout,
                    AttnPool=args['use_path_pooling'])
            )
        self.BN = nn.BatchNorm1d(hidden_size, eps=1e-7)

        threshold = args['threshold_logitdim']
        self.threshold = threshold if threshold < 1 else None
        self.out_size = 1 if threshold < 1 else threshold

        if self.pred_:
            mlp_layers = [
                nn.BatchNorm1d(hidden_size, eps=1e-7),
                nn.Linear(hidden_size, 2 * hidden_size),
                nn.Dropout(p=0.5),
                nn.Linear(2 * hidden_size, self.out_size)
            ]
            if self.threshold is not None:
                mlp_layers.append(nn.Sigmoid())
            else:
                mlp_layers.append(nn.Softmax())

            self.mlp = nn.Sequential(* mlp_layers)

            self.gnn_loss = FocalLoss(gamma=args['fl_param'][0],
                                      alpha=args['fl_param'][1],
                                      size_average=False, )

    def forward(self, g, h, path_mask, pred_=True):
        self.pred_ = pred_
        path_attns = []
        gat_attns = []

        residual = h

        for gnn in self.layers:
            h_out, p_attn, g_attn = gnn(g, h, path_mask)

            path_attns.append(p_attn)
            gat_attns.append(g_attn)
            h = self.BN(h_out + h)

        # h = self.MP(torch.stack(emb_layers,dim=1).permute(0,2,1)).squeeze()

        if self.args['emb_init'] != 'None':
            # if  randomly initializing node embedding, the model is better without adding residual
            h += residual

        if self.pred_:
            logit = self.mlp(h).squeeze(1)
            return logit, h, path_attns[0].squeeze(-1), gat_attns[0]
        else:
            return h, path_attns[0].squeeze(-1), gat_attns[0]
