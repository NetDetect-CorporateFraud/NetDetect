import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnSum3d(nn.Module):
    def __init__(self,
                 dim,
                 TRAINABLE=False):
        super(AttnSum3d, self).__init__()

        self.dim = dim
        self.TRAINABLE = TRAINABLE
        self.LN = nn.LayerNorm(dim)

        if self.TRAINABLE:
            self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=1)

    def forward(self, input, mask=None):
        # input : [b, m, dim]
        # mask: [b, m], eg:
        #    [[1,1,1,0],
        #     [1,1,0,0],
        #     [1,0,0,0]]
        # out: [b, 1, dim]
        if self.TRAINABLE:
            # Q, K, V
            out, attn_weights = self.attn(input, input, input)
            out = out.mean(dim=1)
            norm_attn_weights = attn_weights.mean(dim=1)
        else:
            attn_weights = torch.bmm(input, input.permute(0, 2, 1)).mean(2, keepdim=True)

            if mask is not None:
                attn_weights = attn_weights.masked_fill((1 - mask.unsqueeze(-1)).bool(), float('-inf'))
                norm_attn_weights = F.softmax(attn_weights, dim=1)

                norm_attn_weights = norm_attn_weights.masked_fill((1 - mask.unsqueeze(-1)).bool(), 0.0)
            else:
                norm_attn_weights = F.softmax(attn_weights, dim=1)
            out = torch.bmm(norm_attn_weights.permute(0, 2, 1), input)
            out = self.LN(out.squeeze(1))
        return out, norm_attn_weights.detach()

class AttnSum2d(nn.Module):
    def __init__(self,
                 dim,
                 TRAINABLE=False):
        super(AttnSum2d, self).__init__()

        self.dim = dim
        self.TRAINABLE = TRAINABLE
        if self.TRAINABLE:

            self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=1)
        self.LN = nn.LayerNorm(dim)

    def forward(self, input):
        # input : [m, dim]
        # out: [1, dim]
        if self.TRAINABLE:
            # Q, K, V
            out, attn_weights = self.attn(input, input, input)
            out = out.mean(dim=0)
        else:
            # [m, m] <- [m, dim] * [dim, m]
            attn_weights = torch.mm(input, input.permute(1, 0))
            attn_weights = torch.softmax(attn_weights, dim=1)
            # (m, m) x (m, dim) -> (m, dim) -> (1, dim)
            # out = torch.mm(attn_weights.mean(1).unsqueeze(0), input)

            out = torch.mm(attn_weights.permute(1, 0), input).mean(dim=0, keepdim=True)
            out = self.LN(out.squeeze())
        return out, attn_weights.detach()

class AttnPooling(nn.Module):
    def __init__(self,
                 dim, dropout=0):
        super(AttnPooling, self).__init__()
        # [D, 1]
        self.U = nn.Parameter(torch.randn(dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.LN = nn.LayerNorm(dim)

    def forward(self, emb, mask=None):
        '''
        :param emb: [N, L, D]
        :param mask: [N, L]
        :return: [N, D]
        '''
        # [N, L, 1]
        w = torch.matmul(emb, self.U)

        if mask is not None:
            # [N, L, 1]
            mask = (1 - mask.unsqueeze(2)).bool()
            w = w.masked_fill(mask, -1e9)

        w = F.softmax(w, dim=1)

        # [N, 1, L] * [N, L, D] -> [N, D]
        output = torch.matmul(w.permute(0, 2, 1), emb).squeeze()
        output = self.dropout(output)
        output = self.LN(output)

        return output, w


class MeanPooling(nn.Module):
    def __init__(self,):
        super(MeanPooling, self).__init__()

    def forward(self, emb, mask):
        # emb: [N, L, D], mask: [N, L]
        normed_mask = mask / (mask.sum(1) + 1e-9).unsqueeze(1)
        # [N, D]
        emb = torch.matmul(normed_mask.unsqueeze(1), emb).squeeze()
        return emb, normed_mask
