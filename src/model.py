import torch
import torch.nn as nn
import json
from hgan import HAN
import sys
sys.path.append('..')
from loss import FocalLoss, ReweightLoss, ReweightRevisionLoss
from attention import AttnSum3d, AttnPooling, MeanPooling
import utils as _U

class Model(nn.Module):
    def __init__(self, embs, args,
                 num_sample,
                 num_path,
        ):
        super(Model, self).__init__()
        fin_embs, nonfin_embs, mda_bert_embs, kg_embs = embs[0], embs[1], embs[2], embs[3]
        self.args = args
        self.device = args['device']
        self.data = args['data']
        self.n_graph = kg_embs.shape[0]
        self.n_entities = kg_embs.shape[1]
        self.n_path = num_path
        self.hidden_dim = args["hidden_size"]


        self.max_seq_len = 6 if args['MAX_SEQ']=='[]' else len(json.loads(args['MAX_SEQ']))

        self.dropout = self.args['dropout']
        self.num_heads = self.args['num_heads']
        self.fin_size = fin_embs.size(-1)
        self.nfin_size = nonfin_embs.size(-1)
        self.bert_size = mda_bert_embs.size(-1)
        self.fraud_dim = self.args['fraud_dim']
        self.numeric_size = self.fin_size + self.nfin_size

        threshold = self.args['threshold_logitdim']
        self.threshold = threshold if threshold < 1 else None
        self.out_size = 1 if threshold < 1 else threshold

        if self.fin_size > 0:
            self.fin_embs = nn.Embedding(fin_embs.shape[0] + 1, self.fin_size, padding_idx=-1)
            self.fin_embs.weight.data[:-1].copy_(fin_embs)
        if self.nfin_size > 0:
            self.nfin_embs = nn.Embedding(nonfin_embs.shape[0] + 1, self.nfin_size, padding_idx=-1)
            self.nfin_embs.weight.data[:-1].copy_(nonfin_embs)
        if self.bert_size > 0:
            self.bert_embs = nn.Embedding(mda_bert_embs.shape[0] + 1, self.bert_size, padding_idx=-1)
            self.bert_embs.weight.data[:-1].copy_(mda_bert_embs)
        if self.fraud_dim > 0:
            self.fraud_embs = nn.Embedding(2, self.fraud_dim)

        if self.args['use_conven_seq'] or self.args['emb_init'] == 'Conven':
            if self.bert_size > 0:
                self.text_dim = max( self.hidden_dim - self.numeric_size - self.fraud_dim, 1)

                self.text_encoder = nn.Sequential(
                        nn.BatchNorm1d(self.bert_size, eps=1e-7),
                        nn.Linear(self.bert_size, self.text_dim),
                        nn.ReLU(),
                )
            else:
                self.text_dim = self.bert_size

            self.conv_dim = self.numeric_size + self.fraud_dim + self.text_dim

            self.conven_proj = nn.Sequential(
                nn.BatchNorm1d(self.conv_dim, eps=1e-7),
                nn.Linear(self.conv_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            )

        self.seq_dim = 0
        if self.args['use_conven_seq']:
            self.seq_dim += self.hidden_dim

        self.use_graph = False if 'NN' in self.args['model'] else True

        if self.use_graph:
            self.out_embs = nn.Embedding(num_sample + 1, self.hidden_dim)

            if self.args['emb_init'] == 'Conven':
                self.in_dim = self.conv_dim
            else:
                self.in_dim = self.hidden_dim
                # nn.init.kaiming_normal_(self.out_embs.weight, nonlinearity='relu')

            if 'NetDetect' in self.args['model']:
                self.gnn = HAN(in_size = None,
                               hidden_size = self.hidden_dim,
                               num_meta_paths=self.n_path, num_heads = self.num_heads, dropout=self.dropout, args=self.args, )

            self.seq_dim += self.hidden_dim

        if self.args['use_rnn']:
            self.LN = nn.LayerNorm(self.seq_dim)

            self.rnn = nn.LSTM(input_size = self.seq_dim,
                                  hidden_size = int(self.seq_dim / 2),
                                  num_layers = 1,
                                  dropout = self.dropout,
                                  bidirectional = True,
                                  batch_first = True)

        self.mlp_in_dim = int(self.seq_dim / 2) * 2 if self.args['use_rnn'] else self.seq_dim
        if self.args['use_seq_pooling']:
            self.seq_pooling = AttnSum3d(self.mlp_in_dim)
        else:
            self.seq_pooling = MeanPooling()

        mlp_layers = [
            nn.BatchNorm1d(self.mlp_in_dim, eps=1e-7),
            nn.Linear(self.mlp_in_dim, int(self.mlp_in_dim * 2)),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.mlp_in_dim * 2), self.out_size)
        ]
        if self.threshold is not None:
            mlp_layers.append(nn.Sigmoid())


        self.mlp = nn.Sequential(*mlp_layers)


        if self.args['use_noisy'] == 'reweight':
            self.rl_ = ReweightLoss(gamma=args['fl_param'][0],
                                alpha=args['fl_param'][1],
                                size_average=False)

        elif self.args['use_noisy'] == 'reweight_revision':
            self.rrl_ = ReweightRevisionLoss(gamma=args['fl_param'][0],
                                    alpha=args['fl_param'][1],
                                    size_average=False)
            self.T_revision = nn.Linear(2, 2, False)

        else: # focal loss
            self.fl_ = FocalLoss(gamma=args['fl_param'][0],
                                 alpha=args['fl_param'][1],
                                 size_average=False,)

    def forward(self, year,
                adjs, path_mask, t_node_ids,
                node_seq,
                fin_seq, nfin_seq, mda_seq, fraud_seq, seq_len, seq_mask):

        output = []
        path_attns, gat_attns, seq_attn_weights = None, None, None

        if self.use_graph:
            # [N, D]
            output.append( self.out_embs(node_seq) )

        if self.args['use_conven_seq']:

            conv_emb = []
            if self.fin_size > 0:
                fin_emb = self.fin_embs(fin_seq)
                conv_emb.append(fin_emb)
            if self.nfin_size > 0:
                nfin_emb = self.nfin_embs(nfin_seq)
                conv_emb.append(nfin_emb)
            if self.bert_size > 0:
                bert_emb = self.text_encoder(
                    self.bert_embs(mda_seq).view(-1, self.bert_size)).view(
                        -1, self.max_seq_len, self.text_dim)
                conv_emb.append(bert_emb)
            if self.fraud_dim > 0:
                fraud_emb = self.fraud_embs(fraud_seq)
                conv_emb.append(fraud_emb)

            conv_emb = torch.cat(conv_emb, dim=-1)
            conv_emb = self.conven_proj(conv_emb.view(-1, self.conv_dim)).view(-1, self.max_seq_len ,self.hidden_dim)
            output.append(conv_emb)

        output = torch.cat(output, dim=-1)

        if self.args['use_rnn']:
            output, _ = self.rnn(output)

        if self.args['use_seq_pooling']:

            output, seq_attn_weights = self.seq_pooling(output, seq_mask)
        else:
            output, seq_attn_weights = self.seq_pooling(output, seq_mask)

        mlp_in = output.squeeze()
        logits = self.mlp(mlp_in).squeeze(1)

        if self.args['print_param']:
            out_w = [path_attns, gat_attns, seq_attn_weights.squeeze().detach()]
        else:
            out_w = None

        self.correction = self.T_revision.weight if self.args['use_noisy'] == 'reweight_revision' else None

        return logits, out_w

    def gnn_forward(self, adjs, t_node_ids, path_mask,
                    t_fin_id, t_nfin_id, t_mda_id, t_fraud_id):

        path_attns, gat_attns = None, None

        if self.args['emb_init'] == 'Conven':
            t_node_embs = [
                self.fin_embs(t_fin_id),
            ]
            if self.nfin_size > 0:
                nfin_emb = self.nfin_embs(t_nfin_id)
                t_node_embs.append(nfin_emb)
            if self.bert_size > 0:
                bert_emb = self.text_encoder(
                        self.bert_embs(t_mda_id))
                t_node_embs.append(bert_emb)

            if self.fraud_dim > 0:
                t_node_embs.append(self.fraud_embs(t_fraud_id))

            t_node_embs = torch.cat(t_node_embs, dim=-1)
            t_node_embs = self.conven_proj(t_node_embs)
        else:
            t_node_embs = self.out_embs(t_node_ids)
            # t_node_embs = self.node_proj(self.out_embs(t_node_ids))

        if 'NetDetect' in self.args['model']:
            # [G, node_emb_size]
            gnn_logit, gnn_output, path_attns, gat_attns = self.gnn(adjs, t_node_embs, path_mask)

        self.out_embs.weight.data[t_node_ids] = gnn_output

        if self.args['print_param']:
            out_w = [path_attns, gat_attns]
        else:
            out_w = None

        return gnn_logit, out_w

    def cal_loss(self, logits, labels, T_=None):

        if self.args['use_noisy']=='reweight':
            if logits.dim() == 1:
                logits = torch.cat([(1 - logits).view(-1,1), logits.view(-1,1)], dim=1)
            loss = self.rl_(logits, labels, T_)
        elif self.args['use_noisy']=='reweight_revision':
            loss = self.rrl_(logits, labels, T_, self.correction)
        else:
            loss = self.fl_(logits, labels)

        return loss

    def score(self, logits, labels, loss=None):

        if logits.dim() > 1:
            _, indices = torch.max(logits, dim=1)
            y_prob = logits[:, 1].detach().cpu().numpy()
            y_pred_binary = indices.long().cpu().numpy()
        else:
            if self.threshold is not None:
                y_prob = logits.detach().cpu().numpy()
                y_pred_binary = (logits >= self.threshold).long().cpu().numpy()

        if 'Prob' in self.args and self.args['Prob'] is False:
            y_prob = y_pred_binary

        labels = labels.cpu().numpy()


        auc, acc, fr_prec, fr_recall, le_prec, le_recall, macro_f1, micro_f1, pos_neg_rate = _U.evaluate(
            y_prob, y_pred_binary, labels)

        output_dict = {
            'loss': round(loss, 7),
            'auc': auc, 'fr_prec': fr_prec, 'fr_re': fr_recall,
            'le_prec': le_prec, 'le_re': le_recall,
            'mac_f1': macro_f1,
            'pos_neg': pos_neg_rate
        }
        return output_dict

