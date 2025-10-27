import torch
from sklearn.preprocessing import StandardScaler
import utils as _U
import argparse
from model import Model
from mlp import MLP
from stagn import STAGN
from smgcn import SMGCN
from hinsage import HINSAGE
from hgt import HGT
import numpy as np
import os
import json
import yaml
import time
from sklearn.model_selection import ParameterGrid
from torch.optim import lr_scheduler


def time_span_seq_align(corp_data, args):

    if args['data'] == 'CN':
        fin_seq, nfin_seq, mda_seq, node_seq, fraud_seq, seq_lens, seq_mask = \
            corp_data['fin_seq'], corp_data['nonfin_seq'], corp_data['mda_seq'], corp_data['node_seq'], corp_data['fraud_series'], corp_data['seq_len'], corp_data['seq_mask']
    else:
        fin_seq, nfin_seq, mda_seq, node_seq, fraud_seq, seq_lens, seq_mask = corp_data['fin_seq'], corp_data['nonfin_seq'], corp_data['mda_seq'], corp_data['node_seq'], corp_data[
                'fraud_series_1'], corp_data['seq_len'], corp_data['seq_mask']

    DEVICE = args["device"]


    new_fin, new_nfin, new_mda, new_node, new_fraud, new_lens, new_seq_mask = [], [], [], [], [], [], []
    SeqPositionOffset = []

    SUB_SEQ = torch.tensor(json.loads(args['MAX_SEQ'])) - 1

    if SUB_SEQ.tolist() == []:
        new_fin, new_nfin, new_mda, new_node, new_fraud, new_lens, new_seq_mask = \
            fin_seq, nfin_seq, mda_seq, node_seq, fraud_seq, seq_lens, seq_mask
        SeqPositionOffset = torch.tensor(SeqPositionOffset)

    elif -6 in SUB_SEQ:
        for i in range(0, len(seq_lens)):
            if seq_lens[i] < SUB_SEQ.shape[0]:
                subseq = SUB_SEQ + 6 - len(SUB_SEQ)
            else:
                subseq = SUB_SEQ + 6 - seq_lens[i]

            new_fin.append(fin_seq[i][subseq])
            new_nfin.append(nfin_seq[i][subseq])
            new_mda.append(mda_seq[i][subseq])
            new_node.append(node_seq[i][subseq])
            new_fraud.append(fraud_seq[i][subseq])
            new_lens.append(min(seq_lens[i], torch.tensor(subseq.shape[0])))
            SeqPositionOffset.append(subseq)

        new_fin, new_nfin, new_mda, new_node, new_fraud, new_lens = torch.stack(new_fin), torch.stack(new_nfin), torch.stack(
            new_mda), torch.stack(new_node), torch.stack(new_fraud), torch.stack(new_lens)

        new_seq_mask = [[0 for _ in range(SUB_SEQ.shape[0]-l)] + [1 for _ in range(l)] for l in new_lens]
        new_seq_mask = torch.tensor(new_seq_mask)
        SeqPositionOffset = torch.stack(SeqPositionOffset)

    elif -1 in SUB_SEQ:
        for i in range(0, len(seq_lens)):
            new_fin.append(fin_seq[i][SUB_SEQ])
            new_nfin.append(nfin_seq[i][SUB_SEQ])
            new_mda.append(mda_seq[i][SUB_SEQ])
            new_node.append(node_seq[i][SUB_SEQ])
            new_fraud.append(fraud_seq[i][SUB_SEQ])
            new_lens.append(min(seq_lens[i], torch.tensor(SUB_SEQ.shape[0])))
            SeqPositionOffset.append(SUB_SEQ)

        new_fin, new_nfin, new_mda, new_node, new_fraud, new_lens = torch.stack(new_fin), torch.stack(new_nfin), torch.stack(
            new_mda), torch.stack(new_node), torch.stack(new_fraud), torch.stack(new_lens)

        new_seq_mask = [[0 for _ in range(SUB_SEQ.shape[0] - l)] + [1 for _ in range(l)] for l in new_lens]
        new_seq_mask = torch.tensor(new_seq_mask)
        SeqPositionOffset = torch.stack(SeqPositionOffset)

    return SeqPositionOffset.to(DEVICE), new_fin.to(DEVICE), new_nfin.to(DEVICE), new_mda.to(DEVICE), new_node.to(DEVICE), new_fraud.to(DEVICE), new_lens.to(DEVICE), new_seq_mask.to(DEVICE)

def select_path(y, args, gs):
    if args['path'] == 'total':
        adj = [adj.to(args["device"]) for adj in gs[y]['adj']]
        path_mask = gs[y]['path_mask'].to(args["device"])
    else:
        adj = []
        path_mask = []
        if 'Equity' in args['path']:
            adj += [gs[y]['adj'][j].to(args["device"]) for j in range(0, 4)]
            path_mask.append(gs[y]['path_mask'][:,:4])
        if 'Employ' in args['path']:
            adj += [gs[y]['adj'][j].to(args["device"]) for j in range(4, 7)]
            path_mask.append(gs[y]['path_mask'][:, 4:7])
        if 'Kinship' in args['path']:
            adj += [gs[y]['adj'][j].to(args["device"]) for j in range(7, 10)]
            path_mask.append(gs[y]['path_mask'][:, 7:10])
        if 'RPTs' in args['path']:
            adj += [gs[y]['adj'][j].to(args["device"]) for j in range(10, 26)]
            path_mask.append(gs[y]['path_mask'][:, 10:])

        path_mask = torch.cat(path_mask, dim=-1).to(args["device"])

    return adj, path_mask

def train_graphs(args, model, optimizer,
                 sample_ids, fraud_labels, gs, corp_data, b_y, e_y):

    losses, logits, labels = [], [], []
    path_attn = []
    gat_attn = []
    DEVICE = args["device"]

    raw_fin_seq, raw_nfin_seq, raw_mda_seq, raw_node_seq, raw_fraud_seq = \
        corp_data['fin_seq'], corp_data['nonfin_seq'], corp_data['mda_seq'], corp_data['node_seq'], corp_data[
            'fraud_series']

    for y_ in range(b_y, e_y):
        adj, path_mask = select_path(y_, args, gs)

        node_id = sample_ids[y_].to(args["device"])

        if args['data'] == 'US':
            sample_id_in_listedCorp = corp_data['hetegraphs'][y_]['sample_id_in_listedCorp'].to(args["device"])
        else:
            sample_id_in_listedCorp = corp_data['hetegraphs'][y_]['sample_id_in_listedCorp'].to(args["device"])

        if args['model'] in ['HinSAGE', 'HGT']:
            logit, attn_w= model.gnn_forward(y_, node_id,
                        sample_id_in_listedCorp,
                        t_fin_id = raw_fin_seq[node_id][:, -1].squeeze().to(DEVICE),
                        t_nfin_id = raw_nfin_seq[node_id][:, -1].squeeze().to(DEVICE),
                        t_mda_id = raw_mda_seq[node_id][:, -1].squeeze().to(DEVICE),
                        t_fraud_id = raw_fraud_seq[node_id][:, -1].to(DEVICE),
                        pred_=True)
            label = raw_fraud_seq[node_id][:, -2].squeeze().to(args["device"])
            loss = model.gnns[y_-b_y].gnn_loss(logit, label)
        else:
            logit, attn_w = model.gnn_forward(adj, node_id, path_mask,
                                        t_fin_id = raw_fin_seq[node_id][:, -1].squeeze().to(DEVICE),
                                        t_nfin_id = raw_nfin_seq[node_id][:, -1].squeeze().to(DEVICE),
                                        t_mda_id = raw_mda_seq[node_id][:, -1].squeeze().to(DEVICE),
                                        t_fraud_id = raw_fraud_seq[node_id][:, -1].to(DEVICE))
            label = raw_fraud_seq[node_id][:, -2].squeeze().to(args["device"])
            loss = model.gnn.gnn_loss(logit, label)

        losses.append(loss.item())
        logits.append(logit)
        labels.append(label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args['print_param']:
            path_attn.append(attn_w[0].cpu())
            gat_attn.append([p.cpu() for p in attn_w[1]])

    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)

    results = model.score(logits, labels, np.mean(losses))

    return results, path_attn, gat_attn

def train_model(args, model, optimizer, sample_ids, fraud_labels,
                gs, fin_seqs, nfin_seqs, mda_seqs, node_seqs, fraud_seqs,
                seq_lens, seq_masks, transition_matrix,
                start_y, end_y, IS_TRAIN = True):
    losses, logits, labels = [], [], []

    gru_attn = []

    for y_ in range(start_y, end_y):

        adj, path_mask = select_path(y_, args, gs)

        node_id = sample_ids[y_].to(args["device"])

        y_ = torch.tensor(y_, dtype=torch.int32).to(args["device"])

        if IS_TRAIN:
            logit, attn_w = model(y_,
                adj, path_mask, node_id,
                node_seqs[node_id], fin_seqs[node_id],
                nfin_seqs[node_id], mda_seqs[node_id],
                fraud_seqs[node_id],
                seq_lens[node_id], seq_masks[node_id])
        else:
            with torch.no_grad():
                logit, attn_w = model(y_,
                    adj, path_mask, node_id,
                    node_seqs[node_id], fin_seqs[node_id],
                    nfin_seqs[node_id], mda_seqs[node_id],
                    fraud_seqs[node_id],
                    seq_lens[node_id], seq_masks[node_id])

        label = fraud_labels[node_id]

        loss = model.cal_loss(logit, label, transition_matrix)
        losses.append(loss.item())
        logits.append(logit)

        labels.append(label)

        if IS_TRAIN:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if args['print_param']:
            gru_attn.append(attn_w[2].cpu())

    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)

    # print(logits)

    results = model.score(logits, labels, np.mean(losses))

    return results, logits.detach().cpu(), labels.cpu(), gru_attn


def main(args, log_name):
    models = {
        "NN": MLP,
        "NetDetect": Model,
        "SMGCN": SMGCN,
        "STAGN": STAGN,
        "HinSAGE": HINSAGE,
        "HGT": HGT,
    }

    if args['data'] == "CN":
        corp_data = torch.load(f"../../CN_data/CN_data.pt")
    elif args['data'] == "US":
        corp_data = torch.load(f"../../US_data/US_data.pt")

    seq_posit_offset, fin_seqs, nfin_seqs, mda_seqs, node_seqs, fraud_seqs, seq_lens, seq_masks = time_span_seq_align(corp_data, args)

    sample_ids = corp_data['sample_idx_by_year']
    fraud_labels = corp_data['label'].to(args["device"])

    scaler = StandardScaler()
    fin_emb = torch.tensor(scaler.fit_transform(corp_data['fin_ratio'])).to(args["device"])

    nonfin_emb = corp_data['nonfin_ratio'].to(args["device"])

    try:
        mda_bert_embs = corp_data['mda_bert'].to(args["device"])
    except:
        mda_bert_embs = torch.zeros(1,0)

    gs = corp_data['graphs']

    if args['data'] == 'US':
        hetegraphs = [hg['graph'].to(args["device"]) for year, hg in corp_data['hetegraphs'].items()]
    else:
        hetegraphs = [hg['graph'].to(args["device"]) for year, hg in corp_data['hetegraphs'].items()]

    kg_embs = torch.cat([gs[y]['kg_embs'] for y in gs], dim=0).to(args["device"])
    '''
    --------------------------------- initialize model ---------------------------------
    '''
    if args["path"] == 'total':
        num_path = 26 if args["data"] == 'CN' else 7
    else:
        num_path = 0
        if 'Equity' in args['path']:
            num_path += 4
        if 'Employ' in args['path']:
            num_path += 3
        if 'Kinship' in args['path']:
            num_path += 3
        if 'RPTs' in args['path']:
            num_path += 16

    if args['model'] in ['HinSAGE', 'HGT']:
        model = models[args["model"]](
            hetegraphs = hetegraphs,
            embs=[fin_emb, nonfin_emb, mda_bert_embs, kg_embs],
            args=args,
            num_sample=fraud_labels.shape[0],
            num_path=num_path,
        ).to(args["device"])
    else:
        model = models[args["model"]](
            embs = [fin_emb, nonfin_emb, mda_bert_embs, kg_embs],
            args=args,
            num_sample = fraud_labels.shape[0],
            num_path = num_path,
        ).to(args["device"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"], eps=1e-4
    )
    logger = _U.EarlyStopping(dir_path=args["log_dir"], args=args)


    train_loss = []

    if args["use_noisy"] is not None or args["use_noisy"]!='None':
        T_M = _U.transition_matrix_generate(noise_rate=args['noise_rate'], num_classes=2)
        T_M = torch.from_numpy(T_M).float().to(args["device"])
    else:
        T_M = None

    b_y = 2003 if args['data'] == 'CN' else 2000
    e_y = 2022 if args['data'] == 'CN' else 2020

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args['lr_adjust'][0], gamma=args['lr_adjust'][0])

    total_time, get_para_num = 0, False

    if not args["init_checkpoint"]:
        for epoch in range(args["num_epochs"]):
            print('*' * 60 + 'Epoch {:d}'.format(epoch) + '*' * 60)
            with open(os.path.join(args["log_dir"], f'train_{log_name}.log'), 'a+') as f:
                f.write("Epoch: {}\n".format(epoch))

            path_attns = []
            gat_attns = []
            gru_attns = []
            logits = []

            # ------------------- train --------------------
            model.train()

            if model.use_graph:

                start_time = time.time()
                gnn_results, path_attn, gat_attn = train_graphs(
                    args, model, optimizer, sample_ids, fraud_labels, gs, corp_data, b_y, args["tvt_split"][0])
                end_time = time.time()
                graph_time = end_time - start_time
                gnn_results['train_time'] = graph_time

                '''print('---NeD---1st stage Train: {}'.format(gnn_results))
                with open(os.path.join(args["log_dir"], f'train_{log_name}.log'), 'a+') as f:
                    f.write("\t 1st stage GNN Train: {}\n".format(gnn_results))
                '''
                if args['print_param']:
                    path_attns += path_attn
                    gat_attns += gat_attn


            train_results, train_logit, _, train_gru_attn = train_model(
                args, model, optimizer,
                sample_ids, fraud_labels, gs,
                fin_seqs, nfin_seqs, mda_seqs, node_seqs, fraud_seqs,
                seq_lens, seq_masks, T_M,
                start_y = b_y, end_y=args["tvt_split"][0], IS_TRAIN=True)

            train_loss.append(train_results['loss'])

            scheduler.step()

            print('---NeD---Train: {}'.format(train_results))
            time.sleep(1)

            with open(os.path.join(args["log_dir"], f'train_{log_name}.log'), 'a+') as f:
                f.write("\t NeD Train: {}\n".format(train_results))

            if not get_para_num:
                total_params = sum(p.numel() for p in model.parameters())
                print("Total Parameters:", total_params)
                with open(os.path.join(args["log_dir"], f'train_{log_name}.log'), 'a+') as f:
                    f.write("Total Parameters: {}\n".format(total_params))
                get_para_num = True

            # -------------------- valid ----------------------
            if epoch % args["valid_epoch"] == 0:
                model.eval()

                valid_results, valid_logit, _, valid_gru_attn = train_model(
                    args, model, optimizer,
                    sample_ids, fraud_labels, gs,
                    fin_seqs, nfin_seqs, mda_seqs, node_seqs, fraud_seqs,
                    seq_lens, seq_masks, T_M,
                    start_y=args["tvt_split"][0], end_y=args["tvt_split"][1], IS_TRAIN=False)

                print('---NeD---Valid: {}'.format(valid_results))

                time.sleep(1)
                with open(os.path.join(args["log_dir"], f'train_{log_name}.log'), 'a+') as f:
                    f.write("\t NeD Valid: {}\n".format(valid_results))

                # -------------------- test ----------------------
                model.eval()

                start_time = time.time()

                test_results, test_logit, _, test_gru_attn = train_model(
                    args, model, optimizer,
                    sample_ids, fraud_labels, gs,
                    fin_seqs, nfin_seqs, mda_seqs, node_seqs, fraud_seqs,
                    seq_lens, seq_masks, T_M,
                    start_y=args["tvt_split"][1], end_y=e_y, IS_TRAIN=False)

                end_time = time.time()
                test_results['infer_time'] = end_time - start_time

                if args['print_param']:
                    logits += [train_logit, valid_logit, test_logit]
                    gru_attns = train_gru_attn + valid_gru_attn + test_gru_attn

                early_stop = logger.step(loss=test_results['loss'], auc=test_results['auc'],
                                         model=model, epoch=epoch, optimizer=optimizer,
                                         checkpoint_name = log_name)


                print('---NeD---Test: {}'.format(test_results))

                with open(os.path.join(args["log_dir"], f'train_{log_name}.log'), 'a+') as f:
                    f.write("\t NeD Test: {}\n".format(test_results))

                if logger.best_epoch == epoch and args['print_param']:
                    param_dict = {
                        'logits' : torch.cat(logits, dim=0).cpu(),
                        'path_attn': {y : path_attns[y- b_y] for y in range(b_y, e_y)},
                        'gat_attn': {y: gat_attns[y- b_y] for y in range(b_y, e_y)},
                        'gru_attn': {y: gru_attns[y- b_y] for y in range(b_y, e_y)},
                    }
                    torch.save(param_dict, f'{args["log_dir"]}/hidden_states_{args["seed"]}.pt')

        logger.load_checkpoint(model, optimizer, checkpoint_name=log_name)

    else:
        # no need to train and validate.
        logger.load_checkpoint(model, optimizer,
            path=f'../../output/{args["data"]}/results_replicate/' + args["init_checkpoint"], checkpoint_name=log_name)

    # best test
    model.eval()
    best_test_results, _, _, _ = train_model(
        args, model, optimizer,
        sample_ids, fraud_labels, gs,
        fin_seqs, nfin_seqs, mda_seqs, node_seqs, fraud_seqs,
        seq_lens, seq_masks, T_M,
        start_y=args["tvt_split"][1], end_y=e_y, IS_TRAIN=False)

    print("BEST Epoch: {}".format(logger.best_epoch))
    print('---NeD---Best Test: {}'.format(best_test_results))

    if True:
        with open(os.path.join(args["log_dir"], f'train_{log_name}.log'), 'a+') as f:
            f.write("BEST Epoch: {}\n".format(logger.best_epoch))
            f.write("\t NeD BEST Test: {}\n".format(best_test_results))


def set_up(args, default_configure):
    args.update(default_configure)
    _U.set_random_seed(args["seed"])
    args['model_name'] = _U.generate_model_name(args)
    args["log_dir"] = _U.setup_log_dir(args)

    return args

def run_n_fold(args, params_grid):
    for p_dict in params_grid:
        for key in p_dict:
            args[key] = p_dict[key]

        log_name = '_'.join([str(p_dict[k]) for k in p_dict])
        print('-' * 50 + log_name + '-' * 50)

        _U.set_random_seed(args["seed"])
        main(args, log_name)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NetDetect")
    # data configuration
    parser.add_argument("--data", type=str, default='CN', help="dataset source: US, CN")
    parser.add_argument("--init_checkpoint", type=str,
        # default=None,
        default="base-NetDetect_Conven_('reweight', 0.04)_2e-05_0.6_0.05_[4]_0.485_[4, 0.038]_25_[0]_256",
        help='initial checkpoint path')

    args = parser.parse_args().__dict__

    # overwrite previous config
    config_yaml = f'config/{args["data"]}/NetDetect(T-0).yaml'  # please manually define path

    assigned_config = load_config(config_yaml)

    # update args with input yaml file
    args = set_up(args, default_configure = assigned_config)

    # specify the parameters e.g., random seeds
    params_grid = {
        'seed': [2020, 2028, 2036, 2037, 2040],
    }

    params_grid = list(ParameterGrid(params_grid))
    run_n_fold(args, params_grid=params_grid)

    save_path = args["log_dir"]
    # You can specify the save_path to manually calculate average results by yourself.
    # save_path = f'../../output/{args["data"]}/results_replicate/' + "base-NetDetect_Conven_('reweight', 0.04)_2e-05_0.7_0.05_[4, 4]_0.485_[4, 0.049]_30_[]_256"
    _U.calculate_avg_results(save_path=save_path)
