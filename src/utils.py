import datetime
import errno
import os
import random
import numpy as np
import torch
import json
import re
import collections
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, roc_auc_score

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.bool()

def transition_matrix_generate(noise_rate=0.1, num_classes=2):
    # https://github.com/xiaoboxia/T-Revision/blob/master/tools.py#L4
    # generate symmetric matrix
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, num_classes-1):
            P[i, i] = 1. - n
        P[num_classes-1, num_classes-1] = 1. - n

    return P

def generate_model_name(args):
    full_name = []

    if args['use_conven_seq'] is True:
        full_name.append('base')

    if args['model'] == 'han' or args['model']=='NetDetect':
        if args['path'] == 'total':
            full_name.append(args['model'])
        else:
            # use single meta-path
            full_name.append(args['model']+'({})'.format(args['path']))
    elif args['model'] is None:
        full_name.append('None')
    else:
        full_name.append(args['model'])

    # if args['use_seq_pooling'] is False:
    #     full_name.append('NoTempAttn')
    # if args['use_path_pooling'] is False:
    #     full_name.append('NoHierAttn')

    return '-'.join(full_name)

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def setup_log_dir(args):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args["log_dir"].format(args["data"]), "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            args["model_name"], args["emb_init"],
            (args['use_noisy'],args['noise_rate']),
            args["lr"], args["dropout"],
            args["weight_decay"], args["num_heads"],
            args['threshold_logitdim'],
            args["fl_param"], args["num_epochs"],
            args["MAX_SEQ"], args["hidden_size"], args["fraud_dim"],
            date_postfix)
    )
    mkdir_p(log_dir)
    return log_dir

def get_date_postfix():
    """
    Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    # post_fix = "{}_{:02d}-{:02d}-{:02d}".format(dt.date(), dt.hour, dt.minute, dt.second)
    post_fix = dt.date()
    return post_fix

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print("Directory {} already exists.".format(path))
        else:
            raise

def evaluate(y_prob, y_pred_binary, labels):
    '''
        accuracy = (y_pred_binary == labels).sum() / len(y_pred_binary)
        prec = precision_score(labels, y_pred_binary)
        recall = recall_score(labels, y_pred_binary)
        f1 = f1_score(labels, y_pred_binary, average="binary")
    '''

    result = classification_report(labels, y_pred_binary, target_names=['0', '1'], output_dict=True)
    auc = roc_auc_score(labels, y_prob)
    accuracy = result['accuracy']
    # result of the fraud set
    fraud_prec = result['1']['precision']
    fraud_recall = result['1']['recall']

    # result of the nonfraud set
    legit_prec = result['0']['precision']
    legit_recall = result['0']['recall']

    # macro f1
    macro_avg_f1 = result['macro avg']['f1-score']

    # micro-f1
    weight_avg_f1 = result['weighted avg']['f1-score']

    pos_neg_rate = result['0']['support'] / result['1']['support']
    return round(auc, 7), round(accuracy, 5), round(fraud_prec, 5), round(fraud_recall, 5), round(legit_prec, 5), \
           round(legit_recall, 5), round(macro_avg_f1, 5), round(weight_avg_f1, 5), round(pos_neg_rate, 2)

def calculate_avg_results(save_path):
    results = collections.defaultdict(list)
    for root, dirs, files in os.walk(save_path):

        # read '*.log' files only
        log_files = sorted([f for f in files if '.log' in f])

        for log in log_files:
            with open(os.path.join(save_path, log), 'r') as f:
                output = f.readlines()[-1].replace('NeD BEST Test:', '')
            output = re.sub(r'\t| ', '', output)
            output = re.sub(r"'", r'"', output)
            output = json.loads(output)
            results['auc'].append(output['auc'])
            results['fr_prec'].append(output['fr_prec'])
            results['fr_re'].append(output['fr_re'])
            results['le_prec'].append(output['le_prec'])
            results['le_re'].append(output['le_re'])
            results['mac_f1'].append(output['mac_f1'])

    result_avg = {
            'auc': round(np.mean(results['auc']), 4),
            'fr_prec': round(np.mean(results['fr_prec']), 4),
            'fr_re': round(np.mean(results['fr_re']), 4),
            'le_prec': round(np.mean(results['le_prec']), 4),
            'le_re': round(np.mean(results['le_re']), 4),
    }
    fr_f1 = 2 * result_avg['fr_prec'] * result_avg['fr_re'] / (result_avg['fr_prec'] + result_avg['fr_re'])
    le_f1 = 2 * result_avg['le_prec'] * result_avg['le_re'] / (result_avg['le_prec'] + result_avg['le_re'])
    result_avg['mac_f1'] = round((fr_f1 + le_f1) / 2, 4)

    with open(os.path.join(save_path, 'result.json'), 'w') as f:
        json.dump({'variants': log_files,
                   'results_list': results,
                   'result_avg': result_avg,
       }, f, indent=4)
    return results

class EarlyStopping(object):
    def __init__(self, dir_path, args):
        dt = datetime.datetime.now()
        # save model
        self.filepath = dir_path

        self.args = args
        self.patience = args["patience"]
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.best_epoch = None
        self.early_stop = False

    def step(self, loss, auc, model, epoch, optimizer, checkpoint_name=''):
        if self.best_loss is None:
            self.best_auc = auc
            self.best_loss = loss
            self.best_epoch = epoch
            self.save_checkpoint(model, optimizer, checkpoint_name)
        elif (loss > self.best_loss) and (auc < self.best_auc):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        elif auc >= self.best_auc:
            # save model
            self.best_epoch = epoch
            self.best_auc = auc
            self.save_checkpoint(model, optimizer, checkpoint_name)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop



    def save_checkpoint(self, model, optimizer, checkpoint_name):
        """Saves model when validation loss decreases."""

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(self.filepath, f'checkpoint_{checkpoint_name}.pth'))
        self.save_config()

    def load_checkpoint(self, model, optimizer, path=None, checkpoint_name=None):
        """Load the latest checkpoint."""
        if path==None:
            print('Loading checkpoint %s...' % checkpoint_name)
            checkpoint = torch.load(os.path.join(self.filepath, f'checkpoint_{checkpoint_name}.pth'))
        else:
            print('Loading checkpoint %s...' % checkpoint_name)
            checkpoint = torch.load(os.path.join(path, f'checkpoint_{checkpoint_name}.pth'))

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save_config(self):
        with open(os.path.join(self.filepath,'config.json'), 'w') as f:
            json.dump(self.args, f, indent=4)

