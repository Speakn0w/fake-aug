from functools import partial
from itertools import product

import re
import sys
import argparse
from utils import logger, logger_gcn, logger_binary, logger_gcn_binary
from datasets import get_dataset, get_gacl_dataset
from train_eval import cross_validation_with_label, cross_validation_gcn, cross_validation_gacl
from res_gcn import ResGCN
import os
import json
import time
import warnings

warnings.filterwarnings('ignore')

str2bool = lambda x: x.lower() == "true"
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default="benchmark")
parser.add_argument('--data_root', type=str, default="/data/zhangjiajun/autoaug_fakenews/benchmark_dataset/tudatasets/")
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=500)
parser.add_argument('--epoch_select', type=str, default='test_max')
parser.add_argument('--n_layers_feat', type=int, default=1)
parser.add_argument('--n_layers_conv', type=int, default=3)
parser.add_argument('--n_layers_fc', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--res_branch', type=str, default="BNConvReLU")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--edge_norm', type=str2bool, default=True)
parser.add_argument('--with_eval_mode', type=str2bool, default=True)
parser.add_argument('--dataset', type=str, default="twitter16")
# parser.add_argument('--aug1', type=str, default="None")
# parser.add_argument('--aug_ratio1', type=float, default=0.2)
# parser.add_argument('--aug2', type=str, default="None")
# parser.add_argument('--aug_ratio2', type=float, default=0.2)
# parser.add_argument('--suffix', type=int, default=0)
parser.add_argument('--n_percents', type=int, default=5)
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--sep_by_event', type=int, default=0)
args = parser.parse_args()


def create_n_filter_triples(datasets, feat_strs, nets, gfn_add_ak3=False,
                            gfn_reall=True, reddit_odeg10=False,
                            dd_odeg10_ak1=False):
    triples = [(d, f, n) for d, f, n in product(datasets, feat_strs, nets)]
    triples_filtered = []
    for dataset, feat_str, net in triples:
        # Add ak3 for GFN.
        if gfn_add_ak3 and 'GFN' in net:
            feat_str += '+ak3'
        # Remove edges for GFN.
        if gfn_reall and 'GFN' in net:
            feat_str += '+reall'
        # Replace degree feats for REDDIT datasets (less redundancy, faster).
        if reddit_odeg10 and dataset in [
                'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
        # Replace degree and akx feats for dd (less redundancy, faster).
        if dd_odeg10_ak1 and dataset in ['DD']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
            feat_str = feat_str.replace('ak3', 'ak1')
        triples_filtered.append((dataset, feat_str, net))
    return triples_filtered


def get_model_with_default_configs(model_name,
                                   num_feat_layers=args.n_layers_feat,
                                   num_conv_layers=args.n_layers_conv,
                                   num_fc_layers=args.n_layers_fc,
                                   residual=args.skip_connection,
                                   hidden=args.hidden):
    # More default settings.
    res_branch = args.res_branch
    global_pool = args.global_pool
    dropout = args.dropout
    edge_norm = args.edge_norm


    if model_name.startswith('ResGFN'):
        collapse = True if 'flat' in model_name else False
        def foo(dataset):
            return ResGCN(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=True, collapse=collapse,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    elif model_name.startswith('ResGCN'):
        def foo(dataset):
            return ResGCN(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=False, collapse=False,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    else:
        raise ValueError("Unknown model {}".format(model_name))
    return foo


def run_exp_lib(dataset_feat_net_triples,
                get_model=get_model_with_default_configs):
    print("PID: %d" % os.getpid())
    args_setting = json.dumps(vars(args), sort_keys=True, indent=2)
    print("============= GLA ========================\n{}".format(args_setting))

    results = []
    exp_nums = len(dataset_feat_net_triples)
    print("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        print('{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
    print("Here we go..")
    sys.stdout.flush()
    if args.dataset == 'pheme':
        mylogger = logger_binary
    else:
        mylogger = logger

    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        print('-----\n{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()

        get_data_start = time.time()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)

        get_data_end = time.time()
        print("get_data_time", get_data_end - get_data_start)
        print(dataset)
        print(dataset.data)
        # sys.exit()

        model_func = get_model(net)
        cross_validation_with_label(
                dataset,
                model_func,
                folds=10,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                epoch_select=args.epoch_select,
                random_seed=args.random_seed,
                seed=args.seed,
                sep_by_event=args.sep_by_event,
                with_eval_mode=args.with_eval_mode,
                logger=mylogger,
                dataset_name=args.dataset,
                eta=args.eta, n_percents=args.n_percents)

        """
        summary1 = 'data={}, model={}, feat={}, eval={}'.format(
            dataset_name, net, feat_str, args.epoch_select)
        summary2 = 'train_acc={:.2f}, test_acc={:.2f} ± {:.2f}, sec={}'.format(
            train_acc*100, acc*100, std*100, round(duration, 2))
        results += ['{}: {}, {}'.format('fin-result', summary1, summary2)]
        print('{}: {}, {}'.format('mid-result', summary1, summary2))
        sys.stdout.flush()
    print('-----\n{}'.format('\n'.join(results)))
    sys.stdout.flush()
        """



def run_exp_benchmark():
    # Run GFN, GFN (light), GCN
    print('[INFO] running standard benchmarks..')
    # datasets = DATA_BIO + DATA_SOCIAL
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    # nets = ['ResGFN', 'ResGFN_conv0_fc2', 'ResGCN']
    nets = ['ResGCN']
    run_exp_lib(create_n_filter_triples(datasets, feat_strs, nets,
                                        gfn_add_ak3=True,
                                        reddit_odeg10=True,
                                        dd_odeg10_ak1=True))



def run_train_gcn():
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    nets = ['ResGCN']
    dataset_feat_net_triples = create_n_filter_triples(datasets, feat_strs, nets,
                            gfn_add_ak3=True,
                            reddit_odeg10=True,
                            dd_odeg10_ak1=True)
    get_model = get_model_with_default_configs
    print("PID: %d" % os.getpid())
    args_setting = json.dumps(vars(args), sort_keys=True, indent=2)
    print("============= GLA ========================\n{}".format(args_setting))

    results = []
    exp_nums = len(dataset_feat_net_triples)
    print("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        print('{}/{} - {} - {} - {}'.format(
            exp_id + 1, exp_nums, dataset_name, feat_str, net))
    print("Here we go..")
    sys.stdout.flush()
    print(args.dataset)
    print(args.dataset == 'pheme')
    if args.dataset == 'pheme':
        mylogger = logger_gcn_binary
    else:
        mylogger = logger_gcn
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        print('-----\n{}/{} - {} - {} - {}'.format(
            exp_id + 1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()

        get_data_start = time.time()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)

        get_data_end = time.time()
        print("get_data_time", get_data_end - get_data_start)
        print(dataset)
        print(dataset.data)
        # sys.exit()

        model_func = get_model(net)
        cross_validation_gcn(
            dataset,
            model_func,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            epoch_select=args.epoch_select,
            random_seed=args.random_seed,
            seed=args.seed,
            sep_by_event=args.sep_by_event,
            with_eval_mode=args.with_eval_mode,
            logger=mylogger,
            dataset_name=args.dataset,
            eta=args.eta, n_percents=args.n_percents)
        
def run_train_gacl():
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    nets = ['ResGCN']
    dataset_feat_net_triples = create_n_filter_triples(datasets, feat_strs, nets,
                            gfn_add_ak3=True,
                            reddit_odeg10=True,
                            dd_odeg10_ak1=True)
    get_model = get_model_with_default_configs
    print("PID: %d" % os.getpid())
    args_setting = json.dumps(vars(args), sort_keys=True, indent=2)
    print("============= GLA ========================\n{}".format(args_setting))

    results = []
    exp_nums = len(dataset_feat_net_triples)
    print("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        print('{}/{} - {} - {} - {}'.format(
            exp_id + 1, exp_nums, dataset_name, feat_str, net))
    print("Here we go..")
    sys.stdout.flush()
    print(args.dataset)
    print(args.dataset == 'pheme')
    if args.dataset == 'pheme':
        mylogger = logger_gcn_binary
    else:
        mylogger = logger_gcn
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        print('-----\n{}/{} - {} - {} - {}'.format(
            exp_id + 1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()

        get_data_start = time.time()
        dataset = get_gacl_dataset(dataset_name)

        get_data_end = time.time()
        print("get_data_time", get_data_end - get_data_start)
        print(dataset)
        print(dataset.data)
        # sys.exit()

        model_func = get_model(net)
        cross_validation_gacl(
            dataset,
            model_func,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            epoch_select=args.epoch_select,
            random_seed=args.random_seed,
            seed=args.seed,
            sep_by_event=args.sep_by_event,
            with_eval_mode=args.with_eval_mode,
            logger=mylogger,
            dataset_name=args.dataset,
            eta=args.eta, n_percents=args.n_percents)


if __name__ == '__main__':
    if args.exp == 'benchmark':
        run_exp_benchmark()
    elif args.exp == 'train_gcn':
        run_train_gcn()
    elif args.exp == 'train_gacl':
        run_train_gacl()
    else:
        raise ValueError('Unknown exp {} to run'.format(args.exp))
    pass
