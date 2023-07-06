import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
import pandas as pd
from utils import print_weights
from sklearn import metrics
from torch_geometric.data import DataLoader
import time
from torch_geometric.nn import GCNConv
import torch_geometric
import torch.nn as nn
import random
from torch_scatter import scatter_mean
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from copy import deepcopy




def k_fold(dataset, folds, epoch_select, sep_by_event=0, random_seed=1, seed=0, n_percents=1):
    n_splits = folds - 2
    
    if sep_by_event==1:
        test_indices, train_indices, train_indices_unlabel = [], [], []
        train_all = []
        if len(dataset) == 661:
            pathname = "/data/zhangjiajun/autoaug_fakenews/data/T16.csv"
        else:
            pathname = "/data/zhangjiajun/autoaug_fakenews/data/T15.csv"
        datadf = pd.read_csv(pathname)
        train_all = datadf[datadf.sep_labels == 0].index.tolist()
        test_index = datadf[datadf.sep_labels == 1].index.tolist()
        for i in range(folds):
            test_indices.append(torch.tensor(test_index))
        val_indices = test_indices
        skf_semi = StratifiedKFold(n_splits, shuffle=True, random_state=12345)
        for i in range(folds):
            train_mask = torch.ones(len(dataset), dtype=torch.uint8)
            train_mask[test_indices[i].long()] = 0
            train_mask[val_indices[i].long()] = 0
            idx_train_all = train_mask.nonzero(as_tuple=False).view(-1)
            idx_train = []
            for _, idx in skf_semi.split(torch.zeros(idx_train_all.size()[0]), dataset.data.y[idx_train_all]):
                idx_train.append(idx_train_all[idx])
                if len(idx_train) >= n_percents:
                    break
            idx_train = torch.concat(idx_train).view(-1)
            train_indices.append(idx_train)
            train_mask[train_indices[i].long()] = 0
            idx_train_unlabel = train_mask.nonzero(as_tuple=False).view(-1)
            train_indices_unlabel.append(idx_train_unlabel)
            print("Train:", len(train_indices[i]), "Val:", len(val_indices[i]), "Test:", len(test_indices[i]))
            for j in range(10):
                print("Train({:d}):".format(j),train_indices[i])
                print("Train_unlabel({:d}):".format(j), train_indices_unlabel[i])
                print("Val({:d}):".format(j),val_indices[i])
                print("Test({:d}):".format(j),test_indices[i])

        
        return train_indices, test_indices, val_indices, train_indices_unlabel

    

    # if n_percents == 10:
    #     all_indices = torch.arange(0,len(dataset),1, dtype=torch.long)
    #     return [all_indices], [all_indices], [all_indices], [all_indices]
    print("folds", folds)
    if random_seed == 1:
        seed = random.randint(1, 100000)
    print("seed",seed)
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed) #7815

    test_indices, train_indices, train_indices_unlabel = [], [], []
    save_test,  save_train, save_val, save_train_unlabel = [], [], [], []


    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))
    # for _, idx in skf.split(torch.zeros(len(dataset)), all_data.y):
    #     test_indices.append(torch.from_numpy(idx))
        if len(save_test) > 0 and len(list(idx)) < len(save_test[0]):
            save_test.append(list(idx) + [list(idx)[-1]])
        else:
            save_test.append(list(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
        save_val = [save_test[i] for i in range(folds)]
        # n_splits += 1
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]
        save_val = [save_test[i - 1] for i in range(folds)]

    skf_semi = StratifiedKFold(n_splits, shuffle=True, random_state=12345)
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train_all = train_mask.nonzero(as_tuple=False).view(-1)

        idx_train = []
        for _, idx in skf_semi.split(torch.zeros(idx_train_all.size()[0]), dataset.data.y[idx_train_all]):
            idx_train.append(idx_train_all[idx])
        # for _, idx in skf_semi.split(torch.zeros(idx_train_all.size()[0]), all_data.y[idx_train_all]):
        #     idx_train.append(idx_train_all[idx])
            if len(idx_train) >= n_percents:
                break
        idx_train = torch.concat(idx_train).view(-1)
        
        train_indices.append(idx_train)
        cur_idx = list(idx_train.cpu().detach().numpy())
        if i > 0 and len(cur_idx) < len(save_train[0]):
            save_train.append(cur_idx + [cur_idx[-1]])
        else:
            save_train.append(cur_idx)

        # train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[train_indices[i].long()] = 0
        idx_train_unlabel = train_mask.nonzero(as_tuple=False).view(-1)
        train_indices_unlabel.append(idx_train_unlabel) # idx_train_all, idx_train_unlabel
        cur_idx = list(idx_train_unlabel.cpu().detach().numpy())
        if i > 0 and len(cur_idx) < len(save_train_unlabel[0]):
            save_train_unlabel.append(cur_idx + [cur_idx[-1]])
        else:
            save_train_unlabel.append(cur_idx)

    print("Train:", len(train_indices[i]), "Val:", len(val_indices[i]), "Test:", len(test_indices[i]))
    for j in range(10):
        print("Train({:d}):".format(j),train_indices[i])
        print("Train_unlabel({:d}):".format(j), train_indices_unlabel[i])
        print("Val({:d}):".format(j),val_indices[i])
        print("Test({:d}):".format(j),test_indices[i])


    return train_indices, test_indices, val_indices, train_indices_unlabel


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def computing_metrics(true_labels: list, predict_labels: list, dataset_name):
        """
        Computing classification metrics for 2/4 category classification

        Parameters
        ----------
        true_labels: :class:`list`
        predict_labels: :class:`list`

        """

        assert len(true_labels) == len(predict_labels)
        if "twitter" in dataset_name:
            accuracy = metrics.accuracy_score(true_labels, predict_labels)
            f1_U = metrics.f1_score(true_labels, predict_labels, average=None, labels=[0])[0]
            f1_NR = metrics.f1_score(true_labels, predict_labels, average=None, labels=[1])[0]
            f1_T = metrics.f1_score(true_labels, predict_labels, average=None, labels=[2])[0]
            f1_F = metrics.f1_score(true_labels, predict_labels, average=None, labels=[3])[0]


        else:
            accuracy = metrics.accuracy_score(true_labels, predict_labels)
            f1_U = metrics.f1_score(true_labels, predict_labels, average=None, labels=[0])[0]
            f1_NR = metrics.f1_score(true_labels, predict_labels, average=None, labels=[1])[0]
            f1_T = metrics.precision_score(true_labels, predict_labels)
            f1_F = metrics.recall_score(true_labels, predict_labels)
        result = {
            # total
            'f1_U': f1_U,
            'f1_NR': f1_NR,
            'f1_T': f1_T,
            'f1_F': f1_F,
            'accuracy': accuracy,
        }
        return result


def eval_f1(model, loader, device, with_eval_mode, dataset_name, suffix=0, eta=1.0):
    if with_eval_mode:
        model.eval()

    true_labels, predict_labels = [], []
    correct, correct_invariant = 0, 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred, _, _, _ = model(data)
            pred = pred.max(1)[1]

        # print(data.y)
        # print(pred)
        true_labels.extend(data.y.view(-1).cpu().tolist())
        predict_labels.extend(pred.cpu().tolist())
    # print(true_labels)
    # print(predict_labels)
    return computing_metrics(true_labels, predict_labels, dataset_name)


def eval_acc(model, loader, device, with_eval_mode, suffix=0, eta=1.0):
    if with_eval_mode:
        model.eval()

    correct, correct_invariant = 0, 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred, _, _, _ = model(data)
            pred = pred.max(1)[1]

            if suffix == 10:
                out2, _, pred2, _ = model.forward_cl(data, True, get_one_hot_encoding(data, model.n_class), grads=None, eta=eta)
                pred2 = pred2.max(1)[1]
        
        if suffix == 10:
            correct_invariant += pred.eq(pred2.view(-1)).sum().item()
        correct += pred.eq(data.y.view(-1)).sum().item()

    if suffix == 10:
        return correct / len(loader.dataset), correct_invariant / len(loader.dataset)
    return correct / len(loader.dataset)


def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, _, _, _ = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)




def get_one_hot_encoding(data, n_class):
    y = data.y.view(-1)
    encoding = np.zeros([len(y), n_class])
    for i in range(len(y)):
        encoding[i, int(y[i])] = 1
    return torch.from_numpy(encoding).to(device)


def train(model, optimizer, dataset, device, batch_size, eta):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    #dataset1 = dataset
    dataset1.aug, dataset1.aug_ratio = "none", 0.0
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = "none", 0.0


    loader1 = DataLoader(dataset1, batch_size, shuffle=False, num_workers=1)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False, num_workers=1)
    # print(dataset1)

    model.train()

    total_loss = 0

    # train_unlabel_epoch_start = time.time()
    for data1, data2 in zip(loader1, loader2):
        # print(data1, data2)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        # print(data1)

        # cl1_start = time.time()
        out1, x1, pred1, pred_gcn1 = model.forward_cl(data1, False, None, eta=eta)
        # cl1_end = time.time()
        # print("cl1_time", cl1_end - cl1_start)

        # grad_start = time.time()
        graph_grad = torch.autograd.grad(out1,x1,retain_graph=True, grad_outputs=torch.ones_like(out1))[0]
        # grad_end = time.time()
        # print("grad_time", grad_end - grad_start)

        # cl2_start = time.time()
        out2, x2, pred2, pred_gcn2 = model.forward_cl(data2, True, None, grads=graph_grad, eta=eta)
        # cl2_end = time.time()
        # print("cl2_time", cl2_end - cl2_start)
        # print(x1.shape)
        # print(x2.shape)
        # print(out1.shape)
        # print(out2.shape)

        eq = torch.argmax(pred1, axis=-1) - torch.argmax(pred2, axis=-1)
        indices = (eq == 0).nonzero().reshape(-1)
        loss = model.loss_cl(out1[indices], out2[indices])

        # optimize_start = time.time()
        if len(indices) > 1:
            loss.backward()
            total_loss += loss.item() * num_graphs(data1)
            optimizer.step()
        # optimize_end = time.time()
        # print("optimize_time", optimize_end - optimize_start)

    # train_unlabel_epoch_end = time.time()
    # print("unlabel_epoch", train_unlabel_epoch_end - train_unlabel_epoch_start)

    return total_loss / len(loader1.dataset)



def train_label(model, optimizer, dataset, device, batch_size, eta):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset1.aug, dataset1.aug_ratio = "none", 0.0
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = "none", 0.0
    # print(dataset1)

    loader1 = DataLoader(dataset1, batch_size, shuffle=False, num_workers=1)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False, num_workers=1)

    model.train()
    total_loss = 0
    correct = 0
    # train_label_epoch_start = time.time()
    for data1, data2 in zip(loader1, loader2):
        # infor_start = time.time()

        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)

        out1, x1, pred1, pred_gcn1 = model.forward_cl(data1, False, get_one_hot_encoding(data1, model.n_class), eta=eta)
        graph_grad = torch.autograd.grad(out1,x1,retain_graph=True, grad_outputs=torch.ones_like(out1))[0]
        out2, _, pred2, pred_gcn2 = model.forward_cl(data2, True, get_one_hot_encoding(data2, model.n_class), grads=graph_grad, eta=eta)

        eq = torch.argmax(pred1, axis=-1) - torch.argmax(pred2, axis=-1)
        indices = (eq == 0).nonzero().reshape(-1)
        loss = model.loss_cl(out1[indices], out2[indices])

        out, _, hidden, pred_gcn = model(data1)
        loss += (F.nll_loss(pred1, data1.y.view(-1))+F.nll_loss(pred2[indices], data2.y.view(-1)[indices])) #* 0.01

        pred = out.max(1)[1]
        correct += pred.eq(data1.y.view(-1)).sum().item()

        if len(indices) > 1:
            loss.backward()
            total_loss += loss.item() * num_graphs(data1)
            optimizer.step()

        # infor_end = time.time()
        # print("infor_time", infor_end - infor_start)

    # train_label_epoch_end = time.time()
    # print("label_epoch", train_label_epoch_end - train_label_epoch_start)

    return total_loss / len(loader1.dataset), correct / len(loader1.dataset)


def cross_validation_with_label(dataset,
                                  model_func,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  random_seed=True,
                                  seed=0,
                                  sep_by_event=0,
                                  with_eval_mode=True,
                                  logger=None,
                                  dataset_name=None,
                                  eta=1.0, n_percents=None):
    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    test_f1 = []
    best_test_acc = 0.0
    for fold, (train_idx, test_idx, val_idx, train_idx_unlabel) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select, sep_by_event, random_seed, seed, n_percents=int(n_percents)))):

        train_idx[train_idx < 0] = train_idx[0]
        train_idx[train_idx >= len(dataset)] = train_idx[0]
        test_idx[test_idx < 0] = test_idx[0]
        test_idx[test_idx >= len(dataset)] = test_idx[0]
        val_idx[val_idx < 0] = val_idx[0]
        val_idx[val_idx >= len(dataset)] = val_idx[0]


        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_dataset_unlabel = dataset[train_idx_unlabel]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        train_loader_unlabel = DataLoader(train_dataset_unlabel, batch_size, shuffle=True)

        dataset.aug = "none"
        model = model_func(dataset).to(device)
        optimizer_label = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = Adam(model.parameters(), lr=lr/512, weight_decay=weight_decay)   # 0.7 51200

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        patience = 0

        for epoch in range(1, epochs+1):
            #start_time =  time.time()

            train_loss = train(
                model, optimizer, train_dataset_unlabel, device, batch_size, eta)

            train_label_loss, train_acc = train_label(
                model, optimizer_label, train_dataset, device, batch_size, eta)


            train_accs.append(train_acc)
            val_losses.append(eval_loss(
                model, val_loader, device, with_eval_mode))
            test_accs.append(eval_acc(
                model, test_loader, device, with_eval_mode))
            test_f1.append(eval_f1(
                model, test_loader, device, with_eval_mode, dataset_name))

            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_label_loss': train_label_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
                'test_f1': test_f1[-1],
            }
            if test_accs[-1] >= best_test_acc:
                patience = 0
                best_test_acc = test_accs[-1]
                # torch.save(model.state_dict(), './models/twitter16_527_bench.pkl')
            else:
                patience += 1

            if patience > 300:
                if len(test_accs) < (fold + 1) * epochs:
                    test_accs += [test_accs[-1]] * ((fold + 1) * epochs - len(test_accs))
                    train_accs += [train_accs[-1]] * ((fold + 1) * epochs - len(train_accs))
                    test_f1 += [test_f1[-1]] * ((fold + 1) * epochs - len(test_f1))

                break

            # if epoch == 1 or epoch % 10 == 0:
            #     end_time = time.time()
            #     print("epoch", epoch, "time", end_time - start_time)
            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        print(fold, "finish run")


        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    # val_loss = val_loss.view(folds, epochs)

    # ----- f1 -------
    test_f1_u = [d["f1_U"] for d in test_f1]
    test_f1_nr = [d["f1_NR"] for d in test_f1]
    test_f1_t = [d["f1_T"] for d in test_f1]
    test_f1_f = [d["f1_F"] for d in test_f1]
    test_f1_u = tensor(test_f1_u).view(folds, epochs)
    test_f1_nr = tensor(test_f1_nr).view(folds, epochs)
    test_f1_t = tensor(test_f1_t).view(folds, epochs)
    test_f1_f = tensor(test_f1_f).view(folds, epochs)

    if epoch_select == 'test_max':  # take epoch that yields best test results.
        # _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        # selected_epoch = selected_epoch.repeat(folds)
        _, selected_epoch = test_acc.max(dim=1)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    print(selected_epoch)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    print(test_acc)
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    print('Train Acc: {:.4f}, Test Acc: {:.4f} ± {:.4f}, Duration(per fold): {:.4f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    test_f1_u = test_f1_u[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_nr = test_f1_nr[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_t = test_f1_t[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_f = test_f1_f[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_u_mean = test_f1_u.mean().item()
    test_f1_u_std = test_f1_u.std().item()
    test_f1_nr_mean = test_f1_nr.mean().item()
    test_f1_nr_std = test_f1_nr.std().item()
    test_f1_t_mean = test_f1_t.mean().item()
    test_f1_t_std = test_f1_t.std().item()
    test_f1_f_mean = test_f1_f.mean().item()
    test_f1_f_std = test_f1_f.std().item()

    print('f1_U: {:.4f} ± {:.4f}, f1_NR: {:.4f} ± {:.4f}, f1_T: {:.4f} ± {:.4f}, f1_F: {:.4f} ± {:.4f}'.
          format(test_f1_u_mean, test_f1_u_std, test_f1_nr_mean, test_f1_nr_std, test_f1_t_mean, test_f1_t_std, test_f1_f_mean, test_f1_f_std))

    sys.stdout.flush()

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_classes=4):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.fc = torch.nn.Linear(out_dim, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        y = data.y
        return x, y


def eval_acc_gcn(model, loader, device, with_eval_mode, suffix=0, eta=1.0):
    if with_eval_mode:
        model.eval()

    correct, correct_invariant = 0, 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out_labels, y = model(data)
            _, pred = out_labels.max(dim=-1)

        correct += pred.eq(y).sum().item()

    return correct / len(loader.dataset)


def eval_loss_gcn(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, y = model(data)
        loss += F.nll_loss(out, y).item()
    return loss / len(loader.dataset)

def eval_f1_gcn(model, loader, device, with_eval_mode, dataset_name, suffix=0, eta=1.0):
    if with_eval_mode:
        model.eval()

    true_labels, predict_labels = [], []
    correct, correct_invariant = 0, 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out_labels, y = model(data)
            _, pred = out_labels.max(dim=-1)

        # print(data.y)
        # print(pred)
        true_labels.extend(data.y.view(-1).cpu().tolist())
        predict_labels.extend(pred.cpu().tolist())
    # print(true_labels)
    # print(predict_labels)
    return computing_metrics(true_labels, predict_labels, dataset_name)


def train_gcn(model, optimizer, dataset, device, batch_size, eta):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset1.aug, dataset1.aug_ratio = "none", 0.0
    loader1 = DataLoader(dataset1, batch_size, shuffle=False, num_workers=1)
    model.train()
    total_loss = 0
    correct = 0

    for data1 in loader1:
        optimizer.zero_grad()
        data1 = data1.to(device)
        out_labels, y = model(data1)
        loss = F.nll_loss(out_labels, y)
        _, pred = out_labels.max(dim=-1)
        correct += pred.eq(y).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()

    return total_loss / len(loader1.dataset), correct / len(loader1.dataset)


def cross_validation_gcn(dataset,
                                  model_func,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  random_seed=True,
                                  seed=0,
                                  sep_by_event=0,
                                  with_eval_mode=True,
                                  logger=None,
                                  dataset_name=None,
                                  eta=1.0, n_percents=None):
    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    test_f1 = []
    best_test_acc = 0.0
    for fold, (train_idx, test_idx, val_idx, train_idx_unlabel) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select, sep_by_event, random_seed, seed, n_percents=int(n_percents)))):

        train_idx[train_idx < 0] = train_idx[0]
        train_idx[train_idx >= len(dataset)] = train_idx[0]
        test_idx[test_idx < 0] = test_idx[0]
        test_idx[test_idx >= len(dataset)] = test_idx[0]
        val_idx[val_idx < 0] = val_idx[0]
        val_idx[val_idx >= len(dataset)] = val_idx[0]

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_dataset_unlabel = dataset[train_idx_unlabel]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        train_loader_unlabel = DataLoader(train_dataset_unlabel, batch_size, shuffle=True)

        dataset.aug = "none"
        #model = model_func(dataset).to(device)
        model = GCN(768, 64, 64, dataset.num_classes).to(device)
        optimizer_label = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = Adam(model.parameters(), lr=lr / 5, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_gcn(
                model, optimizer, train_dataset, device, batch_size, eta)

            train_accs.append(train_acc)
            val_losses.append(eval_loss_gcn(
                model, val_loader, device, with_eval_mode))
            test_accs.append(eval_acc_gcn(
                model, test_loader, device, with_eval_mode))
            test_f1.append(eval_f1_gcn(
                model, test_loader, device, with_eval_mode, dataset_name))

            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                # 'train_label_loss': train_label_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
                'test_f1': test_f1[-1],
            }

            # if test_accs[-1] > best_test_acc:
            #     best_test_acc = test_accs[-1]
            #     torch.save(model.state_dict(), './models/twitter16_527_bench.pkl')


            # if epoch == 1 or epoch % 10 == 0:
            #     end_time = time.time()
            #     print("epoch", epoch, "time", end_time - start_time)
            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        print(fold, "finish run")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    durations.append(t_end - t_start)


    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    val_loss = val_loss.view(folds, epochs)

    # ----- f1 -------
    test_f1_u = [d["f1_U"] for d in test_f1]
    test_f1_nr = [d["f1_NR"] for d in test_f1]
    test_f1_t = [d["f1_T"] for d in test_f1]
    test_f1_f = [d["f1_F"] for d in test_f1]
    test_f1_u = tensor(test_f1_u).view(folds, epochs)
    test_f1_nr = tensor(test_f1_nr).view(folds, epochs)
    test_f1_t = tensor(test_f1_t).view(folds, epochs)
    test_f1_f = tensor(test_f1_f).view(folds, epochs)

    if epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    print('Train Acc: {:.4f}, Test Acc: {:.4f} ± {:.4f}, Duration(per fold): {:.4f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    test_f1_u = test_f1_u[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_nr = test_f1_nr[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_t = test_f1_t[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_f = test_f1_f[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_u_mean = test_f1_u.mean().item()
    test_f1_u_std = test_f1_u.std().item()
    test_f1_nr_mean = test_f1_nr.mean().item()
    test_f1_nr_std = test_f1_nr.std().item()
    test_f1_t_mean = test_f1_t.mean().item()
    test_f1_t_std = test_f1_t.std().item()
    test_f1_f_mean = test_f1_f.mean().item()
    test_f1_f_std = test_f1_f.std().item()

    print('f1_U: {:.4f} ± {:.4f}, f1_NR: {:.4f} ± {:.4f}, f1_T: {:.4f} ± {:.4f}, f1_F: {:.4f} ± {:.4f}'.
          format(test_f1_u_mean, test_f1_u_std, test_f1_nr_mean, test_f1_nr_std, test_f1_t_mean, test_f1_t_std,
                 test_f1_f_mean, test_f1_f_std))

    sys.stdout.flush()

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean


# --------- GACL -----------
class hard_fc(torch.nn.Module):
    def __init__(self, d_in,d_hid, DroPout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DroPout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='hard_fc1.'): # T15: epsilon = 0.2
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='hard_fc1.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class GCN_Net(torch.nn.Module): 
    def __init__(self,in_feats,hid_feats,out_feats): 
        super(GCN_Net, self).__init__() 
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        self.fc=torch.nn.Linear(2*out_feats,4)
        self.hard_fc1 = hard_fc(out_feats, out_feats)
        self.hard_fc2 = hard_fc(out_feats, out_feats) # optional

    def forward(self, data):
        if self.training:
            init_x0, init_x, edge_index1, edge_index2 = data.x0, data.x, data.edge_index, data.edge_index2
        else:
            init_x0, init_x, edge_index1, edge_index2 = data.x0, data.x0, data.edge_index, data.edge_index2
        x1 = self.conv1(init_x0, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1(x1)
        x1_t = x1
        x1 = torch.cat((x1_g, x1_t), 1)

        x2 = self.conv1(init_x, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0)
        x2_g = x2
        x2 = self.hard_fc1(x2)
        x2_t = x2 
        x2 = torch.cat((x2_g, x2_t), 1)
        x = torch.cat((x1, x2), 0)
        y = torch.cat((data.y1, data.y2), 0)

        x_T = x.t()
        dot_matrix = torch.mm(x, x_T)
        x_norm = torch.norm(x, p=2, dim=1)
        x_norm = x_norm.unsqueeze(1)
        norm_matrix = torch.mm(x_norm, x_norm.t())
        
        t = 0.3 # pheme: t = 0.6
        cos_matrix = (dot_matrix / norm_matrix) / t
        cos_matrix = torch.exp(cos_matrix)
        diag = torch.diag(cos_matrix)
        cos_matrix_diag = torch.diag_embed(diag)
        cos_matrix = cos_matrix - cos_matrix_diag
        y_matrix_T = y.expand(len(y), len(y))
        y_matrix = y_matrix_T.t()
        y_matrix = torch.ne(y_matrix, y_matrix_T).float()
        #y_matrix_list = y_matrix.chunk(3, dim=0)
        #y_matrix = y_matrix_list[0]
        neg_matrix = cos_matrix * y_matrix
        neg_matrix_list = neg_matrix.chunk(2, dim=0)
        #neg_matrix = neg_matrix_list[0]
        pos_y_matrix = y_matrix * (-1) + 1
        pos_matrix_list = (cos_matrix * pos_y_matrix).chunk(2,dim=0)
        #print('cos_matrix: ', cos_matrix.shape, cos_matrix)
        #print('pos_y_matrix: ', pos_y_matrix.shape, pos_y_matrix)
        pos_matrix = pos_matrix_list[0]
        #print('pos shape: ', pos_matrix.shape, pos_matrix)
        neg_matrix = (torch.sum(neg_matrix, dim=1)).unsqueeze(1)
        sum_neg_matrix_list = neg_matrix.chunk(2, dim=0)
        p1_neg_matrix = sum_neg_matrix_list[0]
        p2_neg_matrix = sum_neg_matrix_list[1]
        neg_matrix = p1_neg_matrix
        #print('neg shape: ', neg_matrix.shape)
        div = pos_matrix / neg_matrix 
        div = (torch.sum(div, dim=1)).unsqueeze(1)  
        div = div / 128
        log = torch.log(div)
        SUM = torch.sum(log)
        cl_loss = -SUM

        x = self.fc(x) 
        x = F.log_softmax(x, dim=1)

        return x, cl_loss, y

def train_gacl(model, optimizer, dataset, device, batch_size, eta):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset1.aug, dataset1.aug_ratio = "none", 0.0
    loader1 = DataLoader(dataset1, batch_size, shuffle=False, num_workers=1)
    model.train()
    total_loss = 0
    correct = 0
    beta=0.001
    for data1 in loader1:
        optimizer.zero_grad()
        data1 = data1.to(device)
        out_labels, cl_loss, y = model(data1)
        loss = F.nll_loss(out_labels, y) + beta*cl_loss
        _, pred = out_labels.max(dim=-1)
        correct += pred.eq(y).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()

    return total_loss / len(loader1.dataset), correct / (2*len(loader1.dataset))

def eval_acc_gacl(model, loader, device, with_eval_mode, suffix=0, eta=1.0):
    if with_eval_mode:
        model.eval()

    correct, correct_invariant = 0, 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out_labels, _, y = model(data)
            _, pred = out_labels.max(dim=-1)

        correct += pred.eq(y).sum().item()

    return correct / (2*len(loader.dataset))


def eval_loss_gacl(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, _, y = model(data)
        loss += F.nll_loss(out, y).item()
    return loss / len(loader.dataset)

def eval_f1_gacl(model, loader, device, with_eval_mode, dataset_name, suffix=0, eta=1.0):
    if with_eval_mode:
        model.eval()

    true_labels, predict_labels = [], []
    correct, correct_invariant = 0, 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out_labels, _, y = model(data)
            _, pred = out_labels.max(dim=-1)

        # print(data.y)
        # print(pred)
        true_labels.extend(y.view(-1).cpu().tolist())
        predict_labels.extend(pred.cpu().tolist())
    # print(true_labels)
    # print(predict_labels)
    return computing_metrics(true_labels, predict_labels, dataset_name)

def cross_validation_gacl(dataset,
                                  model_func,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  random_seed=True,
                                  seed=0,
                                  sep_by_event=0,
                                  with_eval_mode=True,
                                  logger=None,
                                  dataset_name=None,
                                  eta=1.0, n_percents=None):
    assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    test_f1 = []
    best_test_acc = 0.0
    for fold, (train_idx, test_idx, val_idx, train_idx_unlabel) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select, sep_by_event, random_seed, seed, n_percents=int(n_percents)))):

        train_idx[train_idx < 0] = train_idx[0]
        train_idx[train_idx >= len(dataset)] = train_idx[0]
        test_idx[test_idx < 0] = test_idx[0]
        test_idx[test_idx >= len(dataset)] = test_idx[0]
        val_idx[val_idx < 0] = val_idx[0]
        val_idx[val_idx >= len(dataset)] = val_idx[0]

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_dataset_unlabel = dataset[train_idx_unlabel]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        train_loader_unlabel = DataLoader(train_dataset_unlabel, batch_size, shuffle=True)

        dataset.aug = "none"
        model = GCN_Net(768,64,64).to(device) 
        fgm = FGM(model)
        for para in model.hard_fc1.parameters():
            para.requires_grad = False
        for para in model.hard_fc2.parameters():
            para.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, weight_decay=1e-4)
        # optional ------ S1 ----------
        for para in model.hard_fc1.parameters():
            para.requires_grad = True
        for para in model.hard_fc2.parameters():
            para.requires_grad = True

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        patience = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_gacl(
                model, optimizer, train_dataset, device, batch_size, eta)

            train_accs.append(train_acc)
            val_losses.append(eval_loss_gacl(
                model, val_loader, device, with_eval_mode))
            test_accs.append(eval_acc_gacl(
                model, test_loader, device, with_eval_mode))
            test_f1.append(eval_f1_gacl(
                model, test_loader, device, with_eval_mode, dataset_name))

            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                # 'train_label_loss': train_label_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
                'test_f1': test_f1[-1],
            }

            if test_accs[-1] >= best_test_acc:
                patience = 0
                best_test_acc = test_accs[-1]
                # torch.save(model.state_dict(), './models/twitter16_527_bench.pkl')
            else:
                patience += 1

            if patience > 300:
                if len(test_accs) < (fold + 1) * epochs:
                    test_accs += [test_accs[-1]] * ((fold + 1) * epochs - len(test_accs))
                    train_accs += [train_accs[-1]] * ((fold + 1) * epochs - len(train_accs))
                    test_f1 += [test_f1[-1]] * ((fold + 1) * epochs - len(test_f1))

                break
            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']


        print(fold, "finish run")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    durations.append(t_end - t_start)


    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    #val_loss = val_loss.view(folds, epochs)

    # ----- f1 -------
    test_f1_u = [d["f1_U"] for d in test_f1]
    test_f1_nr = [d["f1_NR"] for d in test_f1]
    test_f1_t = [d["f1_T"] for d in test_f1]
    test_f1_f = [d["f1_F"] for d in test_f1]
    test_f1_u = tensor(test_f1_u).view(folds, epochs)
    test_f1_nr = tensor(test_f1_nr).view(folds, epochs)
    test_f1_t = tensor(test_f1_t).view(folds, epochs)
    test_f1_f = tensor(test_f1_f).view(folds, epochs)

    if epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    print('Train Acc: {:.4f}, Test Acc: {:.4f} ± {:.4f}, Duration(per fold): {:.4f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))

    test_f1_u = test_f1_u[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_nr = test_f1_nr[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_t = test_f1_t[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_f = test_f1_f[torch.arange(folds, dtype=torch.long), selected_epoch]
    test_f1_u_mean = test_f1_u.mean().item()
    test_f1_u_std = test_f1_u.std().item()
    test_f1_nr_mean = test_f1_nr.mean().item()
    test_f1_nr_std = test_f1_nr.std().item()
    test_f1_t_mean = test_f1_t.mean().item()
    test_f1_t_std = test_f1_t.std().item()
    test_f1_f_mean = test_f1_f.mean().item()
    test_f1_f_std = test_f1_f.std().item()

    print('f1_U: {:.4f} ± {:.4f}, f1_NR: {:.4f} ± {:.4f}, f1_T: {:.4f} ± {:.4f}, f1_F: {:.4f} ± {:.4f}'.
          format(test_f1_u_mean, test_f1_u_std, test_f1_nr_mean, test_f1_nr_std, test_f1_t_mean, test_f1_t_std,
                 test_f1_f_mean, test_f1_f_std))

    sys.stdout.flush()

    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean