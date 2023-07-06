import os.path as osp
import re

import torch
from torch_geometric.datasets import MNISTSuperpixels, TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import json
from itertools import repeat
import random
import time

class TwitterDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        print("init")
        if self.root == 'twitter16':
            self.data, self.slices = torch.load("/data/zhangjiajun/autoaug_fakenews/data/T16.pt")
        elif self.root == 'twitter15':
            self.data, self.slices = torch.load("/data/zhangjiajun/autoaug_fakenews/data/T15.pt")
        # self.num_features = 768

    def len(self):
        return self.datadf.shape[0]

    @property
    def num_classes(self):
        return 4

    @property
    def num_features(self):
        return 768

    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        process_start = time.time()
        if self.root == 'twitter16':
            self.datadf = pd.read_csv("/data/zhangjiajun/autoaug_fakenews/data/T16.csv")
        else:
            self.datadf = pd.read_csv("/data/zhangjiajun/autoaug_fakenews/data/T15.csv")

        self.datadf['tweets'] = self.datadf['tweets'].apply(lambda x: np.array(json.loads(x)))
        self.datadf['infs'] = self.datadf['infs'].apply(lambda x: np.array(json.loads(x)))
        datalist = []

        for index, row in self.datadf.iterrows():
            x = row['tweets']
            edge_index = row['infs']
            y = row['labels']
            x = torch.tensor(x, dtype=torch.float32)
            edge_index = torch.LongTensor(edge_index)
            y = torch.LongTensor([y])

            sep_label = row['sep_labels']
            sep_label = torch.LongTensor([sep_label])

            # Process your data here

            data = Data(x=x, edge_index=edge_index, y=y, sep_label=sep_label)
            datalist.append(data)
        data, slices = self.collate(datalist)
        if self.root == 'twitter16':
            torch.save((data, slices), "/data/zhangjiajun/autoaug_fakenews/data/T16.pt")
        elif self.root == 'twitter15':
            torch.save((data, slices), "/data/zhangjiajun/autoaug_fakenews/data/T15.pt")

        process_end = time.time()
        print("process time: ", process_end - process_start)
    def get(self, index):
        # return self.data[index]
        # get_start = time.time()
        data = self.data.__class__()
        # get_end = time.time()
        # print("get time: ", get_end - get_start)

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[index]

        for key in self.data.keys:
            if key == 'num_nodes':
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[index],
                                                       slices[index + 1])
            else:
                s = slice(slices[index], slices[index + 1])
            data[key] = item[s]


        return data

def random_pick(list, probabilities): 
    x = random.uniform(0,1)
    cumulative_probability = 0.0 
    for item, item_probability in zip(list, probabilities): 
         cumulative_probability += item_probability 
         if x < cumulative_probability:
               break 
    return item 

class GACLDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        print("init")
        if self.root == 'twitter16':
            self.data, self.slices = torch.load("/data/zhangjiajun/autoaug_fakenews/data/T16_gacl.pt")
        elif self.root == 'twitter15':
            self.data, self.slices = torch.load("/data/zhangjiajun/autoaug_fakenews/data/T15_gacl.pt")
        # self.num_features = 768

    def len(self):
        return self.datadf.shape[0]

    @property
    def num_classes(self):
        return 4

    @property
    def num_features(self):
        return 768

    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        process_start = time.time()
        if self.root == 'twitter16':
            self.datadf = pd.read_csv("/data/zhangjiajun/autoaug_fakenews/data/T16.csv")
        else:
            self.datadf = pd.read_csv("/data/zhangjiajun/autoaug_fakenews/data/T15.csv")

        self.datadf['tweets'] = self.datadf['tweets'].apply(lambda x: np.array(json.loads(x)))
        self.datadf['infs'] = self.datadf['infs'].apply(lambda x: np.array(json.loads(x)))
        datalist = []
         #==================================- dropping + adding + misplacing -===================================#

        choose_list = [1,2,3] # 1-drop 2-add 3-misplace
        probabilities = [0.7,0.2,0.1] # T15: probabilities = [0.5,0.3,0.2] 
        choose_num = random_pick(choose_list, probabilities)
        droprate = 0.4

        for index, rowrow in self.datadf.iterrows():
            x = rowrow['tweets']
            edge_index = rowrow['infs']
            y = rowrow['labels']
            x = torch.tensor(x, dtype=torch.float32)
            x0 = x
            edge_index = torch.LongTensor(edge_index)
            y1 = torch.LongTensor([y])
            y2 = torch.LongTensor([y])
            row, col = edge_index[0], edge_index[1]
            row = row.tolist()
            init_row = row[:int(len(row)/2)]
            init_col = row[int(len(row)/2):]
            #print("equal", len(init_col)==len(init_row))
            col = col.tolist()
            if droprate > 0:
                if choose_num == 1:
                
                    length = len(row)
                    poslist = random.sample(range(length), int(length * (1 - droprate)))
                    poslist = sorted(poslist)
                    row2 = list(np.array(row)[poslist])
                    col2 = list(np.array(col)[poslist])
                    new_edgeindex2 = [row2, col2]
                    #new_edgeindex = [row2, col2]
                    
                    
                elif choose_num == 2:
                    length = len(list(set(sorted(row))))
                    add_row = random.sample(range(length), int(length * droprate)) 
                    add_col = random.sample(range(length), int(length * droprate))
                    row2 = row + add_row + add_col
                    col2 = col + add_col + add_row

                    new_edgeindex2 = [row2, col2]


                            
                elif choose_num == 3: 
                    length = len(init_row)
                    mis_index_list = random.sample(range(length), int(length * droprate))
                    #print('mis_index_list:', mis_index_list)
                    Sort_len = len(list(set(sorted(row))))
                    if Sort_len > int(length * droprate):
                        mis_value_list = random.sample(range(Sort_len), int(length * droprate))
                        #print('mis_valu_list:', mis_value_list)
                        #val_i = 0
                        for i, item in enumerate(init_row):
                            for mis_i,mis_item in enumerate(mis_index_list):
                                if i == mis_item and mis_value_list[mis_i] != item:
                                    init_row[i] = mis_value_list[mis_i]
                        row2 = init_row + init_col
                        col2 = init_col + init_row
                        new_edgeindex2 = [row2, col2]

                    else:
                        length = len(row)
                        poslist = random.sample(range(length), int(length * (1 - droprate)))
                        poslist = sorted(poslist)
                        row2 = list(np.array(row)[poslist])
                        col2 = list(np.array(col)[poslist])
                        new_edgeindex2 = [row2, col2]
            else:
                new_edgeindex2 = [row, col]

            # Process your data here
            x_list = x.tolist()
            if droprate > 0:
                if choose_num == 1:
                    zero_list = [0]*768
                    x_length = len(x_list)
                    r_list = random.sample(range(x_length), int(x_length * droprate))
                    r_list = sorted(r_list)
                    for idex, line in enumerate(x_list):
                        for r in r_list:
                            if idex == r:
                                x_list[idex] = zero_list
                    
                    x2 = np.array(x_list)
                    x = x2
                    x = torch.tensor(x, dtype=torch.float32)

            data = Data(x0=x0, x=x, edge_index=edge_index, edge_index2=torch.LongTensor(new_edgeindex2), y1=y1, y2=y2, y=y)
            datalist.append(data)
        data, slices = self.collate(datalist)
        if self.root == 'twitter16':
            torch.save((data, slices), "/data/zhangjiajun/autoaug_fakenews/data/T16_gacl.pt")
        elif self.root == 'twitter15':
            torch.save((data, slices), "/data/zhangjiajun/autoaug_fakenews/data/T15_gacl.pt")

        process_end = time.time()
        print("process time: ", process_end - process_start)
    def get(self, index):
        # return self.data[index]
        # get_start = time.time()
        data = self.data.__class__()
        # get_end = time.time()
        # print("get time: ", get_end - get_start)

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[index]

        for key in self.data.keys:
            if key == 'num_nodes':
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[index],
                                                       slices[index + 1])
            else:
                s = slice(slices[index], slices[index + 1])
            data[key] = item[s]


        return data

class PhemeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        print("init")
        self.data, self.slices = torch.load("/data/zhangjiajun/pheme/pheme.pt")

        # self.num_features = 768

    def len(self):
        return self.datadf.shape[0]

    @property
    def num_classes(self):
        return 2

    @property
    def num_features(self):
        return 768

    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        process_start = time.time()
        self.datadf = pd.read_csv("/data/zhangjiajun/pheme/pheme.csv")


        self.datadf['tweets'] = self.datadf['tweets'].apply(lambda x: np.array(json.loads(x)))
        self.datadf['infs'] = self.datadf['infs'].apply(lambda x: np.array(json.loads(x)))
        datalist = []

        for index, row in self.datadf.iterrows():
            x = row['tweets']
            edge_index = row['infs']
            y = row['labels']
            x = torch.tensor(x, dtype=torch.float32)
            edge_index = torch.LongTensor(edge_index)
            y = torch.LongTensor([y])

            # Process your data here

            data = Data(x=x, edge_index=edge_index, y=y)
            datalist.append(data)
        data, slices = self.collate(datalist)
        torch.save((data, slices), "/data/zhangjiajun/pheme/pheme.pt")

        process_end = time.time()
        print("process time: ", process_end - process_start)
    def get(self, index):
        # return self.data[index]
        # get_start = time.time()
        data = self.data.__class__()
        # get_end = time.time()
        # print("get time: ", get_end - get_start)

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[index]

        for key in self.data.keys:
            if key == 'num_nodes':
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[index],
                                                       slices[index + 1])
            else:
                s = slice(slices[index], slices[index + 1])
            data[key] = item[s]


        return data



def get_dataset(name, sparse=True, feat_str="deg+ak3+reall", root=None, aug=None, aug_ratio=None):
    print("get_dataset:",name)
    if 'twitter' in name:
       # dataset = TUDatasetTwitter(name,path, pre_transform=pre_transform,use_node_attr=True, processed_filename="data_%s.pt" % feat_str, aug=aug, aug_ratio=aug_ratio)
        dataset = TwitterDataset(name)
    elif 'pheme' in name:
        dataset = PhemeDataset(name)
    return dataset

def get_gacl_dataset(name):
    print("get_dataset:",name)
    dataset = GACLDataset(name)
    return dataset