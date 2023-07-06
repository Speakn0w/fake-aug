import os
import random
from random import shuffle
import pandas as pd
import pickle
import numpy as np
import json

# global
label2id = {
            "rumor": 0,
            "non-rumor": 1,
            }

def gen_dataframe():
    path = '/data/zhangjiajun/pheme/all'
    label_path = '/data/zhangjiajun/pheme/pheme_label.json'
    label_json = json.load(open(label_path))
    t_path = path
    file_list = os.listdir(t_path)
    column_names = ["ids", "tweets", "infs", "labels"]
    ids, tweets, infs, labels = [], [], [], []  # [str]  [np.ndarray]  [list]  [int]
    labelDic = {}
    lenlist = []
    for eid, label in label_json.items():
        if eid in file_list:
            labelDic[eid] = label.lower()
            label = label.lower()
            labels.append(label2id[label])
            ids.append(eid)
            with open('/data/zhangjiajun/pheme/all/' + eid + '/tweets.pkl', 'rb') as t:
                tweet_num = pickle.load(t)
            dict = {}
            for index, tweet in enumerate(tweet_num):
                dict[tweet] = index
            with open('/data/zhangjiajun/pheme/all/' + eid + '/structure.pkl', 'rb') as f:
                inf = pickle.load(f)
            inf = inf[1:]
            new_inf = []
            for pair in inf:
                new_pair = []
                for E in pair:
                    if E == 'ROOT':
                        break
                    E = dict[E]
                    new_pair.append(E)
                if E != 'ROOT':
                    new_inf.append(new_pair)
            new_inf = np.array(new_inf).T
            edgeindex = new_inf
            init_row = list(edgeindex[0])
            init_col = list(edgeindex[1])
            burow = list(edgeindex[1])
            bucol = list(edgeindex[0])
            row = init_row + burow
            col = init_col + bucol
            edge_index = [row, col]
            edge_index = np.array(edge_index)  #!!!!!!!!!!!!!!!
            infs.append(edge_index)

            with open('/data/zhangjiajun/pheme/bert_w2c/PHEME/pheme_mask/' + eid + '.json', 'r') as j_f:
                json_inf = json.load(j_f)

            x_list = json_inf[eid]
            x = np.array(x_list)
            tweets.append(x)

    data_dict = {column_name: l for column_name, l in
                 zip(column_names, [ids, tweets, infs, labels])}
    datadf = pd.DataFrame(data_dict)
    print(type(datadf.iloc[0]['infs']))
    print(datadf.iloc[0]['infs'])
    datadf['tweets'] = datadf['tweets'].apply(lambda x: json.dumps(x.tolist()))
    print(datadf['infs'][1])
    print(type(datadf['infs'][1]))
    datadf['infs'] = datadf['infs'].apply(lambda x: json.dumps(x.tolist()))
    print("tatoldata DataFrame: ")
    print(len(datadf))
    print(datadf.head())
    datadf.to_csv("/data/zhangjiajun/pheme/pheme.csv")

gen_dataframe()
