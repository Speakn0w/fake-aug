import os
import random
from random import shuffle
import pandas as pd
import pickle
import numpy as np
import json

# global
label2id = {
            "unverified": 0,
            "non-rumor": 1,
            "true": 2,
            "false": 3,
            }

def gen_dataframe(dataname):
    if 'twitter16' in dataname:
        path = '/data/zhangjiajun/autoaug_fakenews/data/twitter16/'
        label_path = '/data/zhangjiajun/autoaug_fakenews/data/Twitter16_label_All.txt'
        labelPath = os.path.join(label_path)
        t_path = path
        file_list = os.listdir(t_path)
        column_names = ["ids", "tweets", "infs", "labels", "events", "sep_labels"]
        ids, tweets, infs, labels = [], [], [], [] #[str]  [np.ndarray]  [list]  [int]
        labelDic = {}
        lenlist = []

        events = []
        sep_labels = []

        event_sep_train_path = "/data/zhangjiajun/autoaug_fakenews/data/Twitter16_eventsep.train"
        event_sep_dev_path = "/data/zhangjiajun/autoaug_fakenews/data/Twitter16_eventsep.dev"
        event_sep_test_path = "/data/zhangjiajun/autoaug_fakenews/data/Twitter16_eventsep.test"

        event_sep_train = []
        event_sep_test = []
        with open(event_sep_train_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                _, eid, _, _ = line.strip().split("\t")
                event_sep_train.append(eid)
        with open(event_sep_dev_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                _, eid, _, _ = line.strip().split("\t")
                event_sep_train.append(eid)
        with open(event_sep_test_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                _, eid, _, _ = line.strip().split("\t")
                event_sep_test.append(eid)

        for line in open(labelPath):
            line = line.rstrip()
            label, eid, event = line.split('\t')[0], line.split('\t')[2], line.split('\t')[1]
            if eid in file_list:
                events.append(event)
                sep_labels.append(0 if eid in event_sep_train else 1) # 0: train, 1: test

                labelDic[eid] = label.lower()
                label = label.lower()
                labels.append(label2id[label])
                ids.append(eid)
                with open('/data/zhangjiajun/autoaug_fakenews/data/twitter16/' + eid + '/after_tweets.pkl', 'rb') as t:
                    tweet_num = pickle.load(t)
                dict = {}
                for index, tweet in enumerate(tweet_num):
                    dict[tweet] = index
                with open('/data/zhangjiajun/autoaug_fakenews/data/twitter16/' + eid + '/after_structure.pkl', 'rb') as f:
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
                lenlist.append(len(row))
                # print("len row", len(row))
                # print("len col", len(col))
                new_edgeindex = [row, col]
                new_edgeindex = np.array(new_edgeindex)
                infs.append(new_edgeindex)
                with open('/data/zhangjiajun/autoaug_fakenews/bert_w2c/bert_w2c/T16/t16_mask_00/' + eid + '.json',
                          'r') as j_f:
                    json_inf = json.load(j_f)
                x = json_inf[eid]

                x = np.array(x)
                # print(x.shape)
                tweets.append(x)

        len1 = np.mean(lenlist)
        print(len1)

        data_dict = {column_name: l for column_name, l in
                     zip(column_names, [ids, tweets, infs, labels, events, sep_labels])}
        datadf = pd.DataFrame(data_dict)
        # print(type(datadf.iloc[0]['infs']))
        # print(datadf.iloc[0]['infs'])
        datadf['tweets'] = datadf['tweets'].apply(lambda x: json.dumps(x.tolist()))
        datadf['infs'] = datadf['infs'].apply(lambda x: json.dumps(x.tolist()))
        print("tatoldata DataFrame: ")
        print(len(datadf))
        print(datadf.head())
        print(datadf['sep_labels'].value_counts())
        print(datadf['events'].nunique())
        pd.set_option ('display.max_columns', None)
        print(datadf['events'].value_counts())
        datadf.to_csv("/data/zhangjiajun/autoaug_fakenews/data/T16.csv")

    elif 'twitter15' in dataname:
        path = '/data/zhangjiajun/autoaug_fakenews/data/twitter15/'
        label_path = '/data/zhangjiajun/autoaug_fakenews/data/Twitter15_label_All.txt'
        labelPath = os.path.join(label_path)
        t_path = path
        file_list = os.listdir(t_path)
        column_names = ["ids", "tweets", "infs", "labels", "events", "sep_labels"]
        ids, tweets, infs, labels = [], [], [], []  # [str]  [np.ndarray]  [list]  [int]
        labelDic = {}
        lenlist = []

        events = []
        sep_labels = []

        event_sep_train_path = "/data/zhangjiajun/autoaug_fakenews/data/Twitter15_eventsep.train"
        event_sep_dev_path = "/data/zhangjiajun/autoaug_fakenews/data/Twitter15_eventsep.dev"
        event_sep_test_path = "/data/zhangjiajun/autoaug_fakenews/data/Twitter15_eventsep.test"

        event_sep_train = []
        event_sep_test = []
        with open(event_sep_train_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                _, eid, _, _ = line.strip().split("\t")
                event_sep_train.append(eid)
        with open(event_sep_dev_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                _, eid, _, _ = line.strip().split("\t")
                event_sep_train.append(eid)
        with open(event_sep_test_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                _, eid, _, _ = line.strip().split("\t")
                event_sep_test.append(eid)

        for line in open(labelPath):
            line = line.rstrip()
            label, eid, event = line.split('\t')[0], line.split('\t')[2], line.split('\t')[1]
            if eid in file_list:
                events.append(event)
                sep_labels.append(0 if eid in event_sep_train else 1) # 0: train, 1: test

                labelDic[eid] = label.lower()
                label = label.lower()
                labels.append(label2id[label])
                ids.append(eid)
                with open('/data/zhangjiajun/autoaug_fakenews/data/twitter15/' + eid + '/after_tweets.pkl', 'rb') as t:
                    tweet_num = pickle.load(t)
                dict = {}
                for index, tweet in enumerate(tweet_num):
                    dict[tweet] = index
                with open('/data/zhangjiajun/autoaug_fakenews/data/twitter15/' + eid + '/after_structure.pkl',
                          'rb') as f:
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
                lenlist.append(len(row))
                # print("len row", len(row))
                # print("len col", len(col))
                new_edgeindex = [row, col]
                new_edgeindex = np.array(new_edgeindex)
                infs.append(new_edgeindex)
                with open('/data/zhangjiajun/autoaug_fakenews/bert_w2c/bert_w2c/T15/t15_mask_00/' + eid + '.json',
                          'r') as j_f:
                    json_inf = json.load(j_f)
                x = json_inf[eid]

                x = np.array(x)
                # print(x.shape)
                tweets.append(x)

        len1 = np.mean(lenlist)
        print(len1)

        data_dict = {column_name: l for column_name, l in
                     zip(column_names, [ids, tweets, infs, labels, events, sep_labels])}
        datadf = pd.DataFrame(data_dict)
        datadf['tweets'] = datadf['tweets'].apply(lambda x: json.dumps(x.tolist()))
        datadf['infs'] = datadf['infs'].apply(lambda x: json.dumps(x.tolist()))
        print("tatoldata DataFrame: ")
        print(len(datadf))
        print(datadf.head())
        print(datadf['sep_labels'].value_counts())
        print(datadf['events'].nunique())
        pd.set_option ('display.max_columns', None)
        print(datadf['events'].value_counts())
        datadf.to_csv("/data/zhangjiajun/autoaug_fakenews/data/T15.csv")


gen_dataframe('twitter16')

