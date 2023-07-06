#!/bin/bash -ex

CUDA_VISIBLE_DEVICES=4 python main.py --dataset 'twitter16' --lr 0.001 --epochs 1000 --eta 1.0 --n_percents 7 \
--lr_decay_step_size 11500 --lr_decay_factor 0.1 --batch_size 128 --random_seed 0 --seed 89901 --sep_by_event 1 --exp train_gcn