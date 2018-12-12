#!/bin/bash

batch_size=16

kernel_size=3

padding=1

learning_rate=0.0001

data_path='./NewData'

print_freq=20

resume="./lr_0_001_epoch_15_acc_83_28.pth.tar"

python main2.py \
--res-block \
--batch-size $batch_size \
--print-freq $print_freq \
--data $data_path \
--lr $learning_rate \
| tee a.txt