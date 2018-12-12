#!/bin/bash

#resume='/home/dm/Desktop/OneLayerOutFeatures/best_models/kernel_3/features128/train_blur_clear_1_1/lr_0_001_epoch_8_acc_92_38.pth.tar'
#resume='/home/dm/Desktop/OneLayerOutFeatures/TrainAndTestData2/val/blur'
#resume="./oneCovNewData/lr_0_0001_epoch_19_acc_83_28.pth.tar"
resume="./oneBottleNectNewData/lr_0_0001_epoch_20_acc_86_65.pth.tar"
#scan_folder="/home/dm/Desktop/OneLayerOutFeatures/TrainAndTestData2/val/blur"

#scan_root="/home/dm/Desktop/XMCDATA/etalk-v35"
#scan_root="./oneCovNewData"
#des_folder="/home/dm/Desktop/useModelFilterBlurData/testNewModel"
#des_root='/home/dm/Desktop/useModelFilterBlurData/testNewModel'
des_root='/home/dm/Desktop/useModelFilterBlurData/testNewModel3'
#data_path='./NewData'

python filter.py \
--res-block \
--resume $resume \
--des-root $des_root
#--data $data_path
#--scan-root $scan_root \
#--des-folder $des_folder