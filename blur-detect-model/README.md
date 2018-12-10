## Introduction

blur-detect-model is a neural network model to detect whether one image is blur or clear. The dataset I build with real data from Etalk Stub ( a 1-to-1 online English tutoring platform)



## Model Usage

There are three different neural network model in this code:

1. Single Convolution Layer
2. Single Residual block
3. ResNet18

To train one of these models, You should set many parameters in `train.sh`

`train.sh` is a shell which can run in Linux directly by typing `./train.sh` in terminal, and you should set parameters in `train.sh` instead of `train.py` as far as possible

### Dataset

The input data default size is (3 x 144 x 176), but it's easy to change the input-size by adding a resize transform 

### GPU

If you don't have GPU, you'll get a bug when running the code because of the `.cuda`method. So when there is no GPU in your computer , delete all the `.cuda` method

### Train ResNet18 Model

If you want use the ResNet18 model, you must add `--resnet18` in `train.sh`

What's more important is the input-size must be (3 x 224 x 224) if you wanna use pretrained model, and to use the pretrained model , you should add `--pretrained ` in `train.sh`

### Train Single Residual-Block / Conv-Layer Model

If you want use the Single Residual Block model, you must add `--res-block` in `train.sh`

If you add nothing , the default model is Single-Conv-Layer Model

And there are many other parameter you can find in `train.py`

### evaluate

Add `-e` or`--evaluate` in `train.sh` and the model will just run evaluating function with validation set



### Filter Images

When you train the model, you can get many best model step by step named `XXXX.pth.tar`. If you wanna use the best model to select pictures ,you should add `--resume` and the best model package path in `filter.sh`

Similarly, there are many parameters in `filter.py`



