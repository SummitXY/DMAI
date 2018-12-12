# 模糊检测模型压缩版README

### main2.py / train.sh

模型训练与测试代码

可选择三个模型之一：

1. 单卷积层:

   | 可选参数            | 解释                                                  |
   | ------------------- | ----------------------------------------------------- |
   | --out-features  int | 单卷积输出channel数                                   |
   | --kernel-size  int  | 卷积核大小                                            |
   | --padding  int      | 填充，为了使卷积后图像size不变，需根据kernel-size调节 |
   | --adam              | 如果添加，使用adam优化器，默认使用SGD-Momentum        |
   | --max-pool          | 如果添加，使用池化，默认不使用                        |

2. 单残差快，加参数`--res-block`

3. ResNet18，加参数`--resnet18`

三个模型都有的参数：

| 可选参数          | 解释                         |
| ----------------- | ---------------------------- |
| --lr  float       | learning rate                |
| --batch-size  int | batch size                   |
| --data  str       | 训练数据地址                 |
| --print-freq  int | 输出频率                     |
| --resume  str     | best model 的路径            |
| -e                | 只进行测试，需要加`--resume` |

> 以上参数设置均在`train.sh`文件设置



### filter.py / filter.sh

使用训练好的模型筛选数据

拟采用的best-model:`./best_model.pth.tar`，模型结构与参数：

| 属性         | 值            |
| ------------ | ------------- |
| 网络结构     | 单残差快; out-features = 128 |
| 验证集精度   | 86.65%        |
| 输入图片大小 | 3 × 144 × 176 |
| 输入图片存储格式             |![img](/home/dm/Desktop/blurModelForZhaohui/README_IMAGE/structure.png)               |
| 处理形式             |将高于设定阈值的模糊图片复制到指定文件夹               |

`filter.sh`可选参数：

| 可选参数 | 解释 |
| -------- | ---- |
| --resume  str         |导入best-model      |
|--des-root  str |将高于设定阈值的模糊图片复制到的目标文件夹 |
|--blur-threshold float |模糊阈值，默认为0.95 |

