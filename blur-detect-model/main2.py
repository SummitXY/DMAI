# -*- coding: utf-8 -*-
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch blurDataset Training')
parser.add_argument('--data', metavar='DIR',
					help='path to dataset', default='./TrainAndTestData', type=str)                                         # dataset
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
					choices=model_names,
					help='model architecture: ' +
						' | '.join(model_names) +
						' (default: resnet18)')                                     # architecture

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')             # data loading workers

parser.add_argument('--epochs', default=10000, type=int, metavar='N',
					help='number of total epochs to run')                           # total epochs

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')                # start epoch

parser.add_argument('-b', '--batch-size', default=8, type=int,
					metavar='N', help='mini-batch size (default: 4)')               # batch size

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
					metavar='LR', help='initial learning rate')                     # initial learning rate

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')                                                # momentum

parser.add_argument('--weight-decay', '--wd', default=0, type=float,
					metavar='W', help='weight decay (default: 0)')                  # weight decay

parser.add_argument('--print-freq', '-p', default=100, type=int,
					metavar='N', help='print frequency (default: 1)')               # print frequency

parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')               # resume from latest checkpoint

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')                        # evaluate model on validation set

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')                                   # use pretrained model

parser.add_argument('--world-size', default=1, type=int,
					help='number of distributed processes')                         # number of distributed processes

parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')                 # url used to set up distributed training

parser.add_argument('--dist-backend', default='gloo', type=str,
					help='distributed backend')                                     # distributed backend

parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')                        # seed for initializing training

parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')                                          # gpu

parser.add_argument('--kernel-size',default=3,type=int,help='kernel size')

parser.add_argument('--padding',default=1,type=int,help='padding')

parser.add_argument('--max-pool',action='store_true',
					help='whether has max-pool')

parser.add_argument('--adam' ,action='store_true',
					help='whether to use adam optimizer')

parser.add_argument('--out-features',default=128,type=int,help='one-conv-layer out-features')

parser.add_argument('--res-block',action='store_true', help='whether use ResidualBlock')

parser.add_argument('--resnet18',action='store_true',help='whether use resnet18 model')

args = parser.parse_args()

best_acc1	 = 0

# one-conv-layer network
class OneConvLayerNet(nn.Module):

	def __init__(self,kernel_size,padding,out_features):
		super(OneConvLayerNet,self).__init__()
		self.conv = nn.Conv2d(3, out_features, kernel_size,padding=padding)
		self.batch_norm=nn.BatchNorm2d(out_features)
		#self.fc = nn.Linear(112 * 112 * out_features, 2)
		if args.max_pool:
			self.fc=nn.Linear(72 * 88 * out_features,2)
		else:
			self.fc=nn.Linear(144 * 176 * out_features,2)

	def forward(self, x):
		if args.max_pool:
			x = F.max_pool2d(F.relu(self.batch_norm(self.conv(x))),2)
		else:
			x = F.relu(self.batch_norm(self.conv(x)))
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

#--------------------------------------------------------------------------------------------

# one-layer-resnet
class ResidualBlock(nn.Module):

	def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
		super(ResidualBlock, self).__init__()
		self.left = nn.Sequential(
				nn.Conv2d(inchannel,outchannel,3,stride, 1,bias=False),
				nn.BatchNorm2d(outchannel),
				nn.ReLU(inplace=True),
				nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
				nn.BatchNorm2d(outchannel) )
		self.right = shortcut

	def forward(self, x):
		out = self.left(x)
		residual = x if self.right is None else self.right(x)
		out += residual
		return F.relu(out)
		
class OneLayerResNet(nn.Module):

	def __init__(self, num_classes=2):
		super(OneLayerResNet, self).__init__()
		self.pre = nn.Sequential(
				nn.Conv2d(3, 64, 3,padding=1, bias=False),
				nn.BatchNorm2d(64),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(3, 2, 1))
		
		self.layer1 = self._make_layer( 64, 64, 1)
		#self.layer2 = self._make_layer( 64, 128, 4, stride=2)
		#self.layer3 = self._make_layer( 128, 256, 6, stride=2)
		#self.layer4 = self._make_layer( 256, 512, 3, stride=2)

		self.fc = nn.Linear(64*72*88, num_classes)
	
	def _make_layer(self,  inchannel, outchannel, block_num, stride=1):

		shortcut = nn.Sequential(
				nn.Conv2d(inchannel,outchannel,1,stride, bias=False),
				nn.BatchNorm2d(outchannel))
		
		layers = []
		layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
		
		for i in range(1, block_num):
			layers.append(ResidualBlock(outchannel, outchannel))
		return nn.Sequential(*layers)
		
	def forward(self, x):
		x = self.pre(x)
		
		x = self.layer1(x)
		#x = self.layer2(x)
		#x = self.layer3(x)
		#x = self.layer4(x)

		#x = F.avg_pool2d(x, 7)
		x = x.view(x.size(0), -1)
		return self.fc(x)
		#return x

#---------------------------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18():
    """Constructs a ResNet18-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet18(BasicBlock, [2, 2, 2, 2])
    #if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


# 以上部分为resnet18源码


#----------------------------------------------------------------------------------------------------
def save_checkpoint(state, is_best_model, file_name='checkpoint.pth.tar'):
	torch.save(state, file_name)
	if is_best_model:
		saveBestModelStr='lr_0_'+str(args.lr)[2:]+'_epoch_'+str(state["epoch"])+'_acc_'+str(state["best_acc1"])[0:2]+'_'+str(state["best_acc1"])[3:5]+'.pth.tar'
		shutil.copyfile(file_name, saveBestModelStr)



def main():
	global args, best_acc1
	

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					  'disable data parallelism.')

	args.distributed = args.world_size > 1

	if args.distributed:
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size)

	# create model
	if args.resnet18:
		if args.pretrained:
			print("=> using pre-trained ResNet18 model '{}'".format(args.arch))
			model = models.__dict__[args.arch](pretrained=True)
		else:
			print("=> creating ResNet18 model '{}'".format(args.arch))
			model = resnet18()

		# adjust model
		# model.conv1.requires_grad = False
		# model.bn1.requires_grad = False
		# num_ftrs = model.fc.in_features
		# model.fc = nn.Linear(num_ftrs, 2)
		# model.fc.weight.data.normal_(0, 0.0001)
		# model.fc.bias.data.zero_()
	
	else:
		
		if args.res_block:
			print('=> creating one-res-block network')
			model=OneLayerResNet()
		else:
			print('=> creating one-conv-layer network')
			model = OneConvLayerNet(args.kernel_size ,args.padding ,args.out_features)

	

	if args.gpu is not None:
		model = model.cuda(args.gpu)
	elif args.distributed:
		model.cuda()
		model = torch.nn.parallel.DistributedDataParallel(model)
	else:
		if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
			model.features = torch.nn.DataParallel(model.features)
			model.cuda()
		else:
			model = torch.nn.DataParallel(model).cuda()

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()

	if args.adam:
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	else:
		optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_acc1 = checkpoint['best_acc1']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	# Data loading code
	traindir = os.path.join(args.data, 'train')
	valdir = os.path.join(args.data, 'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	train_dataset = datasets.ImageFolder(
		traindir,
		transforms.Compose([
			#transforms.Resize((224, 224)),
			#transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	else:
		train_sampler = None

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
		num_workers=args.workers, pin_memory=True, sampler=train_sampler)

	val_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(valdir, transforms.Compose([
			#transforms.Resize((224, 224)),
			#transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	if args.evaluate:
		validate(val_loader, model, criterion, args.start_epoch)
		return

	for epoch in range(args.start_epoch, args.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)
		adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch)

		# evaluate on validation set
		acc1 = validate(val_loader, model, criterion, epoch)

		# remember best acc@1 and save checkpoint
		is_best = acc1 > best_acc1
		best_acc1 = max(acc1, best_acc1)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'state_dict': model.state_dict(),
			'best_acc1': best_acc1,
			'optimizer' : optimizer.state_dict(),
		}, is_best)



def train(train_loader, model, criterion, optimizer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	#top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if args.gpu is not None:
			input = input.cuda(args.gpu, non_blocking=True)
		target = target.cuda(args.gpu, non_blocking=True)

		# compute output
		output = model(input)
		loss = criterion(output, target)

		# measure accuracy and record loss
		#acc1, acc5 = accuracy(output, target, topk=(1, 5))
		acc1 = accuracy(output, target)
		losses.update(loss.item(), input.size(0))
		top1.update(acc1[0].item(), input.size(0))
		#top5.update(acc5[0], input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, i_epoch):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	#top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(val_loader):
			if args.gpu is not None:
				input = input.cuda(args.gpu, non_blocking=True)
			target = target.cuda(args.gpu, non_blocking=True)

			# compute output
			output = model(input)
			loss = criterion(output, target)

			# measure accuracy and record loss
			#acc1, acc5 = accuracy(output, target, topk=(1, 5))
			acc1 = accuracy(output, target)
			losses.update(loss.item(), input.size(0))
			top1.update(acc1[0].item(), input.size(0))
			#top5.update(acc5[0], input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
					i, len(val_loader), batch_time=batch_time, loss=losses,
					top1=top1))

		print(' * Acc@1 {top1.avg:.3f}'
			  .format(top1=top1))

		with open('trainLog.txt','a') as fLog:
			logWriteStr='epoch '+str(i_epoch)+': '+str(top1.avg)+'\n'
			fLog.write(logWriteStr)

	return top1.avg

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = float(val)
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	#lr = args.lr * (0.1 ** (epoch // 30))
	#lr = args.lr
	print("lr = %f ,batch_size = %d ,kernel_size = %d ,optimizer = %s" % (args.lr ,args.batch_size ,args.kernel_size, 'Adam' if args.adam else 'SGD-Momentum'))
	for param_group in optimizer.param_groups:
		param_group['lr'] = args.lr


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


if __name__ == '__main__':
	main()
