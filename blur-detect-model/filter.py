# -*- coding: utf-8 -*-
import argparse
import os
import random
import shutil
import time
import warnings
import cv2
from PIL import Image

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
from torch.autograd import Variable


parser = argparse.ArgumentParser()

parser.add_argument('--kernel-size',default=3,type=int,help='kernel size')

parser.add_argument('--padding',default=1,type=int,help='padding')

parser.add_argument('--out-features',default=128,type=int,help='one-conv-layer out-features')

parser.add_argument('--scan-folder',default='/home/dm/Desktop/XMCDATA/etalk-v3/20181119/teacher/frames_7859_43640_0',
					type=str,help='scan folder')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')

parser.add_argument('--max-pool',action='store_true',help='whether has max-pool')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
					metavar='LR', help='initial learning rate')  

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')                                                # momentum

parser.add_argument('--weight-decay', '--wd', default=0, type=float,
					metavar='W', help='weight decay (default: 0)') 

parser.add_argument('--des-folder',default='/home/dm/Desktop/useModelFilterBlurData/modelGetBlur2',type=str)

parser.add_argument('--scan-root',default='/home/dm/Desktop/XMCDATA/etalk-v3',type=str)

parser.add_argument('--src-root',default='/home/dm/Desktop/XMCDATA/20181128',type=str)
parser.add_argument('--des-root',default='/home/dm/Desktop/useModelFilterBlurData/testNewModel2',type=str)

parser.add_argument('--res-block',action='store_true')

parser.add_argument('--blur-threshold',default=0.95,type=float)

args = parser.parse_args()

typeName=['student','teacher']

best_acc1=0.0

checkpoint=None

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

def main():
	if args.res_block:
		model=OneLayerResNet()
	else:
		model = OneConvLayerNet(args.kernel_size ,args.padding ,args.out_features)
	model = torch.nn.DataParallel(model).cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)
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
			return 

	else:
		print('error:you must load a model')
		return 

	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	])
	
	img_sum=0


	# for timeName in os.listdir(args.scan_root):
	# 	for IDTypte in os.listdir(os.path.join(args.scan_root,timeName)):
	# 		for videoName in os.listdir(os.path.join(args.scan_root,timeName,IDTypte)):
	# 			for imgName in os.listdir(os.path.join(args.scan_root,timeName,IDTypte,videoName)):
	# 				img_sum+=1
	# 				img=cv2.cvtColor(cv2.imread(os.path.join(args.scan_root,timeName,IDTypte,videoName,imgName)),cv2.COLOR_BGR2RGB)

	# 				img_tensor=transform(img)

	# 				img_output=model(Variable(img_tensor.unsqueeze(0)))
	# 				sm = nn.Softmax()
	# 				img_output = sm(img_output)

	# 				#print(img_output.size())
	# 				if img_output[0][0] > 0.5:
	# 					shutil.copyfile(os.path.join(args.scan_root,timeName,IDTypte,videoName,imgName),os.path.join(args.des_folder,imgName))
	# 					print(img_output)




	# for imgName in os.listdir(args.scan_folder):
	# 	img_sum+=1
	# 	img=cv2.cvtColor(cv2.imread(os.path.join(args.scan_folder,imgName)),cv2.COLOR_BGR2RGB)

	# 	img_tensor=transform(img)

	# 	img_output=model(Variable(img_tensor.unsqueeze(0)))
	# 	sm = nn.Softmax()
	# 	img_output = sm(img_output)

	# 	#print(img_output.size())
	# 	#if img_output[0][0] > 0.5:
	# 		#shutil.copyfile(os.path.join(args.scan_folder,imgName),os.path.join(args.des_folder,imgName))
	# 	print(img_output)


	# XMCDATA/20181128
	for typefolder in typeName:
		#os.mkdir(os.path.join(des_root,typefolder))
		for idfolder in os.listdir(os.path.join(args.src_root,typefolder)):
			#os.mkdir(os.path.join(des_root,typefolder,idfolder))
			for classfolder in os.listdir(os.path.join(args.src_root,typefolder,idfolder)):
				#os.mkdir(os.path.join(des_root,typefolder,idfolder,classfolder))
				img_sum=0
				for imgName in os.listdir(os.path.join(args.src_root,typefolder,idfolder,classfolder)):
					
					#img_sum+=1
					#img=cv2.cvtColor(cv2.imread(os.path.join(args.src_root,typefolder,idfolder,classfolder,imgName)),cv2.COLOR_BGR2RGB)
					img=Image.open(os.path.join(args.src_root,typefolder,idfolder,classfolder,imgName))
					img_tensor=transform(img)

					img_output=model(Variable(img_tensor.unsqueeze(0)))
					sm = nn.Softmax()
					img_output = sm(img_output)

					#print(img_output.size())
					if img_output[0][0] > args.blur_threshold:
						img_sum+=1
						#shutil.copyfile(os.path.join(args.src_root,typefolder,idfolder,classfolder,imgName),os.path.join(args.des_root,typefolder,idfolder,classfolder,imgName))
						#print('des:',os.path.join(typefolder,idfolder,classfolder),'output:',img_output)	
						shutil.copyfile(os.path.join(args.src_root,typefolder,idfolder,classfolder,imgName),os.path.join(args.des_root,imgName))
						print('output:',img_output)				
				print(os.path.join(typefolder,idfolder,classfolder),'sum:',img_sum)

	print('images sum:',img_sum)



if __name__ == '__main__':
	main()