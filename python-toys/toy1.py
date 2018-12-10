import os,shutil

#src_root='/home/dm/Desktop/XMCDATA/20181128'
src_root='/home/dm/Desktop/useModelFilterBlurData/testNewModel2'
#src_root='/home/dm/Desktop/useModelFilterBlurData/clear'
#src_root='/home/dm/Desktop/XMCDATA/oImages-v4/oImages-v4/20181128'
#src_root='/home/dm/Desktop/XMCDATA/oImages-v3/oImages-v3'
#src_root='/home/dm/Desktop/useModelFilterBlurData/TrainAndTestData2/val/clear'
#src_root='/home/dm/Desktop/useModelFilterBlurData/NewData/clear_sum'
#des_root='/home/dm/Desktop/useModelFilterBlurData/NewData/val/clear'
#des_root='/home/dm/Desktop/useModelFilterBlurData/clear_sum'
#des_root='/home/dm/Desktop/useModelFilterBlurData/testNewModel'
des_root='/home/dm/Desktop/useModelFilterBlurData/testNewModel2'
typeName=['student','teacher']

removeEmptyImgFolder='/home/dm/Desktop/useModelFilterBlurData/clear_sum'

#copy tree folders 

# for typefolder in typeName:
# 	os.mkdir(os.path.join(des_root,typefolder))
# 	for idfolder in os.listdir(os.path.join(src_root,typefolder)):
# 		os.mkdir(os.path.join(des_root,typefolder,idfolder))
# 		for classfolder in os.listdir(os.path.join(src_root,typefolder,idfolder)):
# 			os.mkdir(os.path.join(des_root,typefolder,idfolder,classfolder))


#------------------------------------------------------------------------


# remove images

for typefolder in typeName:
	for idfolder in os.listdir(os.path.join(src_root,typefolder)):
		for classfolder in os.listdir(os.path.join(src_root,typefolder,idfolder)):
			for imgName in os.listdir(os.path.join(src_root,typefolder,idfolder,classfolder)):
				os.remove(os.path.join(src_root,typefolder,idfolder,classfolder,imgName)) 



# for typefolder in typeName:
# 	for videofolder in os.listdir(os.path.join(src_root,typefolder)):
# 		idIndex=0
# 		for imgName in os.listdir(os.path.join(src_root,typefolder,videofolder)):
# 			if idIndex % 20 == 0:
# 				shutil.copyfile(os.path.join(src_root,typefolder,videofolder,imgName),os.path.join(des_root,imgName))
# 				print(os.path.join(typefolder,videofolder,imgName))

# 			idIndex+=1


#------------------------------------------------------------------------

#split data

# idIndex=0
# for imgName in os.listdir(src_root):
# 	if idIndex % 6 == 0:
# 		shutil.copyfile(os.path.join(src_root,imgName),os.path.join(des_root,imgName))
# 		os.remove(os.path.join(src_root,imgName))
# 		print(os.path.join(imgName))

# 	idIndex+=1



#------------------------------------------------------------------------

# idIndex=0
# for imgName in os.listdir(src_root):
# 	idIndex+=1
# 	if idIndex % 3 == 0:
# 		shutil.copyfile(os.path.join(src_root,imgName),os.path.join(des_root,imgName))
# 		os.remove(os.path.join(src_root,imgName))


