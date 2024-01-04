import os
import shutil

source = './data/SOP/original'
trainset = './data/SOP/original/Ebay_train.txt'
testset = './data/SOP/original/Ebay_test.txt'
i,j,k = 0,0,0

with open(trainset) as t:
    target = './data/SOP/class'
    target = os.path.join(target, 'train')
    for row in t:
        if k != 0:
            picture = row.split( )[3]
            image_path = os.path.join(source,picture)
            class_name = row[-20:-7].rstrip('_').lstrip('/')
            target_path = os.path.join(target,class_name)+'/'
            if not os.path.isdir(target_path):
                os.makedirs(target_path)
            shutil.copy(image_path, target_path)
        k = k+1

k = 0
with open(testset) as t:
    target = './data/SOP/class'
    target = os.path.join(target, 'test')
    for row in t:
        if k != 0:
            picture = row.split( )[3]
            image_path = os.path.join(source,picture)
            class_name = row[-20:-7].rstrip('_').lstrip('/')
            target_path = os.path.join(target,class_name)+'/'
            if not os.path.isdir(target_path):
                os.makedirs(target_path)
            shutil.copy(image_path, target_path)
        k = k+1
