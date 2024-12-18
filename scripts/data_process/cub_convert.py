import os
import shutil

source = 'data/CUB200/original/images'
i = 0
for root,dirs,files in os.walk(source):
    target = './data/CUB200/class/'
    i = i+1
    if i != 1 :
        clas=int(root.split('/')[-1].split('.')[0])
        if clas <= 100:
            target = os.path.join(target, 'train')
        else:
            target = os.path.join(target, 'test')
        target_path = os.path.join(target, root.split('/')[-1])
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
        for image in files:
            image_path=os.path.join(root,image)
            shutil.copy(image_path, target_path)
