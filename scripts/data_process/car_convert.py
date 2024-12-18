import scipy.io
import os
import shutil

source = './data/CARS196/original'

data = scipy.io.loadmat('./data/CARS196/original/cars_annos.mat')
class_names = data['class_names']
annotations = data['annotations']

for i in range(annotations.shape[1]):
    target = './data/CARS196/class/'
    name = str(annotations[0, i][0])[2:-2]
    image_path = os.path.join(source, name)
    clas = int(annotations[0, i][5])
    if clas <= 98:
        target = os.path.join(target, 'train')
    else:
        target = os.path.join(target, 'test')
    class_name = str(class_names[0, clas-1][0]).replace(' ', '_')
    class_name = class_name.replace('/', '')
    target_path = os.path.join(target, class_name)
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    shutil.copy(image_path, target_path)
