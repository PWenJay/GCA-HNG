import os
import torch
import numpy as np
import pandas as pd
import PIL.Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import BatchSampler


class MetricDataset(datasets.ImageFolder):
    def __init__(self, data_path, is_train=True, size_crops=(224, 224), scale=(0.08, 1)):
        super(MetricDataset, self).__init__(data_path)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if is_train:
            self.transform = transforms.Compose([
                            transforms.RandomResizedCrop(size_crops, scale),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std)
                    ])
        else:
            self.transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std)
                    ])

    def __getitem__(self, index):
        path, classes = self.samples[index]
        images = self.loader(path)
        trans_images = self.transform(images)
        return trans_images, classes
    

class Inshop_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mode):
        self.root = root + '/IN_SHOP'
        self.mode = mode
        if mode == 'train':
            self.transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
        elif mode == 'query' or mode == 'gallery':
            self.transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
        
        self.train_ys, self.train_im_paths = [], []
        self.query_ys, self.query_im_paths = [], []
        self.gallery_ys, self.gallery_im_paths = [], []
                    
        data_info = np.array(pd.read_table(self.root +'/list_eval_partition.txt', header=1, delim_whitespace=True))[:,:]
        # Separate into training dataset and query/gallery dataset for testing.
        train, query, gallery = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]

        # Generate conversions
        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
        train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])

        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
        query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
        gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])

        # Generate Image-Dicts for training, query and gallery of shape {class_idx:[list of paths to images belong to this class] ...}
        for img_path, key in train:
            self.train_im_paths.append(os.path.join(self.root, img_path))
            self.train_ys += [int(key)]

        for img_path, key in query:
            self.query_im_paths.append(os.path.join(self.root, img_path))
            self.query_ys += [int(key)]

        for img_path, key in gallery:
            self.gallery_im_paths.append(os.path.join(self.root, img_path))
            self.gallery_ys += [int(key)]
            
        if self.mode == 'train':
            self.im_paths = self.train_im_paths
            self.ys = self.train_ys
        elif self.mode == 'query':
            self.im_paths = self.query_im_paths
            self.ys = self.query_ys
        elif self.mode == 'gallery':
            self.im_paths = self.gallery_im_paths
            self.ys = self.gallery_ys

    def nb_classes(self):
        return len(set(self.ys))
            
    def __len__(self):
        return len(self.ys)
            
    def __getitem__(self, index):
        
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im
        
        im = img_load(index)
        target = self.ys[index]

        return im, target
    
    @property
    def targets(self):
        return self.ys
    
    @property
    def classes(self):
        return list(set(self.ys))


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = np.array(list(set(labels)))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            
            for class_ in classes:  
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    for i in range(self.n_samples):
                        chose = np.random.choice(len(self.label_to_indices[class_]), 1 , replace=False)
                        indices.extend(self.label_to_indices[class_][chose])

                else:
                    indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]: self.used_label_indices_count[class_]+self.n_samples])
                
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size
