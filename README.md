# GCA-HNG: Globally Correlation-Aware Hard Negative Generation
This repository serves as the official PyTorch implementation for the paper: Globally Correlation-Aware Hard Negative Generation.

It offers source code for replicating the experiments conducted on four benchmark datasets (CUB-200-2011, Cars196, SOP, and InShop) and releases the pretrained metric models.

## Requirements
+ Python 3.8
+ PyTorch 1.8.1+cu111
+ torch_scatter 2.0.8
+ numpy
+ tqdm
+ tensorboardX
+ scikit-learn
+ scipy

## Datasets
1. Download four benchmark datasets.
    + [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158)
    + [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
    + [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)
    + [InShop Clothes Retrieval](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
2. Extract the tgz or zip file into `./data/dataset_name/original/` folder and run the data convert scripts to transform data format. In particular, the InShop dataset does not require any data transformation.
```
python scripts/data_process/xxx_convert.py
```
3. The data folder is constructed as followed:
```
data:
├──  CUB200/CARS196/SOP
│  └──  class 
│    ├──  train 
│    │  ├──  Catrgory 1
│    │  ├──  Catrgory ...
│    │  └──  Catrgory M
│    └──  test
│       ├──  Catrgory 1
│       ├──  Catrgory ...
│       └──  Catrgory N
└──  IN_SHOP
   ├── img  
   │  ├──  MEN
   │  └──  WOMEN
   └──  list_eval_partition.txt

```

## Training Process
Run the training process by designating the corresponding yml file.
```
python train.py --cfg scripts/cfgs/xxx.yml
```

## Image Retrieval Evaluation
The released metric model is available at [Google Drive](https://drive.google.com/drive/folders/1oGtEQ61MDGtGbSbE7gsYEj1CCfu0ugPL?usp=sharing).

Given the (full model/ metric model) with pretrained weights, run the evaluation process as follows:
```
python eval.py --cfg scripts/cfgs/xxx.yml --model_path xxx.pth
```

## Citation
```
@article{peng2024globally,
  title = {Globally Correlation-Aware Hard Negative Generation},
  author = {Peng, Wenjie and Huang, Hongxiang and Chen, Tianshui and Ke, Quhui and Dai, Gang and Huang, Shuangping}
  journal = {International Journal of Computer Vision},
  volume={133},
  number={5},
  pages = {2441-2462},
  year = {2025},
  doi = {10.1007/s11263-024-02288-0},
}
```