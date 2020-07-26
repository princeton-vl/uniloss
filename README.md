# UniLoss
Code in Pytorch for the paper:

**A Unified Framework of Surrogate Loss by Refactorization and Interpolation**<br/>
Lanlan Liu, Mingzhe Wang, Jia Deng<br/>
ECCV 2020

## Downloading Data
For binary classification and classification tasks, the corresponding MNIST and CIFAR-10/100 datasets are downloaded automatically.

### MPII for Pose Estimation

1. Download the images from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/)

2. Create a symbolic link to the `images` directory of the MPII dataset:
   ```
   ln -s PATH_TO_MPII_IMAGES_DIR pose/data/mpii/images
   ```
   
## Training and Evaluation

### Binary Classification
To download the MNIST dataset and train the binary classification task with UniLoss, run 
```
python train_mnist.py --batch-size 16 
```


### Multiclass Classification
To download the CIFAR-10 dataset and train the multi-class classification task with UniLoss, run 
```
python train_cifar10.py --batch-size 128
```

To download the CIFAR-100 dataset and train the multi-class classification task with UniLoss, run 
```
python train_cifar100.py --batch-size 128
```

### Pose Estimation

After downloading the MPII images, to train the pose estimation task with UniLoss, run 
```
python example/train_mpii.py -a hg --lr 2.5e-4 --schedule 30 40 50
```

