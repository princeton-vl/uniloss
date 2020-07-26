The pose estimation framework is brought from [Pytorch-Pose](https://github.com/bearpaw/pytorch-pose). Thanks to the authors for providing such a good implementation. We implement a new example in examples/uniloss.py.

## Usage
1. Create a symbolic link to the `images` directory of the MPII dataset:
   ```
   ln -s PATH_TO_MPII_IMAGES_DIR data/mpii/images
   ```
2. Run 
```
python example/train_mpii.py -a hg --lr 2.5e-4 --schedule 30 40 50
```






