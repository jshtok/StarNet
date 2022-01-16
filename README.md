## StarNet: Weakly supervised Object detection and classification

<div align="center">
  <img src="gate image.PNG"/ alt="drawing" width="500"/>
</div>

This is a Pytorch implementation of the StarNet paper algorithm:

**Karlinsky, Leonid, Joseph Shtok, Amit Alfassy, Moshe Lichtenstein, Sivan Harary, Eli Schwartz, Sivan Doveh et al. "[StarNet: towards Weakly
 Supervised Few-Shot Object Detection.](https://www.aaai.org/AAAI21Papers/AAAI-9153.KarlinskyL.pdf)" In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 2, pp. 1743-1753. 2021.**
 
The algorithm addresses the task of few-shot object classification and localization, while training on the base classes in the few-shot fashion
  also. Moreover, the training only uses image-level data, so no bounding boxes are needed (except for performance evaluation on the test data).

The code is released and maintaned by the AI Vision group of IBM Research AI, part of the Research Lab in Haifa, Israel. 

## Installation instructions
### Conda environment
The package is tested in the following environment:
```
python 3.9.4
pytorch 1.8.1
cuda 11.1
torchvision 0.9.1
```

Step-by-step instructions are as follows:
```
conda create --name starnet_pub python==3.9
. activate starnet_pub
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c conda-forge opencv tqdm matplotlib psutil
conda install scikit-image
pip install torchnet
conda install -c anaconda h5py
```
## Datasets
We exemplify the algorithm on two datasets, the ImageNet-LOC with a given split of classes into the base and novel categories, and the CUB dataset
 (with bounding boxes). The datasets should be downloaded to your environment and the dataset paths should be updated accordingly:in `data
 * `imagenet_loc_data_path`, `imagenet_loc_anno_path` in `/imagenet_loc.py` 
 and 
 * `DATASET_DIR` in `data/cub_bb.py`, `data/cub_bb.py`
 
 The argument  `--recompute_dataset_dicts 1 ` forces to recreate the lists of classes and samples, prepared once for the experiments.
 
## The training process
The training process for detection and classification differs in that for classification we use two-stage StarnNet, while for detection only stage
 1 is used.
 ### Detection training:
 #### Imagenet-LOC dataset:
 ```
python train.py --do_train \
--save-path "./experiments/train_inloc_1" \
--val_iters 0 --dataset imagenet-loc --image_res=168 --network=ResNet_star_hi \
--two_stage=0 --train-shot=1 --train-query=6 --episodes-per-batch=4 \
--scheduler_regime 1 --num-epoch 150 --recompute_dataset_dicts 1
```
 Here we use the `168` image size, which is a higher than the standard `84` size used by default. The detections are produced during the evaluation
  phase, which is enabled by setting `--val_iters` to a positive value (recommended to do so after the training, in a separate run as detailed below)

#### CUB dataset:
```
python train.py --do_train \
--save-path "./experiments/train_cub_1" \
--dataset cub --network=ResNet_star \
--two_stage=0 --train-shot=1 --train-query=6 --val_iters 0 --episodes-per-batch=4 \
--scheduler_regime 1 --num-epoch 150 --recompute_dataset_dicts 1

```

### Classification training and evaluation:
The test accuracy is computed along the training
```
python train.py --do_train \
 --save-path=./experiments/cls_cub_01 \
--dataset=cub  --network=ResNet_star_2stage \
--train-query=4 --two_stage=1 --scheduler_regime 0
```

## Detection inference and evaluation:
Just remove the `--do_train` argument and set `eval_iters` to a positive value to enable the test performance output:

```
python train.py --save-path "./experiments/train_inloc_1" --resume 1 --resume_model_name last_epoch.pth --dataset imagenet-loc --image_res=168 \
--network=ResNet_star_hi --train-query=6   --num-epoch 150 --val_iters=500
```



