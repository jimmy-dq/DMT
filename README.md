# A Lightweight and Detector-free 3D Single Object Tracker on Point Clouds (DMT)
Pytorch-Lightning implementation of DMT.  

[A Lightweight and Detector-free 3D Single Object Tracker on Point Clouds](https://arxiv.org/pdf/2203.04232.pdf). 

**IEEE Transactions on Intelligent Transportation Systems 2023** 


### Features
+ Modular design. It is easy to config the model and trainng/testing behaviors through just a `.yaml` file.
+ DDP support for both training and testing.
### Setup
Installation
+ Create the environment
  ```
  git clone https://github.com/jimmy-dq/EXPL_BAT.git
  cd BAT
  conda create -n bat  python=3.6
  conda activate bat
  ```
+ Install pytorch
  ```
  conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
  ```
  Our code is well tested with pytorch 1.4.0 and CUDA 10.1. But other platforms may also work. Follow [this](https://pytorch.org/get-started/locally/) to install another version of pytorch.

+ Install other dependencies
  ```
  pip install -r requirement.txt
  ```

KITTI dataset
+ Download the data for [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).
+ Unzip the downloaded files.
+ Put the unzipped files under the same folder as following.
  ```
  [Parent Folder]
  --> [calib]
      --> {0000-0020}.txt
  --> [label_02]
      --> {0000-0020}.txt
  --> [velodyne]
      --> [0000-0020] folders with velodynes .bin files
  ```
### Quick Start
#### Training
To train a model, you must specify the `.yaml` file with `--cfg` argument. The `.yaml` file contains all the configurations of the dataset and the model. Currently, we provide three `.yaml` files under the [*cfgs*](./cfgs) directory. **Note:** Before running the code, you will need to edit the `.yaml` file by setting the `path` argument as the correct root of the dataset.
```bash
python main.py --gpu 0 1 --cfg cfgs/BAT_Car.yaml  --batch_size 50 --epoch 60
```
After you start training, you can start Tensorboard to monitor the training process:
```
tensorboard --logdir=./ --port=6006
```
By default, the trainer runs a full evaluation on the full test split after training every epoch. You can set `--check_val_every_n_epoch` to a larger number to speed up the training.
#### Testing
To test a trained model, specify the checkpoint location with `--checkpoint` argument and send the `--test` flag to the command.
```bash
python main.py --gpu 0 1 --cfg cfgs/BAT_Car.yaml  --checkpoint /path/to/checkpoint/xxx.ckpt --test


### Acknowledgment
+ This repo is built upon [P2B](https://github.com/HaozheQi/P2B) and [SC3D](https://github.com/SilvioGiancola/ShapeCompletion3DTracking).
+ Thank Erik Wijmans for his pytorch implementation of [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)

### License
This repository is released under MIT License (see LICENSE file for details).
# DMT_Tracking
