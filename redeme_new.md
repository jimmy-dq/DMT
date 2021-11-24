### MP3-Tracking: Motion Prediction based 3D Single Object Tracking

------

#### Introduction

MP3-Tracking is a detector-free Motion Prediction based 3D Tracking network that totally removes the usage of complicated 3D detectors, which is lighter, faster, and more accurate than previous methods. MP3-Tracking achieves state-of-the-art performance on both KITTI and NuScenes.
The whole training process can be conducted on a single GPU (e.g., RTX 3090).

#### Main Pre-requisites

- Python3.7
- PyTorch-lightning1.3.8
- CUDA-11.0
- Scipy
- Pandas
- Sklearn
- More pre-requisites can be found in requirements.txt
- Dataset preparation and DAT file generation for KITTI and NuScenes, refer to BAT[1], P2B [2].

#### Generate Tracklet Dataset for Training Motion Prediction Module

```python
python datasets/generate_tracklet_dataset.py --path path_to_dat_training_file
```

#### Training Motion Prediction Module

```python
python datasets/train_lstm_model.py
```

#### Training MP3-Tracking
Specify the path of the trained LSTM model in models/base_model.py, then run:
```python
python main.py --gpu 0 --cfg cfgs/MP3Tracking_NUSCENES_CAR.yaml  --batch_size 100 --epoch 60 --preloading
```

#### Testing
To test a trained model, specify the checkpoint location with `--checkpoint` argument and send the `--test` flag to the command.
```python
python main.py --gpu 0 --cfg cfgs/MP3Tracking_NUSCENES_CAR.yaml  --checkpoint /path/to/checkpoint/xxx.ckpt --test
```

#### Acknowledgement
This is a modified version of the framework based on **BAT** [1] and **P2B** [2], We would like to thank their authors for providing great frameworks and toolkits.

#### References

1. Chaoda Zheng and Xu Yan and Jiantao Gao. Box-Aware Feature Enhancement for Single Object Tracking on Point Clouds. ICCV 2021.
2. P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds. CVPR 2020.

------

