import sys
sys.path.append('../')
import pickle
import numpy as np
import torch
from datasets import points_utils

def sliding_window(data, seq_length):   # data: N * 3
    x = []
    y = []
    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)] # _x: 10*3
        _y = data[i+seq_length]     #  _y: 3
        sample_bb = _x[-1]
        x_temp = []
        y_temp = []
        for j in range(len(_x)):
            x_temp.append(points_utils.generate_single_pc(_x[j].center.reshape(3, 1), sample_bb).points.squeeze())
        # velocity_x = [x_temp[j] - x_temp[j-1] for j in range(1, len(x_temp))]
        y_temp.append(points_utils.generate_single_pc(_y.center.reshape(3, 1), sample_bb).points.squeeze())
        x.append(x_temp)
        y.append(y_temp)
    return np.array(x), np.array(y) # x: N x seq_length x 3; y: N x 3;

# for training tracklet generation
# path = '/home/xiayan/tracking/nuscenes/preload_nuscenes_Car_train_track_v1.0-trainval_10_-1.dat'
# path = '/mnt/data/yan/tracking/training/preload_kitti_All_train_velodyne_-1.dat'
# for testing tracklet generation
# path = '/mnt/data/yan/tracking/training/preload_kitti_Car_test_velodyne_-1.dat'
# path = '/mnt/data/yan/tracking/training/preload_kitti_All_test_velodyne_-1.dat'
path = '/home/xiayan/tracking/nuscenes/preload_nuscenes_Car_val_v1.0-trainval_-1_1.dat'
seq_length = 10
frame_window = 10
with open(path, 'rb') as f:
    training_seqs = pickle.load(f)
count_seq = 0
for v_index in range(len(training_seqs)):
    print('[%d]/[%d]' %(v_index, len(training_seqs)))
    frame_list = training_seqs[v_index]
    if len(frame_list) < seq_length + 1: # seq_length: num of velocity
        continue
    else:
        frame_location_list = []
        for i in range(len(frame_list)):
            frame_location_list.append(frame_list[i]['3d_bbox'])
        x, y = sliding_window(frame_location_list, seq_length)
        if count_seq == 0:
            trainX = torch.from_numpy(x)
            trainY = torch.from_numpy(y)
        else:
            trainX = torch.cat((trainX, torch.from_numpy(x)), dim=0)
            trainY = torch.cat((trainY, torch.from_numpy(y)), dim=0)
        count_seq += 1
print(trainX.size())
print(trainY.size())
# torch.save(trainX, 'nuscenes_car_train_x_' + str(seq_length) + '_normalize_pos.pt')
# torch.save(trainY, 'nuscenes_car_train_label_'+ str(seq_length) +'_normalize_pos.pt')

# for the testing data generation
torch.save(trainX, 'nuscenes_car_test_x_' + str(seq_length) + '_normalize_pos.pt')
torch.save(trainY, 'nuscenes_car_test_label_'+ str(seq_length) +'_normalize_pos.pt')


