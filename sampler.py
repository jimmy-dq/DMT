# Created by zenn at 2021/4/27

import random
from math import sin

import torch
import numpy as np
import datasets.points_utils as points_utils
from datasets.searchspace import KalmanFiltering
from easydict import EasyDict
import copy


def no_processing(data, *args):
    return data


def siamese_processing(data, config):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    """
    first_frame = data['first_frame']           # get the first frame point clouds
    template_frame = data['template_frame']     # get the last frame (before the current frame) point clouds
    search_frame = data['search_frame']         # get the current frame point clouds
    candidate_id = data['candidate_id']         # when candidate_id ==0, there is no offsets

    # get specific data, note: they are in the same coordinate system
    first_pc, first_box = first_frame['pc'], first_frame['3d_bbox']
    template_pc, template_box = template_frame['pc'], template_frame['3d_bbox']
    search_pc, search_box = search_frame['pc'], search_frame['3d_bbox']
    dist = np.sqrt(np.sum((search_box.center - template_box.center)**2))
    if candidate_id == 0:
        samplegt_offsets = np.zeros(3)
    else:
        samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
        samplegt_offsets[2] = samplegt_offsets[2] * (5 if config.degrees else np.deg2rad(5))
    # generate the noisy template box in the previous frame
    template_box = points_utils.getOffsetBB(template_box, samplegt_offsets, limit_box=config.limit_box, degrees=config.degrees)

    # get the previous target center (in the global coordinate system)
    pl = copy.deepcopy(template_box.center)
    # generating template. Merging the object from previous and the first frames.
    model_pc, model_box = points_utils.getModel([first_pc, template_pc], [first_box, template_box],
                                                scale=config.model_bb_scale, offset=config.model_bb_offset)


    assert model_pc.nbr_points() > 20, 'not enough template points'

    # generating search area. Use the current gt box to select the nearby region as the search area.
    if candidate_id == 0:
        sample_offset = np.zeros(3)
    else:
        gaussian = KalmanFiltering(bnd=[1, 1, (5 if config.degrees else np.deg2rad(5))])
        sample_offset = gaussian.sample(1)[0]
    sample_bb = points_utils.getOffsetBB(search_box, sample_offset, limit_box=config.limit_box, degrees=config.degrees)
    search_pc_crop = points_utils.generate_subwindow(search_pc, sample_bb,
                                                     scale=config.search_bb_scale, offset=config.search_bb_offset)
    ################
    # previous_center = pl - sample_bb.center
    previous_center = points_utils.generate_single_pc(pl.reshape(3, 1), sample_bb)
    previous_center = previous_center.points.transpose(1, 0)
    ################
    assert search_pc_crop.nbr_points() > 20, 'not enough search points'
    search_box = points_utils.transform_box(search_box, sample_bb)
    seg_label = points_utils.get_in_box_mask(search_pc_crop, search_box).astype(int)
    # this is the true center location of the gt 3D box
    search_bbox_reg = [search_box.center[0], search_box.center[1], search_box.center[2], -sample_offset[2]]

    # add random shifts to the gt center in the current frame
    # search_box.center
    true_gt_center = np.array([[search_box.center[0], search_box.center[1], search_box.center[2]]])
    # scale = (true_gt_center - previous_center) * 1.5 #1x3

    # random sampling around the GT center
    k_num = 32 #8
    dis_thr = 0.15

    pos_samples = np.zeros((k_num//2, 3))
    neg_samples = np.zeros((k_num//2, 3))
    count_pos = 0
    count_neg = 0
    # sample random offsets for pos positions
    while count_pos < (k_num//2):
        # random_offsets = np.random.uniform(low = -max(np.max(abs(scale)), 1.0), high = max(np.max(abs(scale)), 1.0), size=3) # make sure our offsets are nor too small
        random_offsets = np.random.uniform(low = -0.15, high = 0.15, size=3)
        dis = np.sqrt(np.sum(random_offsets**2))
        if dis < dis_thr:
            pos_samples[count_pos] = random_offsets + true_gt_center[0]
            count_pos += 1

    while count_neg < (k_num//2):
        random_offsets = np.random.uniform(low=-1.0, high=1.0, size=3)
        dis = np.sqrt(np.sum(random_offsets ** 2))
        if dis > dis_thr:
            neg_samples[count_neg] = random_offsets + true_gt_center[0]
            count_neg += 1
    samples = np.concatenate((pos_samples, neg_samples), axis=0) # num//2 pos, num//2 neg, the last is the previous location


    # # random sampling based on the velocity
    # samples_v = np.zeros((k_num, 3))
    # velocity = true_gt_center - previous_center
    # count_sample = 0
    # scale = 1.5
    # while count_sample < k_num:
    #           for i in range(3):
    #               if velocity[0][i] == 0:
    #                   samples_v[count_sample][i] = true_gt_center[0][i]
    #               elif velocity[0][i] > 0:
    #                   samples_v[count_sample][i] = true_gt_center[0][i] + np.random.uniform(low=0, high=velocity[0][i]*scale, size=1)[0]
    #               elif velocity[0][i] < 0:
    #                   samples_v[count_sample][i] = true_gt_center[0][i] + np.random.uniform(low=velocity[0][i] * scale,
    #                                                                                         high=0,
    #                                                                                         size=1)[0]
    #           count_sample += 1
    # samples = np.concatenate((samples, samples_v), axis=0)





    template_points, idx_t = points_utils.regularize_pc(model_pc.points.T, config.template_size)
    search_points, idx_s = points_utils.regularize_pc(search_pc_crop.points.T, config.search_size)
    seg_label = seg_label[idx_s]
    data_dict = {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': search_box.wlh,
        'seg_label': seg_label.astype('float32'),
        'dist': dist.astype('float32'),
        'samples': samples.astype('float32')
    }
    if getattr(config, 'box_aware', False):
        template_bc = points_utils.get_point_to_box_distance(template_points, model_box)
        search_bc = points_utils.get_point_to_box_distance(search_points, search_box)
        previous_location_bc = points_utils.get_point_to_box_distance(previous_center.reshape(1, 3), search_box)
        data_dict.update({'points2cc_dist_t': template_bc.astype('float32'),
                          'points2cc_dist_s': search_bc.astype('float32'),
                          'previous_center': previous_center.reshape(3).astype('float32'),
                          'previous_location_bc': previous_location_bc.astype('float32')})
    return data_dict


class PointTrackingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, random_sample, sample_per_epoch=10000, processing=siamese_processing, config=None,
                 **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.sample_per_epoch = sample_per_epoch
        self.dataset = dataset
        self.processing = processing
        self.config = config
        self.random_sample = random_sample
        self.num_candidates = getattr(config, 'num_candidates', 1)
        if not self.random_sample:
            num_frames_total = 0
            self.tracklet_start_ids = [num_frames_total]
            for i in range(dataset.get_num_tracklets()):
                num_frames_total += dataset.get_num_frames_tracklet(i)
                self.tracklet_start_ids.append(num_frames_total)

    def get_anno_index(self, index):
        return index // self.num_candidates

    def get_candidate_index(self, index):
        return index % self.num_candidates

    def __len__(self):
        if self.random_sample:
            return self.sample_per_epoch * self.num_candidates
        else:
            return self.dataset.get_num_frames_total() * self.num_candidates

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:
            if self.random_sample:
                tracklet_id = torch.randint(0, self.dataset.get_num_tracklets(), size=(1,)).item()
                tracklet_annos = self.dataset.tracklet_anno_list[tracklet_id]
                frame_ids = [0] + points_utils.random_choice(num_samples=2, size=len(tracklet_annos)).tolist()
            else:
                for i in range(0, self.dataset.get_num_tracklets()):
                    if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                        tracklet_id = i
                        this_frame_id = anno_id - self.tracklet_start_ids[i]
                        prev_frame_id = max(this_frame_id - 1, 0)
                        frame_ids = (0, prev_frame_id, this_frame_id)
            first_frame, template_frame, search_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            data = {"first_frame": first_frame,
                    "template_frame": template_frame,
                    "search_frame": search_frame,
                    "candidate_id": candidate_id}

            return self.processing(data, self.config)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]


class TestTrackingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, config=None, **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return self.dataset.get_num_tracklets()

    def __getitem__(self, index):
        tracklet_annos = self.dataset.tracklet_anno_list[index]
        frame_ids = list(range(len(tracklet_annos)))
        return self.dataset.get_frames(index, frame_ids)


