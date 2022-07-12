""" 
baseModel.py
Created by zenn at 2021/5/9 14:40
"""

import torch
from easydict import EasyDict
import pytorch_lightning as pl
from datasets import points_utils
from utils.metrics import TorchSuccess, TorchPrecision
from utils.metrics import estimateOverlap, estimateAccuracy
import torch.nn.functional as F
import numpy as np
from sklearn import *
from sklearn.preprocessing import PolynomialFeatures
import time

class LSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1, seq_len=10):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        self.linear = torch.nn.Linear(hidden_layer_size*seq_len, output_size)

        self.num_layers = num_layers

        # self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
        #                     torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        h_0 = torch.zeros(
            self.num_layers, input_seq.size(0), self.hidden_layer_size).cuda()

        c_0 = torch.zeros(
            self.num_layers, input_seq.size(0), self.hidden_layer_size).cuda()

        # lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        lstm_out, self.hidden_cell = self.lstm(input_seq, (h_0, c_0))
        predictions = self.linear(lstm_out.reshape(len(input_seq), -1))
        return predictions #[-1]


# lstm = LSTM(input_size=3, hidden_layer_size=50, output_size=3, seq_len=10)
# lstm.load_state_dict(torch.load('/home/yan/EXPL_BAT/lstm_models/pedestrian_model_len_10_hidden_50_normalize_position_add_noise.pt'))
# lstm = lstm.cuda()
# lstm.eval()
# print('loading the lstm model')

class BaseModel(pl.LightningModule):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        # testing metrics
        self.prec = TorchPrecision()
        self.success = TorchSuccess()
        self.lstm = LSTM(input_size=3, hidden_layer_size=50, output_size=3, seq_len=10)
        self.lstm.load_state_dict(torch.load('/workspace/tracking/EXPL_BAT_car_kitti_transformer/lstm_models/car_model_len_10_hidden_50_normalize_position_add_noise.pt'))
        self.lstm = self.lstm.cuda()
        self.lstm.eval()

    def configure_optimizers(self):
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, data, output):
        """

        :param data: input data
        :param output:
        :return:
        """
        estimation_boxes = output['estimation_boxes']  # B,num_proposal,5
        estimation_cla = output['estimation_cla']  # B,N
        seg_label = data['seg_label']
        box_label = data['box_label']  # B,4
        proposal_center = output["center_xyz"]  # B,num_proposal,3
        vote_xyz = output["vote_xyz"]

        loss_seg = F.binary_cross_entropy_with_logits(estimation_cla, seg_label)

        loss_vote = F.smooth_l1_loss(vote_xyz, box_label[:, None, :3].expand_as(vote_xyz), reduction='none')  # B,N,3
        loss_vote = (loss_vote.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)

        dist = torch.sum((proposal_center - box_label[:, None, :3]) ** 2, dim=-1)

        dist = torch.sqrt(dist + 1e-6)  # B, K
        objectness_label = torch.zeros_like(dist, dtype=torch.float)
        objectness_label[dist < 0.3] = 1
        objectness_score = estimation_boxes[:, :, 4]  # B, K
        objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1
        loss_objective = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                            pos_weight=torch.tensor([2.0]).cuda())
        loss_objective = torch.sum(loss_objective * objectness_mask) / (
                torch.sum(objectness_mask) + 1e-6)
        loss_box = F.smooth_l1_loss(estimation_boxes[:, :, :4],
                                    box_label[:, None, :4].expand_as(estimation_boxes[:, :, :4]),
                                    reduction='none')
        loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

        return {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
            "loss_seg": loss_seg,
            "loss_vote": loss_vote,
        }



    def generate_template(self, sequence, current_frame_id, results_bbs):
        """
        generate template for evaluating.
        the template can be updated using the previous predictions.
        :param sequence: the list of the whole sequence
        :param current_frame_id:
        :param results_bbs: predicted box for previous frames
        :return:
        """
        first_pc = sequence[0]['pc']
        previous_pc = sequence[current_frame_id - 1]['pc']
        if "firstandprevious".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([first_pc, previous_pc],
                                                               [results_bbs[0], results_bbs[current_frame_id - 1]],
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        elif "first".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(first_pc, results_bbs[0],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "previous".upper() in self.config.hape_aggregation.upper():
            template_pc, canonical_box = points_utils.cropAndCenterPC(previous_pc, results_bbs[current_frame_id - 1],
                                                                      scale=self.config.model_bb_scale,
                                                                      offset=self.config.model_bb_offset)
        elif "all".upper() in self.config.shape_aggregation.upper():
            template_pc, canonical_box = points_utils.getModel([frame["pc"] for frame in sequence[:current_frame_id]],
                                                               results_bbs,
                                                               scale=self.config.model_bb_scale,
                                                               offset=self.config.model_bb_offset)
        return template_pc, canonical_box

    def generate_search_area(self, sequence, current_frame_id, results_bbs):
        """
        generate search area for evaluating.

        :param sequence:
        :param current_frame_id:
        :param results_bbs:
        :return:
        """
        this_bb = sequence[current_frame_id]["3d_bbox"]
        this_pc = sequence[current_frame_id]["pc"]
        if ("previous_result".upper() in self.config.reference_BB.upper()):
            ref_bb = results_bbs[-1]
        elif ("previous_gt".upper() in self.config.reference_BB.upper()):
            previous_bb = sequence[current_frame_id - 1]["3d_bbox"]
            ref_bb = previous_bb
        elif ("current_gt".upper() in self.config.reference_BB.upper()):
            ref_bb = this_bb
        search_pc_crop = points_utils.generate_subwindow(this_pc, ref_bb,
                                                         scale=self.config.search_bb_scale,
                                                         offset=self.config.search_bb_offset)
        return search_pc_crop, ref_bb

    def prepare_input(self, template_pc, search_pc, template_box, *args, **kwargs):
        """
        construct input dict for evaluating
        :param template_pc:
        :param search_pc:
        :param template_box:
        :return:
        """
        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T, self.config.template_size,
                                                            seed=1)
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T, self.config.search_size,
                                                          seed=1)
        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32)
        data_dict = {
            'template_points': template_points_torch[None, ...],
            'search_points': search_points_torch[None, ...],
        }
        return data_dict

    def get_past_avarage_velocity(self, results_bbs, frame_num=2, average_velocity_model = False, linear_regression = False,
                                  polynomial_regression = False, ridge_regression = False, lstm_regression = False):
        frame_num = min(len(results_bbs), frame_num)
        pre_locations = []
        velocity = []
        results_bbs_temp = results_bbs[(len(results_bbs)-frame_num):]
        for i in range(len(results_bbs_temp)):
            rel_location = points_utils.generate_single_pc(results_bbs_temp[i].center.reshape(3, 1), results_bbs_temp[-1])
            pre_locations.append(rel_location.points) #3x1
        if average_velocity_model:
            # if frame==2, it's the constant velocity model
            for i in range(1, len(pre_locations)):
                velocity.append(pre_locations[i] - pre_locations[i-1])
            mean_velocity = np.mean(np.array(velocity), axis=0)
            predicted_location = mean_velocity

        if linear_regression:
            ols = linear_model.LinearRegression()
            times_steps = np.array([i for i in range(len(results_bbs_temp))]).reshape(len(results_bbs_temp), 1)
            pre_locations = np.array(pre_locations).squeeze()
            ols.fit(times_steps, pre_locations)
            predicted_location = ols.predict(np.array([[pre_locations.shape[0]]])) #1x3
            predicted_location = predicted_location.transpose(1, 0)

        if polynomial_regression:
            times_steps = np.array([i for i in range(len(results_bbs_temp))]).reshape(len(results_bbs_temp), 1)
            poly = PolynomialFeatures(degree=2)
            poly.fit(times_steps)
            inputs = poly.transform(times_steps)
            pre_locations = np.array(pre_locations).squeeze()
            ols = linear_model.LinearRegression()
            ols.fit(inputs, pre_locations)

            testing = np.array([[pre_locations.shape[0]]])
            poly.fit(testing)
            testing = poly.transform(testing)

            predicted_location = ols.predict(testing)
            predicted_location = predicted_location.transpose(1, 0)

        if ridge_regression:
            ols = linear_model.RidgeCV(alphas=np.logspace(-5, 15, 30), cv=2)
            times_steps = np.array([i for i in range(len(results_bbs_temp))]).reshape(len(results_bbs_temp), 1)
            pre_locations = np.array(pre_locations).squeeze()
            ols.fit(times_steps, pre_locations)
            predicted_location = ols.predict(np.array([[pre_locations.shape[0]]]))  # 1x3
            predicted_location = predicted_location.transpose(1, 0)



        return predicted_location


    def evaluate_one_sequence(self, sequence):
        """

        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        time_start = time.time()
        ious = []
        distances = []
        results_bbs = []
        # dist_gt = []
        gt_bbs = []
        # for frame_id in range(len(sequence)):
            # if frame_id > 0:
            #     dist_gt.append(np.sqrt(np.sum((sequence[frame_id]["3d_bbox"].center - sequence[frame_id-1]["3d_bbox"].center)**2)))
        for frame_id in range(len(sequence)):  # tracklet
            # print(frame_id)
            # if frame_id == 130:
            #     print('a')
            this_bb = sequence[frame_id]["3d_bbox"]
            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
                gt_bbs.append(this_bb)
            else:

                # preparing search area
                # NOTE: crop pcs based on the previous estimated location
                search_pc_crop, ref_bb = self.generate_search_area(sequence, frame_id, results_bbs)
                # update template
                # crop the previous template PCs and concat them together
                template_pc, canonical_box = self.generate_template(sequence, frame_id, results_bbs)
                # construct input dict
                # randomly sample pcs for template and search
                # template_pc: 512
                # search_pc: 1024
                # BAT prepare_input
                data_dict = self.prepare_input(template_pc, search_pc_crop, canonical_box)

                if frame_id == 1:
                    previous_center = points_utils.generate_single_pc(results_bbs[-1].center.reshape(3, 1), results_bbs[-1])
                    previous_center = previous_center.points.transpose(1, 0)
                    data_dict['previous_center'] = torch.from_numpy(previous_center).float().cuda() # 1x3
                    data_dict['samples'] = None
                else:
                    # previous_center = points_utils.generate_single_pc(results_bbs[-1].center.reshape(3, 1),
                    #                                                   results_bbs[-1])  # 3x1
                    # bef_previous_center = points_utils.generate_single_pc(results_bbs[-2].center.reshape(3, 1),
                    #                                                       results_bbs[-1])  # 3x1
                    # velocity = previous_center.points - bef_previous_center.points
                    #
                    # est_cur_centers = velocity + previous_center.points
                    # est_cur_centers = est_cur_centers.transpose(1, 0)
                    # data_dict['previous_center'] = torch.from_numpy(est_cur_centers).float().cuda()
                    # data_dict['samples'] = None

                    if frame_id < 10:  # 10: # 10 is the seq_len for velocity, i.e., at least have (seq_len+1) frames, except for the current frame
                        previous_center = points_utils.generate_single_pc(results_bbs[-1].center.reshape(3, 1),
                                                                          results_bbs[-1])  # 3x1
                        bef_previous_center = points_utils.generate_single_pc(results_bbs[-2].center.reshape(3, 1),
                                                                              results_bbs[-1])  # 3x1
                        velocity = previous_center.points - bef_previous_center.points

                        est_cur_centers = velocity + previous_center.points
                        est_cur_centers = est_cur_centers.transpose(1, 0)
                        data_dict['previous_center'] = torch.from_numpy(est_cur_centers).float().cuda()
                        data_dict['samples'] = None
                    else:
                        pre_locations = []
                        results_bbs_temp = results_bbs[(len(results_bbs) - 10):]
                        for i in range(len(results_bbs_temp)):
                            rel_location = points_utils.generate_single_pc(results_bbs_temp[i].center.reshape(3, 1),
                                                                           results_bbs_temp[-1])
                            pre_locations.append(rel_location.points)  # 3x1
                        location_input = torch.from_numpy(np.array(pre_locations)).squeeze().unsqueeze(0).cuda()
                        est_cur_centers = self.lstm(location_input.float()).cpu().data.numpy().reshape(3, 1)

                        # # frame_location_list = gt_bbs[(len(gt_bbs)-10-1):]
                        # velocity_list = [frame_location_list[j].center - frame_location_list[j - 1].center for j in range(1, len(frame_location_list))]
                        # velocity_input = torch.from_numpy(np.array(velocity_list)).unsqueeze(0).cuda()
                        # velocity = lstm(velocity_input.float()).cpu().data.numpy().reshape(3, 1)
                        # predicted_location = results_bbs[-1].center.reshape(3, 1) + velocity
                        # # predicted_location = gt_bbs[-1].center.reshape(3, 1) + velocity

                        # est_cur_centers = points_utils.generate_single_pc(predicted_location, results_bbs[-1]) # 3x1
                        # est_cur_centers = points_utils.generate_single_pc(this_bb.center.reshape(3, 1), results_bbs[-1]) # 3x1
                        est_cur_centers = est_cur_centers.transpose(1, 0)  # 1x3
                        data_dict['previous_center'] = torch.from_numpy(est_cur_centers).float().cuda()
                        data_dict['samples'] = None

                # if frame_id <=1:
                end_points = self(data_dict)
                estimation_box = end_points['estimation_boxes']
                estimation_boxes_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

                if estimation_boxes_cpu.shape[1] == 4:
                    estimation_box_cpu = estimation_boxes_cpu[0, 0:4]
                else:
                    best_box_idx = estimation_boxes_cpu[:, 4].argmax()
                    estimation_box_cpu = estimation_boxes_cpu[best_box_idx, 0:4]

                candidate_box = points_utils.getOffsetBB(ref_bb, estimation_box_cpu, degrees=self.config.degrees,
                                                         use_z=self.config.use_z,
                                                         limit_box=self.config.limit_box)
                results_bbs.append(candidate_box)
                gt_bbs.append(this_bb)
            # if np.isnan(results_bbs[-1].center).sum() == 0 and np.isinf(results_bbs[-1].center).sum() == 0:
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=self.config.IoU_space, up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=self.config.IoU_space, up_axis=self.config.up_axis)
            # else:
            #     this_overlap = 0.0
            #     this_accuracy = 0.0
            ious.append(this_overlap)
            distances.append(this_accuracy)
        time_end = time.time()
        fps = len(sequence)/(time_end-time_start)
        print(fps)
        return ious, distances

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success, on_step=True, on_epoch=True)
        self.log('precision/test', self.prec, on_step=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)
