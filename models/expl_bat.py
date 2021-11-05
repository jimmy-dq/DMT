""" 
bat.py
Created by zenn at 2021/7/21 14:16
"""

import torch
from torch import nn
from models.backbone.pointnet import Pointnet_Backbone
from models.head.xcorr import BoxAwareXCorr
# from models.head.expl_rpn_sa import P2BVoteNetRPN
from models.head.expl_rpn import P2BVoteNetRPN
from models import base_model
import torch.nn.functional as F
from datasets import points_utils
from pointnet2.utils import pytorch_utils as pt_utils
import numpy as np
from pyquaternion import Quaternion
from datasets.data_classes import Box
import copy



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


lstm = LSTM(input_size=3, hidden_layer_size=50, output_size=3, seq_len=10)
lstm.load_state_dict(torch.load('/home/visal/Data/Point_cloud_project/BAT/lstm_models/car_model_len_10_hidden_50_normalize_position_add_noise_0.3_78.2_64.4.pt'))
lstm = lstm.cuda()
lstm.eval()
print('loading the lstm model')


class EXPL_BAT(base_model.BaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        self.backbone = Pointnet_Backbone(self.config.use_fps, self.config.normalize_xyz, return_intermediate=False)
        self.conv_final = nn.Conv1d(256, self.config.feature_channel, kernel_size=1)
        self.mlp_bc = (pt_utils.Seq(3 + self.config.feature_channel)
                       .conv1d(self.config.feature_channel, bn=True)
                       .conv1d(self.config.feature_channel, bn=True)
                       .conv1d(self.config.bc_channel, activation=None))

        self.xcorr = BoxAwareXCorr(feature_channel=self.config.feature_channel,
                                   hidden_channel=self.config.hidden_channel,
                                   out_channel=self.config.out_channel,
                                   k=self.config.k,
                                   use_search_bc=self.config.use_search_bc,
                                   use_search_feature=self.config.use_search_feature,
                                   bc_channel=self.config.bc_channel)
        self.rpn = P2BVoteNetRPN(self.config.feature_channel,
                                 vote_channel=self.config.vote_channel,
                                 num_proposal=self.config.num_proposal,
                                 normalize_xyz=self.config.normalize_xyz)

        # # follow this vote net for prediction
        # self.explicit_vote_layer = (
        #     pt_utils.Seq(3 + self.config.feature_channel)
        #         .conv1d(self.config.feature_channel, bn=True)
        #         .conv1d(self.config.feature_channel, bn=True)
        #         .conv1d(self.config.feature_channel, activation=None))


    def prepare_input(self, template_pc, search_pc, template_box):
        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T, self.config.template_size,
                                                            seed=1)
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T, self.config.search_size,
                                                          seed=1)

        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32)
        template_bc = points_utils.get_point_to_box_distance(template_points, template_box)
        template_bc_torch = torch.tensor(template_bc, device=self.device, dtype=torch.float32)
        data_dict = {
            'template_points': template_points_torch[None, ...],
            'search_points': search_points_torch[None, ...],
            'points2cc_dist_t': template_bc_torch[None, ...]
        }
        return data_dict

    def compute_loss(self, data, output):
        estimation_boxes = output['estimation_boxes']  # B,num_proposal,4
        box_label = data['box_label']  # B,4
        estimation_cla = output['estimation_cla']  # B,N
        seg_label = data['seg_label']

        loss_seg = F.binary_cross_entropy_with_logits(estimation_cla, seg_label)

        # box loss
        batch_size = box_label.size()[0]
        k_num = estimation_boxes.size()[0] // batch_size
        # loss_verification = 0
        for i in range(k_num): # also should include the previous_xyz batch (batch samples)
            if i == 0:
                loss_box = F.smooth_l1_loss(estimation_boxes[(i*batch_size):(i*batch_size+batch_size), :, :4],
                                            box_label[:, None, :4].expand_as(estimation_boxes[(i*batch_size):(i*batch_size+batch_size), :, :4]),
                                            reduction='none')
            else:
                loss_box += F.smooth_l1_loss(estimation_boxes[(i*batch_size):(i*batch_size+batch_size), :, :4],
                                            box_label[:, None, :4].expand_as(estimation_boxes[(i*batch_size):(i*batch_size+batch_size), :, :4]),
                                            reduction='none')
            # if i < k_num -1:
            #     if i < (k_num-1)//2:
            #         loss_verification += F.binary_cross_entropy_with_logits(
            #             output['verification_scores'].squeeze(-1)[(i * batch_size):(i * batch_size + batch_size), :],
            #             torch.ones(batch_size, 1).cuda())
            #     else:
            #         loss_verification += F.binary_cross_entropy_with_logits(
            #             output['verification_scores'].squeeze(-1)[(i * batch_size):(i * batch_size + batch_size), :],
            #             torch.zeros(batch_size, 1).cuda())



        loss_box = torch.mean(loss_box.mean(2))
        # loss_verification /= k_num

        # dist = torch.sum((estimation_boxes[:, :, :3] - box_label[:, None, :3]) ** 2, dim=-1)

        # dist = torch.sqrt(dist + 1e-6)  # B, K
        # objectness_label = torch.zeros_like(dist, dtype=torch.float)
        # objectness_label[dist < 0.2] = 1
        # objectness_score = estimation_boxes[:, :, 4]  # B, K
        # objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
        # objectness_mask[dist < 0.2] = 1
        # objectness_mask[dist > 0.6] = 1
        # loss_objective = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
        #                                                     pos_weight=torch.tensor([2.0]).cuda())

        # box-aware loss
        # search_bc = data['previous_location_bc']
        search_bc = data['points2cc_dist_s']
        pred_search_bc = output['pred_search_bc']
        loss_bc = F.smooth_l1_loss(pred_search_bc, search_bc, reduction='none')
        loss_bc = torch.sum(loss_bc.mean(2) * seg_label) / (seg_label.sum() + 1e-6)
        # loss_bc = F.smooth_l1_loss(pred_search_bc, search_bc, reduction='none')
        # loss_bc = torch.mean(loss_bc.mean(2))
        # "loss_objective": loss_objective,
        return {
                "loss_seg": loss_seg,
                "loss_bc": loss_bc,
                "loss_box": loss_box
               }
        #"loss_objective": loss_objective,
    # ,
    #                 "loss_verification": loss_verification
    # ,
    # "loss_verification": loss_verification


    def forward(self, input_dict, search_bboxes = None):
        """
        :param input_dict:
        {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': search_box.wlh,
        'seg_label': seg_label.astype('float32'),
        'points2cc_dist_t': template_bc,
        'points2cc_dist_s': search_bc,
        }

        :return:
        """
        template = input_dict['template_points']        # batchx512x3
        search = input_dict['search_points']            # batchx1024x3
        template_bc = input_dict['points2cc_dist_t']    # batchx512x9
        M = template.shape[1]
        N = search.shape[1]

        samples = input_dict['samples'] #training: batchx8x3


        # if samples == None:
        #     # in the testing stage
        #     # print('we need to do some sampling here')
        #     sampled_samples = torch.from_numpy(
        #         self.sample_nearby_location(input_dict['previous_center'].unsqueeze(0).cpu().data.numpy(),
        #                                     k_num=32)).cuda().unsqueeze(0)
        #     search = torch.cat((search, sampled_samples.float()), dim=1).cuda()
        # else:
        #     sampled_samples = samples[:, 0:int(samples.size()[1] / 2), :].float()
        #     search = torch.cat((search, sampled_samples), dim=1).cuda()
        #     for i in range(sampled_samples.size()[0]):
        #         if i == 0:
        #             seg_label = torch.from_numpy(points_utils.get_in_box_mask_from_numpy(sampled_samples[i].cpu().data.numpy().T, search_bboxes[i]).astype(int)).cuda().unsqueeze(0)
        #             sample_bc =  torch.from_numpy(points_utils.get_point_to_box_distance(sampled_samples[i].cpu().data.numpy(), search_bboxes[i])).cuda().unsqueeze(0)
        #         else:
        #             seg_label = torch.cat((seg_label,
        #                                    torch.from_numpy(points_utils.get_in_box_mask_from_numpy(sampled_samples[i].cpu().data.numpy().T, search_bboxes[i]).astype(int)).cuda().unsqueeze(0)), dim=0)
        #             sample_bc = torch.cat((sample_bc, torch.from_numpy(points_utils.get_point_to_box_distance(sampled_samples[i].cpu().data.numpy(), search_bboxes[i])).cuda().unsqueeze(0)), dim=0)
        #     input_dict['seg_label'] = torch.cat((input_dict['seg_label'], seg_label.float()), dim=1)
        #     input_dict['points2cc_dist_s'] = torch.cat((input_dict['points2cc_dist_s'], sample_bc.float()), dim=1)


        # if template.shape[0]>1:
        #     dist = input_dict['dist']

        # backbone
        # template_xyz: batchx64x3
        # template_feature: batchx256x64
        template_xyz, template_feature, sample_idxs_t = self.backbone(template, [M // 2, M // 4, M // 8])
        search_xyz, search_feature, sample_idxs = self.backbone(search, [N // 2, N // 4, N // 8])
        template_feature = self.conv_final(template_feature) # batchxDxNum.
        search_feature = self.conv_final(search_feature)

        # prepare bc
        pred_search_bc = self.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # B, 9, N // 8
        pred_search_bc = pred_search_bc.transpose(1, 2) #BxNx9
        sample_idxs_t = sample_idxs_t[:, :M // 8, None]
        template_bc = template_bc.gather(dim=1, index=sample_idxs_t.repeat(1, 1, self.config.bc_channel).long())
        # box-aware xcorr
        fusion_feature = self.xcorr(template_feature, search_feature, template_xyz, search_xyz, template_bc,
                                    pred_search_bc)
        previous_xyz = input_dict['previous_center'].unsqueeze(1)  # Nx1x3
        if samples == None:
            estimation_boxes, estimation_cla = self.rpn(search_xyz, fusion_feature, previous_xyz, template_xyz, template_feature, samples=samples)
        else:
            estimation_boxes, estimation_cla, verification_scores = self.rpn(search_xyz, fusion_feature, previous_xyz, template_xyz,
                                                        template_feature, samples=samples)


        # '''
        # written by Jimmy Wu
        # Sep-19
        # vote to a explicit position, previous location
        # '''
        # previous_xyz = input_dict['previous_center'].unsqueeze(1)               # Nx1x3
        #
        # offsets = search_xyz - previous_xyz.repeat(1, search_xyz.size()[1], 1)
        # voted_feature = self.explicit_vote_layer(torch.cat([offsets.transpose(1, 2), search_feature], dim=1))
        # voted_feature = F.max_pool2d(voted_feature, kernel_size=[1, voted_feature.size()[-1]])
        # voted_feature = self.conv_final(voted_feature)
        #
        # # prepare bc
        # # pred_search_bc = self.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # B, 9, N // 8
        # pred_search_bc = self.mlp_bc(torch.cat([previous_xyz.transpose(1, 2), voted_feature], dim=1))  # B, 9, N // 8
        # pred_search_bc = pred_search_bc.transpose(1, 2)
        # sample_idxs_t = sample_idxs_t[:, :M // 8, None]
        # template_bc = template_bc.gather(dim=1, index=sample_idxs_t.repeat(1, 1, self.config.bc_channel).long())
        # # box-aware xcorr
        # # fusion_feature = self.xcorr(template_feature, search_feature, template_xyz, search_xyz, template_bc,
        # #                             pred_search_bc)
        # fusion_feature = self.xcorr(template_feature, voted_feature, template_xyz, previous_xyz, template_bc,
        #                             pred_search_bc)
        # # proposal generation
        # # estimation_boxes, estimation_cla, vote_xyz, center_xyzs = self.rpn(search_xyz, fusion_feature)
        # estimation_boxes = self.rpn(previous_xyz, fusion_feature)
        end_points = {"estimation_boxes": estimation_boxes,
                      "pred_search_bc": pred_search_bc,
                      'estimation_cla': estimation_cla,
                      'sample_idxs': sample_idxs,
                      'verification_scores': verification_scores if samples != None else None
                      }
        return end_points

    def sample_nearby_location(self, center, k_num=32):
        # random sampling around the GT center

        dis_thr = 0.15

        pos_samples = np.zeros((k_num // 2, 3))
        neg_samples = np.zeros((k_num // 2, 3))
        count_pos = 0
        count_neg = 0
        # sample random offsets for pos positions
        while count_pos < (k_num // 2):
            # random_offsets = np.random.uniform(low = -max(np.max(abs(scale)), 1.0), high = max(np.max(abs(scale)), 1.0), size=3) # make sure our offsets are nor too small
            random_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
            dis = np.sqrt(np.sum(random_offsets ** 2))
            if dis < dis_thr:
                pos_samples[count_pos] = random_offsets + center[0]
                count_pos += 1

        while count_neg < (k_num // 2):
            random_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
            dis = np.sqrt(np.sum(random_offsets ** 2))
            if dis > dis_thr:
                neg_samples[count_neg] = random_offsets + center[0]
                count_neg += 1
        samples = np.concatenate((pos_samples, neg_samples),
                                 axis=0)  # num//2 pos, num//2 neg, the last is the previous location
        return samples

    def get_new_position(self, tracklet_bbox, offset):
        rot_quat = Quaternion(matrix=tracklet_bbox.rotation_matrix)
        trans = np.array(tracklet_bbox.center)

        new_box = copy.deepcopy(tracklet_bbox)

        new_box.translate(-trans)
        new_box.rotate(rot_quat.inverse)

        new_box.rotate(
            Quaternion(axis=[0, 0, 1], degrees=offset[2]))
        new_box.translate(np.array([offset[0], offset[1], offset[2]]))
        new_box.rotate(rot_quat)
        new_box.translate(trans)
        return new_box.center



    def training_step(self, batch, batch_idx):
        """
        {"estimation_boxes": estimation_boxs.transpose(1, 2).contiguous(),
                  "vote_center": vote_xyz,
                  "pred_seg_score": estimation_cla,
                  "center_xyz": center_xyzs,
                  "seed_idxs":
                  "seg_label"
                  "pred_search_bc": pred_search_bc
        }
        """
        lstm_input = batch['tracklet_xyz'][(batch['flag'] == 1).squeeze()]
        lstm_prediction = lstm(lstm_input.float()).detach()
        if (batch['flag'] == 0).sum() > 0:
            constant_vel_input = batch['tracklet_xyz'][(batch['flag'] == 0).squeeze()]
            constant_vel_input = constant_vel_input[:,0:2,:]
            pre_location = constant_vel_input[:,0,:]
            current_location = constant_vel_input[:,1,:]
            cvm_prediction = current_location - pre_location + current_location
            sampled_locations = torch.cat((lstm_prediction, cvm_prediction), dim=0)
        else:
            sampled_locations = lstm_prediction

        # convert the xyz
        converted_locations = []
        search_bboxes = []
        for j in range(sampled_locations.size()[0]):
            center = batch['tracklet_ref_bbox_center'][j].cpu().data.numpy().tolist()
            bbox_size = batch['tracklet_ref_wlh'][j].cpu().data.numpy().tolist()
            orientation = Quaternion(
                axis=[0, 0, -1], radians=batch['tracklet_ref_bbox_rotation_y'][j].item()) * Quaternion(axis=[0, 0, -1], degrees=90)
            tracklet_bbox = Box(center, bbox_size, orientation)
            # converted_center = self.get_new_position(tracklet_bbox, sampled_locations[j].cpu().data.numpy())

            # points_utils.getOffsetBB(tracklet_bbox, estimation_box_cpu, degrees=self.config.degrees,
            #                          use_z=self.config.use_z,
            #                          limit_box=self.config.limit_box)

            converted_bb = points_utils.getOffsetBB(tracklet_bbox, sampled_locations[j].cpu().data.numpy(), limit_box=False,
                                                 degrees=True, use_z=True)
            converted_center = converted_bb.center

            search_center = batch['search_bbox_center'][j].cpu().data.numpy().tolist()
            search_bbox_size = batch['search_bbox_wlh'][j].cpu().data.numpy().tolist()
            search_orientation = Quaternion(
                axis=[0, 0, -1], radians=batch['search_bbox_rotation_y'][j].item()) * Quaternion(axis=[0, 0, -1], degrees=90)
            search_bbox = Box(search_center, search_bbox_size, search_orientation)
            sample_bb = points_utils.getOffsetBB(search_bbox, batch['sample_offset'][j].cpu().data.numpy(), limit_box=False,
                                                 degrees=True)
            converted_location = points_utils.generate_single_pc(converted_center.reshape(3, 1), sample_bb).points.reshape(1, 3)
            converted_locations.append(converted_location)
            gt_location = np.array(batch['box_label'][j][0:3].cpu().data.numpy()).reshape(1, 3)
            dis = np.mean((converted_location - gt_location)**2)
            search_bboxes.append(points_utils.transform_box(search_bbox, sample_bb))
        sampled_locations = torch.from_numpy(np.array(converted_locations)).squeeze().cuda()

            # center = [anno["x"], anno["y"] - anno["height"] / 2, anno["z"]]
            # size = [anno["width"], anno["length"], anno["height"]]
            # orientation = Quaternion(
            #     axis=[0, 1, 0], radians=anno["rotation_y"]) * Quaternion(
            #     axis=[1, 0, 0], radians=np.pi / 2)
            # bb = Box(center, size, orientation)

        for i in range(sampled_locations.size()[0]): # loop for batch
           if i == 0:
                sampled_samples = torch.from_numpy(self.sample_nearby_location(sampled_locations[i].unsqueeze(0).cpu().data.numpy(), k_num=32)).cuda().unsqueeze(0)
           else:
                sampled_samples = torch.cat((sampled_samples, torch.from_numpy(self.sample_nearby_location(sampled_locations[i].unsqueeze(0).cpu().data.numpy(), k_num=32)).cuda().unsqueeze(0)), dim=0)

        batch['samples'] = torch.cat((batch['samples'], sampled_samples), dim=1)
        end_points = self(batch, search_bboxes=search_bboxes)

        search_pc = batch['points2cc_dist_s']
        estimation_cla = end_points['estimation_cla']  # B,N
        N = estimation_cla.shape[1]
        seg_label = batch['seg_label']
        sample_idxs = end_points['sample_idxs']  # B,N
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        search_pc = search_pc.gather(dim=1, index=sample_idxs[:, :N, None].repeat(1, 1, self.config.bc_channel).long())
        # update label
        batch['seg_label'] = seg_label
        batch['points2cc_dist_s'] = search_pc
        # compute loss
        loss_dict = self.compute_loss(batch, end_points)

        loss =  loss_dict['loss_box'] * self.config.box_weight \
               + loss_dict['loss_seg'] * self.config.seg_weight \
               + loss_dict['loss_bc'] * self.config.bc_weight

        #+ loss_dict['loss_verification'] * self.config.verification_weight

        #loss_dict['loss_objective'] * self.config.objectiveness_weight \
               # + \
        #                + loss_dict['loss_verification'] * self.config.verification_weight
        # \
        # + loss_dict['loss_verification'] * self.config.verification_weight

        # log
        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_box/train', loss_dict['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        # self.log('loss_ver/train', loss_dict['loss_verification'].item(), on_step=True, on_epoch=True, prog_bar=True,
        #          logger=False)
        # self.log('loss_vote/train', loss_dict['loss_vote'].item(), on_step=True, on_epoch=True, prog_bar=True,
        #          logger=False)
        self.log('loss_bc/train', loss_dict['loss_bc'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        # self.log('loss_objective/train', loss_dict['loss_objective'].item(), on_step=True, on_epoch=True, prog_bar=True,
        #          logger=False)
        # ,
        # 'loss_ver': loss_dict['loss_verification'].item()

        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_box': loss_dict['loss_box'].item(),
                                                    'loss_bc': loss_dict['loss_bc'].item(),
                                                    'loss_seg': loss_dict['loss_seg'].item()},
                                           global_step=self.global_step)
        #      'loss_objective': loss_dict['loss_objective'].item()
        # ,
        #                                                     'loss_ver': loss_dict['loss_verification'].item()

        return loss
