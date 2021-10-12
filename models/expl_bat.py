""" 
bat.py
Created by zenn at 2021/7/21 14:16
"""

import torch
from torch import nn
from models.backbone.pointnet import Pointnet_Backbone
from models.head.xcorr import BoxAwareXCorr
from models.head.expl_rpn import P2BVoteNetRPN
from models import base_model
import torch.nn.functional as F
from datasets import points_utils
from pointnet2.utils import pytorch_utils as pt_utils


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

        # follow this vote net for prediction
        self.explicit_vote_layer = (
            pt_utils.Seq(3 + self.config.feature_channel)
                .conv1d(self.config.feature_channel, bn=True)
                .conv1d(self.config.feature_channel, bn=True)
                .conv1d(self.config.feature_channel, activation=None))


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


    def forward(self, input_dict):
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

        if template.shape[0]>1:
            dist = input_dict['dist']

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

        end_points = self(batch)

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
