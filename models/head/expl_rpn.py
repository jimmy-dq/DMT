""" 
rpn.py
Created by zenn at 2021/5/8 20:55
"""
import torch
from torch import nn
from pointnet2.utils import pytorch_utils as pt_utils
import torch.nn.functional as F

from pointnet2.utils.pointnet2_modules import PointnetSAModule


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1, 128)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        # print ('x', x.size())
        output = torch.mean(x, dim=1)
        # output = self.avg_pool(x)
        # print ('output', output.size())
        y = self.conv(output.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        # print ('y', y.size())
        y = self.sigmoid(y)
        # print ('y', y.size())
        y = y.unsqueeze(1)
        # print ('y', y.size())
        y = y.expand_as(x)
        # print ('y', y.size())
        y = x * y
        # exit()
        return y

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c

        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class P2BVoteNetRPN(nn.Module):

    def __init__(self, feature_channel, vote_channel=256, num_proposal=64, normalize_xyz=False):
        super().__init__()
        # self.num_proposal = num_proposal
        self.FC_layer_cla = (
            pt_utils.Seq(feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, bn=True)
                .conv1d(1, activation=None))
        self.FC_layer_verification = (
            pt_utils.Seq(feature_channel+feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, bn=True)
                .conv1d(1, activation=None))
        # self.vote_layer = (
        #     pt_utils.Seq(3 + feature_channel)
        #         .conv1d(feature_channel, bn=True)
        #         .conv1d(feature_channel, bn=True)
        #         .conv1d(3 + feature_channel, activation=None))
        #
        # self.vote_aggregation = PointnetSAModule(
        #     radius=0.3,
        #     nsample=16,
        #     mlp=[1 + feature_channel, vote_channel, vote_channel, vote_channel],
        #     use_xyz=True,
        #     normalize_xyz=normalize_xyz)

        self.conv_final = nn.Conv1d(256, feature_channel, kernel_size=1)

        self.explicit_vote_layer = (
            pt_utils.Seq(3 + feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, activation=None))

        self.FC_proposal = (
            pt_utils.Seq(feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(feature_channel, bn=True)
                .conv1d(3 + 1, activation=None))

        self.eca = eca_layer()
        self.sa_layer = SA_Layer(feature_channel)

    def forward(self, search_xyz, search_feature, previous_xyz, template_xyz, template_feature, samples = None):
        """

        :param xyz: B,N,3
        :param feature: B,f,N
        :return: B,N,4+1 (xyz,theta,targetnessscore)
        """
        # classify the search seeds on the surface as positive, otherwise negative
        estimation_cla = self.FC_layer_cla(search_feature).squeeze(1)
        # score = estimation_cla.sigmoid()

        # # generate center point template features
        # offsets_template = template_xyz
        # offset_template_features = torch.cat((offsets_template.transpose(1, 2).contiguous(), template_feature), dim=1)
        # voted_template_feature = self.explicit_vote_layer(offset_template_features)
        # voted_template_feature = F.max_pool2d(voted_template_feature, kernel_size=[1, voted_template_feature.size()[-1]])
        # template_center_feature  = self.conv_final(voted_template_feature) # Nx256x1

        # we have multiple point proposals
        # contrsuct one batch Nxfeat_dimx128
        if samples != None:
            n_samples_per_epoch = samples.size()[1]
            for i in range(n_samples_per_epoch):
                offsets = search_xyz - samples[:,i,:].unsqueeze(1).repeat(1, search_xyz.size()[1], 1)
                offset_features = torch.cat((offsets.transpose(1, 2).contiguous(), search_feature), dim=1)
                voted_feature = self.explicit_vote_layer(offset_features)
                # attention
                voted_feature = self.eca(voted_feature)
                voted_feature = self.sa_layer(voted_feature)
                voted_feature = F.max_pool2d(voted_feature, kernel_size=[1, voted_feature.size()[-1]])
                voted_feature = self.conv_final(voted_feature)
                proposal_offsets = self.FC_proposal(voted_feature)
                if i == 0:
                    # verification_scores = self.FC_layer_verification(torch.cat((voted_feature, template_center_feature), dim=1))
                    estimation_boxes = torch.cat(
                        (proposal_offsets[:, 0:3, :] + samples[:,i,:].unsqueeze(1).transpose(1,2).contiguous(),
                         proposal_offsets[:, 3:4, :]),
                        dim=1).transpose(1, 2).contiguous()
                else:
                    # verification_scores = torch.cat((verification_scores, self.FC_layer_verification(torch.cat((voted_feature, template_center_feature), dim=1))), dim=0)
                    estimation_boxes = torch.cat((estimation_boxes, torch.cat(
                        (proposal_offsets[:, 0:3, :] + samples[:,i,:].unsqueeze(1).transpose(1,2).contiguous(),
                         proposal_offsets[:, 3:4, :]),
                        dim=1).transpose(1, 2).contiguous()), dim=0)
            # # we need to also include the previous center
            offsets_previous = search_xyz - previous_xyz.repeat(1, search_xyz.size()[1], 1)
            offset_features_previous = torch.cat((offsets_previous.transpose(1, 2).contiguous(), search_feature), dim=1)
            voted_feature_previous = self.explicit_vote_layer(offset_features_previous)
            # attention
            voted_feature_previous = self.eca(voted_feature_previous)
            voted_feature_previous = self.sa_layer(voted_feature_previous)
            voted_feature_previous = F.max_pool2d(voted_feature_previous, kernel_size=[1, voted_feature_previous.size()[-1]])
            voted_feature_previous = self.conv_final(voted_feature_previous)

            proposal_offsets_previous = self.FC_proposal(voted_feature_previous)
            estimation_boxes_previous = torch.cat(
                (proposal_offsets_previous[:, 0:3, :] + previous_xyz.transpose(1, 2).contiguous(), proposal_offsets_previous[:, 3:4, :]),
                dim=1).transpose(1, 2).contiguous()
            estimation_boxes = torch.cat((estimation_boxes, estimation_boxes_previous), dim=0)
            return estimation_boxes, estimation_cla, None #verification_scores




                # estimation_boxes = estimation_boxes

        offsets = search_xyz - previous_xyz.repeat(1, search_xyz.size()[1], 1) # search_xyz: Nx128x3 # previous_xyz: NxN2x128x3 offsets: NxN2x128x3
        # indices = torch.argsort(torch.sum(offsets ** 2, dim=2), descending=False, dim=1)[:, 0:32]   #BxK(4)
        #
        # for i in range(indices.size()[0]):
        #     if i == 0:
        #         new_offsets = offsets[i, indices[i]].unsqueeze(0)
        #         new_search_feature = search_feature[i, :, indices[i]].unsqueeze(0)
        #     else:
        #         new_offsets = torch.cat((new_offsets, offsets[i, indices[i]].unsqueeze(0)), dim=0)
        #         new_search_feature = torch.cat((new_search_feature, search_feature[i, :, indices[i]].unsqueeze(0)), dim=0)

        offset_features = torch.cat((offsets.transpose(1, 2).contiguous(), search_feature), dim=1) #search_feature: Nx256x128 fang zai batch_size weidu
        # offset_features = torch.cat((new_offsets.transpose(1, 2).contiguous(), new_search_feature), dim=1)

        voted_feature = self.explicit_vote_layer(offset_features)
        # attention
        voted_feature = self.eca(voted_feature)
        voted_feature = self.sa_layer(voted_feature)
        voted_feature = F.max_pool2d(voted_feature, kernel_size=[1, voted_feature.size()[-1]])
        # voted_feature = F.avg_pool2d(voted_feature, kernel_size=[1, voted_feature.size()[-1]])
        voted_feature = self.conv_final(voted_feature)

        proposal_offsets = self.FC_proposal(voted_feature)
        estimation_boxes = torch.cat(
            (proposal_offsets[:, 0:3, :] + previous_xyz.transpose(1, 2).contiguous(), proposal_offsets[:, 3:4, :]),
            dim=1)

        # estimation_boxes = torch.cat(
        #     (proposal_offsets[:, 0:3, :] + previous_xyz.transpose(1, 2).contiguous(), proposal_offsets[:, 3:5, :]),
        #     dim=1)

        # classification model
        # final_offsets = search_xyz - estimation_boxes.cpu().numpy()[:, 0:3, :].transpose((1, 2)).repeat(1, search_xyz.size()[1], 1)
        # final_offset_features = torch.cat((final_offsets.transpose(1, 2).contiguous(), score.unsqueeze(1), search_feature), dim=1)
        # final_voted_feature = self.explicit_vote_layer(final_offset_features)
        # center_scores = self.FC_layer_cla_final(final_voted_feature)


        estimation_boxes = estimation_boxes.transpose(1, 2).contiguous()
        return estimation_boxes, estimation_cla
