import torch
from torch import nn
from models.backbone.pointnet import Pointnet_Backbone
from models.backbone.completion import PCN
# from models.backbone.pointnet_transformer import PointTransformer
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
from typing import Optional, List
from torch import Tensor
from copy import deepcopy
from extensions.chamfer_distance.chamfer_distance import ChamferDistance

CD = ChamferDistance()

def l2_cd(pcs1, pcs2):
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.mean(dist1, dim=1)
    dist2 = torch.mean(dist2, dim=1)
    return torch.sum(dist1 + dist2)


def l1_cd(pcs1, pcs2):
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return torch.sum(dist1 + dist2) / 2

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
        
class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        bs, n, c = src.shape
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=None, pos=pos_embed)
        # hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
        #                   pos=pos_embed, query_pos=query_embed)
        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, n)
        return memory.permute(1, 2, 0).view(bs, c, n)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


# class TransformerDecoder(nn.Module):

#     def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#         self.return_intermediate = return_intermediate

#     def forward(self, tgt, memory,
#                 tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None):
#         output = tgt

#         intermediate = []

#         for layer in self.layers:
#             output = layer(output, memory, tgt_mask=tgt_mask,
#                            memory_mask=memory_mask,
#                            tgt_key_padding_mask=tgt_key_padding_mask,
#                            memory_key_padding_mask=memory_key_padding_mask,
#                            pos=pos, query_pos=query_pos)
#             if self.return_intermediate:
#                 intermediate.append(self.norm(output))

#         if self.norm is not None:
#             output = self.norm(output)
#             if self.return_intermediate:
#                 intermediate.pop()
#                 intermediate.append(output)

#         if self.return_intermediate:
#             return torch.stack(intermediate)

#         return output.unsqueeze(0)        

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def attention(query, key,  value):
    dim = query.shape[1]
    scores_1 = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    scores_2 = torch.einsum('abcd, aced->abcd', key, scores_1)
    prob = torch.nn.functional.softmax(scores_2, dim=-1)
    output = torch.einsum('bnhm,bdhm->bdhn', prob, value)
    return output, prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.down_mlp = MLP(input_dim = self.dim, hidden_dim = 32, output_dim = 1, num_layers = 1)


    def forward(self, query, key, value):
        batch_dim = query.size(0)
        # pdb.set_trace()
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        x = self.down_mlp(x)
        return x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)


# class TransformerDecoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn = MultiHeadedAttention(nhead, d_model)

#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before


#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward_post(self, tgt, memory,
#                      tgt_mask: Optional[Tensor] = None,
#                      memory_mask: Optional[Tensor] = None,
#                      tgt_key_padding_mask: Optional[Tensor] = None,
#                      memory_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None,
#                      query_pos: Optional[Tensor] = None):

#         q = k = self.with_pos_embed(tgt, query_pos)
#         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
#                             key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos).permute(1,2,0),
#                                 key=self.with_pos_embed(memory, pos).permute(1,2,0),
#                                 value=memory.permute(1,2,0))
#         tgt2 = tgt2.permute(2,0,1)
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt

#     def forward_pre(self, tgt, memory,
#                     tgt_mask: Optional[Tensor] = None,
#                     memory_mask: Optional[Tensor] = None,
#                     tgt_key_padding_mask: Optional[Tensor] = None,
#                     memory_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None,
#                     query_pos: Optional[Tensor] = None):
#         tgt2 = self.norm1(tgt)
#         q = k = self.with_pos_embed(tgt2, query_pos)
#         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt2 = self.norm2(tgt)
#         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt2 = self.norm3(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
#         tgt = tgt + self.dropout3(tgt2)
#         return tgt

#     def forward(self, tgt, memory,
#                 tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None):
#         if self.normalize_before:
#             return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
#                                     tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
#         return self.forward_post(tgt, memory, tgt_mask, memory_mask,
#                                  tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer():
    return Transformer(
        d_model=256,
        dropout=0.1,
        nhead=4,
        dim_feedforward=512,
        num_encoder_layers=3,
        num_decoder_layers=1,
        normalize_before=False,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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
        # self.query_embed = nn.Embedding(1, 256)
        # self.transformer = build_transformer()
        self.completion_fc = PCN()
        # self.cd_loss = l1_cd().to(self.device)

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
        estimate_completion_pc = output['estimate_completion_points'] 
        # print ('estimate_completion_pc', estimate_completion_pc.size())
        # print ('completion_points', data['completion_points'].size())

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

        loss_box = torch.mean(loss_box.mean(2))

        search_bc = data['points2cc_dist_s']
        pred_search_bc = output['pred_search_bc']
        loss_bc = F.smooth_l1_loss(pred_search_bc, search_bc, reduction='none')
        loss_bc = torch.sum(loss_bc.mean(2) * seg_label) / (seg_label.sum() + 1e-6)

        loss_cd = l1_cd(estimate_completion_pc, data['completion_points'])
        return {
                "loss_seg": loss_seg,
                "loss_bc": loss_bc,
                "loss_box": loss_box,
                'loss_cd': loss_cd
               }

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
        # completion_PC = input_dict['completion_points'] # batchx1024x3
        M = template.shape[1]
        N = search.shape[1]

        samples = input_dict['samples'] #training: batchx8x3
    
        template_xyz, template_feature, sample_idxs_t = self.backbone(template, [M // 2, M // 4, M // 8])
        # print ('template_xyz', template_xyz.size())
        # print ('template_feature', template_feature.size())
        # print ('sample_idxs_t', sample_idxs_t.size())
        search_xyz, search_feature, sample_idxs = self.backbone(search, [N // 2, N // 4, N // 8])
        template_feature = self.conv_final(template_feature) # batchxDxNum.

        #completion network
        estimate_completion_points = self.completion_fc(template_feature)
        # print ('completion_points', completion_points.size())
        # completion_points = completion_points.unsqueeze(-1)

        search_feature = self.conv_final(search_feature)

        # prepare bc
        pred_search_bc = self.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # B, 9, N // 8
        pred_search_bc = pred_search_bc.transpose(1, 2) #BxNx9
        sample_idxs_t = sample_idxs_t[:, :M // 8, None]
        template_bc = template_bc.gather(dim=1, index=sample_idxs_t.repeat(1, 1, self.config.bc_channel).long())
        # box-aware xcorr
        fusion_feature = self.xcorr(template_feature, search_feature, template_xyz, search_xyz, template_bc,
                                    pred_search_bc) ### 1, 256, 128
        previous_xyz = input_dict['previous_center'].unsqueeze(1)  # Nx1x3
        if samples == None:
            estimation_boxes, estimation_cla = self.rpn(search_xyz, fusion_feature, previous_xyz, template_xyz, template_feature, samples=samples)
        else:
            estimation_boxes, estimation_cla, verification_scores = self.rpn(search_xyz, fusion_feature, previous_xyz, template_xyz,
                                                        template_feature, samples=samples)

        end_points = {"estimation_boxes": estimation_boxes,
                      "pred_search_bc": pred_search_bc,
                      'estimation_cla': estimation_cla,
                      'sample_idxs': sample_idxs,
                      'estimate_completion_points': estimate_completion_points,
                      'verification_scores': verification_scores if samples != None else None
                      }
        return end_points

    def sample_nearby_location(self, center, k_num=32):
        # random sampling around the GT center

        dis_thr = 0.15

        # pos_samples = np.zeros((k_num // 2, 3))
        pos_samples = np.zeros((k_num, 3))
        neg_samples = np.zeros((k_num // 2, 3))
        count_pos = 0
        count_neg = 0
        # sample random offsets for pos positions
        while count_pos < (k_num):
            # random_offsets = np.random.uniform(low = -max(np.max(abs(scale)), 1.0), high = max(np.max(abs(scale)), 1.0), size=3) # make sure our offsets are nor too small
            random_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
            dis = np.sqrt(np.sum(random_offsets ** 2))
            if dis < dis_thr:
                pos_samples[count_pos] = random_offsets + center[0]
                count_pos += 1
        return pos_samples

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
               + loss_dict['loss_bc'] * self.config.bc_weight \
               + loss_dict['loss_cd']

        # log
        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_box/train', loss_dict['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_bc/train', loss_dict['loss_bc'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_bc/train', loss_dict['loss_cd'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_box': loss_dict['loss_box'].item(),
                                                    'loss_bc': loss_dict['loss_bc'].item(),
                                                    'loss_seg': loss_dict['loss_seg'].item(),
                                                    'loss_cd': loss_dict['loss_cd'].item()},
                                           global_step=self.global_step)

        return loss
