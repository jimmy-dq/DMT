import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class ts_up_sampling(nn.Module):
    def __init__(self,input_c=128,mid_c=64):
        super(ts_up_sampling, self).__init__()
        self.input_c=input_c
        self.duplate=(pt_utils.Seq(input_c)
                .conv1d(2*input_c, bn=True))
        self.c1=(pt_utils.Seq(input_c)
                .conv1d(mid_c, bn=True))
        self.c2=pt_utils.SharedMLP([input_c,mid_c,mid_c,mid_c], bn=True)
        self.c3 = (pt_utils.Seq(mid_c)
                   .conv1d(input_c, bn=True)
                   .conv1d(input_c, bn=True))
        self.c4=(pt_utils.Seq(input_c+mid_c)
                .conv1d(2*input_c)
                .conv1d(3, activation=None))
        
    def forward(self,search_keypoint):
        B=search_keypoint.size(0)
        N=search_keypoint.size(2)

        up_sampling_fea=self.duplate(search_keypoint)
        print ('up_sampling_fea', up_sampling_fea.size())
        up_sampling_fea=up_sampling_fea.view(B,self.input_c,int(2048/N),N)
        up_sampling_fea=up_sampling_fea.view(B,self.input_c,-1)
        up_sampling_fea=self.c1(up_sampling_fea)
        #b c 1
        global_fea=F.max_pool1d(up_sampling_fea,kernel_size=2048)
        #b 2c N K
        up_sampling_fea=get_graph_feature(up_sampling_fea,k=4)
        up_sampling_fea=self.c2(up_sampling_fea)
        up_sampling_fea=F.max_pool2d(up_sampling_fea, kernel_size=[1, up_sampling_fea.size(3)])

        global_fea=self.c3(global_fea)
        #b c N
        local_fea=up_sampling_fea.squeeze(-1)
        final_fea=torch.cat((local_fea,global_fea.expand(-1,-1,local_fea.size(2))),dim=1)
        points_coord=self.c4(final_fea)
        return points_coord

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
            
        #shared mlp1:1d conv
        self.shared_mlp1 = nn.Sequential()
        self.shared_mlp1.add_module('mlp0', nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 1))
        self.shared_mlp1.add_module('relu0', nn.ReLU(inplace = True))
        self.shared_mlp1.add_module('mlp1', nn.Conv1d(in_channels = 512, out_channels = 1024, kernel_size = 1))


    def forward(self, x):
        feature_f = x #b*256*n
        global_feature_g, _ = torch.max(feature_f, -1, keepdim = True) #b*256*1
        expanded_global_feature_g = global_feature_g.repeat(1, 1, feature_f.size(2)) #b*256*n
        concated_feature = torch.cat([expanded_global_feature_g, feature_f], 1) #b*512*n
        global_feature_v = self.shared_mlp1(concated_feature) #b*1024*n
        global_feature_v, _ = torch.max(global_feature_v, -1, keepdim = True) #b*1024*1
        del _
        global_feature_v = global_feature_v.view(global_feature_v.size(0), -1) #b*1024
        return global_feature_v

class Decoder(nn.Module):
    def __init__(self, num_coarse = 1024):
        super(Decoder, self).__init__()
        self.num_coarse = num_coarse
        #mlp0
        self.mlp = nn.Sequential()
        self.mlp.add_module('mlp0', nn.Linear(in_features = 1024, out_features = 1024, bias = True))
        self.mlp.add_module('relu0', nn.ReLU(inplace = True))
        self.mlp.add_module('mlp1', nn.Linear(in_features = 1024, out_features = 1024, bias = True))
        self.mlp.add_module('relu1', nn.ReLU(inplace = True))
        self.mlp.add_module('mlp2', nn.Linear(in_features = 1024, out_features = num_coarse * 3 , bias = True))
        
      

    def forward(self, feature):
        coarse = self.mlp(feature) #b * (3 * coarse_size)
        coarse = coarse.view(coarse.size(0), 3, -1) #B * 3 * coarse_size
        coarse = coarse.permute(0, 2, 1) #B*coarse_size*3
        return coarse

class PCN(nn.Module):
    def __init__(self):
        super(PCN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))