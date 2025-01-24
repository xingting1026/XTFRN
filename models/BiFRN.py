import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np
from .backbones import Conv_4,ResNet
from .backbones.FSRM import FSRM
from .backbones.FMRM import FMRM
from networks import resnet_big

class BiFRN(nn.Module):
    
    def __init__(self, way=None, shots=None, resnet=False):
        super().__init__()
        
        self.resolution = 5*5
        if resnet:
            self.num_channel = 640
            self.feature_extractor = ResNet.resnet12()
            self.dim = self.num_channel*5*5
        else:
            self.num_channel = 64
            self.feature_extractor = Conv_4.BackBone(self.num_channel)
            self.dim = self.num_channel*5*5

        self.fsrm = FSRM(
                sequence_length=self.resolution,
                embedding_dim=self.num_channel,
                num_layers=1,
                num_heads=1,
                mlp_dropout_rate=0.,
                attention_dropout=0.,
                positional_embedding='sine')

        self.fmrm = FMRM(
            hidden_size=self.num_channel, 
            inner_size=self.num_channel, 
            num_patch=self.resolution, 
            drop_prob=0.1
        )

        proj_hidden = self.dim  
        proj_out = 64         
        self.proj_head = nn.Sequential(
            nn.Linear(self.dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden, proj_out),
            nn.BatchNorm1d(proj_out)
        )

        self.shots = shots
        self.way = way
        self.resnet = resnet

        # Learnable parameters
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.current_contra_weight = 0.5  # default value
        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
            
    def get_feature_vector(self, inp):
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        fsrm_features = self.fsrm(feature_map)
        
        contra_features = fsrm_features.clone()
        contra_features = contra_features.view(batch_size, -1)  # [B, C*H*W] -> [600, 1600]
        contra_features = self.proj_head(contra_features) # [600, 1600] -> [600, 64]
        contra_features = F.normalize(contra_features, dim=1) # [600, 64]
        
        feature_map = fsrm_features.transpose(1, 2).view(batch_size, self.num_channel, 5, 5)
        
        return feature_map, contra_features
    
    def get_neg_l2_dist(self, inp, way, shot, query_shot):
        feature_map, _ = self.get_feature_vector(inp)
        support = feature_map[:way*shot].view(way, shot, *feature_map.size()[1:])
        support = support.permute(0, 2, 1, 3, 4).contiguous()
        query = feature_map[way*shot:]
        
        sq_similarity, qs_similarity = self.fmrm(support, query)
        l2_dist = self.w1*sq_similarity + self.w2*qs_similarity
        
        return l2_dist

    def meta_test(self, inp, way, shot, query_shot):
        neg_l2_dist = self.get_neg_l2_dist(
            inp=inp,
            way=way,
            shot=shot,
            query_shot=query_shot
        )
        _, max_index = torch.max(neg_l2_dist, 1)
        return max_index

    def forward(self, inp):
        feature_map, contra_features = self.get_feature_vector(inp)
        
        # 分離support和query的特徵
        support_features = contra_features[:self.way*self.shots[0]]  
        query_features = contra_features[self.way*self.shots[0]:]
        
        # 構建對比學習的特徵矩陣
        contrast_features = torch.cat([support_features, query_features], dim=0)
        
        logits = self.get_neg_l2_dist(
            inp=inp,
            way=self.way,
            shot=self.shots[0],
            query_shot=self.shots[1]
        )
        logits = logits/self.dim*self.scale
        
        log_prediction = F.log_softmax(logits, dim=1)
        return log_prediction, contrast_features