import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
affine_par = True
import sys
import units
from .ConvGRU import ConvGRUCell
import time
import math
import cv2
import numpy as np
import os
import torch.utils.model_zoo as model_zoo
from .util import remove_layer
from .util import initialize_weights

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class GDL(nn.Module):
    def __init__(self, drop_rate=0.8, drop_th=0.7):
        super(GDL, self).__init__()
        if not (0 <= drop_rate <= 1):
            raise ValueError("drop-rate must be in range [0, 1].")
        if not (0 <= drop_th <= 1):
            raise ValueError("drop-th must be in range [0, 1].")
        self.drop_rate = drop_rate
        self.drop_th = drop_th
        self.attention = None
        self.drop_mask = None

    def forward(self, input_):
        attention = torch.mean(input_, dim=1, keepdim=True)
        importance_map = torch.sigmoid(attention)
        drop_mask = self._drop_mask(attention)
        selected_map = self._select_map(importance_map, drop_mask)
        return (input_.mul(selected_map) + input_) / 2

    def _select_map(self, importance_map, drop_mask):
        random_tensor = torch.rand([], dtype=torch.float32) + self.drop_rate
        binary_tensor = random_tensor.floor()
        return (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

    def _drop_mask(self, attention):
        b_size = attention.size(0)
        max_val, _ = torch.max(attention.view(b_size, -1), dim=1, keepdim=True)
        thr_val = max_val * self.drop_th
        thr_val = thr_val.view(b_size, 1, 1, 1)
        return (attention < thr_val).float()

    def extra_repr(self):
        return 'drop_rate={}, drop_th={}'.format(
            self.drop_rate, self.drop_th)

class CoattentionModel(nn.Module):
    def  __init__(self, features, num_classes, all_channel=20, att_dir='./runs/', training_epoch=10,**kwargs):
        super(CoattentionModel, self).__init__()
        self.features = features
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,20,1)           
        )
        self.extra_linear_e = nn.Linear(all_channel, all_channel,bias = False)
        self.channel = all_channel
        self.extra_gate = nn.Conv2d(all_channel, 1, kernel_size = 1, bias = False)
        self.extra_gate_s = nn.Sigmoid()
        self.extra_ConvGRU = ConvGRUCell(all_channel, all_channel, kernel_size=1)
        self.extra_conv_fusion = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias= True)
        self.extra_relu_fusion = nn.ReLU(inplace=True)
        self.extra_GDL = GDL(kwargs['drop_rate'], kwargs['drop_th'])
        self.softmax = nn.Sigmoid()
        self.propagate_layers = 3

        d = self.channel // 2
        self.extra_proja = nn.Conv2d(self.channel, d, kernel_size=1)
        self.extra_projb = nn.Conv2d(self.channel, d, kernel_size=1)

        self.training_epoch = training_epoch
        self.att_dir = att_dir
        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()     
    		
    def forward(self, input1, input2, input3,epoch=1, label=None, index=None):
        
        batch_num  = input1.size()[0]
       
        x1 = self.features(input1)
        x1 = self.extra_convs(x1)
        self.map = x1.clone()
        x11 = x1.clone()
        x1ss = F.avg_pool2d(x11, kernel_size=(x11.size(2), x11.size(3)), padding=0)
        x1ss = x1ss.view(-1, 20)

        x2 = self.features(input2)
        x2 = self.extra_convs(x2)
        x22 = x2.clone()
        x2ss = F.avg_pool2d(x22, kernel_size=(x22.size(2), x22.size(3)), padding=0)
        x2ss = x2ss.view(-1, 20)

        x3 = self.features(input3)
        x3 = self.extra_convs(x3)
        x33 = x3.clone()
        x3ss = F.avg_pool2d(x33,kernel_size=(x33.size(2),x33.size(3)),padding=0)
        x3ss = x3ss.view(-1, 20)
        
        for ii in range(1):
            exemplar = x1[ii,:,:,:][None].contiguous().clone()
            query = x2[ii, :, :, :][None].contiguous().clone()
            query1 = x3[ii,:,:,:][None].contiguous().clone()
            for passing_round in range(self.propagate_layers):
                attention1 = self.extra_conv_fusion(torch.cat([self.generate_attention(exemplar, query),
                                         self.generate_attention(exemplar, query1)],1)) 
                attention2 = self.extra_conv_fusion(torch.cat([self.generate_attention(query, exemplar),
                                        self.generate_attention(query, query1)],1))
                attention3 = self.extra_conv_fusion(torch.cat([self.generate_attention(query1, exemplar),
                                        self.generate_attention(query1, query)],1))
                
                h_v1 = self.extra_ConvGRU(attention1, exemplar)
                h_v2 = self.extra_ConvGRU(attention2, query)
                h_v3 = self.extra_ConvGRU(attention3, query1)

                h_v1 = self.extra_GDL(h_v1)
                h_v2 = self.extra_GDL(h_v2)
                h_v3 = self.extra_GDL(h_v3)
                
                exemplar = h_v1.clone()
                query = h_v2.clone()
                query1 = h_v3.clone()

                if passing_round == self.propagate_layers -1:
                    
                    self.map1 = exemplar.clone()
                    x1s = F.avg_pool2d(exemplar, kernel_size=(exemplar.size(2), exemplar.size(3)), padding=0)
                    x1s = x1s.view(-1, 20)

                    x2s = F.avg_pool2d(query, kernel_size=(query.size(2), query.size(3)), padding=0)
                    x2s = x2s.view(-1, 20)

                    x3s = F.avg_pool2d(query1, kernel_size=(query1.size(2), query1.size(3)), padding=0)
                    x3s = x3s.view(-1, 20)

                    pre_probs = x1s.clone() 
                    probs = torch.sigmoid(pre_probs)

                    if index != None and epoch > 0:
                        atts = (self.map1 + self.map) / 2
                        atts[atts < 0] = 0
                        ind = torch.nonzero(label)

                        for i in range(ind.shape[0]):
                
                            batch_index, la = ind[i]
                            accu_map_name = '{}/{}_{}.png'.format(self.att_dir, batch_index+index, la)
                            att = atts[0, la].cpu().data.numpy()
                            att = np.rint(att / (att.max()  + 1e-8) * 255)

                            if epoch == self.training_epoch - 1 and not os.path.exists(accu_map_name):
                                cv2.imwrite(accu_map_name, att)
                                continue

                            if probs[0, la] < 0.1:  
                                continue
                            
                            try:
                                if not os.path.exists(accu_map_name):
                                    cv2.imwrite(accu_map_name, att)
                                else:
                                    accu_at = cv2.imread(accu_map_name, 0)
                                    accu_at_max = np.maximum(accu_at, att)
                                    cv2.imwrite(accu_map_name,  accu_at_max)
                            except Exception as e:
                                print(e)
       
        return x1ss,x1s, x2ss,x2s, x3ss,x3s

    def message_fun(self,input):
        input1 = self.extra_conv_fusion(input)
        input1 = self.extra_relu_fusion(input1)
        return input1

    def generate_attention(self, exemplar, query):
        fea_size = query.size()[2:]
        N1, C1, H1, W1 = exemplar.shape
        exemplar_low = self.extra_proja(exemplar)
        query_low = self.extra_projb(query)
        N,C,H,W = exemplar_low.shape

        exemplar_flat = exemplar_low.view(N, C, H*W)
        query_flat = query_low.view(N, C, H*W)
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()

        A = torch.bmm(exemplar_t, query_flat)
        B = F.softmax(torch.transpose(A,1,2),dim=1)
       
        exemplar_ = exemplar.view(N1, C1, H1 * W1)
        query_ = query.view(N1, C1, H1 * W1)

        exemplar_att = torch.bmm(query_, B).contiguous()    
        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  
        input1_mask = self.extra_gate(input1_att)
        input1_mask = self.extra_gate_s(input1_mask)
        input1_att = input1_att * input1_mask
        return input1_att
 
    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups

def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=True)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model
   
def load_pretrained_model(model, path=None):
    state_dict = model_zoo.load_url(model_urls['vgg16'], progress=True)
    state_dict = remove_layer(state_dict, 'classifier.')
    state_dict = adjust_pretrained_model(state_dict, model)
    model.load_state_dict(state_dict, strict=False)
    return model

def make_layers(cfg, batch_norm=False,**kwargs):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)            
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'D2':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def GNNNet_VGG(pretrained=True, **kwargs):
    model = CoattentionModel(make_layers(cfg['D1'],**kwargs),**kwargs)  
    if pretrained:
         model = load_pretrained_model(model,path=None)
    return model
