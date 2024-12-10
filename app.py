import io
import json
import os
import numpy as np
# import pandas as pd
from torchvision import models
import torchvision.transforms as transforms
# import _imaging
from PIL import Image
from flask import Flask, jsonify, request, render_template, request, redirect
import torch

import base64
import io
from werkzeug.utils import secure_filename

import torch.nn as nn
import torch.nn.functional as F  
from flask_sqlalchemy import SQLAlchemy

from sqlalchemy.sql import func
# [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],

import torch.nn as nn
from os import environ
app = Flask(__name__,template_folder='template')
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'database.db')

# from mmcv.cnn import constant_init, kaiming_init
# def last_zero_init(m):
#     if isinstance(m, nn.Sequential):
#         constant_init(m[-1], val=0)
#     else:
#         constant_init(m, val=0)


# class ContextBlock(nn.Module):

#     def __init__(self,
#                  inplanes,
#                  ratio,
#                  pooling_type='att',
#                  fusion_types=('channel_add', )):
#         super(ContextBlock, self).__init__()
#         assert pooling_type in ['avg', 'att']
#         assert isinstance(fusion_types, (list, tuple))
#         valid_fusion_types = ['channel_add', 'channel_mul']
#         assert all([f in valid_fusion_types for f in fusion_types])
#         assert len(fusion_types) > 0, 'at least one fusion should be used'
#         self.inplanes = inplanes
#         self.ratio = ratio
#         self.planes = int(inplanes * ratio)
#         self.pooling_type = pooling_type
#         self.fusion_types = fusion_types
#         if pooling_type == 'att':
#             self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
#             self.softmax = nn.Softmax(dim=2)
#         else:
#             self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         if 'channel_add' in fusion_types:
#             self.channel_add_conv = nn.Sequential(
#                 nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
#                 nn.LayerNorm([self.planes, 1, 1]),
#                 nn.ReLU(inplace=True),  # yapf: disable
#                 nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
#         else:
#             self.channel_add_conv = None
#         if 'channel_mul' in fusion_types:
#             self.channel_mul_conv = nn.Sequential(
#                 nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
#                 nn.LayerNorm([self.planes, 1, 1]),
#                 nn.ReLU(inplace=True),  # yapf: disable
#                 nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
#         else:
#             self.channel_mul_conv = None
#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.pooling_type == 'att':
#             kaiming_init(self.conv_mask, mode='fan_in')
#             self.conv_mask.inited = True

#         if self.channel_add_conv is not None:
#             last_zero_init(self.channel_add_conv)
#         if self.channel_mul_conv is not None:
#             last_zero_init(self.channel_mul_conv)

#     def spatial_pool(self, x):
#         batch, channel, height, width = x.size()
#         if self.pooling_type == 'att':
#             input_x = x
#             # [N, C, H * W]
#             input_x = input_x.view(batch, channel, height * width)
#             # [N, 1, C, H * W]
#             input_x = input_x.unsqueeze(1)
#             # [N, 1, H, W]
#             context_mask = self.conv_mask(x)
#             # [N, 1, H * W]
#             context_mask = context_mask.view(batch, 1, height * width)
#             # [N, 1, H * W]
#             context_mask = self.softmax(context_mask)
#             # [N, 1, H * W, 1]
#             context_mask = context_mask.unsqueeze(-1)
#             # [N, 1, C, 1]
#             context = torch.matmul(input_x, context_mask)
#             # [N, C, 1, 1]
#             context = context.view(batch, channel, 1, 1)
#         else:
#             # [N, C, 1, 1]
#             context = self.avg_pool(x)

#         return context

#     def forward(self, x):
#         # [N, C, 1, 1]
#         context = self.spatial_pool(x)

#         out = x
#         if self.channel_mul_conv is not None:
#             # [N, C, 1, 1]
#             channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
#             out = out * channel_mul_term
#         if self.channel_add_conv is not None:
#             # [N, C, 1, 1]
#             channel_add_term = self.channel_add_conv(context)
#             out = out + channel_add_term

#         return out
    
    
# import torch.nn as nn
# import torch.nn.functional as F

# import math
# import numpy as np
# from mmcv.cnn import kaiming_init


# class GeneralizedAttention(nn.Module):
#     """GeneralizedAttention module.

#     See 'An Empirical Study of Spatial Attention Mechanisms in Deep Networks'
#     (https://arxiv.org/abs/1711.07971) for details.

#     Args:
#         in_dim (int): Channels of the input feature map.
#         spatial_range (int): The spatial range.
#             -1 indicates no spatial range constraint.
#         num_heads (int): The head number of empirical_attention module.
#         position_embedding_dim (int): The position embedding dimension.
#         position_magnitude (int): A multiplier acting on coord difference.
#         kv_stride (int): The feature stride acting on key/value feature map.
#         q_stride (int): The feature stride acting on query feature map.
#         attention_type (str): A binary indicator string for indicating which
#             items in generalized empirical_attention module are used.
#             '1000' indicates 'query and key content' (appr - appr) item,
#             '0100' indicates 'query content and relative position'
#               (appr - position) item,
#             '0010' indicates 'key content only' (bias - appr) item,
#             '0001' indicates 'relative position only' (bias - position) item.
#     """

#     def __init__(self,
#                  in_dim,
#                  spatial_range=-1,
#                  num_heads=9,
#                  position_embedding_dim=-1,
#                  position_magnitude=1,
#                  kv_stride=2,
#                  q_stride=1,
#                  attention_type='0100'):

#         super(GeneralizedAttention, self).__init__()

#         # hard range means local range for non-local operation
#         self.position_embedding_dim = (
#             position_embedding_dim if position_embedding_dim > 0 else in_dim)

#         self.position_magnitude = position_magnitude
#         self.num_heads = num_heads
#         self.channel_in = in_dim
#         self.spatial_range = spatial_range
#         self.kv_stride = kv_stride
#         self.q_stride = q_stride
#         self.attention_type = [bool(int(_)) for _ in attention_type]
#         self.qk_embed_dim = in_dim // num_heads
#         out_c = self.qk_embed_dim * num_heads

#         if self.attention_type[0] or self.attention_type[1]:
#             self.query_conv = nn.Conv2d(
#                 in_channels=in_dim,
#                 out_channels=out_c,
#                 kernel_size=1,
#                 bias=False)
#             self.query_conv.kaiming_init = True

#         if self.attention_type[0] or self.attention_type[2]:
#             self.key_conv = nn.Conv2d(
#                 in_channels=in_dim,
#                 out_channels=out_c,
#                 kernel_size=1,
#                 bias=False)
#             self.key_conv.kaiming_init = True

#         self.v_dim = in_dim // num_heads
#         self.value_conv = nn.Conv2d(
#             in_channels=in_dim,
#             out_channels=self.v_dim * num_heads,
#             kernel_size=1,
#             bias=False)
#         self.value_conv.kaiming_init = True

#         if self.attention_type[1] or self.attention_type[3]:
#             self.appr_geom_fc_x = nn.Linear(
#                 self.position_embedding_dim // 2, out_c, bias=False)
#             self.appr_geom_fc_x.kaiming_init = True

#             self.appr_geom_fc_y = nn.Linear(
#                 self.position_embedding_dim // 2, out_c, bias=False)
#             self.appr_geom_fc_y.kaiming_init = True

#         if self.attention_type[2]:
#             stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
#             appr_bias_value = -2 * stdv * torch.rand(out_c) + stdv
#             self.appr_bias = nn.Parameter(appr_bias_value)

#         if self.attention_type[3]:
#             stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
#             geom_bias_value = -2 * stdv * torch.rand(out_c) + stdv
#             self.geom_bias = nn.Parameter(geom_bias_value)

#         self.proj_conv = nn.Conv2d(
#             in_channels=self.v_dim * num_heads,
#             out_channels=in_dim,
#             kernel_size=1,
#             bias=True)
#         self.proj_conv.kaiming_init = True
#         self.gamma = nn.Parameter(torch.zeros(1))

#         if self.spatial_range >= 0:
#             # only works when non local is after 3*3 conv
#             if in_dim == 256:
#                 max_len = 84
#             elif in_dim == 512:
#                 max_len = 42

#             max_len_kv = int((max_len - 1.0) / self.kv_stride + 1)
#             local_constraint_map = np.ones(
#                 (max_len, max_len, max_len_kv, max_len_kv), dtype=np.int)
#             for iy in range(max_len):
#                 for ix in range(max_len):
#                     local_constraint_map[iy, ix,
#                                          max((iy - self.spatial_range) //
#                                              self.kv_stride, 0):min(
#                                                  (iy + self.spatial_range +
#                                                   1) // self.kv_stride +
#                                                  1, max_len),
#                                          max((ix - self.spatial_range) //
#                                              self.kv_stride, 0):min(
#                                                  (ix + self.spatial_range +
#                                                   1) // self.kv_stride +
#                                                  1, max_len)] = 0

#             self.local_constraint_map = nn.Parameter(
#                 torch.from_numpy(local_constraint_map).byte(),
#                 requires_grad=False)

#         if self.q_stride > 1:
#             self.q_downsample = nn.AvgPool2d(
#                 kernel_size=1, stride=self.q_stride)
#         else:
#             self.q_downsample = None

#         if self.kv_stride > 1:
#             self.kv_downsample = nn.AvgPool2d(
#                 kernel_size=1, stride=self.kv_stride)
#         else:
#             self.kv_downsample = None

#         self.init_weights()

#     def get_position_embedding(self,
#                                h,
#                                w,
#                                h_kv,
#                                w_kv,
#                                q_stride,
#                                kv_stride,
#                                device,
#                                feat_dim,
#                                wave_length=1000):
#         h_idxs = torch.linspace(0, h - 1, h)
#         h_idxs = h_idxs.view((h, 1)) * q_stride

#         w_idxs = torch.linspace(0, w - 1, w)
#         w_idxs = w_idxs.view((w, 1)) * q_stride

#         h_kv_idxs = torch.linspace(0, h_kv - 1, h_kv)
#         h_kv_idxs = h_kv_idxs.view((h_kv, 1)) * kv_stride

#         w_kv_idxs = torch.linspace(0, w_kv - 1, w_kv)
#         w_kv_idxs = w_kv_idxs.view((w_kv, 1)) * kv_stride

#         # (h, h_kv, 1)
#         h_diff = h_idxs.unsqueeze(1) - h_kv_idxs.unsqueeze(0)
#         h_diff *= self.position_magnitude

#         # (w, w_kv, 1)
#         w_diff = w_idxs.unsqueeze(1) - w_kv_idxs.unsqueeze(0)
#         w_diff *= self.position_magnitude

#         feat_range = torch.arange(0, feat_dim / 4)
#         dim_mat = torch.Tensor([wave_length])
#         dim_mat = dim_mat**((4. / feat_dim) * feat_range)
#         dim_mat = dim_mat.view((1, 1, -1))

#         embedding_x = torch.cat(
#             ((w_diff / dim_mat).sin(), (w_diff / dim_mat).cos()), dim=2)

#         embedding_y = torch.cat(
#             ((h_diff / dim_mat).sin(), (h_diff / dim_mat).cos()), dim=2)

#         return embedding_x, embedding_y

#     def forward(self, x_input):
#         num_heads = self.num_heads

#         # use empirical_attention
#         if self.q_downsample is not None:
#             x_q = self.q_downsample(x_input)
#         else:
#             x_q = x_input
#         n, _, h, w = x_q.shape

#         if self.kv_downsample is not None:
#             x_kv = self.kv_downsample(x_input)
#         else:
#             x_kv = x_input
#         _, _, h_kv, w_kv = x_kv.shape

#         if self.attention_type[0] or self.attention_type[1]:
#             proj_query = self.query_conv(x_q).view(
#                 (n, num_heads, self.qk_embed_dim, h * w))
#             proj_query = proj_query.permute(0, 1, 3, 2)

#         if self.attention_type[0] or self.attention_type[2]:
#             proj_key = self.key_conv(x_kv).view(
#                 (n, num_heads, self.qk_embed_dim, h_kv * w_kv))

#         if self.attention_type[1] or self.attention_type[3]:
#             position_embed_x, position_embed_y = self.get_position_embedding(
#                 h, w, h_kv, w_kv, self.q_stride, self.kv_stride,
#                 x_input.device, self.position_embedding_dim)
#             # (n, num_heads, w, w_kv, dim)
#             position_feat_x = self.appr_geom_fc_x(position_embed_x).\
#                 view(1, w, w_kv, num_heads, self.qk_embed_dim).\
#                 permute(0, 3, 1, 2, 4).\
#                 repeat(n, 1, 1, 1, 1)

#             # (n, num_heads, h, h_kv, dim)
#             position_feat_y = self.appr_geom_fc_y(position_embed_y).\
#                 view(1, h, h_kv, num_heads, self.qk_embed_dim).\
#                 permute(0, 3, 1, 2, 4).\
#                 repeat(n, 1, 1, 1, 1)

#             position_feat_x /= math.sqrt(2)
#             position_feat_y /= math.sqrt(2)

#         # accelerate for saliency only
#         if (np.sum(self.attention_type) == 1) and self.attention_type[2]:
#             appr_bias = self.appr_bias.\
#                 view(1, num_heads, 1, self.qk_embed_dim).\
#                 repeat(n, 1, 1, 1)

#             energy = torch.matmul(appr_bias, proj_key).\
#                 view(n, num_heads, 1, h_kv * w_kv)

#             h = 1
#             w = 1
#         else:
#             # (n, num_heads, h*w, h_kv*w_kv), query before key, 540mb for
#             if not self.attention_type[0]:
#                 energy = torch.zeros(
#                     n,
#                     num_heads,
#                     h,
#                     w,
#                     h_kv,
#                     w_kv,
#                     dtype=x_input.dtype,
#                     device=x_input.device)

#             # attention_type[0]: appr - appr
#             # attention_type[1]: appr - position
#             # attention_type[2]: bias - appr
#             # attention_type[3]: bias - position
#             if self.attention_type[0] or self.attention_type[2]:
#                 if self.attention_type[0] and self.attention_type[2]:
#                     appr_bias = self.appr_bias.\
#                         view(1, num_heads, 1, self.qk_embed_dim)
#                     energy = torch.matmul(proj_query + appr_bias, proj_key).\
#                         view(n, num_heads, h, w, h_kv, w_kv)

#                 elif self.attention_type[0]:
#                     energy = torch.matmul(proj_query, proj_key).\
#                         view(n, num_heads, h, w, h_kv, w_kv)

#                 elif self.attention_type[2]:
#                     appr_bias = self.appr_bias.\
#                         view(1, num_heads, 1, self.qk_embed_dim).\
#                         repeat(n, 1, 1, 1)

#                     energy += torch.matmul(appr_bias, proj_key).\
#                         view(n, num_heads, 1, 1, h_kv, w_kv)

#             if self.attention_type[1] or self.attention_type[3]:
#                 if self.attention_type[1] and self.attention_type[3]:
#                     geom_bias = self.geom_bias.\
#                         view(1, num_heads, 1, self.qk_embed_dim)

#                     proj_query_reshape = (proj_query + geom_bias).\
#                         view(n, num_heads, h, w, self.qk_embed_dim)

#                     energy_x = torch.matmul(
#                         proj_query_reshape.permute(0, 1, 3, 2, 4),
#                         position_feat_x.permute(0, 1, 2, 4, 3))
#                     energy_x = energy_x.\
#                         permute(0, 1, 3, 2, 4).unsqueeze(4)

#                     energy_y = torch.matmul(
#                         proj_query_reshape,
#                         position_feat_y.permute(0, 1, 2, 4, 3))
#                     energy_y = energy_y.unsqueeze(5)

#                     energy += energy_x + energy_y

#                 elif self.attention_type[1]:
#                     proj_query_reshape = proj_query.\
#                         view(n, num_heads, h, w, self.qk_embed_dim)
#                     proj_query_reshape = proj_query_reshape.\
#                         permute(0, 1, 3, 2, 4)
#                     position_feat_x_reshape = position_feat_x.\
#                         permute(0, 1, 2, 4, 3)
#                     position_feat_y_reshape = position_feat_y.\
#                         permute(0, 1, 2, 4, 3)

#                     energy_x = torch.matmul(proj_query_reshape,
#                                             position_feat_x_reshape)
#                     energy_x = energy_x.permute(0, 1, 3, 2, 4).unsqueeze(4)

#                     energy_y = torch.matmul(proj_query_reshape,
#                                             position_feat_y_reshape)
#                     energy_y = energy_y.unsqueeze(5)

#                     energy += energy_x + energy_y

#                 elif self.attention_type[3]:
#                     geom_bias = self.geom_bias.\
#                         view(1, num_heads, self.qk_embed_dim, 1).\
#                         repeat(n, 1, 1, 1)

#                     position_feat_x_reshape = position_feat_x.\
#                         view(n, num_heads, w*w_kv, self.qk_embed_dim)

#                     position_feat_y_reshape = position_feat_y.\
#                         view(n, num_heads, h * h_kv, self.qk_embed_dim)

#                     energy_x = torch.matmul(position_feat_x_reshape, geom_bias)
#                     energy_x = energy_x.view(n, num_heads, 1, w, 1, w_kv)

#                     energy_y = torch.matmul(position_feat_y_reshape, geom_bias)
#                     energy_y = energy_y.view(n, num_heads, h, 1, h_kv, 1)

#                     energy += energy_x + energy_y

#             energy = energy.view(n, num_heads, h * w, h_kv * w_kv)

#         if self.spatial_range >= 0:
#             cur_local_constraint_map = \
#                 self.local_constraint_map[:h, :w, :h_kv, :w_kv].\
#                 contiguous().\
#                 view(1, 1, h*w, h_kv*w_kv)

#             energy = energy.masked_fill_(cur_local_constraint_map,
#                                          float('-inf'))

#         attention = F.softmax(energy, 3)

#         proj_value = self.value_conv(x_kv)
#         proj_value_reshape = proj_value.\
#             view((n, num_heads, self.v_dim, h_kv * w_kv)).\
#             permute(0, 1, 3, 2)

#         out = torch.matmul(attention, proj_value_reshape).\
#             permute(0, 1, 3, 2).\
#             contiguous().\
#             view(n, self.v_dim * self.num_heads, h, w)

#         out = self.proj_conv(out)
#         out = self.gamma * out + x_input
#         return out

#     def init_weights(self):
#         for m in self.modules():
#             if hasattr(m, 'kaiming_init') and m.kaiming_init:
#                 kaiming_init(
#                     m,
#                     mode='fan_in',
#                     nonlinearity='leaky_relu',
#                     bias=0,
#                     distribution='uniform',
#                     a=1)
                
                
# class SELayer(nn.Module):
#     def __init__(self,channel, reduction = 8):
#         super(SELayer, self).__init__()
# #         self.channel=channel
#         self.channel=channel
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc       = nn.Sequential(
#                         nn.Linear(channel, channel//reduction),
#                         nn.ReLU(inplace = True),
#                         nn.Linear(channel//reduction, channel),
#                         nn.Sigmoid()
#                 )

#     def forward(self, x):
#         b, c, _, _ = x.size()
# #         print("x size ",x.size())
#         y = self.avg_pool(x).view(b, c)
# #         print("y size ",y.size())
#         y = self.fc(y).view(b, c, 1, 1)
# #         y = self.fc(y)
# #         print("y2 size ",y.size())
#         return x * y

# class VGG16_net(nn.Module):
#     def __init__(self, in_channels=3, n_classes=4):
#         super().__init__()
#         self.in_channels = in_channels
#         self.conv1 = self.conv_block(in_channels=self.in_channels, block=[64])
#         self.conv2 = self.conv_block(in_channels=64, block=[128])
#         self.conv3 = self.conv_block(in_channels=128, block=[256, 256])
#         self.conv4 = self.conv_block(in_channels=256, block=[512, 512])
#         self.conv5 = self.conv_block(in_channels=512, block=[512, 512])
#         self.conv6 = self.conv_block2(in_channels=512, block=[512])
#         self.conv7 = self.conv_block3(in_channels=512, block=[512])
#         self.se=SELayer(512*3)
#         self.ga=GeneralizedAttention(512)
# #         self.conv8 = self.conv_block3(in_channels=512, block=[512])
        
# #         self.fc1=nn.Linear(512*7*7*3, 4096)
# #         self.skip1=nn.Conv2d(inp, out,kernel_size=1, stride=stride, bias=False),
# #         nn.BatchNorm2d(out),
# #         self.fc1=nn.Linear(6272, 4096)
# #         self.fc2=nn.Linear(4096, 4096)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.relu=nn.ReLU()
#         self.drop=nn.Dropout(0.5)
#         self.fcs = nn.Sequential(
#             nn.Flatten(start_dim=1),
# #             nn.Linear(6272, 4096),
#             nn.Linear(1536, 4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, n_classes)
#         )
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x3 = self.conv3(x)
#         x4 = self.conv4(x3)


#         x5 = self.conv5(x4)
#         x5=self.ga(x5)
#         x6 = self.conv6(x4)
#         x6=self.ga(x6)
#         x7 = self.conv7(x4)
#         x7=self.ga(x7)

#         x4=self.avgpool(x4)
#         x5=self.avgpool(x5)
#         x6=self.avgpool(x6)
#         x7=self.avgpool(x7)
#         x5+=x4
#         x6+=x4
#         x7+=x4
#         x=torch.cat((x5,x6,x7),dim=1)

#         x=self.se(x)

#         x = self.fcs(x)

#         return x

#     def conv_block2(self, in_channels ,block):
#         layers = []
#         for i in block:
            
#             layers += [nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=(5,5), stride=(1,1), padding=(1,1)),
# #                 nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=(5,5), stride=(1,1), padding=(1,1)),
# #                        nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=1),
#                        nn.BatchNorm2d(i),
#                        nn.ReLU()]
#             in_channels = i
#         layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
#         return nn.Sequential(*layers)
#     def conv_block3 (self, in_channels ,block):
#         layers = []
#         for i in block:
#             layers += [nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=(7,7), stride=(1,1), padding=(1,1)),
# #                        nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=1),
#                        nn.BatchNorm2d(i),
#                        nn.ReLU()]
#             in_channels = i
#         layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
#         return nn.Sequential(*layers)
    
#     def conv_block(self, in_channels ,block):
#         layers = []
#         for i in block:
#             layers += [nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
#                        nn.BatchNorm2d(i),
#                        nn.ReLU()]
#             in_channels = i
#         layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
#         return nn.Sequential(*layers)

import timm
# model5= VggNet()
model5= timm.create_model('vgg16_bn', pretrained=True,num_classes=4)  


device = torch.device('cpu')

# app = Flask(__name__,template_folder='template')
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dbdir/test.db'
# basedir = os.path.abspath(os.path.dirname(__file__))
# app.config["UPLOAD_PATH"] = "image_uploads"
app.config["UPLOAD_EXTENSIONS"] = [".jpg", ".png",".Jpeg"]

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
imagenet_class_index = json.load(open('index.json'))
model5.load_state_dict(torch.load('./model5cpu.pt',weights_only=True,map_location=device))


model5.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model5.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name.upper()
# @app.route("/favicon.ico")
# def favicon():
#     return url_for('static', filename='data:,')
# @app.route("/success", methods = ["POST", "GET"])

# @app.route("/success/<img><pred>", methods = ["GET","POST"])
# def success(img,pred):
#     print("here")
#     if request.method == 'POST':
#         text = request.form.get('text')
#         processed_text = text.upper()
#         print(processed_text)
#         imager = Funduspred(img=img,modelpred=pred,actualpred=processed_text)
#         db.session.add(imager)
#         db.session.commit()
#         return render_template('success.html')
#     return render_template('index.html')
  
  
@app.route('/', methods=['GET','POST'])
def predict():
   
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return
        
        filename = secure_filename(file.filename)
        print(filename)
        
       
    
       
        img_bytes = file.read()
        
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        encoded_string = base64.b64encode(img_bytes).decode()
        text = request.form.get('text')
        processed_text = text.upper()
        imager = Funduspred(img=img_bytes,modelpred=class_name,actualpred=processed_text)
        db.session.add(imager)
        db.session.commit()
       
        return render_template('result.html', class_id=class_id,
                               class_name=class_name, pred=processed_text, img_data=encoded_string)
        
    return render_template('index.html')
        # return jsonify({'class_id':
        #  class_id, 'class_name': class_name})

class Funduspred(db.Model):
    column_not_exist_in_db = db.Column(db.Integer, primary_key=True)
    img = db.Column(db.LargeBinary, nullable=False)
    modelpred = db.Column(db.String(100))
    actualpred = db.Column(db.String(100), nullable=False)
     
    # created_at = db.Column(db.DateTime(timezone=True),
    #                        server_default=func.now())
  
    def __repr__(self):
        return f'<Funduspred {self.img}>'
from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy import inspect
if __name__ == '__main__':
    app.run(debug=False)
    
 
    # engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
    # inspector = inspect(engine)
    # for table_name in inspector.get_table_names():
    #     for column in inspector.get_columns(table_name):
    #         print("Column: %s" % column['name'])

#     import pandas as pd
# import sqlite3
# import sqlalchemy 

# try:
#     conn = sqlite3.connect("file.db")    
# except Exception as e:
#     print(e)

# #Now in order to read in pandas dataframe we need to know table name
# cursor = conn.cursor()
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(f"Table Name : {cursor.fetchall()}")

# df = pd.read_sql_query('SELECT * FROM Table_Name', conn)
# conn.close()